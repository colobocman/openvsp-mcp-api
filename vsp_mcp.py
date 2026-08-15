#!/usr/bin/env python3
"""OpenVSP MCP server.

Exposes the official OpenVSP Python API (module ``openvsp``, v3.51.2) over the
Model Context Protocol on stdio.

The server holds one long-lived OpenVSP model in memory, so tool calls compose:
add a wing, tweak its parameters, run VSPAERO, export a mesh. All API access is
serialised behind a lock because the OpenVSP API is not thread-safe.
"""

from __future__ import annotations

import io
import contextlib
import threading
from typing import Any

import openvsp as vsp
from mcp.server.mcpserver import MCPServer

server = MCPServer("openvsp", version="0.1.0")

_LOCK = threading.RLock()
_ERR = vsp.ErrorMgrSingleton.getInstance()
_ERR.SilenceErrors()

# Arrays in analysis results can be enormous (per-node pressures etc.).
_MAX_ITEMS = 200


# --------------------------------------------------------------------------
# error plumbing
# --------------------------------------------------------------------------

def _drain_errors() -> list[str]:
    out = []
    while _ERR.GetNumTotalErrors() > 0:
        out.append(_ERR.PopLastError().GetErrorString())
    return out


@contextlib.contextmanager
def _vsp_call(strict: bool = True):
    """Serialise API access and turn OpenVSP's error stack into exceptions."""
    with _LOCK:
        _drain_errors()  # discard anything left over from a previous call
        yield
        errs = _drain_errors()
        if errs and strict:
            raise RuntimeError("OpenVSP error: " + "; ".join(errs))


@contextlib.contextmanager
def _mesh_cleanup(keep: bool = False):
    """Delete the MeshGeom components OpenVSP leaves behind.

    Meshing operations (CompGeom, MassProp, STL export, ...) add a MeshGeom to
    the model as a side effect. Left in place they accumulate and get pulled
    into every later analysis and export, which silently corrupts results.
    """
    before = set(vsp.FindGeoms())
    try:
        yield
    finally:
        if not keep:
            # Type names, not user names: "Mesh" from CompGeom/export,
            # "NGonMesh" from VSPAEROComputeGeometry.
            stray = [
                g
                for g in vsp.FindGeoms()
                if g not in before and "Mesh" in vsp.GetGeomTypeName(g)
            ]
            for g in stray:
                vsp.DeleteGeom(g)
            if stray:
                vsp.Update()


def _truncate(seq: list) -> Any:
    if len(seq) > _MAX_ITEMS:
        return {"truncated": True, "total": len(seq), "values": seq[:_MAX_ITEMS]}
    return seq


# --------------------------------------------------------------------------
# model lifecycle
# --------------------------------------------------------------------------

@server.tool()
def vsp_info() -> dict:
    """Report OpenVSP version, VSPAERO path, and a summary of the in-memory model."""
    with _vsp_call():
        geoms = list(vsp.FindGeoms())
        return {
            "version": vsp.GetVSPVersion(),
            "vsp_file": vsp.GetVSPFileName(),
            "vspaero_path": vsp.GetVSPAEROPath(),
            "geom_count": len(geoms),
            "geom_ids": geoms,
            "available_geom_types": list(vsp.GetGeomTypes()),
        }


@server.tool()
def vsp_new_model() -> dict:
    """Discard the in-memory model and start a fresh, empty one."""
    with _vsp_call():
        vsp.VSPRenew()
        vsp.Update()
        return {"ok": True, "geom_count": len(vsp.FindGeoms())}


@server.tool()
def vsp_open_model(path: str) -> dict:
    """Load a .vsp3 file, replacing the in-memory model.

    Args:
        path: absolute path to a .vsp3 file.
    """
    with _vsp_call():
        vsp.ClearVSPModel()
        vsp.ReadVSPFile(path)
        vsp.Update()
        return {"ok": True, "path": path, "geoms": _list_geoms()}


@server.tool()
def vsp_save_model(path: str, set_index: int = 0) -> dict:
    """Write the in-memory model to a .vsp3 file.

    Args:
        path: absolute destination path.
        set_index: geometry set to write; 0 = SET_ALL (default), 1 = SET_SHOWN.
    """
    with _vsp_call():
        vsp.SetVSP3FileName(path)
        vsp.Update()
        vsp.WriteVSPFile(path, set_index)
        return {"ok": True, "path": path}


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def _list_geoms() -> list[dict]:
    out = []
    for gid in vsp.FindGeoms():
        out.append(
            {
                "id": gid,
                "name": vsp.GetGeomName(gid),
                "type": vsp.GetGeomTypeName(gid),
                "parent": vsp.GetGeomParent(gid),
                "children": list(vsp.GetGeomChildren(gid)),
            }
        )
    return out


@server.tool()
def vsp_list_geoms() -> list[dict]:
    """List every component in the model with its id, name, type, and hierarchy."""
    with _vsp_call():
        return _list_geoms()


@server.tool()
def vsp_add_geom(geom_type: str, name: str = "", parent_id: str = "") -> dict:
    """Add a component to the model.

    Args:
        geom_type: one of WING, FUSELAGE, POD, STACK, BLANK, ELLIPSOID,
            BODYOFREVOLUTION, HUMAN, PROP, GEAR, HINGE, CONFORMAL, ROUTING,
            AUXILIARY, COBRA.
        name: optional display name for the new component.
        parent_id: optional geom id to attach the new component to.
    """
    with _vsp_call():
        gid = vsp.AddGeom(geom_type, parent_id) if parent_id else vsp.AddGeom(geom_type)
        if name:
            vsp.SetGeomName(gid, name)
        vsp.Update()
        return {
            "id": gid,
            "name": vsp.GetGeomName(gid),
            "type": vsp.GetGeomTypeName(gid),
            "parm_count": len(vsp.GetGeomParmIDs(gid)),
        }


@server.tool()
def vsp_delete_geom(geom_id: str) -> dict:
    """Delete a component from the model."""
    with _vsp_call():
        vsp.DeleteGeom(geom_id)
        vsp.Update()
        return {"ok": True, "remaining": len(vsp.FindGeoms())}


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------

_PARM_TYPES = {
    vsp.PARM_DOUBLE_TYPE: "double",
    vsp.PARM_INT_TYPE: "int",
    vsp.PARM_BOOL_TYPE: "bool",
    vsp.PARM_FRACTION_TYPE: "fraction",
    vsp.PARM_LIMITED_INT_TYPE: "limited_int",
    vsp.PARM_NOTEQ_TYPE: "not_equal",
    vsp.PARM_POWER_INT_TYPE: "power_int",
}


def _parm_detail(pid: str) -> dict:
    return {
        "name": vsp.GetParmName(pid),
        "group": vsp.GetParmDisplayGroupName(pid),
        "value": vsp.GetParmVal(pid),
        "type": _PARM_TYPES.get(vsp.GetParmType(pid), "unknown"),
        "min": vsp.GetParmLowerLimit(pid),
        "max": vsp.GetParmUpperLimit(pid),
        "description": vsp.GetParmDescript(pid),
    }


@server.tool()
def vsp_list_parms(geom_id: str, group: str = "", name_contains: str = "") -> dict:
    """Browse a component's parameters.

    A component typically has hundreds of parameters, so with no filter this
    returns only the group names and the parameter names inside them. Pass
    `group` to get full detail (value, type, limits, description) for one group.

    Args:
        geom_id: component id from vsp_list_geoms.
        group: parameter group to expand in full, e.g. "WingGeom", "XForm".
        name_contains: case-insensitive substring filter on parameter names.
    """
    with _vsp_call():
        pids = vsp.GetGeomParmIDs(geom_id)
        needle = name_contains.lower()

        if group:
            detail = [
                _parm_detail(p)
                for p in pids
                if vsp.GetParmDisplayGroupName(p) == group
                and (not needle or needle in vsp.GetParmName(p).lower())
            ]
            return {"geom_id": geom_id, "group": group, "parms": detail}

        groups: dict[str, list[str]] = {}
        for p in pids:
            pname = vsp.GetParmName(p)
            if needle and needle not in pname.lower():
                continue
            groups.setdefault(vsp.GetParmDisplayGroupName(p), []).append(pname)

        return {
            "geom_id": geom_id,
            "total_parms": len(pids),
            "groups": {g: sorted(n) for g, n in sorted(groups.items())},
            "hint": "call again with group=<name> for values, limits and descriptions",
        }


@server.tool()
def vsp_get_parms(geom_id: str, parms: list[dict]) -> list[dict]:
    """Read specific parameters.

    Args:
        geom_id: component id.
        parms: list of {"group": ..., "name": ...} entries.
    """
    with _vsp_call():
        out = []
        for spec in parms:
            pid = vsp.GetParm(geom_id, spec["name"], spec["group"])
            out.append(_parm_detail(pid))
        return out


@server.tool()
def vsp_set_parms(geom_id: str, parms: list[dict]) -> list[dict]:
    """Set parameters and update the model.

    OpenVSP clamps values to each parameter's limits, so the returned values are
    what was actually applied — check them rather than assuming.

    Args:
        geom_id: component id.
        parms: list of {"group": ..., "name": ..., "value": ...} entries.
    """
    with _vsp_call():
        applied = []
        for spec in parms:
            pid = vsp.GetParm(geom_id, spec["name"], spec["group"])
            vsp.SetParmValUpdate(pid, float(spec["value"]))
            applied.append(pid)
        vsp.Update()
        return [_parm_detail(p) for p in applied]


# --------------------------------------------------------------------------
# sets
# --------------------------------------------------------------------------

# VSPAERO solves these as a vortex lattice; everything else gets panels.
_THIN_TYPES = {"Wing", "Prop"}
# Carry no surface, so they belong in neither aero set.
_NON_SURFACE_TYPES = {"Blank", "Hinge", "Routing"}


@server.tool()
def vsp_list_sets() -> list[dict]:
    """List the geometry sets and which components belong to each.

    Sets are how OpenVSP scopes an analysis to part of the model. Indices 0-2
    are the built-ins All / Shown / Not_Shown; the rest are user sets.
    """
    with _vsp_call():
        return [
            {
                "index": i,
                "name": vsp.GetSetName(i),
                "geoms": [
                    vsp.GetGeomName(g) for g in vsp.FindGeoms() if vsp.GetSetFlag(g, i)
                ],
            }
            for i in range(vsp.GetNumSets())
        ]


@server.tool()
def vsp_assign_set(geom_id: str, set_index: int, member: bool = True, set_name: str = "") -> dict:
    """Add or remove a component from a geometry set.

    Args:
        geom_id: component id.
        set_index: set to modify; use 3 or higher for user sets.
        member: True to add, False to remove.
        set_name: optionally rename the set at the same time.
    """
    with _vsp_call():
        if set_name:
            vsp.SetSetName(set_index, set_name)
        vsp.SetSetFlag(geom_id, set_index, member)
        return {
            "set": vsp.GetSetName(set_index),
            "index": set_index,
            "geoms": [vsp.GetGeomName(g) for g in vsp.FindGeoms() if vsp.GetSetFlag(g, set_index)],
        }


def _auto_aero_sets() -> tuple[int, int, dict]:
    """Split the model into VSPAERO's thin and thick surface sets.

    Wings and props are solved as a vortex lattice, bodies with panels. A
    component in both sets duplicates the geometry and aborts the solver, so
    each is assigned to exactly one. Uses the last two user sets, renaming them
    VSPAERO_Thin and VSPAERO_Thick.
    """
    n = vsp.GetNumSets()
    thin_i, thick_i = n - 2, n - 1
    vsp.SetSetName(thin_i, "VSPAERO_Thin")
    vsp.SetSetName(thick_i, "VSPAERO_Thick")

    split: dict[str, list[str]] = {"thin": [], "thick": [], "skipped": []}
    for g in vsp.FindGeoms():
        t = vsp.GetGeomTypeName(g)
        name = vsp.GetGeomName(g)
        if "Mesh" in t or t in _NON_SURFACE_TYPES:
            vsp.SetSetFlag(g, thin_i, False)
            vsp.SetSetFlag(g, thick_i, False)
            split["skipped"].append(name)
            continue
        is_thin = t in _THIN_TYPES
        vsp.SetSetFlag(g, thin_i, is_thin)
        vsp.SetSetFlag(g, thick_i, not is_thin)
        split["thin" if is_thin else "thick"].append(name)
    return thin_i, thick_i, split


# --------------------------------------------------------------------------
# analyses
# --------------------------------------------------------------------------

@server.tool()
def vsp_list_analyses() -> list[str]:
    """List the analyses OpenVSP can run (MassProp, CompGeom, VSPAEROSweep, ...)."""
    with _vsp_call():
        return sorted(vsp.ListAnalysis())


@server.tool()
def vsp_describe_analysis(name: str) -> dict:
    """Show an analysis's inputs with their types and current default values.

    Args:
        name: analysis name from vsp_list_analyses.
    """
    with _vsp_call():
        vsp.SetAnalysisInputDefaults(name)
        inputs = []
        for key in vsp.GetAnalysisInputNames(name):
            t = vsp.GetAnalysisInputType(name, key)
            if t == vsp.INT_DATA:
                val, tname = list(vsp.GetIntAnalysisInput(name, key)), "int"
            elif t == vsp.DOUBLE_DATA:
                val, tname = list(vsp.GetDoubleAnalysisInput(name, key)), "double"
            elif t == vsp.STRING_DATA:
                val, tname = list(vsp.GetStringAnalysisInput(name, key)), "string"
            else:
                val, tname = None, str(t)
            inputs.append(
                {
                    "name": key,
                    "type": tname,
                    "default": _truncate(val) if val is not None else None,
                    "doc": vsp.GetAnalysisInputDoc(name, key),
                }
            )
        return {"analysis": name, "doc": vsp.GetAnalysisDoc(name), "inputs": inputs}


def _set_analysis_inputs(name: str, inputs: dict) -> None:
    for key, value in inputs.items():
        vals = value if isinstance(value, list) else [value]
        t = vsp.GetAnalysisInputType(name, key)
        if t == vsp.INT_DATA or t == vsp.BOOL_DATA:
            vsp.SetIntAnalysisInput(name, key, [int(v) for v in vals])
        elif t == vsp.DOUBLE_DATA:
            vsp.SetDoubleAnalysisInput(name, key, [float(v) for v in vals])
        elif t == vsp.STRING_DATA:
            vsp.SetStringAnalysisInput(name, key, [str(v) for v in vals])
        elif t == vsp.VEC3D_DATA:
            vsp.SetVec3dAnalysisInput(name, key, [vsp.vec3d(*v) for v in vals])
        else:
            raise ValueError(f"unsupported input type for {name}.{key}")


def _read_results(rid: str) -> dict:
    data: dict[str, Any] = {}
    for key in vsp.GetAllDataNames(rid):
        t = vsp.GetResultsType(rid, key)
        if t == vsp.INT_DATA:
            vals = list(vsp.GetIntResults(rid, key))
        elif t == vsp.DOUBLE_DATA:
            vals = list(vsp.GetDoubleResults(rid, key))
        elif t == vsp.STRING_DATA:
            vals = list(vsp.GetStringResults(rid, key))
        elif t == vsp.VEC3D_DATA:
            vals = [[v.x(), v.y(), v.z()] for v in vsp.GetVec3dResults(rid, key)]
        else:
            continue
        data[key] = vals[0] if len(vals) == 1 else _truncate(vals)
    return data


@server.tool()
def vsp_run_analysis(name: str, inputs: dict | None = None, keep_mesh: bool = False) -> dict:
    """Run any OpenVSP analysis and return its results.

    Call vsp_describe_analysis first to see the available inputs. Inputs left
    unspecified keep their defaults.

    Args:
        name: analysis name, e.g. "MassProp", "CompGeom", "VSPAEROSweep".
        inputs: mapping of input name to value (scalar or list).
        keep_mesh: keep the MeshGeom the analysis generates. Off by default —
            left in the model these accumulate and skew later results.
    """
    with _vsp_call(), _mesh_cleanup(keep_mesh):
        vsp.SetAnalysisInputDefaults(name)
        if inputs:
            _set_analysis_inputs(name, inputs)
        rid = vsp.ExecAnalysis(name)
        return {"analysis": name, "results_id": rid, "results": _read_results(rid)}


@server.tool()
def vsp_mass_properties(set_index: int = 0, num_slices: int = 100) -> dict:
    """Compute mass, centre of gravity, and inertia tensor of the model.

    Args:
        set_index: geometry set; 0 = SET_ALL.
        num_slices: slice count for the numerical integration.
    """
    return vsp_run_analysis("MassProp", {"Set": set_index, "NumMassSlices": num_slices})


@server.tool()
def vsp_comp_geom(set_index: int = 0, half_mesh: bool = False) -> dict:
    """Compute wetted area and volume per component via a triangulated mesh.

    Args:
        set_index: geometry set; 0 = SET_ALL.
        half_mesh: mesh only one side of the symmetry plane.
    """
    return vsp_run_analysis(
        "CompGeom",
        {"Set": set_index, "HalfMeshFlag": int(half_mesh), "WriteCSVFlag": 0, "WriteTXTFlag": 0},
    )


@server.tool()
def vsp_vspaero_sweep(
    alpha_start: float = 0.0,
    alpha_end: float = 10.0,
    alpha_npts: int = 5,
    mach: float = 0.3,
    thin_geom_set: int | None = None,
    thick_geom_set: int | None = None,
    wing_id: str = "",
    ref_area: float = 0.0,
    ref_span: float = 0.0,
    ref_chord: float = 0.0,
    n_cpu: int = 4,
) -> dict:
    """Run a VSPAERO angle-of-attack sweep and return the aerodynamic polar.

    Runs VSPAEROComputeGeometry, then the sweep, then reads the polar.

    VSPAERO splits the model into *thin* surfaces solved as a vortex lattice
    (wings, props) and *thick* surfaces solved with panels (fuselages, pods).
    A component must appear in exactly one of the two sets — putting it in both
    duplicates the geometry and aborts the solver. By default the model is
    classified automatically by component type; pass both set indices to take
    over manually (see vsp_list_sets and vsp_assign_set).

    Reference quantities come from the first Wing in the model unless you pass
    ref_area/ref_span/ref_chord, in which case those are used verbatim.

    Args:
        alpha_start: first angle of attack, degrees.
        alpha_end: last angle of attack, degrees.
        alpha_npts: number of alpha points.
        mach: freestream Mach number.
        thin_geom_set: set of lifting surfaces; None = classify automatically.
        thick_geom_set: set of bodies; None = classify automatically.
        wing_id: geom id to take reference area/span/chord from; defaults to the
            first Wing found.
        ref_area: manual reference area; overrides wing_id when non-zero.
        ref_span: manual reference span.
        ref_chord: manual reference chord.
        n_cpu: solver thread count.
    """
    with _vsp_call(), _mesh_cleanup():
        if thin_geom_set is None or thick_geom_set is None:
            auto_thin, auto_thick, split = _auto_aero_sets()
            thin_geom_set = auto_thin if thin_geom_set is None else thin_geom_set
            thick_geom_set = auto_thick if thick_geom_set is None else thick_geom_set
        else:
            split = None

        geom_sets = {"GeomSet": thick_geom_set, "ThinGeomSet": thin_geom_set}

        vsp.SetAnalysisInputDefaults("VSPAEROComputeGeometry")
        _set_analysis_inputs("VSPAEROComputeGeometry", geom_sets)
        vsp.ExecAnalysis("VSPAEROComputeGeometry")

        sweep_inputs: dict[str, Any] = dict(geom_sets)
        sweep_inputs.update(
            {
                "AlphaStart": alpha_start,
                "AlphaEnd": alpha_end,
                "AlphaNpts": alpha_npts,
                "MachStart": mach,
                "MachEnd": mach,
                "MachNpts": 1,
                "NCPU": n_cpu,
            }
        )

        if ref_area or ref_span or ref_chord:
            ref_mode = "manual"
            sweep_inputs["RefFlag"] = vsp.MANUAL_REF
            if ref_area:
                sweep_inputs["Sref"] = ref_area
            if ref_span:
                sweep_inputs["bref"] = ref_span
            if ref_chord:
                sweep_inputs["cref"] = ref_chord
        else:
            ref_geom = wing_id or next(
                (g for g in vsp.FindGeoms() if vsp.GetGeomTypeName(g) == "Wing"), ""
            )
            if ref_geom:
                ref_mode = f"from wing {vsp.GetGeomName(ref_geom)}"
                sweep_inputs["RefFlag"] = vsp.COMPONENT_REF
                sweep_inputs["WingID"] = ref_geom
            else:
                ref_mode = "VSPAERO defaults (no wing in model)"

        vsp.SetAnalysisInputDefaults("VSPAEROSweep")
        _set_analysis_inputs("VSPAEROSweep", sweep_inputs)

        # Results accumulate across the session, so a stale polar from an earlier
        # sweep would otherwise be returned as if it were this run's.
        polars_before = vsp.GetNumResults("VSPAERO_Polar")
        rid = vsp.ExecAnalysis("VSPAEROSweep")

        if vsp.GetNumResults("VSPAERO_Polar") <= polars_before:
            raise RuntimeError(
                "VSPAERO produced no polar — the solver did not finish. Most often "
                "a component is in both the thin and thick set, or in the wrong one: "
                "wings belong in thin, bodies in thick. Check vsp_list_sets, or pass "
                "thin_geom_set/thick_geom_set explicitly. Solver output is on stderr."
            )
        polar_id = vsp.FindLatestResultsID("VSPAERO_Polar")
        return {
            "results_id": rid,
            "reference": ref_mode,
            "surface_split": split,
            "polar": _read_results(polar_id),
        }


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------

_EXPORT_FORMATS = {
    "STL": vsp.EXPORT_STL,
    "STEP": vsp.EXPORT_STEP,
    "IGES": vsp.EXPORT_IGES,
    "OBJ": vsp.EXPORT_OBJ,
    "X3D": vsp.EXPORT_X3D,
    "DXF": vsp.EXPORT_DXF,
    "SVG": vsp.EXPORT_SVG,
    "GMSH": vsp.EXPORT_GMSH,
    "VSPGEOM": vsp.EXPORT_VSPGEOM,
    "CART3D": vsp.EXPORT_CART3D,
    "NASCART": vsp.EXPORT_NASCART,
    "POVRAY": vsp.EXPORT_POVRAY,
    "BEM": vsp.EXPORT_BEM,
    "SELIG_AIRFOIL": vsp.EXPORT_SELIG_AIRFOIL,
    "BEZIER_AIRFOIL": vsp.EXPORT_BEZIER_AIRFOIL,
}


@server.tool()
def vsp_export(path: str, export_format: str, set_index: int = 0, keep_mesh: bool = False) -> dict:
    """Export the model to a CAD or mesh file.

    Args:
        path: absolute destination path.
        export_format: one of STL, STEP, IGES, OBJ, X3D, DXF, SVG, GMSH,
            VSPGEOM, CART3D, NASCART, POVRAY, BEM, SELIG_AIRFOIL, BEZIER_AIRFOIL.
        set_index: geometry set; 0 = SET_ALL.
        keep_mesh: keep the MeshGeom that mesh-based formats generate. Off by
            default — left in the model these accumulate into later exports.
    """
    fmt = export_format.upper()
    if fmt not in _EXPORT_FORMATS:
        raise ValueError(f"unknown format {export_format}; expected one of {sorted(_EXPORT_FORMATS)}")
    with _vsp_call(), _mesh_cleanup(keep_mesh):
        vsp.Update()
        vsp.ExportFile(path, set_index, _EXPORT_FORMATS[fmt])
        return {"ok": True, "path": path, "format": fmt}


# --------------------------------------------------------------------------
# escape hatch
# --------------------------------------------------------------------------

@server.tool()
def vsp_run_api_script(code: str) -> dict:
    """Run Python against the live OpenVSP model for anything the other tools miss.

    The OpenVSP API has ~1700 functions; this reaches the ones without a
    dedicated tool. The module is bound as both `vsp` and `openvsp`, and the
    model persists across calls. Assign to a variable named `result` to return a
    value; anything printed is captured.

    Args:
        code: Python source to execute.
    """
    with _vsp_call():
        ns: dict[str, Any] = {"vsp": vsp, "openvsp": vsp}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(code, ns)
        value = ns.get("result")
        return {
            "stdout": buf.getvalue(),
            "result": value if isinstance(value, (str, int, float, bool, list, dict, type(None))) else repr(value),
        }


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
