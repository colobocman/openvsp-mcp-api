#!/usr/bin/env python3
"""OpenVSP MCP server.

Exposes the official OpenVSP Python API (module ``openvsp``, v3.51.2) over the
Model Context Protocol on stdio.

The server holds one long-lived OpenVSP model in memory, so tool calls compose:
add a wing, tweak its parameters, run VSPAERO, export a mesh. All API access is
serialised behind a lock because the OpenVSP API is not thread-safe.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import threading
from typing import Any, TypedDict
from xml.etree import ElementTree

import openvsp as vsp
from mcp.server.mcpserver import MCPServer

server = MCPServer("openvsp", version="0.1.0")

_LOCK = threading.RLock()
_ERR = vsp.ErrorMgrSingleton.getInstance()
_ERR.SilenceErrors()

# Arrays in analysis results can be enormous (per-node pressures etc.).
_MAX_ITEMS = 200


# --------------------------------------------------------------------------
# response shapes
#
# Declared so clients get an output schema instead of an opaque object. Only
# the stable shapes are typed: analysis results carry whatever fields the
# analysis emits (30 for CompGeom, 3 for DegenGeom), so those stay open.
# --------------------------------------------------------------------------


class VspInfo(TypedDict):
    version: str
    vsp_file: str
    vspaero_path: str
    geom_count: int
    geom_ids: list[str]
    available_geom_types: list[str]


class GeomInfo(TypedDict):
    id: str
    name: str
    type: str
    parent: str
    children: list[str]


class NewModelResult(TypedDict):
    ok: bool
    geom_count: int


class OpenResult(TypedDict):
    ok: bool
    path: str
    geoms: list[GeomInfo]


class SaveResult(TypedDict):
    ok: bool
    path: str


class AddGeomResult(TypedDict):
    id: str
    name: str
    type: str
    parm_count: int


class DeleteResult(TypedDict):
    ok: bool
    remaining: int


class ParmDetail(TypedDict):
    name: str
    group: str
    value: float
    type: str
    min: float
    max: float
    description: str


class SetInfo(TypedDict):
    index: int
    name: str
    geoms: list[str]


class AssignSetResult(TypedDict):
    set: str
    index: int
    geoms: list[str]


class AnalysisInput(TypedDict):
    name: str
    type: str
    default: Any
    doc: str


class AnalysisDescription(TypedDict):
    analysis: str
    doc: str
    inputs: list[AnalysisInput]


class AnalysisResult(TypedDict):
    analysis: str
    results_id: str
    results: dict[str, Any]


class SweepResult(TypedDict):
    results_id: str
    reference: str
    surface_split: dict[str, list[str]] | None
    polar: dict[str, Any]


class ExportResult(TypedDict):
    ok: bool
    path: str
    format: str


class ApiScriptResult(TypedDict):
    stdout: str
    result: Any


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
                g for g in vsp.FindGeoms() if g not in before and "Mesh" in vsp.GetGeomTypeName(g)
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
def vsp_info() -> VspInfo:
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
def vsp_new_model() -> NewModelResult:
    """Discard the in-memory model and start a fresh, empty one."""
    with _vsp_call():
        vsp.VSPRenew()
        vsp.Update()
        return {"ok": True, "geom_count": len(vsp.FindGeoms())}


@server.tool()
def vsp_open_model(path: str) -> OpenResult:
    """Load a .vsp3 file, replacing the in-memory model.

    The file is validated first. ReadVSPFile has no way to fail cleanly — it
    clears the model before parsing — so an unreadable path would otherwise
    destroy the model already in memory.

    Args:
        path: absolute path to a .vsp3 file.
    """
    if not os.path.isfile(path):
        raise ValueError(f"no such file: {path}")
    try:
        root = ElementTree.parse(path).getroot().tag
    except ElementTree.ParseError as exc:
        raise ValueError(f"not a readable .vsp3 file: {path} ({exc})") from exc
    if root != "Vsp_Geometry":
        raise ValueError(f"not an OpenVSP model: {path} has root <{root}>")

    with _vsp_call():
        vsp.ClearVSPModel()
        vsp.ReadVSPFile(path)
        vsp.Update()
        return {"ok": True, "path": path, "geoms": _list_geoms()}


@server.tool()
def vsp_save_model(path: str, set_index: int = 0) -> SaveResult:
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


def _list_geoms() -> list[GeomInfo]:
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
def vsp_list_geoms() -> list[GeomInfo]:
    """List every component in the model with its id, name, type, and hierarchy."""
    with _vsp_call():
        return _list_geoms()


@server.tool()
def vsp_add_geom(geom_type: str, name: str = "", parent_id: str = "") -> AddGeomResult:
    """Add a component to the model.

    Args:
        geom_type: one of WING, FUSELAGE, POD, STACK, BLANK, ELLIPSOID,
            BODYOFREVOLUTION, HUMAN, PROP, GEAR, HINGE, CONFORMAL, ROUTING,
            AUXILIARY, COBRA.
        name: optional display name for the new component.
        parent_id: geom id to attach the new component to. CONFORMAL requires
            one — it takes its shape from the parent.
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
def vsp_delete_geom(geom_id: str) -> DeleteResult:
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


def _parm_detail(pid: str) -> ParmDetail:
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
def vsp_get_parms(geom_id: str, parms: list[dict]) -> list[ParmDetail]:
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
def vsp_set_parms(geom_id: str, parms: list[dict]) -> list[ParmDetail]:
    """Set parameters and update the model.

    The returned values are what was actually applied — check them rather than
    assuming. A write can land differently than requested for two reasons:
    OpenVSP clamps values to each parameter's limits, and many parameters are
    linked through driver groups (a wing's span, chord, area and aspect ratio
    are one such group — only the current drivers hold a value, the rest are
    recomputed). To choose which quantities drive a wing section, call
    SetDriverGroup via vsp_run_api_script.

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
def vsp_list_sets() -> list[SetInfo]:
    """List the geometry sets and which components belong to each.

    Sets are how OpenVSP scopes an analysis to part of the model. Indices 0-2
    are the built-ins All / Shown / Not_Shown; the rest are user sets.
    """
    with _vsp_call():
        return [
            {
                "index": i,
                "name": vsp.GetSetName(i),
                "geoms": [vsp.GetGeomName(g) for g in vsp.FindGeoms() if vsp.GetSetFlag(g, i)],
            }
            for i in range(vsp.GetNumSets())
        ]


@server.tool()
def vsp_assign_set(
    geom_id: str, set_index: int, member: bool = True, set_name: str = ""
) -> AssignSetResult:
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
def vsp_describe_analysis(name: str) -> AnalysisDescription:
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


def _check_analysis_preconditions(name: str, inputs: dict) -> None:
    """Refuse calls that OpenVSP answers by aborting the process.

    A C++ abort cannot be caught, so it takes the whole server down along with
    the in-memory model. Cheaper to check first.
    """
    if name == "EmintonLord":

        def resolved(key):
            if key in inputs:
                v = inputs[key]
                return v if isinstance(v, list) else [v]
            return list(vsp.GetDoubleAnalysisInput(name, key))

        x, area = resolved("X_vec"), resolved("Area_vec")
        if not x or not area or len(x) != len(area):
            raise ValueError(
                "EmintonLord needs X_vec and Area_vec as non-empty arrays of equal "
                f"length (got {len(x)} and {len(area)}); OpenVSP aborts the process "
                "otherwise. Use the WaveDrag analysis to derive them from geometry."
            )


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
def vsp_run_analysis(
    name: str, inputs: dict | None = None, keep_mesh: bool = False
) -> AnalysisResult:
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
        if name not in vsp.ListAnalysis():
            raise ValueError(f"unknown analysis {name}; see vsp_list_analyses")
        vsp.SetAnalysisInputDefaults(name)
        _check_analysis_preconditions(name, inputs or {})
        if inputs:
            _set_analysis_inputs(name, inputs)
        rid = vsp.ExecAnalysis(name)
        if not rid:
            raise RuntimeError(
                f"{name} produced no results. Some analyses stop working once a "
                "slicing analysis (MassProp, CompGeom, PlanarSlice) has run in the "
                "same session — WaveDrag is one. Call vsp_new_model and rebuild, or "
                "run this analysis before the slicing ones."
            )
        return {"analysis": name, "results_id": rid, "results": _read_results(rid)}


@server.tool()
def vsp_mass_properties(
    set_index: int = 0, num_slices: int = 100, include_slices: bool = False
) -> AnalysisResult:
    """Compute mass, centre of gravity, and inertia tensor of the model.

    Returns totals and per-component values. The slice-by-slice integration
    detail (`Fill_*`, one entry per slice per quantity) is dropped unless asked
    for — with the default 100 slices it is ~30 KB that rarely matters.

    Args:
        set_index: geometry set; 0 = SET_ALL.
        num_slices: slice count for the numerical integration.
        include_slices: keep the per-slice `Fill_*` arrays in the result.
    """
    out = vsp_run_analysis("MassProp", {"Set": set_index, "NumMassSlices": num_slices})
    if not include_slices:
        out["results"] = {k: v for k, v in out["results"].items() if not k.startswith("Fill_")}
    return out


@server.tool()
def vsp_comp_geom(set_index: int = 0, half_mesh: bool = False) -> AnalysisResult:
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
) -> SweepResult:
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

# Format -> (OpenVSP enum, default extension). The extension is not cosmetic:
# the IGES writer aborts the process outright when the path has none.
_EXPORT_FORMATS = {
    "STL": (vsp.EXPORT_STL, ".stl"),
    "STEP": (vsp.EXPORT_STEP, ".stp"),
    "IGES": (vsp.EXPORT_IGES, ".igs"),
    "OBJ": (vsp.EXPORT_OBJ, ".obj"),
    "X3D": (vsp.EXPORT_X3D, ".x3d"),
    "DXF": (vsp.EXPORT_DXF, ".dxf"),
    "SVG": (vsp.EXPORT_SVG, ".svg"),
    "GMSH": (vsp.EXPORT_GMSH, ".msh"),
    "VSPGEOM": (vsp.EXPORT_VSPGEOM, ".vspgeom"),
    "CART3D": (vsp.EXPORT_CART3D, ".tri"),
    "NASCART": (vsp.EXPORT_NASCART, ".dat"),
    "POVRAY": (vsp.EXPORT_POVRAY, ".pov"),
    "BEM": (vsp.EXPORT_BEM, ".bem"),
    "SELIG_AIRFOIL": (vsp.EXPORT_SELIG_AIRFOIL, ".dat"),
    "BEZIER_AIRFOIL": (vsp.EXPORT_BEZIER_AIRFOIL, ".bz"),
}


@server.tool()
def vsp_export(
    path: str,
    export_format: str,
    set_index: int = 0,
    geom_id: str = "",
    keep_mesh: bool = False,
) -> ExportResult:
    """Export the model to a CAD or mesh file.

    A default extension is appended when the path has none. This matters:
    OpenVSP's IGES writer aborts the process on an extensionless path.

    Args:
        path: absolute destination path.
        export_format: one of STL, STEP, IGES, OBJ, X3D, DXF, SVG, GMSH,
            VSPGEOM, CART3D, NASCART, POVRAY, BEM, SELIG_AIRFOIL, BEZIER_AIRFOIL.
        set_index: geometry set; 0 = SET_ALL.
        geom_id: propeller to export; required by BEM, ignored otherwise.
        keep_mesh: keep the MeshGeom that mesh-based formats generate. Off by
            default — left in the model these accumulate into later exports.
    """
    fmt = export_format.upper()
    if fmt not in _EXPORT_FORMATS:
        raise ValueError(
            f"unknown format {export_format}; expected one of {sorted(_EXPORT_FORMATS)}"
        )
    code, default_ext = _EXPORT_FORMATS[fmt]

    written = path if os.path.splitext(path)[1] else path + default_ext

    with _vsp_call(), _mesh_cleanup(keep_mesh):
        if fmt == "BEM":
            prop = geom_id or next(
                (g for g in vsp.FindGeoms() if vsp.GetGeomTypeName(g) == "Propeller"), ""
            )
            if not prop:
                raise ValueError("BEM export needs a propeller; pass geom_id")
            vsp.SetBEMPropID(prop)
        vsp.Update()
        vsp.ExportFile(written, set_index, code)
        return {"ok": True, "path": written, "format": fmt}


# --------------------------------------------------------------------------
# escape hatch
# --------------------------------------------------------------------------


@server.tool()
def vsp_run_api_script(code: str) -> ApiScriptResult:
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
            exec(code, ns)  # noqa: S102 - this tool exists to run caller-supplied API code
        value = ns.get("result")
        return {
            "stdout": buf.getvalue(),
            "result": value
            if isinstance(value, (str, int, float, bool, list, dict, type(None)))
            else repr(value),
        }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="openvsp-mcp-api",
        description="Serve the OpenVSP Python API over the Model Context Protocol.",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        default="stdio",
        help="stdio (default) runs one server per client. The HTTP transports let "
        "several clients share one OpenVSP process — and therefore one model.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="interface to bind for the HTTP transports (default loopback). The "
        "server is unauthenticated and vsp_run_api_script executes arbitrary "
        "Python, so bind beyond loopback only on a trusted network.",
    )
    parser.add_argument("--port", type=int, default=8000, help="port for the HTTP transports")
    parser.add_argument("--path", default="/mcp", help="mount path for the HTTP transports")
    parser.add_argument(
        "--describe",
        action="store_true",
        help="print the server's tools and exit, without starting it",
    )
    args = parser.parse_args(argv)

    if args.describe:
        print(
            json.dumps(
                {
                    "name": "openvsp-mcp-api",
                    "openvsp_version": vsp.GetVSPVersion(),
                    "default_transport": "stdio",
                    "tools": sorted(
                        name
                        for name, fn in globals().items()
                        if name.startswith("vsp_") and callable(fn)
                    ),
                },
                indent=2,
            )
        )
        return

    if args.transport == "stdio":
        server.run()
    elif args.transport == "streamable-http":
        server.run(
            "streamable-http", host=args.host, port=args.port, streamable_http_path=args.path
        )
    else:
        server.run("sse", host=args.host, port=args.port, sse_path=args.path)


if __name__ == "__main__":
    main()
