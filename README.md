# openvsp-mcp-api

An MCP server that exposes the official [OpenVSP](https://openvsp.org) Python
API — the `openvsp` module shipped with the binary release — rather than
shelling out to the `vsp` CLI. Agents get parametric aircraft geometry, mass
properties, wetted areas, VSPAERO polars and CAD export as ordinary tools.

Tested against OpenVSP 3.51.2 / VSPAERO 7.2.2 on macOS ARM64.

## Install

OpenVSP itself is not on PyPI. Install the binary release from
[openvsp.org/download.php](https://openvsp.org/download.php), then install its
Python packages into a virtualenv:

```bash
uv venv --python 3.13 .venv
cd /Applications/OpenVSP/python && VIRTUAL_ENV=/path/to/.venv uv pip install -r requirements.txt
VIRTUAL_ENV=/path/to/.venv uv pip install "mcp>=2.0"
```

Register the server:

```bash
claude mcp add openvsp --scope user -- /path/to/.venv/bin/python /path/to/vsp-mcp/vsp_mcp.py
```

Verify:

```bash
python tests/e2e.py
```

## Tools

**Model** — `vsp_info`, `vsp_new_model`, `vsp_open_model`, `vsp_save_model`

**Geometry** — `vsp_list_geoms`, `vsp_add_geom`, `vsp_delete_geom`

**Parameters** — `vsp_list_parms`, `vsp_get_parms`, `vsp_set_parms`

**Sets** — `vsp_list_sets`, `vsp_assign_set`

**Analysis** — `vsp_list_analyses`, `vsp_describe_analysis`, `vsp_run_analysis`,
`vsp_mass_properties`, `vsp_comp_geom`, `vsp_vspaero_sweep`

**Output** — `vsp_export` (STL, STEP, IGES, OBJ, X3D, DXF, SVG, GMSH, VSPGEOM,
CART3D, NASCART, POVRAY, BEM, airfoil formats)

**Escape hatch** — `vsp_run_api_script` runs Python against the live model, for
the ~1700 API functions without a dedicated tool.

`vsp_run_analysis` reaches every analysis OpenVSP registers — ParasiteDrag,
WaveDrag, CfdMeshAnalysis, Projection, DegenGeom and the rest — with
`vsp_describe_analysis` to discover inputs.

The server keeps one model in memory across calls, so tools compose: add a
wing, tweak parameters, sweep, export.

## Pitfalls this server handles for you

Each of these was found by testing against a live model, and each produced
wrong answers rather than an error.

**Analyses leave mesh residue.** CompGeom, MassProp, VSPAERO and mesh exports
each add a MeshGeom to the model as a side effect. Left in place they
accumulate and get pulled into every later analysis and export — repeating an
export three times grew one STL from 259 KB to 181 MB. New mesh components are
deleted automatically; pass `keep_mesh=true` to keep them.

**VSPAERO needs thin and thick surfaces separated.** VSPAERO 7.x dropped the
old `AnalysisMethod` input. Instead it solves wings and props as a vortex
lattice (*thin*) and bodies with panels (*thick*), through two separate
geometry sets. A component in both sets duplicates the geometry and the solver
aborts with SIGABRT. `vsp_vspaero_sweep` classifies by component type,
reports the split it chose, and explains what to fix when the solver produces
no polar; override with `thin_geom_set` / `thick_geom_set` and `vsp_assign_set`.

**Results accumulate across the session.** `FindLatestResultsID` happily
returns a polar from an earlier sweep, so a failed run could hand back stale
numbers as if they were current. The polar count is checked before and after.

**Section parameters need their indexed group name.** `GetParmGroupName`
reports a wing section's parameters under `XSec`, but `GetParm` only resolves
them under `XSec_0` / `XSec_1`. Listing by the former produced duplicate names
that could not be written back. The server reports
`GetParmDisplayGroupName`, so any listing feeds straight into a write.

## Other things worth knowing

**Parameters are driver-based.** A wing's span, area, chord and aspect ratio are
linked; setting one recomputes the others, and not every combination is
directly settable. `vsp_set_parms` returns the values actually applied — read
them rather than assuming the write took. To choose which three quantities
drive a section, call `SetDriverGroup` through `vsp_run_api_script`.

**Reference quantities decide your coefficients.** `vsp_vspaero_sweep` takes
Sref/bref/cref from the first Wing in the model unless you pass them
explicitly, and reports which it used in the `reference` field. Coefficients
are only meaningful against the area actually used.

**Solver output goes to stderr**, so it never corrupts the JSON-RPC stream.
When a sweep fails, that is where the reason is.

## Validation

Checked against NACA TR-1208 (45° swept wing, aspect ratio 8.02), the test case
OpenVSP ships as its own API tutorial: CL linear in alpha, CD positive and
quadratic, and CL_alpha ≈ 0.068/deg against a DATCOM estimate of 0.065/deg,
rising with aspect ratio as expected.

Reproducing the planform from the report's dimensions gives aspect ratio 8.000
and MAC 16.670 against the published 8.02 and 16.672. Note that OpenVSP's
shipped `TR1208.vspscript` does *not* reproduce it — see
[OpenVSP/OpenVSP](https://github.com/OpenVSP/OpenVSP) issue for the three
defects in that example.

## License

MIT
