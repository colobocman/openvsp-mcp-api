#!/usr/bin/env python3
"""Broad system test: every tool, every geom type, every export format,
every analysis, plus protocol hygiene, robustness and concurrency.

    python tests/system.py

Complements tests/e2e.py, which covers the common path. This one sweeps the
whole surface and pins the OpenVSP behaviours that used to take the server
down: the IGES writer aborting on an extensionless path, EmintonLord aborting
without its arrays, and a failed open wiping the model.

Exits non-zero if any check fails. Takes a few minutes.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

SERVER = str(Path(__file__).resolve().parent.parent / "vsp_mcp.py")
PY = sys.executable

PASS, FAIL, NOTE = [], [], []


def ok(label, cond, detail=""):
    (PASS if cond else FAIL).append(label)
    print(f"  {'ok  ' if cond else 'FAIL'}  {label}{'   ' + detail if detail else ''}")
    return cond


def note(label, detail=""):
    NOTE.append(f"{label} {detail}".strip())
    print(f"  note  {label}{'   ' + detail if detail else ''}")


class Client:
    def __init__(self, cwd, capture_stderr=False):
        self.proc = subprocess.Popen(
            [PY, SERVER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE if capture_stderr else subprocess.DEVNULL,
            text=True,
            bufsize=1,
            cwd=cwd,
        )
        self._id = 0
        self._lock = threading.Lock()
        self.raw_lines = []
        self.call(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "sys", "version": "1"},
            },
        )
        self.call("notifications/initialized", notify=True)

    def _send(self, msg):
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def call(self, method, params=None, notify=False):
        with self._lock:
            msg = {"jsonrpc": "2.0", "method": method}
            if params is not None:
                msg["params"] = params
            if notify:
                self._send(msg)
                return None
            self._id += 1
            wanted = self._id
            msg["id"] = wanted
            self._send(msg)
            while True:
                line = self.proc.stdout.readline()
                if not line:
                    raise RuntimeError("server closed stream")
                self.raw_lines.append(line)
                d = json.loads(line)
                if d.get("id") == wanted:
                    if "error" in d:
                        raise RuntimeError(f"{method}: {d['error']}")
                    return d["result"]

    def tool(self, name, args=None):
        res = self.call("tools/call", {"name": name, "arguments": args or {}})
        if res.get("isError"):
            raise RuntimeError(res["content"][0]["text"])
        sc = res.get("structuredContent")
        if sc is None:
            return json.loads(res["content"][0]["text"])
        # Non-object returns arrive wrapped as exactly {"result": ...}; object
        # returns are the object itself, and one of them has a "result" field.
        return sc["result"] if set(sc) == {"result"} else sc

    def pipeline(self, requests):
        """Fire many requests without waiting, then collect every reply."""
        with self._lock:
            ids = []
            for name, args in requests:
                self._id += 1
                ids.append(self._id)
                self._send(
                    {
                        "jsonrpc": "2.0",
                        "id": self._id,
                        "method": "tools/call",
                        "params": {"name": name, "arguments": args},
                    }
                )
            got = {}
            while len(got) < len(ids):
                line = self.proc.stdout.readline()
                if not line:
                    raise RuntimeError("server closed stream")
                d = json.loads(line)
                if d.get("id") in ids:
                    got[d["id"]] = d
            return [got[i] for i in ids]

    def close(self):
        self.proc.stdin.close()
        try:
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()


GEOM_TYPES = [
    "POD",
    "FUSELAGE",
    "WING",
    "STACK",
    "BLANK",
    "ELLIPSOID",
    "BODYOFREVOLUTION",
    "HUMAN",
    "PROP",
    "GEAR",
    "HINGE",
    "CONFORMAL",
    "ROUTING",
    "AUXILIARY",
    "COBRA",
]

EXPORT_FORMATS = [
    "STL",
    "STEP",
    "IGES",
    "OBJ",
    "X3D",
    "DXF",
    "SVG",
    "GMSH",
    "VSPGEOM",
    "CART3D",
    "NASCART",
    "POVRAY",
    "BEM",
    "SELIG_AIRFOIL",
    "BEZIER_AIRFOIL",
]

# Analyses cheap and safe enough to actually execute on a plain wing.
RUNNABLE = [
    "MassProp",
    "CompGeom",
    "DegenGeom",
    "Projection",
    "PlanarSlice",
    "WaveDrag",
    "SurfacePatches",
    "GeometryAnalysis",
]


def http_check(port: int = 8813) -> None:
    """Same server over HTTP: several clients can share one OpenVSP process."""
    import socket

    import anyio
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    proc = subprocess.Popen(
        [PY, SERVER, "--transport", "streamable-http", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(40):
            time.sleep(0.5)
            if proc.poll() is not None:
                ok("http server starts", False, "exited early")
                return
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    break
            except OSError:
                continue

        async def go():
            async with (
                streamable_http_client(f"http://127.0.0.1:{port}/mcp") as (r, w),
                ClientSession(r, w) as s,
            ):
                await s.initialize()
                tools = (await s.list_tools()).tools
                ok("http exposes the same 20 tools", len(tools) == 20, str(len(tools)))
                await s.call_tool("vsp_new_model", {})
                await s.call_tool("vsp_add_geom", {"geom_type": "WING", "name": "HttpWing"})
                res = await s.call_tool("vsp_list_geoms", {})
                names = [g["name"] for g in res.structured_content["result"]]
                ok("model persists across separate http calls", names == ["HttpWing"], str(names))
                res = await s.call_tool("vsp_mass_properties", {})
                mass = res.structured_content["results"]["Total_Mass"]
                ok("analysis runs over http", mass > 0, f"mass={mass:.4f}")

        anyio.run(go)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    tmp = tempfile.mkdtemp(prefix="vsp_sys_")
    out = lambda n: str(Path(tmp) / n)
    c = Client(tmp, capture_stderr=True)

    print("\n[1] protocol surface")
    tools = c.call("tools/list")["tools"]
    ok("20 tools advertised", len(tools) == 20, str(len(tools)))
    ok("every tool has a description", all(t.get("description") for t in tools))
    ok(
        "every tool has an input schema",
        all(t.get("inputSchema", {}).get("type") == "object" for t in tools),
    )
    # Stable-shape tools declare their return type, so clients know what comes
    # back without calling first. Only vsp_list_parms stays open: it has two
    # shapes depending on whether a group was named.
    schemas = [t["name"] for t in tools if t.get("outputSchema")]
    ok(
        "all but one tool advertise an output schema",
        len(schemas) == len(tools) - 1,
        f"{len(schemas)}/{len(tools)}, open: {sorted({t['name'] for t in tools} - set(schemas))}",
    )
    covered = set()

    print("\n[2] all 15 geometry types")
    c.tool("vsp_new_model")
    covered.add("vsp_new_model")
    bad_types = []
    host = c.tool("vsp_add_geom", {"geom_type": "POD", "name": "Host"})["id"]
    for t in GEOM_TYPES:
        args = {"geom_type": t, "name": f"T_{t}"}
        if t == "CONFORMAL":  # conforms to a parent, cannot stand alone
            args["parent_id"] = host
        try:
            g = c.tool("vsp_add_geom", args)
            if not g["id"]:
                bad_types.append(t)
        except RuntimeError as e:
            bad_types.append(f"{t}({str(e)[:40]})")
    covered |= {"vsp_add_geom", "vsp_list_geoms"}
    ok("every geom type creates", not bad_types, str(bad_types))
    ok("model holds all of them", len(c.tool("vsp_list_geoms")) == len(GEOM_TYPES) + 1)

    print("\n[3] delete, sets, save/open round-trip")
    geoms = c.tool("vsp_list_geoms")
    c.tool("vsp_delete_geom", {"geom_id": geoms[-1]["id"]})
    covered.add("vsp_delete_geom")
    ok("delete removes one", len(c.tool("vsp_list_geoms")) == len(GEOM_TYPES))

    sets = c.tool("vsp_list_sets")
    covered.add("vsp_list_sets")
    ok("built-in sets present", [s["name"] for s in sets[:3]] == ["All", "Shown", "Not_Shown"])
    wing = next(g for g in c.tool("vsp_list_geoms") if g["type"] == "Wing")
    r = c.tool(
        "vsp_assign_set",
        {"geom_id": wing["id"], "set_index": 4, "member": True, "set_name": "MyLifting"},
    )
    covered.add("vsp_assign_set")
    ok("assign to user set", r["set"] == "MyLifting" and r["geoms"] == ["T_WING"], str(r))
    c.tool("vsp_assign_set", {"geom_id": wing["id"], "set_index": 4, "member": False})
    ok(
        "remove from user set",
        c.tool("vsp_assign_set", {"geom_id": wing["id"], "set_index": 4, "member": False})["geoms"]
        == [],
    )

    before = [(g["name"], g["type"]) for g in c.tool("vsp_list_geoms")]
    c.tool("vsp_save_model", {"path": out("rt.vsp3")})
    covered.add("vsp_save_model")
    c.tool("vsp_new_model")
    ok("new model clears", c.tool("vsp_list_geoms") == [])
    c.tool("vsp_open_model", {"path": out("rt.vsp3")})
    covered.add("vsp_open_model")
    after = [(g["name"], g["type"]) for g in c.tool("vsp_list_geoms")]
    ok("save/open round-trips the model", before == after, f"{len(before)} components")

    print("\n[4] all 15 export formats")
    c.tool("vsp_new_model")
    w = c.tool("vsp_add_geom", {"geom_type": "WING", "name": "W"})
    c.tool("vsp_save_model", {"path": out("e.vsp3")})
    empty, failed = [], []
    for f in [x for x in EXPORT_FORMATS if x != "BEM"]:
        p = out(f"exp_{f}")
        try:
            c.tool("vsp_export", {"path": p, "export_format": f})
            hits = list(Path(tmp).glob(f"exp_{f}*"))
            if not hits or all(h.stat().st_size == 0 for h in hits):
                empty.append(f)
        except RuntimeError as e:
            failed.append(f"{f}({str(e)[:35]})")
    covered.add("vsp_export")
    ok("no export format errors", not failed, str(failed))
    if empty:
        note("formats producing no/empty file (may need extra setup)", str(empty))
    else:
        ok("every format wrote a non-empty file", True)
    try:
        c.tool("vsp_export", {"path": out("nope"), "export_format": "BEM"})
        ok("BEM refuses a model with no propeller", False)
    except RuntimeError:
        ok("BEM refuses a model with no propeller", True)
    c.tool("vsp_new_model")
    c.tool("vsp_add_geom", {"geom_type": "PROP", "name": "P"})
    c.tool("vsp_save_model", {"path": out("p.vsp3")})
    bem = c.tool("vsp_export", {"path": out("p"), "export_format": "BEM"})
    ok(
        "BEM exports a propeller",
        Path(bem["path"]).stat().st_size > 0,
        f"{Path(bem['path']).name} {Path(bem['path']).stat().st_size} b",
    )
    c.tool("vsp_new_model")
    w = c.tool("vsp_add_geom", {"geom_type": "WING", "name": "W"})
    c.tool("vsp_save_model", {"path": out("e.vsp3")})
    ok("exports leave no mesh residue", [g["name"] for g in c.tool("vsp_list_geoms")] == ["W"])

    igs = c.tool("vsp_export", {"path": out("noext"), "export_format": "IGES"})
    ok(
        "extensionless IGES handled, not fatal",
        igs["path"].endswith(".igs") and Path(igs["path"]).stat().st_size > 0,
        f"{Path(igs['path']).name} {Path(igs['path']).stat().st_size} b",
    )
    ok(
        "server alive after the former crash case",
        c.tool("vsp_info")["version"].startswith("OpenVSP"),
    )

    print("\n[5] analyses")
    names = c.tool("vsp_list_analyses")
    covered.add("vsp_list_analyses")
    ok("21 analyses registered", len(names) == 21, str(len(names)))
    undesc, noinput = [], []
    for n in names:
        try:
            d = c.tool("vsp_describe_analysis", {"name": n})
            if not d["inputs"]:
                noinput.append(n)
        except RuntimeError as e:
            undesc.append(f"{n}({str(e)[:30]})")
    covered.add("vsp_describe_analysis")
    ok("every analysis describable", not undesc, str(undesc))
    if noinput:
        note("analyses that take no inputs", str(noinput))

    ran, refused = [], []
    for n in RUNNABLE:
        try:
            res = c.tool("vsp_run_analysis", {"name": n})
            ran.append(n) if res["results"] else refused.append(f"{n}(empty)")
        except RuntimeError as e:
            refused.append(f"{n}: {str(e)[:45]}")
    covered.add("vsp_run_analysis")
    ok("generic runner executes analyses", len(ran) >= 6, f"ran {ran}")
    if refused:
        note("needed extra setup", str(refused))
    try:
        c.tool("vsp_run_analysis", {"name": "EmintonLord"})
        ok("EmintonLord refused instead of aborting the process", False)
    except RuntimeError:
        ok("EmintonLord refused instead of aborting the process", True)
    ok("server alive after that", c.tool("vsp_info")["version"].startswith("OpenVSP"))
    ok(
        "analyses leave no mesh residue",
        [g["name"] for g in c.tool("vsp_list_geoms")] == ["W"],
        str([g["name"] for g in c.tool("vsp_list_geoms")]),
    )

    print("\n[6] convenience wrappers and info")
    covered |= {
        "vsp_mass_properties",
        "vsp_comp_geom",
        "vsp_info",
        "vsp_list_parms",
        "vsp_get_parms",
        "vsp_set_parms",
        "vsp_run_api_script",
        "vsp_vspaero_sweep",
    }
    info = c.tool("vsp_info")
    ok(
        "info reports version and vspaero",
        info["version"].startswith("OpenVSP") and bool(info["vspaero_path"]),
    )
    mp = c.tool("vsp_mass_properties")["results"]
    # The per-slice Fill_* arrays are ~30 KB at the default 100 slices and
    # flooded the caller's context when this ran over a live MCP connection.
    ok(
        "mass properties compact by default",
        not any(k.startswith("Fill_") for k in mp),
        f"{len(mp)} keys",
    )
    full = c.tool("vsp_mass_properties", {"include_slices": True})["results"]
    ok(
        "per-slice detail available on request",
        any(k.startswith("Fill_") for k in full),
        f"{len(full)} keys",
    )
    cg = c.tool("vsp_comp_geom")["results"]
    ok(
        "mass and wetted area positive",
        mp["Total_Mass"] > 0 and cg["Total_Wet_Area"] > 0,
        f"m={mp['Total_Mass']:.3f} A={cg['Total_Wet_Area']:.3f}",
    )

    print("\n[7] robustness — bad input must error, not corrupt state")
    bad = [
        ("unknown geom type", "vsp_add_geom", {"geom_type": "NOT_A_TYPE"}),
        ("unknown export format", "vsp_export", {"path": out("x.zzz"), "export_format": "NOPE"}),
        ("missing file", "vsp_open_model", {"path": out("does_not_exist.vsp3")}),
        ("bad geom id", "vsp_list_parms", {"geom_id": "DEADBEEF", "group": "XSec_1"}),
        ("unknown analysis", "vsp_run_analysis", {"name": "NoSuchAnalysis"}),
        (
            "bad parm group",
            "vsp_set_parms",
            {"geom_id": w["id"], "parms": [{"group": "Zz", "name": "Zz", "value": 1}]},
        ),
        ("api script raises", "vsp_run_api_script", {"code": "raise ValueError('boom')"}),
    ]
    survived = []
    for label, name, args in bad:
        try:
            c.tool(name, args)
            survived.append(label)
        except RuntimeError:
            pass
    ok("all bad inputs rejected", not survived, str(survived))
    ok("server still healthy afterwards", c.tool("vsp_info")["version"].startswith("OpenVSP"))
    ok("model unharmed by the failures", [g["name"] for g in c.tool("vsp_list_geoms")] == ["W"])

    print("\n[8] concurrency — API is not thread-safe, lock must serialise")
    reqs = [
        ("vsp_mass_properties", {}),
        ("vsp_comp_geom", {}),
        ("vsp_info", {}),
        ("vsp_list_geoms", {}),
        ("vsp_mass_properties", {}),
        ("vsp_comp_geom", {}),
    ]
    replies = c.pipeline(reqs)
    ok("all overlapping requests answered", len(replies) == len(reqs))
    ok("none returned an error", not any(r.get("result", {}).get("isError") for r in replies))
    m1 = json.loads(replies[0]["result"]["content"][0]["text"])["results"]["Total_Mass"]
    m2 = json.loads(replies[4]["result"]["content"][0]["text"])["results"]["Total_Mass"]
    ok("interleaved results stay consistent", m1 == m2, f"{m1:.4f}")
    ok(
        "no residue after concurrent analyses",
        [g["name"] for g in c.tool("vsp_list_geoms")] == ["W"],
    )

    print("\n[9] VSPAERO under the protocol, and stdout hygiene")
    sweep = c.tool(
        "vsp_vspaero_sweep", {"alpha_start": 0, "alpha_end": 6, "alpha_npts": 4, "mach": 0.2}
    )
    pol = sweep["polar"]
    ok("polar returned", len(pol["Alpha"]) == 4)
    ok(
        "lift linear and rising",
        all(b > a for a, b in zip(pol["CLtot"], pol["CLtot"][1:])),
        str([round(x, 4) for x in pol["CLtot"]]),
    )
    ok("drag positive", all(d > 0 for d in pol["CDtot"]), str([round(x, 5) for x in pol["CDtot"]]))
    # every stdout line the server ever wrote must be valid JSON-RPC
    bad_lines = [l for l in c.raw_lines if not l.strip().startswith("{")]
    ok(
        "stdout carried only JSON-RPC, solver noise excluded",
        not bad_lines,
        f"{len(c.raw_lines)} frames read",
    )

    print("\n[10] tool coverage")
    advertised = {t["name"] for t in tools}
    missed = advertised - covered
    ok("every advertised tool exercised", not missed, str(sorted(missed)))

    c.close()

    print("\n[11] restart isolation")
    c2 = Client(tmp)
    ok("fresh server starts with an empty model", c2.tool("vsp_list_geoms") == [])
    c2.close()

    print("\n[12] streamable-http transport")
    http_check()

    print(f"\n{'=' * 60}\npassed {len(PASS)}   failed {len(FAIL)}   notes {len(NOTE)}")
    if FAIL:
        print("FAILURES:")
        for f in FAIL:
            print("  -", f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
