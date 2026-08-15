#!/usr/bin/env python3
"""End-to-end check: drives the server over real stdio JSON-RPC.

Exercises every tool against a live OpenVSP model, then asserts the properties
that earlier defects violated — repeatability, no geometry residue, and errors
surfacing instead of returning wrong answers.

    python tests/e2e.py

Exits non-zero on the first failure. Takes about a minute; VSPAERO dominates.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

SERVER = Path(__file__).resolve().parent.parent / "vsp_mcp.py"


class Client:
    """Minimal MCP stdio client."""

    def __init__(self, cwd: str):
        self.proc = subprocess.Popen(
            [sys.executable, str(SERVER)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1, cwd=cwd,
        )
        self._id = 0
        self.call("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "e2e", "version": "1"},
        })
        self.call("notifications/initialized", notify=True)

    def call(self, method, params=None, notify=False):
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        if notify:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
            return None
        self._id += 1
        msg["id"] = self._id
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("server closed the stream")
            d = json.loads(line)
            if d.get("id") == self._id:
                if "error" in d:
                    raise RuntimeError(f"{method} -> {d['error']}")
                return d["result"]

    def tool(self, name, args=None):
        res = self.call("tools/call", {"name": name, "arguments": args or {}})
        if res.get("isError"):
            raise RuntimeError(f"{name} -> {res['content'][0]['text']}")
        sc = res.get("structuredContent")
        if sc is None:
            return json.loads(res["content"][0]["text"])
        # Non-object returns arrive wrapped as exactly {"result": ...}; object
        # returns are the object itself, and one of them has a "result" field.
        return sc["result"] if set(sc) == {"result"} else sc

    def close(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=10)


def check(label, condition, detail=""):
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    if not condition:
        raise SystemExit(1)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="vsp_mcp_e2e_") as tmp:
        c = Client(tmp)
        out = lambda n: str(Path(tmp) / n)

        print("model and geometry")
        check("server reports version", c.tool("vsp_info")["version"].startswith("OpenVSP"))
        c.tool("vsp_new_model")
        wing = c.tool("vsp_add_geom", {"geom_type": "WING", "name": "MainWing"})
        c.tool("vsp_add_geom", {"geom_type": "FUSELAGE", "name": "Body"})
        geoms = c.tool("vsp_list_geoms")
        check("two components created", len(geoms) == 2,
              str([g["name"] for g in geoms]))

        print("parameters")
        # Section parms must be listed under the group name GetParm accepts,
        # so a listing can be fed straight back into a write.
        listing = c.tool("vsp_list_parms", {"geom_id": wing["id"]})
        check("sections listed individually",
              {"XSec_0", "XSec_1"} <= set(listing["groups"]),
              str([g for g in listing["groups"] if g.startswith("XSec")]))
        span = next(p for p in c.tool("vsp_list_parms",
                                      {"geom_id": wing["id"], "group": "XSec_1"})["parms"]
                    if p["name"] == "Span")
        applied = c.tool("vsp_set_parms", {"geom_id": wing["id"], "parms": [
            {"group": span["group"], "name": "Span", "value": 8.0}]})
        check("listing round-trips into a write", applied[0]["value"] == 8.0)

        print("analyses are repeatable and leave nothing behind")
        results = []
        for _ in range(2):
            c.tool("vsp_export", {"path": out("m.stl"), "export_format": "STL"})
            results.append((
                c.tool("vsp_comp_geom")["results"]["Total_Wet_Area"],
                c.tool("vsp_mass_properties")["results"]["Total_Mass"],
                Path(out("m.stl")).stat().st_size,
            ))
        check("repeated runs agree", results[0] == results[1], str(results[0]))
        names = [g["name"] for g in c.tool("vsp_list_geoms")]
        check("no mesh residue", names == ["MainWing", "Body"], str(names))

        print("export")
        check("STEP written", c.tool("vsp_export", {
            "path": out("m.step"), "export_format": "STEP"})["ok"])

        print("VSPAERO")
        # A component in both the thin and thick set aborts the solver, so the
        # wing and the fuselage must land in different ones.
        sweep = c.tool("vsp_vspaero_sweep", {
            "alpha_start": 0.0, "alpha_end": 6.0, "alpha_npts": 4, "mach": 0.25})
        split = sweep["surface_split"]
        check("surfaces split by type",
              split["thin"] == ["MainWing"] and split["thick"] == ["Body"], str(split))
        polar = sweep["polar"]
        check("polar has one point per alpha", len(polar["Alpha"]) == 4, str(polar["Alpha"]))
        check("lift rises with alpha",
              all(b > a for a, b in zip(polar["CLtot"], polar["CLtot"][1:])),
              str([round(x, 4) for x in polar["CLtot"]]))

        print("escape hatch and error handling")
        check("api script runs", c.tool("vsp_run_api_script", {
            "code": "result = vsp.GetGeomTypeName(vsp.FindGeoms()[0])"})["result"] == "Wing")
        try:
            c.tool("vsp_get_parms", {"geom_id": wing["id"],
                                     "parms": [{"group": "Nope", "name": "Nope"}]})
            check("bad parameter raises", False)
        except RuntimeError:
            check("bad parameter raises", True)

        c.close()
    print("\nall checks passed")


if __name__ == "__main__":
    main()
