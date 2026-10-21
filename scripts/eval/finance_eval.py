"""Ask a deployed BuffettRAG app every finance_v1 question of one half and store the full responses.

python scripts/eval/finance_eval.py --url URL --name staging1 [--set dev]
-> <runs dir>/<name>/responses.jsonl (one line per qid: answer, citations, served hit ids, retrieved hit ids, seconds)
Resumable: qids already in the file are skipped. The site password is read from the secrets file, never printed.
--set test is the frozen half: it refuses to run without --i-am-the-main-agent-final-run and reads the case file from the
Artifacts workspace (the test half is never copied into the repo).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = Path.home() / "AI-Workplace/Artifacts/BuffettRAG/agent-runs/finance-testset"
CASE_FILES = {
    "dev": REPO / "data/evaluation/finance_v1/dev.json",
    "test": WORKSPACE / "out/finance_v1_test.json",
}
RUNS_DIR = WORKSPACE / "runs"
SECRET = Path.home() / "AI-Workplace/Artifacts/BuffettRAG/cloud/secrets/site_password.env"
FINAL_RUN_FLAG = "--i-am-the-main-agent-final-run"


def case_file(set_name: str, final_run: bool = False) -> Path:
    if set_name not in CASE_FILES:
        raise SystemExit(f"unknown set {set_name!r}")
    if set_name == "test" and not final_run:
        raise SystemExit(f"--set test is the frozen half: refused without {FINAL_RUN_FLAG}")
    return CASE_FILES[set_name]


def load_cases(set_name: str, final_run: bool = False) -> list[dict]:
    return json.loads(case_file(set_name, final_run).read_text())["cases"]


def site_password() -> str:
    for line in SECRET.read_text().splitlines():
        if line.startswith("SITE_PASSWORD="):
            return line.split("=", 1)[1].strip()
    raise SystemExit(f"SITE_PASSWORD not found in {SECRET}")


def record(case: dict, half: str, resp, data: dict, seconds: float) -> dict:
    return {"qid": case["qid"], "half": half, "status": getattr(resp, "status_code", None), "seconds": round(seconds, 1),
            "answer": data.get("answer"), "citations": data.get("citations"), "used_filter": data.get("used_filter"),
            # pre-registered measure: the anchors the reader was served (the API's `hits`)
            "served_ids": [h.get("id") for h in data.get("hits") or []],
            # FINANCE_DEBUG only: anchors plus the neighbour chunks merged into their text (reported separately)
            "served_ids_merged": (data.get("debug") or {}).get("served_ids"),
            "retrieved_ids": [h.get("id") for h in data.get("retrieved_hits") or []],
            "error": data.get("error") or data.get("detail"),
            **({"debug": data["debug"]} if data.get("debug") is not None else {})}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="deployed app base URL")
    ap.add_argument("--name", required=True, help="run name: <runs dir>/<name>/responses.jsonl")
    ap.add_argument("--set", default="dev", choices=["dev", "test"], dest="set_name")
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    ap.add_argument(FINAL_RUN_FLAG, action="store_true", dest="final_run")
    args = ap.parse_args(argv)
    cases = load_cases(args.set_name, args.final_run)
    auth = ("buffett", site_password())
    url = args.url.rstrip("/")
    out_dir = args.runs_dir / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "responses.jsonl"
    done = {json.loads(l)["qid"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    t0 = time.time()
    while time.time() - t0 < 900:                       # wait for a cold start
        try:
            if requests.get(url + "/api/ready", auth=auth, timeout=60).ok:
                break
        except requests.RequestException:
            pass
        time.sleep(5)
    print(f"ready after {time.time() - t0:.0f}s", flush=True)
    with out.open("a") as fh:
        for c in cases:
            if c["qid"] in done:
                continue
            resp, data = None, {}
            for _ in range(3):
                t = time.time()
                try:
                    resp = requests.post(url + "/api/ask", json={"query": c["query"]}, auth=auth, timeout=900)
                    data = resp.json()
                    break
                except Exception as exc:  # noqa: BLE001
                    data = {"error": str(exc)[:300]}
                    time.sleep(10)
            rec = record(c, args.set_name, resp, data, time.time() - t)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"{c['qid']} {args.set_name} {rec['status']} {rec['seconds']}s served={len(rec['served_ids'])}", flush=True)
    print(f"done in {time.time() - t0:.0f}s", file=sys.stdout)


if __name__ == "__main__":
    main()
