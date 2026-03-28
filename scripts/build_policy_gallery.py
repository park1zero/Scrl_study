#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _status(summary: dict) -> str:
    if summary.get("collision", 0):
        return "collision"
    if summary.get("success", 0):
        return "success"
    return "running"


def _discover_entries(base_dir: Path) -> list[tuple[str, dict]]:
    entries: list[tuple[str, dict]] = []
    for summary_path in sorted(base_dir.glob("*_summary.json")):
        name = summary_path.stem.removesuffix("_summary")
        gif_path = base_dir / f"{name}.gif"
        if not gif_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        if isinstance(summary, dict) and "summary" in summary and isinstance(summary["summary"], dict):
            summary = summary["summary"]
        entries.append((name, summary))
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a simple HTML gallery for rendered policy GIFs.")
    parser.add_argument("--dir", type=str, default="artifacts/animations")
    parser.add_argument("--title", type=str, default="SBW shared control animation gallery")
    parser.add_argument("--subtitle", type=str, default="Baselines and trained authority-allocation policies.")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    base_dir = Path(args.dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    entries = _discover_entries(base_dir)

    html = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'/>",
        f"<title>{args.title}</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:20px;} .card{border:1px solid #ccc;border-radius:12px;padding:16px;} img{width:100%;height:auto;border-radius:8px;} pre{white-space:pre-wrap;background:#f8f8f8;padding:12px;border-radius:8px;} code{background:#f0f0f0;padding:2px 6px;border-radius:6px;}</style></head><body>",
        f"<h1>{args.title}</h1>",
        f"<p>{args.subtitle}</p>",
        "<div class='grid'>",
    ]

    for name, summary in entries:
        title = name.replace("_", " ")
        lines = "\n".join(
            f"{k}: {v:.4f}" if isinstance(v, (float, int)) else f"{k}: {v}"
            for k, v in summary.items()
        )
        html.append(
            f"<div class='card'><h2>{title}</h2><p>{_status(summary)}</p><img src='{name}.gif' alt='{title} animation'/><pre>{lines}</pre></div>"
        )

    html.extend(["</div>", "</body></html>"])
    out_path = Path(args.out) if args.out is not None else base_dir / "policy_gallery.html"
    out_path.write_text("".join(html), encoding="utf-8")
    print(f"Saved gallery to {out_path}")


if __name__ == "__main__":
    main()
