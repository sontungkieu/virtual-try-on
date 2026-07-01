from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


RATING_COLUMNS = [
    "sample_id",
    "mode",
    "identity_1_5",
    "garment_fidelity_1_5",
    "realism_1_5",
    "pose_preservation_1_5",
    "artifact_1_5",
    "winner",
    "notes",
]


def _load_rows(benchmark_dir: Path) -> list[dict]:
    summary_path = benchmark_dir / "summary.json"
    if not summary_path.exists():
        return []
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return payload.get("rows", [])


def _relative_image(benchmark_dir: Path, value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        candidate = Path.cwd() / path
    else:
        candidate = path
    try:
        return candidate.resolve().relative_to(benchmark_dir.resolve()).as_posix()
    except ValueError:
        return value.replace("\\", "/")


def _image_cell(benchmark_dir: Path, value: str | None, label: str) -> str:
    rel = _relative_image(benchmark_dir, value)
    if not rel:
        return f"<div class='placeholder'>{html.escape(label)} skipped</div>"
    return f"<img src='{html.escape(rel)}' alt='{html.escape(label)}'>"


def _write_manual_ratings(benchmark_dir: Path, rows: list[dict]) -> Path:
    path = benchmark_dir / "manual_ratings.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RATING_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in RATING_COLUMNS})
    return path


def build_gallery(benchmark_dir: Path) -> Path:
    benchmark_dir = Path(benchmark_dir)
    rows = _load_rows(benchmark_dir)
    _write_manual_ratings(benchmark_dir, rows)
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row.get("sample_id", "unknown"), []).append(row)

    sections: list[str] = []
    for sample_id, sample_rows in sorted(grouped.items()):
        sample_rows = sorted(sample_rows, key=lambda row: row.get("mode", ""))
        first = sample_rows[0]
        person = _image_cell(benchmark_dir, first.get("input_person_path"), "person")
        garment = _image_cell(benchmark_dir, first.get("input_garment_path"), "garment")
        mode_cards = []
        for row in sample_rows:
            mode = html.escape(str(row.get("mode", "")))
            status = html.escape(str(row.get("status", "")))
            notes = html.escape(str(row.get("notes", "")))
            runtime = html.escape(str(row.get("runtime_seconds", "")))
            final_choice = html.escape(str(row.get("final_choice", "")))
            scores = (
                f"bg={row.get('background_preservation_score')} "
                f"face={row.get('face_preservation_score')} "
                f"garment={row.get('garment_change_score')} "
                f"over={row.get('over_edit_score')}"
            )
            image = _image_cell(benchmark_dir, row.get("output_path"), mode)
            mode_cards.append(
                "<article class='mode'>"
                f"<h3>{mode}</h3>{image}"
                f"<p><strong>Status:</strong> {status}</p>"
                f"<p><strong>Runtime:</strong> {runtime}s</p>"
                f"<p><strong>Final:</strong> {final_choice}</p>"
                f"<p><strong>Scores:</strong> {html.escape(scores)}</p>"
                f"<p><strong>Notes:</strong> {notes}</p>"
                "</article>"
            )
        sections.append(
            "<section class='sample'>"
            f"<h2>{html.escape(sample_id)}</h2>"
            "<div class='inputs'>"
            f"<article><h3>Person</h3>{person}</article>"
            f"<article><h3>Garment</h3>{garment}</article>"
            "</div>"
            "<div class='modes'>"
            + "".join(mode_cards)
            + "</div></section>"
        )

    body = "".join(sections) if sections else "<p>No benchmark rows found.</p>"
    html_text = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Virtual Try-On Review Gallery</title>"
        "<style>"
        "body{font-family:Arial,sans-serif;margin:24px;background:#f6f7f9;color:#20242a}"
        "h1,h2,h3{letter-spacing:0;margin:0 0 10px}"
        ".sample{margin:0 0 28px;padding:18px;background:white;border:1px solid #d7dce3;border-radius:6px}"
        ".inputs,.modes{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}"
        "article{border:1px solid #e0e4ea;border-radius:6px;padding:12px;background:#fff}"
        "img{max-width:100%;height:auto;display:block;border:1px solid #dde2e8;background:#fafafa}"
        ".placeholder{height:280px;display:flex;align-items:center;justify-content:center;background:#eef1f5;color:#667085;border:1px dashed #b7c0cc}"
        "p{font-size:13px;line-height:1.4;margin:6px 0;word-break:break-word}"
        "</style></head><body>"
        "<h1>Virtual Try-On Review Gallery</h1>"
        f"<p>Rows: {len(rows)}. Manual rating template: manual_ratings.csv</p>"
        f"{body}</body></html>"
    )
    output_path = benchmark_dir / "index.html"
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an offline benchmark review gallery.")
    parser.add_argument("--benchmark-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = build_gallery(Path(args.benchmark_dir))
    print(f"gallery={output_path}")
    print(f"manual_ratings={output_path.parent / 'manual_ratings.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
