"""Deterministic engineering checks for VisionForge AI.

These functions perform dimensional fit screening and transparent checklist scoring.
They are not CAD collision analysis, tolerance analysis, or engineering certification.
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE_DIR / "outputs" / "engineering_reviews"


def _positive_number(value: Any, label: str, allow_zero: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number in millimeters.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number.")
    if number < 0 if allow_zero else number <= 0:
        relation = "zero or greater" if allow_zero else "greater than zero"
        raise ValueError(f"{label} must be {relation}.")
    if number > 100000:
        raise ValueError(f"{label} is implausibly large; enter dimensions in millimeters.")
    return number


def parse_component_specs(text: str) -> list[dict[str, Any]]:
    """Parse CSV-style rows: name, length_mm, width_mm, height_mm."""
    if not (text or "").strip():
        raise ValueError(
            "Enter at least one component row: Name, length_mm, width_mm, height_mm."
        )
    rows: list[dict[str, Any]] = []
    reader = csv.reader((line for line in text.splitlines() if line.strip()))
    for line_number, columns in enumerate(reader, start=1):
        if len(columns) != 4:
            raise ValueError(
                f"Component line {line_number} needs exactly four comma-separated fields: "
                "name, length, width, height."
            )
        name = columns[0].strip()
        if not name:
            raise ValueError(f"Component line {line_number} needs a name.")
        if len(name) > 100:
            raise ValueError(f"Component name on line {line_number} is too long.")
        dimensions = [
            _positive_number(columns[i].strip(), f"{name} dimension {i}")
            for i in range(1, 4)
        ]
        rows.append({"name": name, "length_mm": dimensions[0], "width_mm": dimensions[1], "height_mm": dimensions[2]})
        if len(rows) > 100:
            raise ValueError("Enter no more than 100 components per check.")
    return rows


def split_features(text: str) -> list[str]:
    """Accept one requirement per line, or comma/semicolon-separated values."""
    parts = re.split(r"[\n,;]+", text or "")
    result: list[str] = []
    seen: set[str] = set()
    for part in parts:
        item = part.strip().strip("-•* ")
        key = re.sub(r"\s+", " ", item.casefold())
        if item and key not in seen:
            result.append(item)
            seen.add(key)
    return result


def _feature_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def _brief_audit(description: str, priorities: list[str] | None, required: str, envelope: str, avoid: str) -> dict[str, Any]:
    checks = [
        ("Product purpose", len((description or "").split()) >= 8,
         "Describe the main job and intended user in a complete sentence."),
        ("Design priorities", bool(priorities),
         "Select the most important size, usability, lifecycle, or engineering goals."),
        ("Must-have features", bool(split_features(required)),
         "List the controls, ports, modules, and features that must remain present."),
        ("Dimensions and interfaces", bool((envelope or "").strip()),
         "Add approximate external dimensions and critical interfaces."),
        ("Explicit exclusions", bool((avoid or "").strip()),
         "Describe design approaches or features that should be avoided."),
    ]
    return {
        "score_pct": round(sum(1 for _, okay, _ in checks if okay) / len(checks) * 100),
        "items": [
            {"name": name, "addressed": bool(okay), "suggestion": suggestion}
            for name, okay, suggestion in checks
        ],
    }


def evaluate_design(
    enclosure_length: Any,
    enclosure_width: Any,
    enclosure_height: Any,
    clearance: Any,
    component_specs: str,
    required_features: str,
    observed_features: str,
    human_rating: Any,
    description: str = "",
    priorities: list[str] | None = None,
    envelope_notes: str = "",
    avoid_notes: str = "",
) -> dict[str, Any]:
    """Calculate per-component fit, manual feature coverage, and a disclosed review index.

    Each component is tested independently against the internal envelope in all six
    axis-aligned orientations. Components are not packed together, so this does not
    assess collisions or whether all parts can coexist in the enclosure.
    """
    length = _positive_number(enclosure_length, "Enclosure length")
    width = _positive_number(enclosure_width, "Enclosure width")
    height = _positive_number(enclosure_height, "Enclosure height")
    wall_clearance = _positive_number(clearance, "Clearance per side", allow_zero=True)
    usable = (length - 2 * wall_clearance, width - 2 * wall_clearance, height - 2 * wall_clearance)
    if any(dim <= 0 for dim in usable):
        raise ValueError("Clearance leaves no positive internal space. Reduce clearance or enlarge the enclosure.")

    components = parse_component_specs(component_specs)
    results: list[dict[str, Any]] = []
    for component in components:
        dims = (component["length_mm"], component["width_mm"], component["height_mm"])
        viable: list[tuple[float, float, float]] = []
        for orientation in set(itertools.permutations(dims)):
            if all(oriented <= room for oriented, room in zip(orientation, usable)):
                viable.append(orientation)
        if viable:
            # Choose the orientation with the smallest total unused linear margin.
            best = min(viable, key=lambda orientation: sum(room - part for room, part in zip(usable, orientation)))
            margins = tuple(round(room - part, 2) for room, part in zip(usable, best))
            results.append({
                **component,
                "fits_individually": True,
                "oriented_dimensions_mm": [round(v, 2) for v in best],
                "remaining_margin_mm": list(margins),
                "status": "Fits individually",
            })
        else:
            min_dimensions = sorted(dims)
            usable_sorted = sorted(usable)
            results.append({
                **component,
                "fits_individually": False,
                "oriented_dimensions_mm": None,
                "remaining_margin_mm": None,
                "status": "Does not fit in any axis-aligned orientation",
                "smallest_dimension_gaps_mm": [round(max(0.0, part - room), 2) for part, room in zip(min_dimensions, usable_sorted)],
            })

    fit_pct = round(100 * sum(r["fits_individually"] for r in results) / len(results))
    required = split_features(required_features)
    observed = split_features(observed_features)
    observed_keys = {_feature_key(value) for value in observed}
    matched = [value for value in required if _feature_key(value) in observed_keys]
    missing = [value for value in required if _feature_key(value) not in observed_keys]
    coverage_pct: int | None = round(100 * len(matched) / len(required)) if required else None
    try:
        visual = int(human_rating)
    except (TypeError, ValueError) as exc:
        raise ValueError("Human visual rating must be an integer from 1 to 5.") from exc
    if visual < 1 or visual > 5:
        raise ValueError("Human visual rating must be from 1 to 5.")
    visual_pct = (visual - 1) * 25
    audit = _brief_audit(description, priorities, required_features, envelope_notes, avoid_notes)

    # Weights are a transparent internal review index, not an external contest score.
    scored_metrics: list[tuple[str, float, float]] = [
        ("Individual component fit", float(fit_pct), 0.35),
        ("Brief completeness", float(audit["score_pct"]), 0.20),
        ("Human visual rating", float(visual_pct), 0.15),
    ]
    if coverage_pct is not None:
        scored_metrics.append(("Manually confirmed feature coverage", float(coverage_pct), 0.30))
    denominator = sum(weight for _, _, weight in scored_metrics)
    review_index = round(sum(score * weight for _, score, weight in scored_metrics) / denominator) if denominator else 0

    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "enclosure_external_mm": {"length": length, "width": width, "height": height},
        "clearance_per_side_mm": wall_clearance,
        "estimated_internal_envelope_mm": {"length": round(usable[0], 2), "width": round(usable[1], 2), "height": round(usable[2], 2)},
        "component_count": len(results),
        "component_fit_pct": fit_pct,
        "component_results": results,
        "required_features": required,
        "manually_observed_features": observed,
        "matched_features": matched,
        "missing_features": missing,
        "feature_coverage_pct": coverage_pct,
        "brief_audit": audit,
        "human_visual_rating_1_to_5": visual,
        "human_visual_rating_pct": visual_pct,
        "review_index_0_to_100": review_index,
        "weighting": {name: weight for name, _, weight in scored_metrics},
        "interpretation": (
            "The review index is an explicitly weighted screening aid using individual component fit, "
            "brief topic coverage, manually confirmed features when supplied, and the user's visual rating. "
            "It is not a validated engineering score or an AI claim of correctness."
        ),
        "limitations": [
            "Each component is tested individually against the internal envelope; component-to-component packing and collisions are not checked.",
            "Only axis-aligned 90-degree orientations are considered.",
            "Cable routing, connectors, screw bosses, wall thickness details, tolerances, heat, acoustics, battery safety, and electrical function are not simulated.",
            "Feature coverage is a manual observation supplied by the user, not automatically detected ground truth.",
            "A passing dimensional check does not prove that the design is safe, manufacturable, or functional.",
        ],
    }


def render_engineering_report(result: dict[str, Any]) -> str:
    box = result["estimated_internal_envelope_mm"]
    lines = [
        "# VisionForge AI — Engineering Design Review",
        "",
        f"- **Created:** {result['created_at']}",
        f"- **External enclosure (L × W × H):** {result['enclosure_external_mm']['length']} × {result['enclosure_external_mm']['width']} × {result['enclosure_external_mm']['height']} mm",
        f"- **Estimated internal envelope:** {box['length']} × {box['width']} × {box['height']} mm",
        f"- **Assumed clearance per side:** {result['clearance_per_side_mm']} mm",
        f"- **Components individually fitting:** {sum(r['fits_individually'] for r in result['component_results'])}/{result['component_count']} ({result['component_fit_pct']}%)",
        f"- **Brief completeness topics detected:** {result['brief_audit']['score_pct']}%",
        f"- **Manually confirmed must-have features:** {result['feature_coverage_pct'] if result['feature_coverage_pct'] is not None else 'Not scored — no must-have features entered'}",
        f"- **Human visual rating:** {result['human_visual_rating_1_to_5']}/5",
        f"- **Transparent review index:** {result['review_index_0_to_100']}/100",
        "",
        "> The review index is a local screening aid, not a validated engineering grade.",
        "",
        "## Component fit results",
        "| Component | Input L × W × H (mm) | Result | Chosen orientation / notes |",
        "|---|---:|---|---|",
    ]
    for item in result["component_results"]:
        dims = f"{item['length_mm']} × {item['width_mm']} × {item['height_mm']}"
        if item["fits_individually"]:
            orientation = " × ".join(str(v) for v in item["oriented_dimensions_mm"]) + " mm; margins " + " × ".join(str(v) for v in item["remaining_margin_mm"])
        else:
            orientation = "No tested orientation fits"
        lines.append(f"| {item['name']} | {dims} | {'PASS' if item['fits_individually'] else 'FAIL'} | {orientation} |")
    lines += ["", "## Required-feature checklist", "| Feature | Status |", "|---|---|"]
    for item in result["required_features"]:
        lines.append(f"| {item} | {'Manually marked observed' if item in result['matched_features'] else 'Not marked observed'} |")
    if not result["required_features"]:
        lines.append("| No requirements entered | Not scored |")
    lines += ["", "## Brief completeness checklist", "| Topic | Addressed by current text? | Improvement suggestion |", "|---|---|---|"]
    for item in result["brief_audit"]["items"]:
        lines.append(f"| {item['name']} | {'Yes' if item['addressed'] else 'No'} | {item['suggestion']} |")
    lines += ["", "## Limitations and assumptions"]
    lines.extend(f"- {item}" for item in result["limitations"])
    lines += ["", result["interpretation"], ""]
    return "\n".join(lines)


def save_engineering_report(result: dict[str, Any], output_root: str | Path | None = None) -> str:
    """Persist an auditable JSON/Markdown report and a ZIP bundle; returns the ZIP path."""
    root = Path(output_root) if output_root else REPORTS_DIR
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"review_{stamp}"
    suffix = 2
    while run_dir.exists():
        run_dir = root / f"review_{stamp}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    json_path = run_dir / "engineering_review.json"
    md_path = run_dir / "engineering_review.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_engineering_report(result), encoding="utf-8")
    zip_path = run_dir / "engineering_review_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(json_path, arcname=json_path.name)
        archive.write(md_path, arcname=md_path.name)
    return str(zip_path)
