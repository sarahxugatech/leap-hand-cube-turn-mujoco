"""Render the Prompt 29 diagnostic trace as a dependency-free SVG."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "logs" / "prompt29_ring_breakthrough_trace.csv"
OUTPUT = ROOT / "logs" / "prompt29_ring_breakthrough_diagnostic.svg"
OUTPUT_PNG = ROOT / "logs" / "prompt29_ring_breakthrough_diagnostic.png"
WIDTH, HEIGHT = 1200, 1120
LEFT, RIGHT = 92, 1155
PANEL_H = 205
PANEL_TOPS = (100, 355, 610, 865)
COLORS = {"mcp": "#2563eb", "pip": "#d97706", "dip": "#059669"}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}", size)


def draw_dashed(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    color: str,
    width: int = 2,
) -> None:
    for index in range(0, len(points) - 1, 2):
        draw.line(points[index : index + 2], fill=color, width=width)


def render_png(
    joint_values: dict[str, list[float]],
    errors: dict[str, list[float]],
    actuator_forces: dict[str, list[float]],
    normal_force: list[float],
    angles: list[float],
    ranges: tuple[float, float, float, float, float, float, float, float],
) -> None:
    joint_lo, joint_hi, error_limit, actuator_limit, force_lo, force_hi, angle_lo, angle_hi = ranges
    image = Image.new("RGB", (WIDTH, HEIGHT), "#f8fafc")
    draw = ImageDraw.Draw(image)
    draw.text((LEFT, 24), "Ring-finger breakthrough diagnostic", fill="#172033", font=font(26, True))
    draw.text((LEFT, 58), "KF02 9.539 deg + (MCP +0.08, PIP -0.06, DIP -0.10 rad); dashed vertical = transition/hold", fill="#526174", font=font(15))

    def axes(top: float, title: str, lo: float, hi: float, unit: str) -> None:
        draw.text((LEFT, top - 24), title, fill="#172033", font=font(17, True))
        draw.rectangle((LEFT, top, RIGHT, top + PANEL_H), fill="white", outline="#94a3b8", width=1)
        for tick in range(5):
            value = lo + tick * (hi - lo) / 4
            y = scale(value, lo, hi, top + PANEL_H, top)
            draw.line((LEFT, y, RIGHT, y), fill="#e2e8f0", width=1)
            label = f"{value:.3f}"
            box = draw.textbbox((0, 0), label, font=font(12))
            draw.text((LEFT - 12 - (box[2] - box[0]), y - 7), label, fill="#64748b", font=font(12))
        for step in (1, 150, 300, 450, 650):
            x = scale(step, 1, 650, LEFT, RIGHT)
            draw.line((x, top, x, top + PANEL_H), fill="#e2e8f0", width=1)
            label = str(step)
            box = draw.textbbox((0, 0), label, font=font(12))
            draw.text((x - (box[2] - box[0]) / 2, top + PANEL_H + 5), label, fill="#64748b", font=font(12))
        boundary = scale(150, 1, 650, LEFT, RIGHT)
        for y in range(int(top), int(top + PANEL_H), 10):
            draw.line((boundary, y, boundary, min(y + 5, top + PANEL_H)), fill="#7c3aed", width=2)
        draw.text((18, top + PANEL_H / 2 - 7), unit, fill="#475569", font=font(12))

    def points(values: list[float], lo: float, hi: float, top: float) -> list[tuple[float, float]]:
        return [
            (scale(i + 1, 1, len(values), LEFT, RIGHT), scale(value, lo, hi, top + PANEL_H, top))
            for i, value in enumerate(values)
        ]

    axes(PANEL_TOPS[0], "Joint targets (solid) vs actual angles (dashed)", joint_lo, joint_hi, "rad")
    for short in ("mcp", "pip", "dip"):
        draw.line(points(joint_values[f"{short}_target"], joint_lo, joint_hi, PANEL_TOPS[0]), fill=COLORS[short], width=3)
        draw_dashed(draw, points(joint_values[f"{short}_actual"], joint_lo, joint_hi, PANEL_TOPS[0]), COLORS[short])
    draw.text((900, PANEL_TOPS[0] + 8), "MCP blue / PIP orange / DIP green", fill="#526174", font=font(12))

    axes(PANEL_TOPS[1], "Target tracking error", -error_limit, error_limit, "mrad")
    for short in ("mcp", "pip", "dip"):
        draw.line(points(errors[short], -error_limit, error_limit, PANEL_TOPS[1]), fill=COLORS[short], width=2)

    axes(PANEL_TOPS[2], "Actuator scalar force (no configured force limit)", -actuator_limit, actuator_limit, "force")
    for short in ("mcp", "pip", "dip"):
        draw.line(points(actuator_forces[short], -actuator_limit, actuator_limit, PANEL_TOPS[2]), fill=COLORS[short], width=2)

    axes(PANEL_TOPS[3], "Ring/face normal force (red) and face angle (purple)", force_lo, force_hi, "N")
    draw.line(points(normal_force, force_lo, force_hi, PANEL_TOPS[3]), fill="#dc2626", width=3)
    draw.line(points(angles, angle_lo, angle_hi, PANEL_TOPS[3]), fill="#7c3aed", width=3)
    for threshold, color, label in ((0.35, "#ea580c", "0.35 N cap"), (0.50, "#dc2626", "0.50 N")):
        y = scale(threshold, force_lo, force_hi, PANEL_TOPS[3] + PANEL_H, PANEL_TOPS[3])
        for x in range(LEFT, RIGHT, 14):
            draw.line((x, y, min(x + 8, RIGHT), y), fill=color, width=2)
        draw.text((RIGHT - 110, y - 18), label, fill=color, font=font(12, True))
    draw.text((RIGHT + 8, PANEL_TOPS[3]), f"{angle_hi:.0f} deg", fill="#7c3aed", font=font(12))
    draw.text((RIGHT + 8, PANEL_TOPS[3] + PANEL_H - 14), f"{angle_lo:.0f} deg", fill="#7c3aed", font=font(12))
    image.save(OUTPUT_PNG)


def scale(value: float, lo: float, hi: float, a: float, b: float) -> float:
    return a + (value - lo) * (b - a) / (hi - lo)


def path(values: list[float], lo: float, hi: float, top: float) -> str:
    points = []
    for index, value in enumerate(values):
        x = scale(index + 1, 1, len(values), LEFT, RIGHT)
        y = scale(value, lo, hi, top + PANEL_H, top)
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def panel(lines: list[str], title: str, top: float, lo: float, hi: float, unit: str) -> None:
    lines.append(f'<text x="{LEFT}" y="{top - 16}" class="panel-title">{title}</text>')
    lines.append(f'<rect x="{LEFT}" y="{top}" width="{RIGHT-LEFT}" height="{PANEL_H}" class="frame"/>')
    for tick in range(5):
        value = lo + tick * (hi - lo) / 4
        y = scale(value, lo, hi, top + PANEL_H, top)
        lines.append(f'<line x1="{LEFT}" y1="{y:.2f}" x2="{RIGHT}" y2="{y:.2f}" class="grid"/>')
        lines.append(f'<text x="{LEFT-10}" y="{y+4:.2f}" text-anchor="end" class="tick">{value:.3f}</text>')
    for step in (1, 150, 300, 450, 650):
        x = scale(step, 1, 650, LEFT, RIGHT)
        lines.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+PANEL_H}" class="grid"/>')
        lines.append(f'<text x="{x:.2f}" y="{top+PANEL_H+20}" text-anchor="middle" class="tick">{step}</text>')
    transition_x = scale(150, 1, 650, LEFT, RIGHT)
    lines.append(f'<line x1="{transition_x:.2f}" y1="{top}" x2="{transition_x:.2f}" y2="{top+PANEL_H}" class="phase"/>')
    lines.append(f'<text x="{LEFT-66}" y="{top+PANEL_H/2:.2f}" text-anchor="middle" class="axis" transform="rotate(-90 {LEFT-66} {top+PANEL_H/2:.2f})">{unit}</text>')


def main() -> None:
    with INPUT.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    steps = len(rows)
    if steps != 650:
        raise RuntimeError(f"expected 650 samples, got {steps}")

    joint_values: dict[str, list[float]] = {}
    errors: dict[str, list[float]] = {}
    actuator_forces: dict[str, list[float]] = {}
    for short in ("mcp", "pip", "dip"):
        joint_values[f"{short}_target"] = [float(row[f"rf_{short}_target_rad"]) for row in rows]
        joint_values[f"{short}_actual"] = [float(row[f"rf_{short}_actual_rad"]) for row in rows]
        errors[short] = [1000 * float(row[f"rf_{short}_error_rad"]) for row in rows]
        actuator_forces[short] = [float(row[f"rf_{short}_actuator_force"]) for row in rows]
    normal_force = [float(row["normal_force_n"]) for row in rows]
    angles = [float(row["face_angle_deg"]) for row in rows]

    joint_lo = min(min(values) for values in joint_values.values()) - 0.04
    joint_hi = max(max(values) for values in joint_values.values()) + 0.04
    error_limit = max(10.0, math.ceil(max(abs(value) for values in errors.values() for value in values)))
    actuator_limit = 1.1 * max(abs(value) for values in actuator_forces.values() for value in values)
    force_hi = 0.70
    angle_lo = math.floor(min(angles))
    angle_hi = math.ceil(max(angles))

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<style>",
        "text{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;fill:#172033}",
        ".title{font-size:24px;font-weight:600}.subtitle{font-size:15px;fill:#526174}",
        ".panel-title{font-size:17px;font-weight:600}.tick{font-size:12px;fill:#64748b}",
        ".axis{font-size:13px;fill:#475569}.frame{fill:#ffffff;stroke:#94a3b8;stroke-width:1}",
        ".grid{stroke:#e2e8f0;stroke-width:1}.phase{stroke:#7c3aed;stroke-width:1.5;stroke-dasharray:6 5}",
        ".series{fill:none;stroke-width:2}.actual{fill:none;stroke-width:1.5;stroke-dasharray:5 4}",
        ".threshold{stroke:#dc2626;stroke-width:1.5;stroke-dasharray:7 5}.limit{stroke:#ea580c;stroke-width:1.5;stroke-dasharray:4 4}",
        "</style>",
        '<rect width="1200" height="1120" fill="#f8fafc"/>',
        '<text x="92" y="42" class="title">Ring-finger breakthrough diagnostic</text>',
        '<text x="92" y="68" class="subtitle">KF02 9.539° + (MCP +0.08, PIP −0.06, DIP −0.10 rad) · vertical dashed line = transition/hold boundary</text>',
    ]

    panel(svg, "Joint targets (solid) vs actual angles (dashed)", PANEL_TOPS[0], joint_lo, joint_hi, "joint angle (rad)")
    for short in ("mcp", "pip", "dip"):
        color = COLORS[short]
        svg.append(f'<polyline points="{path(joint_values[f"{short}_target"], joint_lo, joint_hi, PANEL_TOPS[0])}" fill="none" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<polyline points="{path(joint_values[f"{short}_actual"], joint_lo, joint_hi, PANEL_TOPS[0])}" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="7 5"/>')
    svg.append('<text x="930" y="118" class="tick">MCP blue · PIP orange · DIP green</text>')

    panel(svg, "Target tracking error", PANEL_TOPS[1], -error_limit, error_limit, "target − actual (mrad)")
    for short in ("mcp", "pip", "dip"):
        svg.append(f'<polyline points="{path(errors[short], -error_limit, error_limit, PANEL_TOPS[1])}" fill="none" stroke="{COLORS[short]}" stroke-width="2"/>')

    panel(svg, "Actuator scalar force (no configured force limit)", PANEL_TOPS[2], -actuator_limit, actuator_limit, "actuator force")
    for short in ("mcp", "pip", "dip"):
        svg.append(f'<polyline points="{path(actuator_forces[short], -actuator_limit, actuator_limit, PANEL_TOPS[2])}" fill="none" stroke="{COLORS[short]}" stroke-width="2"/>')

    panel(svg, "Ring/face normal force and face-layer angle", PANEL_TOPS[3], 0.0, force_hi, "normal force (N)")
    svg.append(f'<polyline points="{path(normal_force, 0.0, force_hi, PANEL_TOPS[3])}" fill="none" stroke="#dc2626" stroke-width="3"/>')
    for threshold, css, label in ((0.35, "limit", "0.35 N strict cap"), (0.50, "threshold", "0.50 N")):
        y = scale(threshold, 0.0, force_hi, PANEL_TOPS[3] + PANEL_H, PANEL_TOPS[3])
        svg.append(f'<line x1="{LEFT}" y1="{y:.2f}" x2="{RIGHT}" y2="{y:.2f}" class="{css}"/>')
        svg.append(f'<text x="{RIGHT-8}" y="{y-5:.2f}" text-anchor="end" class="tick">{label}</text>')
    angle_points = path(angles, angle_lo, angle_hi, PANEL_TOPS[3])
    svg.append(f'<polyline points="{angle_points}" fill="none" stroke="#7c3aed" stroke-width="3"/>')
    svg.append(f'<text x="{RIGHT+8}" y="{PANEL_TOPS[3]+14}" class="tick">angle {angle_hi}°</text>')
    svg.append(f'<text x="{RIGHT+8}" y="{PANEL_TOPS[3]+PANEL_H}" class="tick">{angle_lo}°</text>')
    svg.append(f'<text x="{LEFT}" y="{HEIGHT-16}" class="axis">simulation step</text>')
    svg.append("</svg>")
    OUTPUT.write_text("\n".join(svg) + "\n", encoding="utf-8")
    render_png(
        joint_values,
        errors,
        actuator_forces,
        normal_force,
        angles,
        (joint_lo, joint_hi, error_limit, actuator_limit, 0.0, force_hi, angle_lo, angle_hi),
    )
    print(OUTPUT)
    print(OUTPUT_PNG)


if __name__ == "__main__":
    main()
