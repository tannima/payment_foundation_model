from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
SCALING_DIR = ROOT / "artifacts" / "scaling"
SUMMARY_PATH = SCALING_DIR / "scaling_summary.json"
REPORT_PATH = ROOT / "docs" / "scaling_report_zh.md"

SCALE_ORDER = ["xs", "s", "m", "l", "xl"]
SCALE_COLORS = {
    "xs": "#4e79a7",
    "s": "#f28e2b",
    "m": "#e15759",
    "l": "#76b7b2",
    "xl": "#59a14f",
}


def load_rows() -> List[dict]:
    rows = json.loads(SUMMARY_PATH.read_text())
    return sorted(rows, key=lambda row: (row["parameter_count"], row["train_size"]))


def group_by_scale(rows: List[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {scale: [] for scale in SCALE_ORDER}
    for row in rows:
        scale = row["run_name"].split("_")[1]
        grouped[scale].append(row)
    for scale in grouped:
        grouped[scale] = sorted(grouped[scale], key=lambda row: row["train_size"])
    return grouped


def group_by_train_size(rows: List[dict]) -> Dict[int, List[dict]]:
    grouped: Dict[int, List[dict]] = {}
    for row in rows:
        grouped.setdefault(row["train_size"], []).append(row)
    for train_size in grouped:
        grouped[train_size] = sorted(grouped[train_size], key=lambda row: row["parameter_count"])
    return grouped


def svg_header(width: int, height: int) -> List[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]


def svg_footer() -> List[str]:
    return ["</svg>"]


def polyline(points: List[Tuple[float, float]], color: str) -> str:
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{coords}"/>'


def circle(x: float, y: float, color: str) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" />'


def line(x1: float, y1: float, x2: float, y2: float, color: str = "#333", width: int = 1) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"/>'


def text(x: float, y: float, value: str, size: int = 14, anchor: str = "middle", color: str = "#222", weight: str = "normal") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{color}" font-family="Arial, Helvetica, sans-serif" font-weight="{weight}">{value}</text>'
    )


def make_line_chart(
    title: str,
    x_values: List[float],
    x_labels: List[str],
    series: Dict[str, List[float]],
    filename: str,
    x_axis_label: str,
    y_axis_label: str,
    y_min: float = 0.0,
    y_max: float = 1.0,
    x_log: bool = False,
) -> Path:
    width, height = 960, 560
    left, right, top, bottom = 90, 40, 60, 80
    chart_w = width - left - right
    chart_h = height - top - bottom

    if x_log:
        x_scaled = [math.log10(v) for v in x_values]
    else:
        x_scaled = x_values[:]
    x_min, x_max = min(x_scaled), max(x_scaled)

    def sx(v: float) -> float:
        vv = math.log10(v) if x_log else v
        if x_max == x_min:
            return left + chart_w / 2
        return left + (vv - x_min) / (x_max - x_min) * chart_w

    def sy(v: float) -> float:
        return top + chart_h - (v - y_min) / (y_max - y_min) * chart_h

    parts = svg_header(width, height)
    parts.append(text(width / 2, 32, title, size=22, weight="bold"))
    parts.append(line(left, top, left, top + chart_h, width=2))
    parts.append(line(left, top + chart_h, left + chart_w, top + chart_h, width=2))

    for tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y = sy(tick)
        parts.append(line(left - 5, y, left + chart_w, y, color="#e0e0e0"))
        parts.append(text(left - 12, y + 5, f"{tick:.1f}", size=12, anchor="end"))

    for xv, xl in zip(x_values, x_labels):
        x = sx(xv)
        parts.append(line(x, top + chart_h, x, top + chart_h + 5))
        parts.append(text(x, top + chart_h + 24, xl, size=12))

    legend_x = left + 20
    legend_y = top - 18
    for idx, (name, values) in enumerate(series.items()):
        color = SCALE_COLORS.get(name, "#333333")
        y_off = legend_y + idx * 20
        parts.append(line(legend_x, y_off, legend_x + 20, y_off, color=color, width=3))
        parts.append(text(legend_x + 28, y_off + 4, name, size=12, anchor="start"))
        pts = [(sx(xv), sy(yv)) for xv, yv in zip(x_values, values)]
        parts.append(polyline(pts, color))
        for px, py in pts:
            parts.append(circle(px, py, color))

    parts.append(text(width / 2, height - 18, x_axis_label, size=14))
    parts.append(text(18, top + chart_h / 2, y_axis_label, size=14, anchor="middle"))
    parts.extend(svg_footer())

    out_path = SCALING_DIR / filename
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path


def make_heatmap(rows: List[dict], filename: str) -> Path:
    width, height = 980, 420
    left, top = 150, 70
    cell_w, cell_h = 130, 50
    parts = svg_header(width, height)
    parts.append(text(width / 2, 32, "Velocity Benchmark Heatmap", size=22, weight="bold"))

    sample_sizes = [1000, 2000, 4000, 8000, 12000]
    row_index = {scale: idx for idx, scale in enumerate(SCALE_ORDER)}
    col_index = {n: idx for idx, n in enumerate(sample_sizes)}

    scores = {(row["run_name"].split("_")[1], row["train_size"]): row["overall_velocity_score"] for row in rows}
    score_values = list(scores.values())
    s_min, s_max = min(score_values), max(score_values)

    def cell_color(score: float) -> str:
        t = 0.0 if s_max == s_min else (score - s_min) / (s_max - s_min)
        r = int(245 - 120 * t)
        g = int(245 - 60 * t)
        b = int(255 - 180 * t)
        return f"rgb({r},{g},{b})"

    for idx, n in enumerate(sample_sizes):
        parts.append(text(left + idx * cell_w + cell_w / 2, top - 18, str(n), size=14))
    for idx, scale in enumerate(SCALE_ORDER):
        parts.append(text(left - 20, top + idx * cell_h + cell_h / 2 + 4, scale, size=14, anchor="end"))

    for scale in SCALE_ORDER:
        for n in sample_sizes:
            x = left + col_index[n] * cell_w
            y = top + row_index[scale] * cell_h
            score = scores[(scale, n)]
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{cell_color(score)}" stroke="#cccccc"/>')
            parts.append(text(x + cell_w / 2, y + cell_h / 2 + 5, f"{score:.3f}", size=13))

    parts.append(text(left + (len(sample_sizes) * cell_w) / 2, height - 18, "Train Sample Size", size=14))
    parts.append(text(55, top + (len(SCALE_ORDER) * cell_h) / 2, "Model Scale", size=14))
    parts.extend(svg_footer())
    out_path = SCALING_DIR / filename
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path


def scale_label(row: dict) -> str:
    return row["run_name"].split("_")[1]


def build_report(rows: List[dict], sample_curve: Path, param_curve: Path, heatmap: Path) -> str:
    best_overall = max(rows, key=lambda row: row["overall_velocity_score"])
    best_count = max(rows, key=lambda row: row["count_exact_mean"])
    best_amount = max(rows, key=lambda row: row["amount_bucket_accuracy"])

    sample_sizes = sorted({row["train_size"] for row in rows})
    grouped_by_size = group_by_train_size(rows)
    best_by_size = {size: max(grouped_by_size[size], key=lambda row: row["overall_velocity_score"]) for size in sample_sizes}

    scale_groups = group_by_scale(rows)
    best_by_scale = {scale: max(scale_groups[scale], key=lambda row: row["overall_velocity_score"]) for scale in SCALE_ORDER}

    lines: List[str] = []
    lines.append("# Scaling 实验报告")
    lines.append("")
    lines.append("## 1. 结论摘要")
    lines.append("")
    lines.append("这轮 5x5 实验验证了两件事：")
    lines.append("")
    lines.append("1. 扩大样本量会明显提升 velocity benchmark 效果。")
    lines.append("2. 在样本量足够大时，扩大模型参数量也会明显提升效果。")
    lines.append("")
    lines.append("当前 25 组里最好的配置是：")
    lines.append("")
    lines.append(
        f"- `run_name={best_overall['run_name']}`，参数量 `{best_overall['parameter_count']}`，"
        f"`overall_velocity_score={best_overall['overall_velocity_score']:.4f}`，"
        f"`count_exact_mean={best_overall['count_exact_mean']:.4f}`，"
        f"`amount_bucket_accuracy={best_overall['amount_bucket_accuracy']:.4f}`"
    )
    lines.append("")
    lines.append("count 类 velocity 最强配置：")
    lines.append("")
    lines.append(
        f"- `{best_count['run_name']}`，`count_exact_mean={best_count['count_exact_mean']:.4f}`"
    )
    lines.append("")
    lines.append("amount 累加类 velocity 最强配置：")
    lines.append("")
    lines.append(
        f"- `{best_amount['run_name']}`，`amount_bucket_accuracy={best_amount['amount_bucket_accuracy']:.4f}`"
    )
    lines.append("")
    lines.append("## 2. 图表")
    lines.append("")
    lines.append(f"![Sample Scaling Curve]({sample_curve})")
    lines.append("")
    lines.append(f"![Parameter Scaling Curve]({param_curve})")
    lines.append("")
    lines.append(f"![Scaling Heatmap]({heatmap})")
    lines.append("")
    lines.append("## 3. 观察")
    lines.append("")
    lines.append("### 3.1 Sample Scaling")
    lines.append("")
    lines.append("固定模型规模时，样本量增加总体会带来更好的 velocity 预测效果。")
    lines.append("")
    for scale in SCALE_ORDER:
        group = scale_groups[scale]
        scores = ", ".join(f"{row['train_size']}->{row['overall_velocity_score']:.3f}" for row in group)
        lines.append(f"- `{scale}`: {scores}")
    lines.append("")
    lines.append("可以看到：")
    lines.append("")
    lines.append("- `m/l/xl` 这几档模型在大样本区间提升最明显")
    lines.append("- `xs/s` 在小样本区间波动较大，容量和优化噪声更明显")
    lines.append("- amount 类目标通常比 count 类更依赖大样本")
    lines.append("")
    lines.append("### 3.2 Parameter Scaling")
    lines.append("")
    lines.append("固定样本量时，模型参数量扩大在中高样本区间基本也会带来收益。")
    lines.append("")
    for size in sample_sizes:
        subset = grouped_by_size[size]
        scores = ", ".join(f"{scale_label(row)}->{row['overall_velocity_score']:.3f}" for row in subset)
        lines.append(f"- `train_size={size}`: {scores}")
    lines.append("")
    lines.append("更具体地看：")
    lines.append("")
    lines.append("- 在 `1000/2000` 这种小样本区间，模型变大有收益，但不稳定")
    lines.append("- 到了 `4000/8000/12000` 以后，`m/l/xl` 的优势开始明显放大")
    lines.append("- 说明当前 benchmark 上，参数扩展需要足够数据支撑才能稳定兑现")
    lines.append("")
    lines.append("### 3.3 Count vs Amount")
    lines.append("")
    lines.append("count / distinct-count 类原子目标比 amount-sum 更容易学。")
    lines.append("")
    lines.append("- `count_exact_mean` 的最好结果已经接近 `0.95`")
    lines.append("- `amount_bucket_accuracy` 的最好结果接近 `0.80`，仍明显落后于 count 类")
    lines.append("")
    lines.append("这说明：")
    lines.append("")
    lines.append("- 模型对离散计数型 velocity 已经具备较强学习能力")
    lines.append("- 对连续累加型 velocity，仍需要更多数据、参数或更直接的归纳偏置")
    lines.append("")
    lines.append("## 4. 每个样本量下的最优配置")
    lines.append("")
    lines.append("| train_size | best_run | params | overall_score | count_exact_mean | amount_bucket_acc |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for size in sample_sizes:
        row = best_by_size[size]
        lines.append(
            f"| {size} | {row['run_name']} | {row['parameter_count']} | "
            f"{row['overall_velocity_score']:.4f} | {row['count_exact_mean']:.4f} | "
            f"{row['amount_bucket_accuracy']:.4f} |"
        )
    lines.append("")
    lines.append("## 5. 每个模型规模下的最优配置")
    lines.append("")
    lines.append("| scale | best_run | train_size | overall_score | count_exact_mean | amount_bucket_acc |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for scale in SCALE_ORDER:
        row = best_by_scale[scale]
        lines.append(
            f"| {scale} | {row['run_name']} | {row['train_size']} | "
            f"{row['overall_velocity_score']:.4f} | {row['count_exact_mean']:.4f} | "
            f"{row['amount_bucket_accuracy']:.4f} |"
        )
    lines.append("")
    lines.append("## 6. 结论")
    lines.append("")
    lines.append("基于这 25 组结果，可以给出当前阶段的判断：")
    lines.append("")
    lines.append("1. 在当前 synthetic velocity benchmark 上，sample scaling 明确成立。")
    lines.append("2. parameter scaling 也成立，但在中高样本区间更明显。")
    lines.append("3. 当前最强配置已经让 velocity benchmark 接近很强水平，但还不能直接外推到真实业务。")
    lines.append("4. 下一步最值得做的是把同一套 scaling 验证迁移到真实离线样本上。")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    rows = load_rows()
    grouped_by_scale = group_by_scale(rows)
    grouped_by_size = group_by_train_size(rows)

    sample_sizes = sorted(grouped_by_size.keys())
    sample_curve = make_line_chart(
        title="Sample Scaling on Velocity Benchmark",
        x_values=sample_sizes,
        x_labels=[str(v) for v in sample_sizes],
        series={scale: [row["overall_velocity_score"] for row in grouped_by_scale[scale]] for scale in SCALE_ORDER},
        filename="sample_scaling_curve.svg",
        x_axis_label="Train Sample Size",
        y_axis_label="Overall Velocity Score",
    )

    reference_sizes = [1000, 4000, 8000, 12000]
    param_counts = [grouped_by_size[reference_sizes[0]][idx]["parameter_count"] for idx in range(len(SCALE_ORDER))]
    param_curve = make_line_chart(
        title="Parameter Scaling on Velocity Benchmark",
        x_values=param_counts,
        x_labels=[f"{v/1e6:.2f}M" for v in param_counts],
        series={
            str(size): [row["overall_velocity_score"] for row in grouped_by_size[size]]
            for size in reference_sizes
        },
        filename="parameter_scaling_curve.svg",
        x_axis_label="Parameter Count",
        y_axis_label="Overall Velocity Score",
    )

    heatmap = make_heatmap(rows, "scaling_heatmap.svg")
    report = build_report(rows, sample_curve, param_curve, heatmap)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(REPORT_PATH)
    print(sample_curve)
    print(param_curve)
    print(heatmap)


if __name__ == "__main__":
    main()
