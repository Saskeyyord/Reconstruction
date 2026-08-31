from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLORS = {
    "ink": "#25313C",
    "muted": "#68737D",
    "line": "#7B8791",
    "clean": "#DCEAF7",
    "clean_edge": "#477AA8",
    "noise": "#F7DFDD",
    "noise_edge": "#B85C58",
    "process": "#ECEAF4",
    "process_edge": "#746D9B",
    "input": "#DDF0EE",
    "input_edge": "#3E8F91",
    "loss": "#F4E8F0",
    "loss_edge": "#9A4D8E",
    "neutral": "#F3F5F6",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "Microsoft JhengHei",
                "SimHei",
                "Arial",
                "DejaVu Sans",
            ],
            "font.size": 6.5,
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def add_box(
    axis: mpl.axes.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    facecolor: str,
    edgecolor: str,
    title_size: float = 7.0,
    body_size: float = 6.0,
    linestyle: str = "-",
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=0.9,
        edgecolor=edgecolor,
        facecolor=facecolor,
        linestyle=linestyle,
        transform=axis.transAxes,
        clip_on=False,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height * 0.72,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=COLORS["ink"],
        transform=axis.transAxes,
    )
    axis.text(
        x + width / 2,
        y + height * 0.34,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        color=COLORS["ink"],
        linespacing=1.35,
        transform=axis.transAxes,
    )


def add_arrow(
    axis: mpl.axes.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str | None = None,
    connectionstyle: str = "arc3,rad=0",
    linestyle: str = "-",
    label_offset: tuple[float, float] = (0.0, 0.018),
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        transform=axis.transAxes,
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=0.9,
        color=COLORS["line"],
        connectionstyle=connectionstyle,
        linestyle=linestyle,
        shrinkA=2,
        shrinkB=2,
        clip_on=False,
    )
    axis.add_patch(arrow)
    if label:
        axis.text(
            (start[0] + end[0]) / 2 + label_offset[0],
            (start[1] + end[1]) / 2 + label_offset[1],
            label,
            ha="center",
            va="center",
            fontsize=5.5,
            color=COLORS["muted"],
            transform=axis.transAxes,
        )


def panel_label(axis: mpl.axes.Axes, x: float, y: float, label: str, title: str) -> None:
    axis.text(
        x,
        y,
        label,
        fontsize=8,
        fontweight="bold",
        color=COLORS["ink"],
        va="center",
        transform=axis.transAxes,
    )
    axis.text(
        x + 0.028,
        y,
        title,
        fontsize=8,
        fontweight="bold",
        color=COLORS["ink"],
        va="center",
        transform=axis.transAxes,
    )


def draw_pipeline(output: Path) -> list[Path]:
    setup_style()
    fig, axis = plt.subplots(figsize=(7.09, 5.55))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    axis.text(
        0.5,
        0.973,
        "CycloSCGNet：从20秒标准记录到实际训练输入",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        color=COLORS["ink"],
        transform=axis.transAxes,
    )
    axis.text(
        0.5,
        0.935,
        "一个训练样本 = 一个干净12心搏窗口 + 一个等长随机噪声窗口",
        ha="center",
        va="top",
        fontsize=7,
        color=COLORS["muted"],
        transform=axis.transAxes,
    )

    panel_label(axis, 0.018, 0.885, "a", "按受试者划分后的训练数据池")
    add_box(
        axis,
        0.035,
        0.715,
        0.22,
        0.13,
        "干净静息SCG池",
        "25名训练受试者 × 每人20 s\nSCG + ECG，共500 s",
        COLORS["clean"],
        COLORS["clean_edge"],
    )
    add_box(
        axis,
        0.035,
        0.535,
        0.22,
        0.13,
        "运动噪声代理池",
        "4名训练噪声受试者 × 4位置 × 20 s\n16条记录，共320 s",
        COLORS["noise"],
        COLORS["noise_edge"],
    )
    add_box(
        axis,
        0.305,
        0.715,
        0.22,
        0.13,
        "随机选干净心搏窗口",
        "ECG检测R峰 → 随机13个连续R峰\n形成12个心搏（约7–12 s）",
        COLORS["clean"],
        COLORS["clean_edge"],
    )
    add_box(
        axis,
        0.305,
        0.535,
        0.22,
        0.13,
        "随机选等长噪声窗口",
        "随机人/位置/起点；长度与12心搏一致\n超出20 s末尾时循环取样",
        COLORS["noise"],
        COLORS["noise_edge"],
    )
    add_box(
        axis,
        0.575,
        0.615,
        0.18,
        0.17,
        "动态混合",
        "x = s + polarity × αn\nSNR ~ U(−18, 2) dB；极性 ±\n输出一对：(污染输入x，干净标签s)\n15%概率：x = s（identity）",
        COLORS["process"],
        COLORS["process_edge"],
    )
    add_box(
        axis,
        0.805,
        0.595,
        0.17,
        0.21,
        "同一R峰相位归一化",
        "每个心搏插值为256点\n污染输入 X：[12,256]\n干净标签 S：[12,256]\n二者除以同一clean RMS",
        COLORS["input"],
        COLORS["input_edge"],
    )

    add_arrow(axis, (0.255, 0.78), (0.305, 0.78))
    add_arrow(axis, (0.255, 0.60), (0.305, 0.60))
    add_arrow(axis, (0.525, 0.78), (0.575, 0.73))
    add_arrow(axis, (0.525, 0.60), (0.575, 0.67))
    add_arrow(axis, (0.755, 0.70), (0.805, 0.70))
    add_box(
        axis,
        0.035,
        0.415,
        0.49,
        0.065,
        "真实步行SCG不进入上述监督训练",
        "70条仅做ECG质控；主外部验证使用5名测试受试者的10条步行记录",
        COLORS["neutral"],
        COLORS["line"],
        title_size=6.5,
        body_size=5.4,
        linestyle="--",
    )
    add_box(
        axis,
        0.575,
        0.415,
        0.40,
        0.065,
        "在线组合空间",
        "25 × 16 = 400种基础记录配对；再随机心搏位置、噪声起点、SNR和极性",
        COLORS["neutral"],
        COLORS["line"],
        title_size=6.5,
        body_size=5.2,
        linestyle="--",
    )

    panel_label(axis, 0.018, 0.365, "b", "DataLoader组成批次并更新轻量级网络")
    y, h = 0.205, 0.115
    add_box(
        axis,
        0.035,
        y,
        0.13,
        h,
        "一个batch",
        "动态生成32个样本\n输入X：[32×12×256]\n标签S：[32×12×256]",
        COLORS["input"],
        COLORS["input_edge"],
        body_size=5.0,
    )
    add_box(
        axis,
        0.195,
        y,
        0.13,
        h,
        "共享心搏编码器",
        "12个心搏共享一套参数\n1 → 16 → 32 → 64",
        COLORS["process"],
        COLORS["process_edge"],
        body_size=5.4,
    )
    add_box(
        axis,
        0.355,
        y,
        0.13,
        h,
        "跨心搏注意力",
        "沿12个心搏交换信息\n4 heads",
        COLORS["process"],
        COLORS["process_edge"],
        body_size=5.4,
    )
    add_box(
        axis,
        0.515,
        y,
        0.13,
        h,
        "心搏共识模块",
        "学习可靠性权重w1…w12\nsum w(k) = 1",
        COLORS["process"],
        COLORS["process_edge"],
        body_size=5.4,
    )
    add_box(
        axis,
        0.675,
        y,
        0.12,
        h,
        "共享解码器",
        "64 → 32 → 16 → 1\n残差重构",
        COLORS["process"],
        COLORS["process_edge"],
        body_size=5.4,
    )
    add_box(
        axis,
        0.825,
        y,
        0.15,
        h,
        "预测干净SCG",
        "Ŝ = X + ΔS\n输出：[32,12,256]",
        COLORS["input"],
        COLORS["input_edge"],
        body_size=5.5,
    )
    for left, right in ((0.165, 0.195), (0.325, 0.355), (0.485, 0.515), (0.645, 0.675), (0.795, 0.825)):
        add_arrow(axis, (left, y + h / 2), (right, y + h / 2))
    add_arrow(
        axis,
        (0.89, y),
        (0.78, 0.13),
        connectionstyle="arc3,rad=-0.12",
    )
    add_arrow(
        axis,
        (0.10, y),
        (0.23, 0.13),
        label="干净标签S",
        connectionstyle="arc3,rad=0.10",
        label_offset=(0.0, -0.012),
    )
    add_box(
        axis,
        0.225,
        0.055,
        0.555,
        0.085,
        "综合训练损失",
        "SmoothL1波形 + 跨心搏相关矩阵 + phase-Gram + 奇异值谱 + 多分辨率STFT + identity保持",
        COLORS["loss"],
        COLORS["loss_edge"],
        title_size=6.8,
        body_size=5.25,
    )
    add_box(
        axis,
        0.825,
        0.055,
        0.15,
        0.085,
        "反向传播",
        "更新139,010个参数",
        COLORS["neutral"],
        COLORS["line"],
        title_size=6.5,
        body_size=5.5,
    )
    add_arrow(axis, (0.78, 0.098), (0.825, 0.098))
    axis.text(
        0.5,
        0.012,
        "v2：1024个动态样本/epoch，batch=32，24 epochs；24,576是随机窗口数，不是独立受试者数。",
        ha="center",
        va="bottom",
        fontsize=5.5,
        color=COLORS["muted"],
        transform=axis.transAxes,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    paths = [
        output.with_suffix(".svg"),
        output.with_suffix(".pdf"),
        output.with_suffix(".tiff"),
        output.with_suffix(".png"),
    ]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    fig.savefig(paths[2], dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(paths[3], dpi=300, bbox_inches="tight")
    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw the CycloSCGNet training-data pipeline")
    parser.add_argument(
        "--output",
        default="results/method_figures/training_data_pipeline",
    )
    args = parser.parse_args()
    paths = draw_pipeline(Path(args.output))
    print("Saved: " + ", ".join(str(path) for path in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
