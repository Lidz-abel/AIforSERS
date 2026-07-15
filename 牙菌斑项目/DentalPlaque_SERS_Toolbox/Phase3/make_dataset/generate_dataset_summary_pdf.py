"""Generate a PDF summary for the Phase3 SERS dataset.

The report is built directly from frozen Phase3 artifacts:
  - Results/Phase3/dataset/spectra.npz
  - Results/Phase3/dataset/wavenumber.npy
  - Results/Phase3/splits/split_seed42.json

It intentionally does not modify any dataset or split file.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


TOOLBOX = Path(__file__).resolve().parents[2]
DATASET_PATH = TOOLBOX / "Results" / "Phase3" / "dataset" / "spectra.npz"
WAVENUMBER_PATH = TOOLBOX / "Results" / "Phase3" / "dataset" / "wavenumber.npy"
SPLIT_PATH = TOOLBOX / "Results" / "Phase3" / "splits" / "split_seed42.json"
OUT_PATH = TOOLBOX / "Results" / "Phase3" / "DATASET_SUMMARY_REPORT.pdf"


def get_font() -> FontProperties:
    candidates = [
        Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    for path in candidates:
        if path.exists():
            return FontProperties(fname=str(path))
    return FontProperties()


FONT = get_font()
BOLD_FONT = FontProperties(fname=FONT.get_file(), weight="bold") if FONT.get_file() else FONT


def add_title(ax, title: str, subtitle: str | None = None):
    ax.text(0.04, 0.94, title, fontproperties=BOLD_FONT, fontsize=22, va="top", color="#15202b")
    if subtitle:
        ax.text(0.04, 0.885, subtitle, fontproperties=FONT, fontsize=10.5, va="top", color="#4a5568")


def add_footer(fig, page: int):
    fig.text(
        0.04,
        0.025,
        "Data source: Results/Phase3/dataset/spectra.npz, Results/Phase3/dataset/wavenumber.npy, Results/Phase3/splits/split_seed42.json",
        fontproperties=FONT,
        fontsize=7.5,
        color="#6b7280",
    )
    fig.text(0.955, 0.025, f"{page}", fontproperties=FONT, fontsize=8, color="#6b7280", ha="right")


def draw_box(ax, xy, width, height, title, body, color="#e8f3ff", edge="#2563eb"):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=color,
        edgecolor=edge,
        linewidth=1.2,
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height - 0.035, title, fontproperties=BOLD_FONT, fontsize=11, ha="center", va="top", color="#111827")
    ax.text(xy[0] + width / 2, xy[1] + height / 2 - 0.015, body, fontproperties=FONT, fontsize=8.7, ha="center", va="center", color="#374151")


def draw_arrow(ax, start, end):
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16, linewidth=1.3, color="#64748b")
    ax.add_patch(arrow)


def draw_split_summary_box(ax, xy, width, height, title, lines):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor="#f8fafc",
        edgecolor="#64748b",
        linewidth=1.2,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height - 0.040,
        title,
        fontproperties=BOLD_FONT,
        fontsize=11.5,
        ha="center",
        va="top",
        color="#111827",
    )
    for i, line in enumerate(lines):
        ax.text(
            xy[0] + width / 2,
            xy[1] + height - 0.105 - i * 0.060,
            line,
            fontproperties=FONT,
            fontsize=9.2,
            ha="center",
            va="top",
            color="#475569",
        )


def load_data():
    data = np.load(DATASET_PATH, allow_pickle=True)
    wavenumber = np.load(WAVENUMBER_PATH)
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))

    labels = data["labels"].astype(int)
    patient_index = data["patient_index"].astype(int)
    patient_uids = data["patient_uids"].astype(str)
    x_spectra = data["X_spectra"]
    x_raw = data["X_raw_spectra"]

    unique_pids = np.arange(len(patient_uids))
    patient_labels = np.array([labels[patient_index == pid][0] for pid in unique_pids], dtype=int)
    patient_counts = np.array([(patient_index == pid).sum() for pid in unique_pids], dtype=int)
    uid_to_idx = {uid: i for i, uid in enumerate(patient_uids)}

    split_patient_indices = {}
    split_spectrum_counts = {}
    split_label_counts = {}
    for name, key in [("train", "train_patients"), ("val", "val_patients"), ("test", "test_patients")]:
        idx = np.array([uid_to_idx[uid] for uid in split[key]], dtype=int)
        split_patient_indices[name] = idx
        split_spectrum_counts[name] = int(patient_counts[idx].sum())
        split_label_counts[name] = {
            0: int((patient_labels[idx] == 0).sum()),
            1: int((patient_labels[idx] == 1).sum()),
        }

    return {
        "data": data,
        "wavenumber": wavenumber,
        "labels": labels,
        "patient_index": patient_index,
        "patient_uids": patient_uids,
        "patient_labels": patient_labels,
        "patient_counts": patient_counts,
        "split": split,
        "split_patient_indices": split_patient_indices,
        "split_spectrum_counts": split_spectrum_counts,
        "split_label_counts": split_label_counts,
        "x_spectra": x_spectra,
        "x_raw": x_raw,
    }


def page_overview(pdf: PdfPages, ctx: dict):
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    add_title(ax, "牙菌斑 SERS 数据集成果总结", "Phase3A_v1.0 | 患者级监督学习数据集 | 二分类: 阴性=0, 阳性=1")

    facts = [
        ("患者数", f"{len(ctx['patient_uids'])}", "临床独立样本单位"),
        ("光谱数", f"{len(ctx['labels'])}", "训练实例可使用单条 spectrum"),
        ("谱维度", f"{ctx['x_spectra'].shape[1]}", "每条光谱 732 个 Raman shift 点"),
        ("Raman shift", f"{ctx['wavenumber'].min():.2f}-{ctx['wavenumber'].max():.2f} cm$^{{-1}}$", "来自 wavenumber.npy"),
    ]
    x0, y0 = 0.055, 0.72
    for i, (k, v, note) in enumerate(facts):
        x = x0 + i * 0.225
        ax.add_patch(Rectangle((x, y0), 0.2, 0.12, facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1))
        ax.text(x + 0.02, y0 + 0.087, k, fontproperties=FONT, fontsize=10, color="#475569")
        ax.text(x + 0.02, y0 + 0.045, v, fontproperties=BOLD_FONT, fontsize=18, color="#0f172a")
        ax.text(x + 0.02, y0 + 0.018, note, fontproperties=FONT, fontsize=7.5, color="#64748b")

    ax.text(0.055, 0.64, "核心贡献", fontproperties=BOLD_FONT, fontsize=15, color="#111827")
    contributions = [
        "1. 将原始 CSV SERS 光谱整理为统一的建模张量: X_spectra = (1970, 732), 同时保留 X_raw_spectra。",
        "2. 明确患者级标签继承规则: 每条 spectrum 的标签来自其所属患者, 避免人为给单条谱重复定义标签。",
        "3. 固定患者级 train/val/test 划分, 同一患者全部光谱只进入一个集合, 防止患者级数据泄露。",
        "4. 同时记录 patient_index、patient_uids、spectrum_ids, 支持 spectrum-level 训练和 patient-level 评估。",
        "5. 暴露并量化类别不平衡与患者光谱数不均衡, 为后续 patient-balanced loss 和 patient aggregation 提供依据。",
    ]
    yy = 0.59
    for item in contributions:
        ax.text(0.07, yy, item, fontproperties=FONT, fontsize=10.5, va="top", color="#1f2937")
        yy -= 0.052

    ax.text(0.055, 0.30, "数据集定义", fontproperties=BOLD_FONT, fontsize=15, color="#111827")
    ax.add_patch(Rectangle((0.065, 0.165), 0.86, 0.10, facecolor="#fefce8", edgecolor="#facc15", linewidth=1.2))
    ax.text(
        0.09,
        0.225,
        "D = { (x_i, y_p, p_i) }",
        fontproperties=BOLD_FONT,
        fontsize=18,
        color="#713f12",
        va="center",
    )
    ax.text(
        0.34,
        0.225,
        "x_i: 第 i 条 SERS 光谱 | y_p: 患者标签 | p_i: 所属患者 ID",
        fontproperties=FONT,
        fontsize=11,
        color="#713f12",
        va="center",
    )
    ax.text(
        0.07,
        0.125,
        "训练时可使用单条 spectrum 作为 instance; 但验证和测试必须回到 patient-level prediction, 因为临床决策单位是患者。",
        fontproperties=FONT,
        fontsize=10,
        color="#374151",
    )

    add_footer(fig, 1)
    pdf.savefig(fig)
    plt.close(fig)


def page_pipeline(pdf: PdfPages):
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_title(ax, "数据集构造流程", "从原始 SERS CSV 到可训练、可审计、患者级无泄露的数据集")

    boxes = [
        ((0.06, 0.62), "原始数据", "Raw CSV\n按患者文件夹组织\n阳性/阴性两组"),
        ((0.25, 0.62), "光谱抽取", "读取 Raman shift\n读取 intensity\n统一 732 点"),
        ((0.44, 0.62), "患者映射", "生成 patient_index\n生成 spectrum_id\n保留 patient_uid"),
        ((0.63, 0.62), "标签继承", "患者标签 -> 光谱标签\n阴性=0, 阳性=1\n二分类任务"),
        ((0.82, 0.62), "冻结数据集", "spectra.npz\nmetadata.csv\nwavenumber.npy"),
    ]
    for xy, title, body in boxes:
        draw_box(ax, xy, 0.13, 0.16, title, body)
    for i in range(len(boxes) - 1):
        draw_arrow(ax, (boxes[i][0][0] + 0.13, 0.70), (boxes[i + 1][0][0], 0.70))

    lower = [
        ((0.16, 0.35), "SNV 归一化", "X_spectra\n逐条光谱独立标准化\n不使用患者/类别信息", "#ecfdf5", "#10b981"),
        ((0.43, 0.35), "患者级划分", "train/val/test\n按 patient_uid 划分\n禁止患者跨集合", "#eff6ff", "#3b82f6"),
        ((0.70, 0.35), "建模使用", "spectrum-level 训练\npatient-level 聚合评估\n防止泄露", "#fff7ed", "#f97316"),
    ]
    for xy, title, body, color, edge in lower:
        draw_box(ax, xy, 0.19, 0.16, title, body, color=color, edge=edge)
    draw_arrow(ax, (0.50, 0.62), (0.255, 0.51))
    draw_arrow(ax, (0.885, 0.62), (0.525, 0.51))
    draw_arrow(ax, (0.62, 0.43), (0.70, 0.43))
    draw_arrow(ax, (0.35, 0.43), (0.43, 0.43))

    ax.text(0.07, 0.23, "严谨性约束", fontproperties=BOLD_FONT, fontsize=14, color="#111827")
    notes = [
        "Normalization: SNV 是逐条光谱独立操作, 不利用标签、患者分组或全局统计量。",
        "Split first: 所有光谱扩展和训练实例构造都必须在患者级划分之后进行。",
        "Evaluation unit: 任何最终性能结论应基于 patient-level aggregation, 不能只报告 spectrum-level 指标。",
    ]
    y = 0.19
    for note in notes:
        ax.text(0.085, y, note, fontproperties=FONT, fontsize=10.5, color="#374151")
        y -= 0.045

    add_footer(fig, 2)
    pdf.savefig(fig)
    plt.close(fig)


def page_composition(pdf: PdfPages, ctx: dict):
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.subplots_adjust(left=0.08, right=0.96, top=0.83, bottom=0.10, wspace=0.28, hspace=0.36)
    fig.text(0.04, 0.94, "数据组成与不平衡结构", fontproperties=BOLD_FONT, fontsize=22, color="#15202b")
    fig.text(0.04, 0.895, "关键结论: 数据是患者级独立, 但光谱数在类别和患者之间明显不均衡。", fontproperties=FONT, fontsize=10.5, color="#4a5568")

    patient_labels = ctx["patient_labels"]
    labels = ctx["labels"]
    patient_counts = ctx["patient_counts"]
    colors = ["#64748b", "#ef4444"]

    ax = axes[0, 0]
    patient_count_by_label = [int((patient_labels == 0).sum()), int((patient_labels == 1).sum())]
    ax.bar(["阴性\nlabel=0", "阳性\nlabel=1"], patient_count_by_label, color=colors, width=0.55)
    ax.set_title("患者级类别分布", fontproperties=BOLD_FONT, fontsize=13)
    ax.set_ylabel("患者数", fontproperties=FONT)
    for i, v in enumerate(patient_count_by_label):
        ax.text(i, v + 0.6, str(v), ha="center", fontproperties=BOLD_FONT, fontsize=12)
    ax.tick_params(axis="x", labelsize=10)

    ax = axes[0, 1]
    spectrum_count_by_label = [int((labels == 0).sum()), int((labels == 1).sum())]
    ax.bar(["阴性\nlabel=0", "阳性\nlabel=1"], spectrum_count_by_label, color=colors, width=0.55)
    ax.set_title("光谱级类别分布", fontproperties=BOLD_FONT, fontsize=13)
    ax.set_ylabel("光谱数", fontproperties=FONT)
    for i, v in enumerate(spectrum_count_by_label):
        ax.text(i, v + 30, str(v), ha="center", fontproperties=BOLD_FONT, fontsize=12)

    ax = axes[1, 0]
    neg_counts = patient_counts[patient_labels == 0]
    pos_counts = patient_counts[patient_labels == 1]
    ax.hist(neg_counts, bins=np.arange(19.5, 83.5, 5), alpha=0.75, color=colors[0], label="阴性患者")
    ax.hist(pos_counts, bins=np.arange(19.5, 83.5, 5), alpha=0.65, color=colors[1], label="阳性患者")
    ax.set_title("每位患者贡献的光谱数", fontproperties=BOLD_FONT, fontsize=13)
    ax.set_xlabel("每位患者光谱数", fontproperties=FONT)
    ax.set_ylabel("患者数", fontproperties=FONT)
    ax.legend(prop=FONT, frameon=False)

    ax = axes[1, 1]
    idx = np.arange(len(patient_counts))
    order = np.argsort(patient_counts)
    sorted_counts = patient_counts[order]
    sorted_labels = patient_labels[order]
    ax.bar(idx, sorted_counts, color=[colors[l] for l in sorted_labels], width=0.8)
    ax.set_title("患者光谱数排序图", fontproperties=BOLD_FONT, fontsize=13)
    ax.set_xlabel("患者 (按光谱数排序)", fontproperties=FONT)
    ax.set_ylabel("光谱数", fontproperties=FONT)
    ax.text(
        0.03,
        0.92,
        f"min={patient_counts.min()}, median={np.median(patient_counts):.0f}, mean={patient_counts.mean():.1f}, max={patient_counts.max()}",
        transform=ax.transAxes,
        fontproperties=FONT,
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="#cbd5e1", alpha=0.9),
    )

    for ax in axes.ravel():
        ax.grid(axis="y", alpha=0.25)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(FONT)

    add_footer(fig, 3)
    pdf.savefig(fig)
    plt.close(fig)


def page_split(pdf: PdfPages, ctx: dict):
    fig = plt.figure(figsize=(11.69, 8.27))
    gs = fig.add_gridspec(2, 2, left=0.08, right=0.96, top=0.82, bottom=0.12, wspace=0.30, hspace=0.34)
    fig.text(0.04, 0.94, "患者级划分与无泄露约束", fontproperties=BOLD_FONT, fontsize=22, color="#15202b")
    fig.text(0.04, 0.895, "seed=42, split unit=patient, ratios=60%/20%/20%; 同一患者的所有光谱只进入一个集合。", fontproperties=FONT, fontsize=10.5, color="#4a5568")

    split_names = ["train", "val", "test"]
    split_labels = ["Train", "Val", "Test"]
    neg = [ctx["split_label_counts"][s][0] for s in split_names]
    pos = [ctx["split_label_counts"][s][1] for s in split_names]
    spectra = [ctx["split_spectrum_counts"][s] for s in split_names]
    colors = {"neg": "#64748b", "pos": "#ef4444", "spec": "#2563eb"}

    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(3)
    ax.bar(x, neg, label="阴性患者", color=colors["neg"], width=0.55)
    ax.bar(x, pos, bottom=neg, label="阳性患者", color=colors["pos"], width=0.55)
    ax.set_xticks(x, split_labels)
    ax.set_ylabel("患者数", fontproperties=FONT)
    ax.set_title("患者级 train/val/test 分布", fontproperties=BOLD_FONT, fontsize=13)
    for i in range(3):
        ax.text(i, neg[i] / 2, str(neg[i]), ha="center", va="center", color="white", fontproperties=BOLD_FONT)
        ax.text(i, neg[i] + pos[i] / 2, str(pos[i]), ha="center", va="center", color="white", fontproperties=BOLD_FONT)
        ax.text(i, neg[i] + pos[i] + 0.6, str(neg[i] + pos[i]), ha="center", fontproperties=BOLD_FONT)
    ax.legend(prop=FONT, frameon=False)
    ax.grid(axis="y", alpha=0.25)

    ax = fig.add_subplot(gs[0, 1])
    ax.bar(split_labels, spectra, color=colors["spec"], width=0.55)
    ax.set_ylabel("光谱数", fontproperties=FONT)
    ax.set_title("各集合对应光谱数", fontproperties=BOLD_FONT, fontsize=13)
    for i, v in enumerate(spectra):
        ax.text(i, v + 20, str(v), ha="center", fontproperties=BOLD_FONT)
    ax.grid(axis="y", alpha=0.25)

    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    draw_split_summary_box(ax, (0.05, 0.46), 0.22, 0.28, "Train patients", ["31 patients", "18 阳性 / 13 阴性", "1197 spectra"])
    draw_split_summary_box(ax, (0.39, 0.46), 0.22, 0.28, "Validation patients", ["10 patients", "6 阳性 / 4 阴性", "381 spectra"])
    draw_split_summary_box(ax, (0.73, 0.46), 0.22, 0.28, "Test patients", ["11 patients", "7 阳性 / 4 阴性", "392 spectra"])
    ax.text(0.50, 0.29, "Leakage check: patient overlap = 0", fontproperties=BOLD_FONT, fontsize=18, color="#166534", ha="center")
    ax.text(
        0.50,
        0.17,
        "严格规则: 不能用 spectrum-level random split。否则同一患者的相似光谱可能同时出现在训练和测试中, 导致性能虚高。",
        fontproperties=FONT,
        fontsize=11,
        color="#374151",
        ha="center",
    )

    for ax in fig.axes:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(FONT)

    add_footer(fig, 4)
    pdf.savefig(fig)
    plt.close(fig)


def page_examples_and_limits(pdf: PdfPages, ctx: dict):
    fig = plt.figure(figsize=(11.69, 8.27))
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.96, top=0.82, bottom=0.12, wspace=0.28, hspace=0.34)
    fig.text(0.04, 0.94, "光谱表示与使用边界", fontproperties=BOLD_FONT, fontsize=22, color="#15202b")
    fig.text(0.04, 0.895, "本页展示 raw spectrum 与 SNV spectrum 的形态, 并说明该数据集在后续建模中的正确使用方式。", fontproperties=FONT, fontsize=10.5, color="#4a5568")

    w = ctx["wavenumber"]
    raw = ctx["x_raw"]
    snv = ctx["x_spectra"]
    labels = ctx["labels"]
    patient_index = ctx["patient_index"]

    ax = fig.add_subplot(gs[0, 0])
    for label, color, name in [(0, "#64748b", "阴性"), (1, "#ef4444", "阳性")]:
        idx = np.where(labels == label)[0][:25]
        mean = raw[idx].mean(axis=0)
        ax.plot(w, mean, color=color, linewidth=1.8, label=f"{name} raw mean")
    ax.set_title("Raw spectra: 类别均值示例", fontproperties=BOLD_FONT, fontsize=13)
    ax.set_xlabel("Raman shift (cm$^{-1}$)", fontproperties=FONT)
    ax.set_ylabel("Raw intensity", fontproperties=FONT)
    ax.legend(prop=FONT, frameon=False)
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[0, 1])
    for label, color, name in [(0, "#64748b", "阴性"), (1, "#ef4444", "阳性")]:
        idx = np.where(labels == label)[0][:25]
        mean = snv[idx].mean(axis=0)
        ax.plot(w, mean, color=color, linewidth=1.8, label=f"{name} SNV mean")
    ax.set_title("SNV spectra: 建模输入示例", fontproperties=BOLD_FONT, fontsize=13)
    ax.set_xlabel("Raman shift (cm$^{-1}$)", fontproperties=FONT)
    ax.set_ylabel("SNV intensity", fontproperties=FONT)
    ax.legend(prop=FONT, frameon=False)
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    left = [
        ("推荐使用", "spectrum-level training + patient-level aggregation"),
        ("推荐指标", "AUC, Acc, BalAcc, Sens, Spec, Brier, ECE"),
        ("推荐权重", "patient-balanced loss"),
    ]
    right = [
        ("禁止", "spectrum-level random split"),
        ("谨慎", "仅报告单次 seed 结果"),
        ("限制", "无外部验证; 当前为内部患者级验证数据集"),
    ]
    ax.text(0.06, 0.86, "后续建模建议", fontproperties=BOLD_FONT, fontsize=14, color="#111827")
    ax.text(0.62, 0.86, "风险与限制", fontproperties=BOLD_FONT, fontsize=14, color="#111827")
    y = 0.72
    for key, val in left:
        ax.text(0.07, y, key, fontproperties=BOLD_FONT, fontsize=11, color="#166534")
        ax.text(0.20, y, val, fontproperties=FONT, fontsize=9.8, color="#374151")
        y -= 0.13
    y = 0.72
    for key, val in right:
        ax.text(0.62, y, key, fontproperties=BOLD_FONT, fontsize=11, color="#991b1b")
        ax.text(0.72, y, val, fontproperties=FONT, fontsize=9.8, color="#374151")
        y -= 0.13

    ax.text(
        0.06,
        0.16,
        f"NaN/Inf check: NaN={bool(np.isnan(snv).any())}, Inf={bool(np.isinf(snv).any())}. "
        f"Unique patients={len(np.unique(patient_index))}, unique spectra={len(labels)}.",
        fontproperties=FONT,
        fontsize=10.5,
        color="#374151",
    )

    for ax in fig.axes:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(FONT)

    add_footer(fig, 5)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ctx = load_data()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    with PdfPages(OUT_PATH) as pdf:
        page_overview(pdf, ctx)
        page_pipeline(pdf)
        page_composition(pdf, ctx)
        page_split(pdf, ctx)
        page_examples_and_limits(pdf, ctx)
    print(f"Saved PDF: {OUT_PATH}")


if __name__ == "__main__":
    main()
