"""Rebuild the paper's three vector figures from the retained LaTeX tables.

Run: python Figures/build_figures.py
Requires matplotlib and numpy. Compilation uses the included PDFs directly.
"""
from pathlib import Path
import csv
import json
import os
import re
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tdd-paper-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WIDTH = 5.45
INK = "#222222"
MUTED = "#555555"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
MODELS = ["Qwen2.5-7B", "Qwen3.5-9B", "Qwen3.6-27B", "Qwen3.8-27B", "DeepSeek-6.7B"]
DATASETS = ["HumanEval+", "MBPP+", "LCB v6-new"]
CONTRASTS = ["Correct - NL", "Correct - Inputs", "Correct - Wrong"]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
    "ytick.color": INK, "axes.edgecolor": "#999999",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.facecolor": "white", "axes.unicode_minus": True,
})


def parse_tables():
    families, values, effects = [], [], []
    for line in (ROOT / "Tables/semantic_families.tex").read_text(encoding="utf-8").splitlines():
        cells = [x.strip() for x in line.removesuffix(r"\\").split("&")]
        if len(cells) == 1+4*len(MODELS) and re.fullmatch(r"\d+\.\d", cells[1]):
            families.append(cells[0])
            values.append([float(x) for x in cells[1:]])
    for line in (ROOT / "Tables/main_effects.tex").read_text(encoding="utf-8").splitlines():
        cells = [x.strip() for x in line.removesuffix(r"\\").split("&")]
        if len(cells) != 5 or cells[0] not in MODELS:
            continue
        for contrast, cell in zip(CONTRASTS, cells[2:]):
            assert re.fullmatch(r"[+-]?\d+\.\d", cell), cell
            effects.append(dict(model=cells[0], dataset=cells[1], contrast=contrast,
                                estimate=float(cell)))
    matrix = np.asarray(values)
    assert matrix.shape == (20, 4*len(MODELS))
    assert np.all((matrix >= 0) & (matrix <= 6))
    assert len(effects) == 3*3*len(MODELS)
    return families, matrix, effects


def save(fig, name):
    fig.savefig(HERE / f"{name}.pdf", metadata={"Creator": "Matplotlib", "CreationDate": None, "ModDate": None})
    fig.savefig(HERE / f"{name}.svg", metadata={"Creator": "Matplotlib", "Date": None})
    fig.savefig(HERE / f"{name}.png", dpi=220)
    plt.close(fig)


def schematic():
    visible = [3, 1, 3, 2]
    unseen = [4, 2, 4, 1]
    assert list(dict.fromkeys(visible)) == [3, 1, 2]
    assert sorted(set(visible)) == [1, 2, 3]
    assert list(dict.fromkeys(unseen)) == [4, 2, 1]
    assert sorted(set(unseen)) == [1, 2, 4]
    fig = plt.figure(figsize=(WIDTH, 3.18))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 392.4), ylim=(0, 229))
    ax.axis("off")

    def text(x, y, value, size=8.3, color=INK, bold=False, mono=False):
        ax.text(x, y, value, ha="center", va="center", fontsize=size,
                color=color, weight="bold" if bold else "normal",
                family="DejaVu Sans Mono" if mono else "DejaVu Sans")

    def arrow(x1, y1, x2, y2, color=MUTED):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                    mutation_scale=8, linewidth=.8, color=color,
                                    shrinkA=0, shrinkB=0))

    # Each stage owns its labels; connectors occupy only the gaps between boxes.
    ax.add_patch(Rectangle((5, 179), 382, 48, facecolor="#F5F6F7",
                           edgecolor="#BBBBBB", lw=.7))
    text(196.2, 217, "FIXED: description, interface, visible input", size=8, bold=True)
    text(196.2, 202, '"Remove duplicates from a list."   dedup(data)', size=8)
    text(196.2, 188, "data = [3, 1, 3, 2]", size=8.5, mono=True)
    for x, center, color, tint, branch, output, hidden, rule in [
        (5, 94, BLUE, "#EFF7FB", "A", "[3, 1, 2]", "[4, 2, 1]", "First-occurrence order"),
        (209, 298, ORANGE, "#FFF4EC", "B", "[1, 2, 3]", "[1, 2, 4]", "Ascending order"),
    ]:
        arrow(center, 177, center, 168, color)
        ax.add_patch(Rectangle((x, 133), 178, 33, facecolor=tint, edgecolor=color, lw=.9))
        text(center, 156, f"Prompt {branch}: change expected output", size=7.7)
        text(center, 141, output, size=10, mono=True, color=color, bold=True)
        arrow(center, 131, center, 122, color)
        ax.add_patch(Rectangle((x, 96), 178, 24, facecolor="white", edgecolor="#999999", lw=.7))
        text(center, 108, f"Same model generates program P_{branch}", size=7.7)
        arrow(center, 94, center, 85, color)
        ax.add_patch(Rectangle((x, 31), 178, 52, facecolor=tint, edgecolor=color, lw=.9))
        text(center, 74, "Held-out evaluation", size=8, bold=True)
        text(center, 61, "Input: [4, 2, 4, 1]", size=8, mono=True)
        text(center, 48, f"Expected: {hidden}", size=8, mono=True, color=color)
        text(center, 37, rule, size=7.4, color=MUTED)
        ax.plot([center, center], [29, 23], color=MUTED, lw=.8)
    ax.plot([94, 298], [23, 23], color=MUTED, lw=.8)
    arrow(196.2, 23, 196.2, 17)
    text(196.2, 7, "Switch success: BOTH programs pass their assigned rules", size=8, bold=True)
    save(fig, "paired_intervention")


def heatmap(families, values):
    # Annotated matrix grammar: fixed scale, direct labels, model grouping.
    height = 4.38
    fig = plt.figure(figsize=(WIDTH, height))
    cmap = plt.get_cmap("Blues")
    norm = Normalize(0, 6)
    left, right, gap = 1.19, .055, .065
    panel_w = (WIDTH-left-right-(len(MODELS)-1)*gap)/len(MODELS)
    for m, model in enumerate(MODELS):
        ax = fig.add_axes([(left+m*(panel_w+gap))/WIDTH, .55/height, panel_w/WIDTH, 3.00/height])
        v = values[:, 4*m:4*m+4]
        mesh = ax.pcolormesh(np.arange(5), np.arange(21), v, cmap=cmap, norm=norm,
                             edgecolors="white", linewidth=.45, rasterized=False)
        ax.set(xlim=(0,4), ylim=(20,0), xticks=np.arange(4)+.5,
               xticklabels=["A", "B", "Switch", "Explicit"], yticks=np.arange(20)+.5)
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", length=0, pad=4, labelsize=7.3, labelrotation=90)
        ax.tick_params(axis="y", length=0, pad=4, labelsize=7.8)
        ax.set_yticklabels([x[0].upper()+x[1:] for x in families] if m==0 else [])
        fig.text((left+m*(panel_w+gap)+panel_w/2)/WIDTH,4.14/height,
                 model.replace('-', '\n'),fontsize=8.0,weight="bold",ha="center",va="center",linespacing=1.1)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for row in range(20):
            for col in range(4):
                rgba = cmap(norm(v[row,col]))
                rgb = np.asarray(rgba[:3])
                linear = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)
                luminance = float(linear @ np.array([.2126,.7152,.0722]))
                color = "white" if luminance < .179 else "#111111"
                ax.text(col+.5, row+.5, f"{v[row,col]:.1f}", ha="center", va="center",
                        color=color, fontsize=7.0)
    cax = fig.add_axes([.38, .28/height, .40, .075/height])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", ticks=[0,1,2,3,4,5,6])
    cb.outline.set_linewidth(.5)
    cb.ax.tick_params(labelsize=7, length=2, pad=2)
    cb.solids.set_rasterized(False)
    fig.text(.58, .022/height, "Mean successful instances (out of 6)", ha="center", fontsize=7.5)
    save(fig, "semantic_families_heatmap")


def forest(effects):
    height = 4.30
    fig = plt.figure(figsize=(WIDTH, height))
    styles = [(BLUE, "o"), (ORANGE, "s"), (GREEN, "^")]
    left, right, gap = 1.08, .07, .22
    panel_w = (WIDTH-left-right-2*gap)/3
    for d, dataset in enumerate(DATASETS):
        ax = fig.add_axes([(left+d*(panel_w+gap))/WIDTH, .80/height, panel_w/WIDTH, 3.1/height])
        ax.set(xlim=(-10,40), ylim=(.6,15.4), xticks=[-10,0,10,20,30,40], yticks=[14,11,8,5,2])
        ax.set_yticklabels(MODELS if d==0 else [])
        ax.tick_params(axis="y", length=0, pad=7, labelsize=8)
        ax.tick_params(axis="x", length=3, labelsize=7.2, pad=3)
        ax.set_title(dataset, fontsize=8.4, weight="bold", pad=10)
        for y in [12.5,9.5,6.5,3.5]:
            ax.axhline(y, color="#E4E4E4", lw=.6, zorder=0)
        ax.axvline(0, color="#777777", ls=(0,(3,3)), lw=.8, zorder=0)
        ax.grid(axis="x", color="#EAEAEA", lw=.45, zorder=0)
        for spine in ["top","right","left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_linewidth(.6)
        for m, model in enumerate(MODELS):
            for k, contrast in enumerate(CONTRASTS):
                row = next(e for e in effects if (e['model'],e['dataset'],e['contrast'])==(model,dataset,contrast))
                y=14-3*m+[.65,0,-.65][k]
                val = row['estimate']
                color, marker = styles[k]
                ax.plot(val, y, marker=marker, linestyle="none", markersize=4.4,
                        color=color, zorder=3)
                ax.annotate(f"{val:+.1f}", (val, y), xytext=(5, 0),
                            textcoords="offset points", va="center", fontsize=6.3,
                            color=color)
    fig.text(.58,.105,"Paired accuracy difference (percentage points)",ha="center",fontsize=8)
    handles=[Line2D([0],[0],color=c,marker=m,lw=0,markersize=3.5) for c,m in styles]
    fig.legend(handles,["Correct − NL-only", "Correct − Inputs-only", "Correct − Wrong I/O"],
               loc="lower center",bbox_to_anchor=(.51,.017),ncol=3,frameon=False,
               fontsize=7.6,handlelength=1.4,columnspacing=1.2,handletextpad=.45)
    save(fig, "main_effects_forest")


def main():
    families, values, effects = parse_tables()
    data = {"families": families, "models": MODELS,
            "family_columns_per_model": ["A", "B", "Switch", "Explicit"],
            "family_values": values.tolist(), "effects": effects,
            "source_tables": ["Tables/semantic_families.tex", "Tables/main_effects.tex"]}
    (HERE/"figure_data.json").write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")
    with (HERE/"main_effects.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["model","dataset","contrast","estimate"])
        writer.writeheader(); writer.writerows(effects)
    schematic()
    heatmap(families, values)
    forest(effects)
    print("Rendered 3 PDF/SVG/PNG figures from 400 heatmap values and 45 mean paired differences.")


if __name__ == "__main__":
    main()
