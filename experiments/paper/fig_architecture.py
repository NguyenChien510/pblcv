"""
Figure 2 of the paper: overall architecture of SpaceTime-DSCA (MFGF backbones + DSCA + hierarchical alignment),
drawn in the visual style of the MFGF figure [24] (serif type, pastel panels, hatched feature glyphs). The frames and
the caption are a real TVPReid-PRID sample (person0001). Same aspect as the image in the template (721 x 401).

    python experiments/paper/fig_architecture.py results/figures/fig2_architecture.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

FRAMES = Path(__file__).resolve().parents[2] / "results" / "figures" / "frames"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "hatch.linewidth": 0.9})

INK = "#222222"
RED, BLUE, PURP = "#c8102e", "#1b4f9c", "#5e2ca5"
APP, MOT, GLB = "#e06a00", "#1f5fb8", "#6a3fb0"            # feature colours (appearance / motion / global)
APP_L, MOT_L, GLB_L = "#f9c89a", "#a9c9f0", "#cdb6ee"
FLOW = "#6a4caf"                                         # projection arrows
LOSS = "#0f8a7d"                                         # contrastive alignment arrows
W, H = 14.42, 8.02

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def rbox(x, y, w, h, fc, ec=None, lw=1.0, r=0.12, z=2, ls="-", alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec or fc, lw=lw,
                                zorder=z, linestyle=ls, alpha=alpha))


def txt(x, y, s, size=10, color=INK, weight="normal", ha="center", va="center", z=10, **kw):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, zorder=z, **kw)


def arrow(p0, p1, color=FLOW, lw=1.3, rad=0.0, ls="-", style="-|>", z=6, ms=11):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=0, shrinkB=0))


def feat_sq(x, y, c, cl, s=0.3, z=8):                    # video feature: hatched square
    ax.add_patch(FancyBboxPatch((x - s / 2, y - s / 2), s, s, boxstyle="round,pad=0,rounding_size=0.05",
                                fc=cl, ec=c, lw=1.2, hatch="////", zorder=z))


def feat_ci(x, y, c, cl, r=0.16, z=8):                   # text feature: hatched circle
    ax.add_patch(Circle((x, y), r, fc=cl, ec=c, lw=1.2, hatch="////", zorder=z))


def colored_paragraph(x, y, width, words, size=9.2, lh=0.27):
    """words: list of (word, colour); simple greedy wrap inside `width` (data units)."""
    fig.canvas.draw()
    inv = ax.transData.inverted()
    cx, cy = x, y
    space = None
    for w, c in words:
        t = ax.text(cx, cy, w, fontsize=size, color=c, ha="left", va="center", zorder=10,
                    fontweight="bold" if w == "Text:" else "normal")
        bb = t.get_window_extent(fig.canvas.get_renderer()).transformed(inv)
        if bb.x1 > x + width and cx > x:
            cx, cy = x, cy - lh
            t.set_position((cx, cy))
            bb = t.get_window_extent(fig.canvas.get_renderer()).transformed(inv)
        if space is None:
            s = ax.text(0, 0, "a a", fontsize=size)
            s2 = ax.text(0, 0, "aa", fontsize=size)
            space = (s.get_window_extent(fig.canvas.get_renderer()).transformed(inv).width
                     - s2.get_window_extent(fig.canvas.get_renderer()).transformed(inv).width)
            s.remove()
            s2.remove()
        cx = bb.x1 + space
    return cy


def image(path, x, y, w, h, z=6, ec="#666", lw=0.8):
    ax.imshow(mpimg.imread(path), extent=(x, x + w, y, y + h), zorder=z, aspect="auto")
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=ec, lw=lw, zorder=z + 0.1))


# ------------------------------------------------------------------------------------------------ panels
rbox(0.08, 4.28, 9.32, 3.66, "#f8d3bd", "#e29a72", lw=1.6, r=0.25, z=0)
rbox(0.08, 0.08, 9.32, 4.08, "#cfe8c3", "#86bf6e", lw=1.6, r=0.25, z=0)
rbox(9.52, 1.62, 4.82, 6.32, "#ddd2f0", "#a48ad4", lw=1.6, r=0.25, z=0)
txt(4.74, 7.66, "Disentangled Concept Learning of Text", 16, weight="bold")
txt(4.74, 3.86, "Two-stream Learning of Video", 16, weight="bold")
txt(11.93, 7.66, "Hierarchical Multi-level Alignment", 14.5, weight="bold")

# ------------------------------------------------------------------------------------------------ text branch
rbox(0.28, 5.3, 2.62, 2.07, "white", "#999", lw=0.9, r=0.06, z=3)
cap = [("Text:", INK), ("The", INK), ("woman", INK), ("is", INK), ("wearing", INK), ("a", INK), ("white", RED),
       ("jacket", RED), ("and", INK), ("blue", RED), ("trousers.", RED), ("She", INK), ("has", INK), ("short", RED),
       ("black", RED), ("hair", RED), ("and", INK), ("black", RED), ("shoes.", RED), ("She", INK), ("is", INK),
       ("walking", BLUE), ("with", BLUE), ("a", BLUE), ("book", RED), ("in", BLUE), ("her", BLUE), ("hand", BLUE),
       ("and", INK), ("a", INK), ("red", RED), ("backpack.", RED)]
colored_paragraph(0.42, 7.1, 2.4, cap, size=11.2, lh=0.37)

rbox(3.15, 6.12, 2.35, 1.2, "#fff3e8", "#d0661c", lw=1.0, r=0.08, z=3)
rbox(3.15, 6.98, 2.35, 0.34, "#f39a5c", "#d0661c", lw=1.0, r=0.08, z=4)
txt(4.325, 7.15, "DSCA Decomposer", 11, weight="bold", z=11)
txt(3.27, 6.74, r"$Q_{\rm app}$: A woman wearing white", 8.6, RED, ha="left")
txt(3.27, 6.54, "jacket, blue trousers, red backpack ...", 8.6, RED, ha="left")
txt(3.27, 6.28, r"$Q_{\rm mot}$: A video of a woman walking ...", 8.6, BLUE, ha="left")
arrow((2.9, 6.6), (3.15, 6.6), INK, 1.2)

ax.add_patch(Ellipse((6.35, 6.7), 1.15, 0.62, fc="white", ec="#888", lw=0.9, zorder=3))
for (dx, dy, c) in ((-0.3, 0.1, RED), (-0.1, -0.12, RED), (0.12, 0.13, BLUE), (0.3, -0.05, BLUE), (-0.38, -0.08, "#4caf50"),
                    (0.0, 0.05, "#e8b52f"), (0.22, -0.18, RED), (-0.2, 0.2, BLUE)):
    ax.add_patch(Circle((6.35 + dx, 6.7 + dy), 0.045, fc=c, ec="none", zorder=4))
txt(6.35, 6.22, "Concept lexicon", 9, weight="bold")
txt(6.35, 6.03, "(5 groups)", 8, "#555")
arrow((5.5, 6.75), (5.78, 6.75), INK, 1.0, style="<|-|>", ms=8)


# token lanes
rbox(3.15, 4.98, 5.3, 0.98, "none", "#555", lw=0.9, r=0.1, z=3, ls=(0, (4, 3)))
txt(3.0, 5.12, "+Position", 8, "#444", ha="right")
txt(3.0, 4.95, "Embedding", 8, "#444", ha="right")
for k, (lab, c) in enumerate(((r"$Q_{\rm full}$", PURP), (r"$Q_{\rm app}$", RED), (r"$Q_{\rm mot}$", BLUE))):
    y = 5.77 - k * 0.3
    txt(3.42, y, lab, 9.5, c)
    for i in range(12):
        xx = 3.85 + i * 0.36
        if i == 9:
            txt(xx, y, "· · ·", 9, "#555")
            continue
        ax.add_patch(Circle((xx, y), 0.1, fc="white", ec=c, lw=0.9, zorder=5))
        if i == 0:
            txt(xx, y, "CLS", 4.6, c, "bold", z=6)
arrow((4.325, 6.12), (4.325, 5.96), INK, 1.0)
rbox(3.15, 4.48, 5.3, 0.36, "#f2a9bf", "#c0587a", lw=1.0, r=0.08, z=3)
txt(5.8, 4.66, "BERT (shared weights)", 11.5, weight="bold")
for xx in (4.5, 5.8, 7.1):
    arrow((xx, 4.98), (xx, 4.84), INK, 1.0)
for k, (c, cl, lab) in enumerate(((PURP, GLB_L, r"$T_{\rm full}$"), (RED, "#f6d0cf", r"$T_{\rm app}$"),
                                  (BLUE, MOT_L, r"$T_{\rm mot}$"))):
    xx = 9.05
    yy = 5.62 - k * 0.42
    feat_ci(xx, yy, c, cl)
    txt(8.86, yy, lab, 10, c, ha="right")
arrow((8.45, 4.62), (8.93, 4.66), INK, 1.0)

# ------------------------------------------------------------------------------------------------ video branch
for k, idx in enumerate((2, 6, 10)):                     # frame stack with time arrow
    image(FRAMES / f"f{idx:02d}.png", 0.3 + k * 0.22, 1.95 - k * 0.22, 0.78, 1.5, z=5 + k)
arrow((0.32, 1.62), (0.1, 1.35), INK, 1.0)
txt(0.38, 1.35, "time", 9, INK, style="italic", ha="left")

for i in range(16):                                      # sampled clip strip (bottom)
    x = 0.3 + i * 0.27
    disc = i in (1, 5, 9, 13)
    image(FRAMES / f"f{i:02d}.png", x, 0.33, 0.23, 0.46, z=5, ec=APP if disc else "#888", lw=1.6 if disc else 0.6)
ax.add_patch(Rectangle((0.26, 0.29), 16 * 0.27 - 0.0, 0.54, fc="none", ec=MOT, lw=1.1, ls=(0, (4, 3)), zorder=7))
txt(0.3, 0.17, "16 consecutive frames → S3D", 8.5, MOT, ha="left", style="italic")
txt(2.45, 0.17, "4 discrete frames (orange) → Space-Time ViT", 8.5, APP, ha="left", style="italic")

# Space-Time ViT (visual encoder)
rbox(2.2, 2.05, 4.25, 1.5, "#eaf5b8", "#6f9a1f", lw=1.0, r=0.1, z=2)
txt(4.35, 3.36, "Visual Encoder: Space-Time ViT-B/16", 11.5, weight="bold")
rbox(2.35, 2.2, 0.32, 1.0, "white", "#777", lw=0.8, r=0.08, z=3)
txt(2.51, 2.7, "Patch Embedding", 7.2, rotation=90)
for row, (lab, y) in enumerate((("Temporal\nAttn", 2.82), ("Spatial\nAttn", 2.35))):
    for j in range(3):
        rbox(2.9 + j * 0.86, y - 0.19, 0.78, 0.4, "#9fd07a", "#3f7f1f", lw=0.9, r=0.08, z=4)
        txt(3.29 + j * 0.86, y + 0.01, lab, 7.6, z=10)
        if j < 2:
            arrow((3.68 + j * 0.86, y), (3.76 + j * 0.86, y), "#5f9a3a", 0.8, ms=6)
    txt(5.6, y + 0.01, "· · ·", 11, "#555")
txt(6.05, 2.6, "×12", 12, "#3d6b22", "bold")
arrow((2.67, 2.7), (2.9, 2.82), INK, 0.9)
arrow((2.67, 2.7), (2.9, 2.35), INK, 0.9)
arrow((1.38, 2.6), (2.35, 2.6), INK, 1.1)

# S3D (motion encoder)
rbox(2.2, 1.0, 4.25, 0.9, "#c4e1f5", "#2f7fb8", lw=1.0, r=0.1, z=2)
txt(4.35, 1.71, "Motion Encoder: S3D (Kinetics-400)", 11.5, weight="bold")
for j in range(4):
    rbox(2.45 + j * 0.86, 1.1, 0.78, 0.38, "#86bfe8", "#1f6aa5", lw=0.9, r=0.08, z=4)
    txt(2.84 + j * 0.86, 1.29, "3D Conv", 7.6, z=10)
txt(6.05, 1.29, "· · ·", 11, "#555")
arrow((3.4, 0.83), (3.4, 1.0), MOT, 1.4)

# heads + fusion
feat_sq(6.9, 2.8, APP, APP_L)
feat_sq(6.9, 1.45, MOT, MOT_L)
arrow((6.45, 2.8), (6.73, 2.8), INK, 1.2)
arrow((6.45, 1.45), (6.73, 1.45), INK, 1.2)
txt(6.9, 3.12, "[CLS]", 8.5, "#333")
txt(6.9, 2.47, r"$V_{\rm app}$", 10.5, APP, "bold")
txt(6.9, 1.13, "AvgPool", 8.5, "#333")
txt(6.9, 1.79, r"$V_{\rm mot}$", 10.5, MOT, "bold")
rbox(7.45, 1.25, 0.42, 1.75, "#a7d88a", "#3f7f1f", lw=1.2, r=0.08, z=3)
txt(7.66, 2.12, "Fusion MLP", 9.5, rotation=90, weight="bold")
arrow((7.07, 2.8), (7.45, 2.8), INK, 1.2)
arrow((7.07, 1.45), (7.45, 1.45), INK, 1.2)
feat_sq(8.2, 2.12, GLB, GLB_L)
arrow((7.87, 2.12), (8.03, 2.12), INK, 1.2)
txt(8.2, 1.79, r"$V_{\rm video}$", 10.5, GLB, "bold")

# ------------------------------------------------------------------------------------------------ links between panels
arrow((9.24, 5.2), (10.22, 5.2), FLOW, 2.4, ms=16)
ax.plot([8.4, 9.72, 9.72], [2.12, 2.12, 4.4], color=FLOW, lw=2.4, zorder=6, solid_joinstyle="miter")
arrow((9.72, 4.4), (10.22, 4.4), FLOW, 2.4, ms=16)
txt(9.0, 2.32, r"$V_{\rm app},V_{\rm mot},V_{\rm video}$", 8.5, "#5a3c8f")

# ------------------------------------------------------------------------------------------------ alignment panel
ox, oy = 9.95, 2.4                                         # axes of the joint space
arrow((ox, oy), (ox, 6.9), INK, 1.4, ms=12)
arrow((ox, oy), (14.0, oy), INK, 1.4, ms=12)
txt(ox + 0.12, 6.88, "Joint Embedding Space", 10.5, weight="bold", ha="left")

groups = [(6.05, PURP, GLB, GLB_L, r"$T_{\rm full}$", r"$V_{\rm video}$", r"$\mathcal{L}_{\rm joint}$"),
          (4.95, RED, APP, APP_L, r"$T_{\rm app}$", r"$V_{\rm app}$", r"$\lambda_{\rm app}\mathcal{L}_{\rm app}$"),
          (3.85, BLUE, MOT, MOT_L, r"$T_{\rm mot}$", r"$V_{\rm mot}$", r"$\lambda_{\rm mot}\mathcal{L}_{\rm mot}$")]
for y, tc, vc, vl, tl, vlab, loss in groups:
    rbox(10.3, y - 0.42, 2.6, 0.84, "white", "#666", lw=0.9, r=0.1, z=3, ls=(0, (4, 3)), alpha=0.85)
    feat_ci(10.65, y, tc, "#ffffff")
    feat_sq(12.2, y, vc, vl)
    arrow((10.85, y), (12.0, y), LOSS, 1.4, style="<|-|>", ls=(0, (4, 2)), ms=9)
    txt(11.42, y + 0.25, loss, 10, LOSS)
    txt(10.65, y - 0.3, tl, 8.5, tc)
    txt(12.4, y, vlab, 9, vc, ha="left")
txt(11.6, 6.6, "Contrastive Alignment (InfoNCE)", 8.5, LOSS, style="italic")

# orthogonality between V_app and V_mot
cx, cy = 13.35, 4.1
arrow((cx, cy), (cx + 0.75, cy), APP, 1.6)
arrow((cx, cy), (cx, cy + 0.75), MOT, 1.6)
ax.add_patch(Polygon([(cx, cy), (cx + 0.14, cy), (cx + 0.14, cy + 0.14), (cx, cy + 0.14)], fc="none", ec=INK, lw=0.9,
                     zorder=8))
txt(cx + 0.78, cy - 0.17, r"$V_{\rm app}$", 9, APP)
txt(cx - 0.05, cy + 0.88, r"$V_{\rm mot}$", 9, MOT)
txt(cx + 0.4, cy - 0.5, r"$\mathcal{L}_{\rm MIM}$", 12, "#b0245a")
txt(cx + 0.4, cy - 0.76, "orthogonality", 8, "#b0245a", style="italic")

# scoring
rbox(10.3, 1.82, 3.85, 0.5, "white", PURP, lw=1.0, r=0.08, z=3)
txt(12.22, 2.07, r"Query: $S=w_1\cos(T_{\rm full},V_{\rm video})+w_2\cos(T_{\rm app},V_{\rm app})+w_3\cos(T_{\rm mot},V_{\rm mot})$",
    8.3, PURP)

# ------------------------------------------------------------------------------------------------ legend
rbox(9.52, 0.08, 4.82, 1.42, "white", "#777", lw=1.0, r=0.1, z=2, ls=(0, (4, 3)))
leg = [(lambda x, y: feat_sq(x, y, APP, APP_L, s=0.22), "Appearance feature"),
       (lambda x, y: feat_ci(x, y, RED, "#f6d0cf", r=0.11), "Appearance text feature"),
       (lambda x, y: feat_sq(x, y, MOT, MOT_L, s=0.22), "Motion feature"),
       (lambda x, y: feat_ci(x, y, BLUE, MOT_L, r=0.11), "Motion text feature"),
       (lambda x, y: feat_sq(x, y, GLB, GLB_L, s=0.22), "Global video feature"),
       (lambda x, y: feat_ci(x, y, PURP, GLB_L, r=0.11), "Full-sentence feature")]
for k, (draw, lab) in enumerate(leg):
    x = 9.8 + (k % 2) * 2.35
    y = 1.22 - (k // 2) * 0.3
    draw(x, y)
    txt(x + 0.22, y, lab, 9, ha="left")
arrow((9.68, 0.25), (10.15, 0.25), FLOW, 1.2)
txt(10.25, 0.25, "Feature projection", 9, ha="left")
arrow((12.03, 0.25), (12.5, 0.25), LOSS, 1.2, style="<|-|>", ls=(0, (4, 2)), ms=8)
txt(12.6, 0.25, "Contrastive alignment", 9, ha="left")

fig.savefig(sys.argv[1], dpi=150, facecolor="white")
print("saved", sys.argv[1])
