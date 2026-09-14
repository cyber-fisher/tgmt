"""
Plot fig:eps-sweep — epsilon vs compression rate C for case17 and case20.
Data source: table tab:eps-sweep in paper-claude-new/main.tex (§4.5).

IEEE-style figure (TVCG/CG&A conventions):
- Single-column width ~3.3 in, aspect ~4:3
- Serif fonts (Times), mathtext STIX
- No title (caption is set in LaTeX)
- Inward ticks, minor ticks, thin axes
- Grayscale-safe: distinguish by marker/linestyle in addition to hue
"""
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import os

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
rcParams['mathtext.fontset'] = 'stix'
rcParams['axes.unicode_minus'] = False
rcParams['axes.linewidth'] = 0.5
rcParams['xtick.major.width'] = 0.5
rcParams['ytick.major.width'] = 0.5
rcParams['xtick.minor.width'] = 0.4
rcParams['ytick.minor.width'] = 0.4
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['xtick.top'] = True
rcParams['ytick.right'] = True
rcParams['xtick.major.size'] = 3.0
rcParams['ytick.major.size'] = 3.0
rcParams['xtick.minor.size'] = 1.6
rcParams['ytick.minor.size'] = 1.6
rcParams['legend.frameon'] = False
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

eps = [0.1, 0.3, 0.5, 0.7]
case17_C = [60.0, 70.5, 74.8, 80.0]
case20_C = [66.8, 82.4, 86.4, 90.0]

# Muted, grayscale-safe palette (IEEE-style)
C17 = '#1f3a68'   # deep blue
C20 = '#a63a2a'   # brick red

fig, ax = plt.subplots(figsize=(3.3, 2.4), dpi=200)

ax.plot(eps, case17_C,
        color=C17, linewidth=0.9, linestyle='-',
        marker='o', markersize=3.6, markerfacecolor='white',
        markeredgewidth=0.9, markeredgecolor=C17,
        label='case17', clip_on=False)

ax.plot(eps, case20_C,
        color=C20, linewidth=0.9, linestyle='--',
        marker='s', markersize=3.4, markerfacecolor='white',
        markeredgewidth=0.9, markeredgecolor=C20,
        label='case20', clip_on=False)

ax.set_xlabel(r'Tolerance $\varepsilon$', fontsize=8.5, labelpad=2)
ax.set_ylabel(r'Compression rate $C$ (\%)'.replace(r'\%', '%'),
              fontsize=8.5, labelpad=2)

ax.set_xlim(0.05, 0.75)
ax.set_ylim(55, 95)
ax.set_xticks(eps)
ax.set_yticks([60, 70, 80, 90])
ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(2))
ax.tick_params(axis='both', labelsize=7.5, pad=2)

for spine in ax.spines.values():
    spine.set_color('black')

leg = ax.legend(loc='lower right', fontsize=7.5,
                handlelength=2.4, handletextpad=0.5,
                borderpad=0.3, labelspacing=0.35)

plt.tight_layout(pad=0.2)

out_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(out_dir, 'fig-eps-sweep.pdf')
png_path = os.path.join(out_dir, 'fig-eps-sweep.png')
plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.02)
plt.savefig(png_path, bbox_inches='tight', pad_inches=0.02, dpi=400)
print(f'Saved: {pdf_path}')
print(f'Saved: {png_path}')
