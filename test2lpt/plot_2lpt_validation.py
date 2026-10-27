"""
Figures for the 2LPT reconstruction checks.

Inputs (produced by the validation scripts):
- synth_results.npz: (2LPT - 1LPT) displacement on a 64^3 synthetic field, and expected grad lap^{-1} S
- cmass_results.npz: Zeldovich / 2LPT shifts on BOSS DR12 CMASS South (FFT and multigrid)
Usage: python plot_2lpt_validation.py <dir with npz files> <output dir>
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

indir, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)

BLUE, ORANGE, AQUA, MAGENTA, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#e87ba4', '#6250d6'
GRAY, INK, INK2 = '#8a8984', '#0b0b0b', '#52514e'
SEQ = LinearSegmentedColormap.from_list('blue', ['#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#104281', '#0d366b'])
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.6,
                     'legend.frameon': False, 'axes.titlesize': 10.5, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left'})


def save(fig, name):
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(outdir, '{}.{}'.format(name, ext)), dpi=160, bbox_inches='tight')
    plt.close(fig)


def hexcompare(ax, x, y, xlabel, ylabel, lines=((1., '1:1', INK2, '--'),), gridsize=70):
    lim = np.percentile(np.abs(np.concatenate([x, y])), 99.9)
    inside = (np.abs(x) < lim) & (np.abs(y) < lim)
    ax.hexbin(x[inside], y[inside], gridsize=gridsize, bins='log', cmap=SEQ, mincnt=1, linewidths=0., extent=(-lim, lim, -lim, lim))
    xx = np.array([-lim, lim])
    for slope, label, color, ls in lines:
        ax.plot(xx, slope * xx, color=color, ls=ls, lw=1.5, label=label)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    r = np.corrcoef(x, y)[0, 1]
    slope = np.sum(x * y) / np.sum(x * x)
    ax.text(0.04, 0.96, 'r = {:.4f}\nfitted slope = {:.3f}'.format(r, slope), transform=ax.transAxes, va='top', color=INK)
    ax.legend(loc='lower right')


# ---------------------------------------------------------------- Figure 1: synthetic tests
# Exact-inversion test (check_2lpt.py, 128^3, L = 1000 Mpc/h, k < 0.05 h/Mpc): rms residual / rms Psi
sigma = np.array([0.025, 0.050, 0.101, 0.206])
inversion = {'Zeldovich only (c = 0)': ([6.612e-3, 1.321e-2, 2.642e-2, 5.342e-2], GRAY, 'o', '-'),
             'forward 2LPT (c = D2 = -3/7)': ([9.442e-3, 1.887e-2, 3.791e-2, 7.827e-2], MAGENTA, 's', '-'),
             'as committed (c = +3/7)': ([3.783e-3, 7.552e-3, 1.500e-2, 2.913e-2], ORANGE, 'D', '-'),
             'fixed (c = 1)': ([1.637e-4, 6.524e-4, 2.652e-3, 1.204e-2], BLUE, 'o', '-')}

synth = np.load(os.path.join(indir, 'synth_results.npz'))
fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.))

ax = axes[0, 0]
for label, (values, color, marker, ls) in inversion.items():
    lw = 2.5 if 'fixed' in label else 2.
    ax.loglog(sigma, values, color=color, marker=marker, ms=7, lw=lw, ls=ls, label=label)
ax.loglog(sigma, 0.35 * inversion['Zeldovich only (c = 0)'][0][0] * sigma / sigma[0], color=INK2, ls=':', lw=1.2)
ax.loglog(sigma, 0.45 * inversion['fixed (c = 1)'][0][0] * (sigma / sigma[0])**2, color=INK2, ls=':', lw=1.2)
ax.text(sigma[0] * 1.04, 0.35 * inversion['Zeldovich only (c = 0)'][0][0] * 0.82, r'$\propto \sigma_\delta$', color=INK2, va='top')
ax.text(sigma[0] * 1.04, 0.45 * inversion['fixed (c = 1)'][0][0] * 0.82, r'$\propto \sigma_\delta^2$', color=INK2, va='top')
ax.set_xlabel(r'$\sigma_\delta$ of the evolved field')
ax.set_ylim(4e-5, 0.15)
ax.set_ylabel(r'rms$(\Psi_{\rm rec} - \Psi_{\rm true})$ / rms$(\Psi_{\rm true})$')
ax.set_title('(a) Exact inversion of a known displacement field')
ax.legend(loc='upper left', fontsize=8.5, title=r'$\Psi_{\rm rec} = \Psi_{\rm ZA} + c\,\nabla\nabla^{-2}S$', title_fontsize=8.5)

ax = axes[0, 1]
groups = ['Iterative FFT', 'Multigrid']
before, after = [0.4321, -0.7970], [1.0000, 0.9959]
xpos = np.arange(len(groups))
w = 0.34
b1 = ax.bar(xpos - w / 2 - 0.01, before, w, color=ORANGE, label='as committed (95e79b0)')
b2 = ax.bar(xpos + w / 2 + 0.01, after, w, color=BLUE, label='fixed')
ax.axhline(1., color=INK2, ls='--', lw=1.2, label='exact (c = 1)')
ax.axhline(0., color=INK2, lw=0.8)
for bar, v in zip(list(b1) + list(b2), before + after):
    ax.text(bar.get_x() + bar.get_width() / 2, v + (0.03 if v >= 0 else -0.03), '{:+.3f}'.format(v), ha='center', va='bottom' if v >= 0 else 'top', color=INK, fontsize=9)
ax.annotate('uncorrelated with S (r = -0.007):\nFFT read a stale multigrid buffer', xy=(1 - w - 0.01, -0.45), xytext=(-0.55, -0.45),
            fontsize=8.5, color=INK2, va='center', arrowprops=dict(arrowstyle='->', color=INK2, lw=0.8))
ax.set_xticks(xpos); ax.set_xticklabels(groups)
ax.set_xlim(-0.6, 1.6); ax.set_ylim(-1.05, 1.25)
ax.set_ylabel(r'measured $c$ in $\Psi_{\rm 2LPT} - \Psi_{\rm ZA} = c\,\nabla\nabla^{-2}S$')
ax.set_title('(b) Second-order coefficient of the classes')
ax.legend(loc='lower left', fontsize=8.5)
ax.grid(axis='x', visible=False)

G = synth['G'].ravel()
for ax, name, label in [(axes[1, 0], 'FFT', 'Iterative FFT'), (axes[1, 1], 'MG', 'Multigrid')]:
    hexcompare(ax, G, synth[name].ravel(), r'expected $\nabla\nabla^{-2}S$ [Mpc/$h$]', r'$\Psi_{\rm 2LPT} - \Psi_{\rm ZA}$ [Mpc/$h$]',
               lines=((1., 'exact (c = 1)', INK2, '--'), (3. / 7., 'as committed (c = 3/7)', ORANGE, '-')))
    ax.set_title('({}) {} 2LPT, fixed: 64$^3$ synthetic field'.format('c' if name == 'FFT' else 'd', label))
fig.tight_layout()
save(fig, '2lpt_synthetic_tests')

# ---------------------------------------------------------------- Figure 2: CMASS South
path = os.path.join(indir, 'cmass_results.npz')
if not os.path.exists(path):
    sys.exit(0)
c = np.load(path)
rms = lambda x: np.sqrt(np.mean(np.sum(x**2, axis=-1)))
psi1 = c['FFT1_disp']
psi2 = c['FFT2_disp'] - c['FFT1_disp']
psi2_mg = c['MG2_disp'] - c['MG1_disp']
psi2_mg_orig = c['MG2orig_disp'] - c['MG1orig_disp']
nz, z, los = c['nz'], c['z'], c['los']

fig, axes = plt.subplots(2, 3, figsize=(15., 9.2))

ax = axes[0, 0]
edges = np.percentile(nz, np.linspace(0, 100, 11))
centers, ratios = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (nz >= lo) & (nz <= hi)
    centers.append(np.median(nz[m]) * 1e4)
    ratios.append(rms(psi2[m]) / rms(psi1[m]))
ax.plot(centers, ratios, color=BLUE, marker='o', ms=7, lw=2)
ax.axhline(rms(psi2) / rms(psi1), color=INK2, ls='--', lw=1.2)
ax.text(centers[-1], rms(psi2) / rms(psi1) + 0.004, 'all galaxies', color=INK2, ha='right', va='bottom', fontsize=9)
ax.set_xlabel(r'$\bar n(z)$ [$10^{-4}\,(h/{\rm Mpc})^3$], median per decile')
ax.set_ylabel(r'rms$|\Psi^{(2)}|$ / rms$|\Psi^{(1)}|$')
ax.set_ylim(0, None)
ax.set_title(r'(a) Second-order term grows where $\bar n$ is low')

ax = axes[0, 1]
data = c['data']
dec = c['dec']
slab = np.abs(dec - np.median(dec)) < 1.5
x, y = c['ra'][slab], np.sqrt(np.sum(data[slab]**2, axis=-1))
x = np.where(x > 180., x - 360., x)
amp = np.sqrt(np.sum(psi2[slab]**2, axis=-1))
order = np.argsort(amp)
sc = ax.scatter(x[order], y[order], c=amp[order], s=3, cmap=SEQ, vmin=0, vmax=np.percentile(amp, 98), linewidths=0)
cb = fig.colorbar(sc, ax=ax, pad=0.01)
cb.set_label(r'$|\Psi^{(2)}|$ [Mpc/$h$]')
cb.outline.set_visible(False)
ax.set_xlabel('RA [deg]'); ax.set_ylabel(r'comoving distance [Mpc/$h$]')
ax.grid(False)
ax.set_title(r'(b) $|\Psi^{(2)}|$ in a 3$^\circ$ Dec slice')

ax = axes[0, 2]
v1, v2 = c['FFT1_vlos'], c['FFT2_vlos']
hexcompare(ax, v1, v2, r'$u_\parallel/(faH)$, Zeldovich [Mpc/$h$]', r'$u_\parallel/(faH)$, 2LPT [Mpc/$h$]')
ax.set_title('(c) Line-of-sight velocity: Zeldovich vs 2LPT (FFT)')

ax = axes[1, 0]
hexcompare(ax, psi2.ravel(), psi2_mg.ravel(), r'$\Psi^{(2)}$ iterative FFT [Mpc/$h$]', r'$\Psi^{(2)}$ multigrid [Mpc/$h$]')
ax.set_title('(d) FFT vs multigrid, same density field')

ax = axes[1, 1]
hexcompare(ax, psi2.ravel(), psi2_mg_orig.ravel(), r'$\Psi^{(2)}$ iterative FFT [Mpc/$h$]', r'$\Psi^{(2)}$ multigrid [Mpc/$h$]')
ax.set_title("(e) FFT vs multigrid, multigrid's own ran_min density")

ax = axes[1, 2]
dpsi_los = np.sum(psi2 * los, axis=-1)
dv = v2 - v1
D2 = -3. / 7.
hexcompare(ax, dpsi_los, dv, r'$\Psi^{(2)}\cdot\hat{r}$ [Mpc/$h$]', r'$\Delta u_\parallel/(faH)$, 2LPT $-$ ZA [Mpc/$h$]',
           lines=((1. + D2, r'expected $1 + D_2 = 4/7$', INK2, '--'), (1., r'slope 1 (if $f$ applied to all)', ORANGE, '-')))
ax.set_title('(f) Velocity uses (1 + D2) x second-order displacement')

fig.suptitle('2LPT reconstruction on BOSS DR12 CMASS South (0.43 < z < 0.7; b = 2.0, f = 0.76, R = 15 Mpc/h)', x=0.01, ha='left', fontweight='bold')
fig.tight_layout()
save(fig, '2lpt_cmass_south')
