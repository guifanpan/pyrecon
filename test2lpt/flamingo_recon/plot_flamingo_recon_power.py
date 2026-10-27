"""
Figures for the reconstruction power of Zeldovich vs 2LPT reconstruction on a FLAMINGO L1_m10 sub-box (z = 0.35, DM particles).
Input: recon_power_results.npz from recon_power.py.
Usage: python plot_flamingo_recon_power.py <recon_power_results.npz> <output dir>
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

fn, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
d = np.load(fn)
h = float(d['h'])
k = d['kmid'] / h

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2 = '#0b0b0b', '#52514e'
DIV = LinearSegmentedColormap.from_list('div', ['#104281', '#3987e5', '#9ec5f4', '#f0efec', '#f3a3a2', '#e34948', '#a52a2a'])
SEQ = LinearSegmentedColormap.from_list('blue', ['#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#104281', '#0d366b'])
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.6,
                     'legend.frameon': False, 'axes.titlesize': 10.5, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left'})
radii = [(5, BLUE), (10, ORANGE), (20, AQUA)]
kmax = 0.75


def save(fig, name):
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(outdir, '{}.{}'.format(name, ext)), dpi=160, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------- Figure 1: r(k), Delta r(k), particle statistics
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
sel = k < kmax
for ax, field, title in [(axes[0, 0], 'disp', r'(a) Displacement: $r(k)$ with truth $\mathbf{x} - \mathbf{q}$'),
                         (axes[0, 1], 'vel', r'(b) Velocity: $r(k)$ with true peculiar velocity')]:
    for R, color in radii:
        ax.plot(k[sel], d['ZA_R{}_rk_{}'.format(R, field)][sel], color=color, lw=2, ls='--')
        ax.plot(k[sel], d['2LPT_R{}_rk_{}'.format(R, field)][sel], color=color, lw=2, label=r'$R$ = {} Mpc/$h$'.format(R))
    ax.plot([], [], color=INK2, ls='--', lw=2, label='Zeldovich')
    ax.plot([], [], color=INK2, ls='-', lw=2, label='2LPT')
    ax.set_xlabel(r'$k$ [$h$/Mpc]')
    ax.set_ylabel(r'$r(k) = P_{\rm rec,true} / \sqrt{P_{\rm rec} P_{\rm true}}$')
    ax.set_ylim(-0.05, 1.02)
    ax.set_xlim(0, kmax)
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=9)

ax = axes[1, 0]
ax.axhline(0., color=INK2, lw=1)
for R, color in radii:
    ax.plot(k[sel], (d['2LPT_R{}_rk_vel'.format(R)] - d['ZA_R{}_rk_vel'.format(R)])[sel], color=color, lw=2, label=r'$R$ = {} Mpc/$h$'.format(R))
    ax.plot(k[sel], (d['2LPT_R{}_rk_disp'.format(R)] - d['ZA_R{}_rk_disp'.format(R)])[sel], color=color, lw=1.2, ls=':')
ax.plot([], [], color=INK2, lw=2, label='velocity')
ax.plot([], [], color=INK2, lw=1.2, ls=':', label='displacement')
ax.text(0.03, 0.018, '2LPT better', color=INK2, fontsize=9, va='bottom')
ax.text(0.03, -0.018, '2LPT worse', color=INK2, fontsize=9, va='top')
ax.set_xlabel(r'$k$ [$h$/Mpc]')
ax.set_ylabel(r'$r_{\rm 2LPT}(k) - r_{\rm ZA}(k)$')
ax.set_xlim(0, kmax)
ax.set_title('(c) Gain from the second-order term')
ax.legend(loc='lower left', fontsize=9, ncol=2)

ax = axes[1, 1]
Rs = d['smoothings_h']
for name, color, marker in [('ZA', INK2, 's'), ('2LPT', BLUE, 'o')]:
    vel = np.array([d['{}_R{:g}_vel'.format(name, R)] for R in Rs])
    disp = np.array([d['{}_R{:g}_disp'.format(name, R)] for R in Rs])
    ax.plot(Rs, vel[:, 0], color=color, marker=marker, ms=7, lw=2, label='{}: velocity'.format(name))
    ax.plot(Rs, disp[:, 0], color=color, marker=marker, ms=7, lw=1.2, ls=':', mfc='white', label='{}: displacement'.format(name))
ax.set_xlabel(r'smoothing radius $R$ [Mpc/$h$]')
ax.set_ylabel('per-particle correlation with truth')
ax.set_title('(d) Per-particle correlation (includes virial motions)')
ax.legend(loc='upper right', fontsize=9)

fig.suptitle(r'FLAMINGO L1_m10, z = {:.2f}: DM particles in a 250 Mpc (170 Mpc/$h$) sub-box, real space, interior 190 Mpc scored'.format(float(d['z'])),
             x=0.01, ha='left', fontweight='bold')
fig.tight_layout()
save(fig, 'flamingo_recon_power')

# ---------------------------------------------------------------- Figure 2: maps and scatter at R = 10 Mpc/h
fig = plt.figure(figsize=(18, 4.6))
gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 0.05, 0.28, 1.05], wspace=0.25)
axes = [fig.add_subplot(gs[0, i]) for i in [0, 1, 2, 5]]
cax = fig.add_subplot(gs[0, 3])
ext = d['field_extent']
extent = [ext[0], ext[1], ext[0], ext[1]]
true = d['field_true']
lim = np.percentile(np.abs(true), 99)
for ax, img, title in [(axes[0], true, r'(a) true $\Psi_x$'), (axes[1], d['field_za'], r'(b) Zeldovich $\Psi_x$'),
                       (axes[2], d['field_2lpt'], r'(c) 2LPT $\Psi_x$')]:
    im = ax.imshow((img - img.mean()).T, origin='lower', extent=extent, cmap=DIV, vmin=-lim, vmax=lim)
    ax.set_title(title + r', $R$ = 10 Mpc/$h$' if 'true' not in title else title)
    ax.set_xlabel(r'$x$ [Mpc]'); ax.set_ylabel(r'$y$ [Mpc]')
    ax.grid(False)
cb = fig.colorbar(im, cax=cax)
cb.set_label(r'$\Psi_x$ [Mpc], 5 Mpc cells, mean removed')
cb.outline.set_visible(False)
ax = axes[3]
a, H, f, D2 = float(d['a']), float(d['H']), float(d['f']), float(d['D2'])
vt = d['sample_v'][:, 0] - d['sample_v'][:, 0].mean()
vr = f * a * H * (d['sample_2lpt'][:, 0] - d['sample_2lpt'][:, 0].mean())
vl = np.percentile(np.abs(vt), 99.5)
inside = (np.abs(vt) < vl) & (np.abs(vr) < vl)
ax.hexbin(vr[inside], vt[inside], gridsize=60, bins='log', cmap=SEQ, mincnt=1, linewidths=0., extent=(-vl, vl, -vl, vl))
ax.plot([-vl, vl], [-vl, vl], color=INK2, ls='--', lw=1.5, label='1:1')
ax.set_xlim(-vl, vl); ax.set_ylim(-vl, vl)
ax.set_xlabel(r'reconstructed $v_x$, 2LPT [km/s]'); ax.set_ylabel(r'true particle $v_x$ [km/s]')
ax.text(0.04, 0.96, 'r = {:.3f}'.format(np.corrcoef(vr, vt)[0, 1]), transform=ax.transAxes, va='top', color=INK)
ax.set_title(r'(d) Particle velocities, $R$ = 10 Mpc/$h$')
ax.legend(loc='lower right')
save(fig, 'flamingo_recon_maps')
