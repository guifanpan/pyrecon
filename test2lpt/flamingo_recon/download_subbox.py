"""Stream DM particles of a sub-box of FLAMINGO L1_m10 (z = 0.35) via hdfstream; save positions, velocities, Lagrangian positions."""
import sys, time
import numpy as np
import hdfstream

ncell_side = int(sys.argv[1]) if len(sys.argv) > 1 else 8
out = sys.argv[2] if len(sys.argv) > 2 else 'flamingo_subbox.npz'
root = hdfstream.open("cosma", "/")
f = root["FLAMINGO/L1_m10/L1_m10/snapshots/flamingo_0070/flamingo_0070.hdf5"]
L, N = float(f["Header"].attrs['BoxSize'][0]), int(f["ICs_parameters"].attrs['Grid Resolution'])
csize = float(f["Cells/Meta-data"].attrs['size'][0])
centres = f["Cells/Centres"][...]
counts = f["Cells/Counts/PartType1"][...].astype('i8')
offsets = f["Cells/OffsetsInFile/PartType1"][...].astype('i8')
idx = np.floor(centres / csize).astype('i8')
lo = 16 - ncell_side // 2
sel = np.all((idx >= lo) & (idx < lo + ncell_side), axis=1) & (counts > 0)
order = np.argsort(offsets[sel])
starts, ns = offsets[sel][order], counts[sel][order]
# merge contiguous ranges
ranges = []
for s, n in zip(starts, ns):
    if ranges and ranges[-1][1] == s: ranges[-1][1] = s + n
    else: ranges.append([s, s + n])
ntot = int(ns.sum())
print(f'{sel.sum()} cells, {len(ranges)} contiguous ranges, {ntot} particles, box [{lo * csize:.2f}, {(lo + ncell_side) * csize:.2f}] Mpc', flush=True)
x = np.empty((ntot, 3), 'f8'); v = np.empty((ntot, 3), 'f4'); ids = np.empty(ntot, 'u8'); m = np.empty(ntot, 'f4')
i, t0 = 0, time.time()
for ir, (s, e) in enumerate(ranges):
    n = e - s
    x[i:i + n] = f["PartType1/Coordinates"][s:e]
    v[i:i + n] = f["PartType1/Velocities"][s:e]
    ids[i:i + n] = f["PartType1/ParticleIDs"][s:e]
    m[i:i + n] = f["PartType1/Masses"][s:e]
    i += n
    if ir % 20 == 0: print(f'  {ir + 1}/{len(ranges)} ranges, {i / ntot:.0%}, {time.time() - t0:.0f} s', flush=True)
i0, i1, i2 = ids // (N * N), (ids // N) % N, ids % N
q = np.stack([i0, i1, i2], axis=1) * (L / N)
h = f["Header"].attrs; c = f["Cosmology"].attrs
np.savez(out, x=x, v=v, q=q, m=m, boxsize=L, lo=lo * csize, hi=(lo + ncell_side) * csize, redshift=float(h['Redshift'][0]),
         Omega_m=float(c['Omega_m'][0]), h=float(c['h'][0]), H=float(c['H [internal units]'][0]) if 'H [internal units]' in c else np.nan)
print(f'done in {time.time() - t0:.0f} s -> {out}', flush=True)
