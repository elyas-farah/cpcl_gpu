"""Regenerate the forecast analytic covariance (masked_fixed, 50k sources)
with the fixed pipeline: NaMaster decoupling operator + sharp correlator."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import pymaster as nmt
from astropy.coordinates import SkyCoord
from astropy import units as u
import cpcl_cov
from brute_cov import build_binning_matrix, pad_binning_matrix

d = np.load("plots/mock_masked_fixed.npz")
RA, DEC, DM = d['ra'], d['dec'], d['DM']
tr = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree, frame='icrs').galactic
gl, gb = np.array(tr.l), np.array(tr.b)
pos = np.array([gl, gb]); w = np.ones(gl.size)

edges = d['pcl_edges']; lmax = int(edges[-1])
b = nmt.NmtBin.from_edges(edges[:-1], edges[1:])
fmask = nmt.NmtFieldCatalog(positions=[gl, gb], weights=w, field=None,
                            lmax=b.lmax, spin=0, lonlat=True)
wasp = nmt.NmtWorkspace.from_fields(fmask, fmask, b)
mcm = np.array(wasp.get_coupling_matrix())

Binning, ell_eff = build_binning_matrix(edges)
Bpad = pad_binning_matrix(Binning, int(edges[0]), lmax)
U = np.zeros((lmax, len(edges) - 1))
for bi in range(len(edges) - 1):
    U[edges[bi]:edges[bi+1], bi] = 1.0
TB = np.linalg.inv(Bpad @ mcm @ U) @ Bpad
_p = np.zeros(lmax); _p[2:] = 1.0/(np.arange(2, lmax)+1.0)
_c = wasp.couple_cell(_p[None, :])
assert np.allclose(TB @ _c[0], wasp.decouple_cell(_c)[0], atol=1e-10)

full_ells = np.arange(0, edges[-1], dtype=float)
cl_th = d['cell']
cl_sharp = np.zeros(lmax); n = min(len(cl_th), lmax)
cl_sharp[:n] = cl_th[:n]; cl_sharp[0] = cl_sharp[1] = 0.0
field_variance = np.sum((2*full_ells + 1) * cl_sharp) / 4.0 / np.pi
noise_variance = np.var(DM) - field_variance
print(f"N={gl.size} lmax={lmax} nbins={len(ell_eff)} noise_variance={noise_variance:.4g}", flush=True)

import time; t0 = time.time()
cov = cpcl_cov.compute_covariance(pos, w, TB, lmax, full_ells, cl_sharp, noise_variance)
print(f"kernel done in {time.time()-t0:.1f}s", flush=True)

old = np.load("plots/covariance_forecast_analytic.npz")
np.savez("plots/covariance_forecast_analytic.npz",
         cov_theory=cov, ell_eff=d['pcl_ell_eff'],
         cov_err=old['cov_err'] if 'cov_err' in old.files else np.zeros_like(cov),
         sigma_data=old['sigma_data'] if 'sigma_data' in old.files else np.sqrt(np.diag(cov)),
         sigma_data_err=old['sigma_data_err'] if 'sigma_data_err' in old.files else np.zeros(len(ell_eff)))
print("saved plots/covariance_forecast_analytic.npz", flush=True)
