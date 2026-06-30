#!/usr/bin/env python
"""Brute-force catalog pseudo-Cl covariance calculation."""

import numpy as np
import matplotlib.pyplot as plt
import pymaster as nmt
from astropy.coordinates import SkyCoord
from astropy import units as u

import cpcl_cov
from brute_cov import build_binning_matrix, pad_binning_matrix
from utils2 import *

# ---------------------------------------------------------------------------
# Load data  ← edit this path as needed
# ---------------------------------------------------------------------------

data = np.load("mock_masked_fixed.npz")
# data = np.load("mock_catalog.npz")

print(data.files)

RA  = data['ra']
DEC = data['dec']
DM  = data['DM']

transformation = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree, frame='icrs').galactic
gl, gb = np.array(transformation.l), np.array(transformation.b)
pos = np.array([gl, gb])

# ---------------------------------------------------------------------------
# Binning and mode-coupling matrix
# ---------------------------------------------------------------------------

edges       = data['pcl_edges']
lmin        = np.min(edges)
lmax_nside  = np.max(edges)
lmax        = lmax_nside
b           = nmt.NmtBin.from_edges(edges[:-1], edges[1:])
w           = np.ones(gl.size)

f_mask = nmt.NmtFieldCatalog(
    positions=[gl, gb], weights=w, field=None,
    lmax=b.lmax, spin=0, lonlat=True,
)
wasp    = nmt.NmtWorkspace.from_fields(f_mask, f_mask, b)
mcm     = np.array(wasp.get_coupling_matrix())
mcm_inv = np.linalg.inv(mcm)

Binning, ell_eff         = build_binning_matrix(edges)
Binning_matrix_padded    = pad_binning_matrix(Binning, lmin, lmax)
TB = Binning_matrix_padded @ mcm_inv

# ---------------------------------------------------------------------------
# Power spectra and noise
# ---------------------------------------------------------------------------

f = nmt.NmtFieldCatalog(
    positions=[gl, gb], weights=w, field=DM[None, :],
    lmax=b.lmax, lonlat=True, spin=0,
)
Sl_coupled = nmt.compute_coupled_cell(f, f)
Nf         = f.Nf
Sl         = wasp.decouple_cell(Sl_coupled)
Sl_unbinned = b.unbin_cell(Sl)[0]

ells      = b.get_effective_ells()
full_ells = np.arange(0, edges[-1], dtype=float)

cl_th                    = data['cell']
cl_th_coupled            = wasp.couple_cell(cl_th[None, :edges[-1]])
cl_th_decoupled          = wasp.decouple_cell(cl_th_coupled)
cl_th_decoupled_unbinned = b.unbin_cell(cl_th_decoupled)[0]

cov = np.cov(data['pcl_dm'].T)

field_variance = np.sum((2 * full_ells + 1) * cl_th[:edges[-1]]) / 4.0 / np.pi
var_f          = np.var(DM) + np.mean(DM) ** 2
noise_variance = var_f - field_variance

# ---------------------------------------------------------------------------
# Block jackknife error on the sample covariance
# ---------------------------------------------------------------------------

n_blocks   = 100
pcl_dm     = data['pcl_dm']
N_real     = pcl_dm.shape[0]
block_size = N_real // n_blocks

cov_jk = np.zeros((n_blocks, cov.shape[0], cov.shape[1]))
for k in range(n_blocks):
    mask = np.ones(N_real, dtype=bool)
    mask[k * block_size : (k + 1) * block_size] = False
    cov_jk[k] = np.cov(pcl_dm[mask].T)

cov_jk_mean = cov_jk.mean(axis=0)
cov_err = np.sqrt(
    (n_blocks - 1) / n_blocks * np.sum((cov_jk - cov_jk_mean) ** 2, axis=0)
)

sigma_data     = np.sqrt(np.diag(cov))
sigma_data_err = np.diag(cov_err) / (2 * sigma_data)

print('Diagonal fractional jackknife error (cov_err / cov):',
      np.diag(cov_err) / np.diag(cov))

# ---------------------------------------------------------------------------
# Run brute-force covariance
# ---------------------------------------------------------------------------

cov_gpu_th = cpcl_cov.compute_covariance(
    pos, w, TB, int(edges[-1]), full_ells,
    cl_th_decoupled_unbinned, noise_variance,
)

# ---------------------------------------------------------------------------
# Plot and save
# ---------------------------------------------------------------------------

ell_eff = data['pcl_ell_eff']
Cov_sim = cov

fig, ax = plt.subplots()
ax.semilogx(ell_eff, np.diag(cov_gpu_th) / np.diag(Cov_sim), lw=2)
ax.fill_between(
    ell_eff,
    (np.diag(cov_gpu_th) / np.diag(Cov_sim)) * (1 - np.sqrt(np.diag(cov_err) / np.diag(cov))),
    (np.diag(cov_gpu_th) / np.diag(Cov_sim)) * (1 + np.sqrt(np.diag(cov_err) / np.diag(cov))),
    color='gray', alpha=0.5, label='Jackknife error on data covariance',
)
ax.axhline(1.0, color='k', ls='--', lw=0.8)
ax.set_xlabel(r'$\ell_{\rm eff}$')
ax.set_ylabel('Analytic / Data sample covariance')
ax.set_title('Diagonal ratio (target: 1)')
ax.legend()
plt.tight_layout()
plt.savefig('covariance_ratio.png', dpi=150)
plt.show()

np.savez(
    'covariance_forecast.npz',
    cov=cov_gpu_th,
    ell_eff=ell_eff,
    cov_err=cov_err,
    sigma_data=sigma_data,
    sigma_data_err=sigma_data_err,
)
print('Saved covariance_forecast.npz')
