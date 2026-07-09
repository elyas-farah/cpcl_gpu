#!/usr/bin/env python
"""Brute-force catalog pseudo-Cl covariance calculation."""

# Must be set before numpy (and therefore OpenBLAS) is imported so that
# OpenBLAS initialises its thread pool to 1.  Our C++ kernel parallelises
# with OpenMP; each thread drives its own single-threaded BLAS call.
# Without this, OpenBLAS on nodes with many cores either warns
# "precompiled NUM_THREADS exceeded" or deadlocks.
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("GOTO_NUM_THREADS", "1")     # older OpenBLAS alias
os.environ.setdefault("MKL_NUM_THREADS", "1")      # MKL clusters

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

#data = np.load("./plots/mock_masked_fixed.npz")
data = np.load("./plots/mock_catalog.npz")

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
Binning_matrix_padded    = pad_binning_matrix(Binning, lmin, lmax)   # B_w  (n_bins x lmax), 1/dl weights

# Decoupling operator that maps a COUPLED pseudo-Cl to decoupled bandpowers.
# This MUST be identical to what NaMaster's `decouple_cell` applies to the
# mock pseudo-Cls, otherwise the analytic and sample covariances describe
# different estimators.  NaMaster bins the mode-coupling matrix and inverts it
# in bandpower space, i.e.  Chat_b = [ (B_w M U)^{-1} B_w ]_{bl} Ctilde_l ,
# NOT  B_w M^{-1}  (bin after a full-l inverse). 
_unbin = np.zeros((lmax, len(edges) - 1))
for _bi in range(len(edges) - 1):
    _unbin[edges[_bi]:edges[_bi + 1], _bi] = 1.0
TB = np.linalg.inv(Binning_matrix_padded @ mcm @ _unbin) @ Binning_matrix_padded

# Sanity check: TB must reproduce NaMaster's decoupling to machine precision.
_cl_probe = np.zeros(lmax); _cl_probe[2:] = 1.0 / (np.arange(2, lmax) + 1.0)
_coupled_probe = wasp.couple_cell(_cl_probe[None, :])
assert np.allclose(TB @ _coupled_probe[0], wasp.decouple_cell(_coupled_probe)[0], atol=1e-10), \
    "TB does not match NaMaster decouple_cell -- covariance estimator mismatch"

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

# Correlator spectrum for xi(theta) = sum_l (2l+1) C_l P_l / 4pi.  This must be
# the SHARP, per-l true signal spectrum.  Do NOT feed the bandpower-averaged
# (piecewise-constant) spectrum here: the covariance ~ C_l^2, so over a wide,
# steep low-l bandpower <C^2> >> <C>^2 and a bin-smoothed spectrum under-predicts
# the low-l covariance by up to ~3x (widest, lowest bin).  [The eq.(A-'S_l^B')
# constant-per-band approximation in the paper's Appendix is only adequate for
# narrow bins / flat spectra; use the sharp spectrum in general.]
cl_th_sharp = np.zeros(int(edges[-1]))
_ncl = min(len(cl_th), int(edges[-1]))
cl_th_sharp[:_ncl] = cl_th[:_ncl]
cl_th_sharp[0] = cl_th_sharp[1] = 0.0

cov = np.cov(data['pcl_dm'].T)

# White-noise floor sigma_N^2 to put on the diagonal of the field correlator.
# It must be the residual per-source *variance* NOT captured by the signal
# spectrum that the kernel uses to build xi(theta).  The kernel's
# xi(1) = sum_l (2l+1) Sl / 4pi with Sl = cl_th_sharp, so the signal-variance
# baseline is computed from the SAME spectrum, and we use the field variance
# (mean is subtracted in the pseudo-Cl, so no +mean**2).
field_variance = np.sum((2 * full_ells + 1) * cl_th_sharp) / 4.0 / np.pi
var_f          = np.var(DM)                       # field is mean-subtracted in the pCl
noise_variance = var_f - field_variance
# NOTE: estimating this from a single stored realisation (DM = dm_all[0]) is
# noisy; for a controlled test prefer the self-consistent validator below,
# where sigma_N^2 is known exactly by construction.

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
    cl_th_sharp, noise_variance,
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

np.savez(
    './plots/covariance_catalog_analytic.npz',
    cov_theory=cov_gpu_th,
    ell_eff=ell_eff,
    cov_err=cov_err,
    sigma_data=sigma_data,
    sigma_data_err=sigma_data_err,
)
print('Saved covariance_catalog_analytic.npz')


# ===========================================================================
# Self-consistent quick validation (namaster + healpy)
# ---------------------------------------------------------------------------
# Draws a Gaussian catalogue field whose source-to-source correlation is
# EXACTLY the model the brute-force kernel assumes,
#     <a_i a_j> = xi(theta_ij) + sigma_N^2 delta_ij,
#     xi(theta) = sum_l (2l+1) C_l P_l(cos theta) / 4pi ,
# measures its debiased pseudo-Cl with NaMaster over many realisations, and
# compares the sample covariance to the analytic one.  Because the mock field
# obeys the kernel's model by construction, agreement (to sampling noise) is a
# clean check of the kernel + the TB decoupling operator, with no ambiguity
# from band-limits, aux-variance padding, or noise bookkeeping.
#
# Run with:  RUN_SELFCHECK=1 python run_covariance.py
# ===========================================================================
def selfconsistency_check(lon, lat, cl, sigma_N2, edges, n_real=4000, seed=0,
                          max_sources=1500):
    rng = np.random.default_rng(seed)
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    cl = np.clip(np.asarray(cl, float), 0.0, None)        # a valid model spectrum is non-negative
    if lon.size > max_sources:                            # subsample for a quick check
        sel = rng.choice(lon.size, max_sources, replace=False)
        lon, lat = lon[sel], lat[sel]
    Nsrc = lon.size
    w = np.ones(Nsrc)
    lmax = len(cl) - 1
    # Use a strictly positive white-noise floor so Sigma is well-conditioned.
    sigma_N2 = max(float(sigma_N2), 1e-2 * np.sum((2 * np.arange(lmax + 1) + 1) * cl) / (4 * np.pi))

    bb = nmt.NmtBin.from_edges(edges[:-1], edges[1:])
    fmask = nmt.NmtFieldCatalog(positions=[lon, lat], weights=w, field=None,
                                lmax=bb.lmax, spin=0, lonlat=True)
    wsp = nmt.NmtWorkspace.from_fields(fmask, fmask, bb)
    M = np.array(wsp.get_coupling_matrix())
    nell = M.shape[0]
    Bn, leff = build_binning_matrix(edges)
    Bpad = pad_binning_matrix(Bn, edges[0], nell)
    Ub = np.zeros((nell, len(edges) - 1))
    for bi in range(len(edges) - 1):
        Ub[edges[bi]:edges[bi + 1], bi] = 1.0
    TBv = np.linalg.inv(Bpad @ M @ Ub) @ Bpad

    # exact correlation matrix Sigma = xi + sigma_N^2 I  (Legendre recurrence)
    lon_r, lat_r = np.radians(lon), np.radians(lat)
    xyz = np.array([np.cos(lat_r) * np.cos(lon_r),
                    np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)])
    cm = np.clip(xyz.T @ xyz, -1.0, 1.0)
    Plm1 = np.ones_like(cm); Pl = cm.copy()
    xi = cl[0] * np.ones_like(cm) + (3.0 * cl[1] * cm if lmax >= 1 else 0.0)
    for l in range(1, lmax):
        Plp1 = ((2 * l + 1) * cm * Pl - l * Plm1) / (l + 1)
        xi += (2 * (l + 1) + 1) * cl[l + 1] * Plp1
        Plm1, Pl = Pl, Plp1
    xi /= 4.0 * np.pi
    # Cholesky of Sigma = xi + sigma_N^2 I, with adaptive jitter if the
    # truncated/discretised xi is marginally non-positive-definite.
    Sigma = xi + sigma_N2 * np.eye(Nsrc)
    jit = 1e-9 * np.trace(Sigma) / Nsrc
    for _ in range(8):
        try:
            Lc = np.linalg.cholesky(Sigma + jit * np.eye(Nsrc))
            break
        except np.linalg.LinAlgError:
            jit *= 10.0
    else:
        raise np.linalg.LinAlgError("Sigma not positive definite even with jitter")

    cov_an = cpcl_cov.compute_covariance(np.array([lon, lat]), w, TBv, nell,
                                         np.arange(nell, dtype=float), cl, sigma_N2)

    pcls = np.zeros((n_real, len(leff)))
    for i in range(n_real):
        a = Lc @ rng.standard_normal(Nsrc)
        ff = nmt.NmtFieldCatalog(positions=[lon, lat], weights=w, field=a[None, :],
                                 lmax=bb.lmax, spin=0, lonlat=True)
        # NaMaster's compute_coupled_cell ALREADY subtracts the per-realisation
        # shot-noise bias Nf (= sum w_i^2 a_i^2 / 4pi) for catalog auto-spectra,
        # i.e. it returns the debiased coupled pCl that the i!=j brute-force
        # covariance models.  Do NOT subtract it again here.
        coupled = nmt.compute_coupled_cell(ff, ff)
        pcls[i] = wsp.decouple_cell(coupled)[0]
    cov_sim = np.cov(pcls.T)

    ratio = np.diag(cov_an) / np.diag(cov_sim)
    print('\n[selfcheck] ell_eff:', np.round(leff, 1))
    print('[selfcheck] analytic/sim diagonal:', np.round(ratio, 3))
    print('[selfcheck] (top 1-2 bandpowers near lmax are edge-affected; '
          'give lmax headroom above the science range and drop them)')
    return leff, cov_an, cov_sim


if os.environ.get('RUN_SELFCHECK'):
    # Band-limit the theory spectrum to the analysis lmax and use a known,
    # exact white-noise level so the comparison is fully self-consistent.
    _cl_bl = np.zeros(int(edges[-1]))
    _cl_bl[:len(cl_th)] = cl_th[:len(_cl_bl)]
    _cl_bl[0] = _cl_bl[1] = 0.0
    selfconsistency_check(gl, gb, _cl_bl, float(noise_variance), edges,
                          n_real=int(os.environ.get('N_SELFCHECK', '3000')))
