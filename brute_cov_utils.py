import numpy as np
import jax.numpy as jnp
from jax import lax, vmap, jit
from astropy.io import fits
from pathlib import Path


def build_binning_matrix(edges):
    nbins = len(edges) - 1
    lmin = edges[0]
    lmax = edges[-1]
    ncols = lmax - lmin  # half-open bins: [lmin, lmax)
    B = np.zeros((nbins, ncols))
    ell_eff = np.zeros(nbins)
    for b in range(nbins):
        ell_min, ell_max = edges[b], edges[b + 1]
        ells_in_bin = np.arange(ell_min, ell_max)
        Δℓ = len(ells_in_bin)
        if Δℓ > 0:
            B[b, ells_in_bin - lmin] = 1.0 / Δℓ
            ell_eff[b] = np.mean(ells_in_bin)
    return B, ell_eff



def pad_binning_matrix(B, lmin, lmax_mcm):
    """
    Pad B (shape: nbins × (lmax_bins − lmin)) to (nbins × (lmax_mcm + 1))
    so it aligns with a mode coupling matrix running from ℓ=0 to lmax_mcm.
    """
    nbins, ncols = B.shape
    total_cols = lmax_mcm
    B_padded = np.zeros((nbins, total_cols))
    print()
    B_padded[:, lmin:lmin + ncols] = B  # place values at the correct ℓ positions
    return B_padded



def cos_theta_pair(gl_i, gb_i, gl_j, gb_j):
    return (
        jnp.sin(gb_i) * jnp.sin(gb_j)
        + jnp.cos(gb_i) * jnp.cos(gb_j) * jnp.cos(gl_i - gl_j)
    )

# This funciton applies the operation from the ith index one-by-one to the jth index.
cos_theta_row = vmap(
    cos_theta_pair,
    in_axes=(None, None, 0, 0)
)


# This function applies the
cos_theta_matrix = vmap(
    cos_theta_row,
    in_axes=(0, 0, None, None)
)

@jit
def compute_cos_theta(gl_i, gb_i, gl_j, gb_j):
    deg = jnp.pi / 180.0
    return cos_theta_matrix(gl_i * deg, gb_i * deg,
                             gl_j * deg, gb_j * deg)

@jit
def mcm_binning( mcm_binned_inv, BP):
    return jnp.einsum('la, aij->lij', mcm_binned_inv, BP)

@jit
def corr(s, P):
    return jnp.einsum('l, lij->ij', s, P)


def compute_legendre(cos_theta_ia, lmax):
    def body_fn(carry, l):
        P_lm1, P_l = carry
        P_lp1 = ((2*l + 1)*cos_theta_ia*P_l - l*P_lm1) / (l + 1)
        return (P_l, P_lp1), P_lp1
    
    carry_init = (jnp.ones_like(cos_theta_ia), cos_theta_ia)
    _, P_ia = lax.scan(body_fn, carry_init, jnp.arange(1, lmax - 1))
    P_ia = jnp.concatenate([
        jnp.ones_like(cos_theta_ia)[None, :],   # P_0
        cos_theta_ia[None, :],                  # P_1
        P_ia                        # P_2 ... P_{lmax-1}
    ], axis=0)
    
    return P_ia


def batch_covariace(posi, posj, posa, posb, wi, wj, wa, wb, TB, lmax, full_ells, Sl_unbinned, noise_variance):
    compute_legendre_jitted = jit(compute_legendre, static_argnames='lmax')
    
    
    # it may cause some issues for the array modification since w could be treated as jax array.
    
   
    w_ij = wi[:, None] * wj[None, :]
    w_ij = lax.cond(jnp.all(posi == posj), lambda x: x - jnp.diag(jnp.diag(x)), lambda x: x, w_ij)
    
    
        
    w_ab = wa[:, None] * wb[None, :]
    w_ab = lax.cond(jnp.all(posa == posb), lambda x: x - jnp.diag(jnp.diag(x)), lambda x: x, w_ab)
    
    
    
    
        
    cos_theta_ij = compute_cos_theta(posi[0], posi[1], posj[0], posj[1])
    cos_theta_ab = compute_cos_theta(posa[0], posa[1], posb[0], posb[1])

    cos_theta_ia = compute_cos_theta(posi[0], posi[1], posa[0], posa[1])
    cos_theta_jb = compute_cos_theta(posj[0], posj[1], posb[0], posb[1])


    P_ij = compute_legendre_jitted(cos_theta_ij, lmax)
    
    TBP_ij = mcm_binning(TB, P_ij)
    del P_ij, cos_theta_ij
    
    

    P_ab = compute_legendre_jitted(cos_theta_ab, lmax)    
    TBP_ab = mcm_binning(TB, P_ab)
    del P_ab, cos_theta_ab
    
    
    # Preparing the correlators
    P_jb = compute_legendre_jitted(cos_theta_jb, lmax)
    signal_corr_jb = corr((2*full_ells + 1) *Sl_unbinned, P_jb)/4./np.pi

    del cos_theta_jb, P_jb
    

    P_ia = compute_legendre_jitted(cos_theta_ia, lmax)
    
    
    signal_corr_ia = corr((2*full_ells + 1) *Sl_unbinned, P_ia)/4./np.pi
    # signal_corr_ia = jnp.sum(((2*full_ells + 1) *Sl_unbinned)[:, None, None] * P_ia, axis = 0)/4./np.pi
    del P_ia, cos_theta_ia
    
    
    
    
    
    
    noise_variance_matrix = noise_variance*jnp.eye(wi.size)
    
    full_field_corr_ia = lax.cond(jnp.all(posi == posa), lambda s, n: s + n, lambda s, n: s, signal_corr_ia, noise_variance_matrix)
    full_field_corr_jb = lax.cond(jnp.all(posj == posb), lambda s, n: s + n, lambda s, n: s, signal_corr_jb, noise_variance_matrix)
    del signal_corr_ia, signal_corr_jb, noise_variance_matrix
    
    
    Cov_binned = 2*jnp.einsum('ij, aij, km, bkm, ik, jm->ab', w_ij, TBP_ij, w_ab, TBP_ab, full_field_corr_ia, full_field_corr_jb)/(4*np.pi)**2

   
    return Cov_binned


def sample_covariance_healpy(
    gl, gb, w, wasp, b, Cl_signal, noise_variance, n_realizations=500, seed=42
):
    """
    Monte Carlo benchmark via NaMaster's catalog-based pseudo-Cl estimator.

    For each realisation:
      1. Draw alm from Cl_signal via hp.synalm up to lmax_signal = len(Cl_signal)-1.
         This can exceed b.lmax so the full input power spectrum is included.
      2. Evaluate the field at exact source positions using ducc0's
         synthesis_general -- the same irregular-grid SHT backend that
         NaMaster uses internally for NmtFieldCatalog.  This avoids the
         pixel-window-function attenuation that hp.get_interp_val introduces
         (bilinear interpolation suppresses power at high ell and would cause
         the simulation covariance to be systematically low there).
      3. Add per-source Gaussian noise, then run the NaMaster pipeline:
             NmtFieldCatalog -> compute_coupled_cell -> wasp.decouple_cell
    The sample covariance of the resulting binned Cl estimates is a direct
    benchmark for the analytic covariance from sum_matrices_jitted.

    Args:
        gl, gb:          Galactic longitude / latitude of sources (degrees), shape (N,)
        w:               Source weights, shape (N,)
        wasp:            NmtWorkspace built from the catalog geometry
        b:               NmtBin object defining the bandpower binning
        Cl_signal:       Signal power spectrum C_ell[0..lmax_signal], starting from ell=0.
                         Pass the full input spectrum (e.g. up to ell_max_catalog) so
                         signal power at ell > b.lmax is correctly included in the field.
        noise_variance:  White-noise variance added per source (set 0 to disable).
                         Use the true per-source noise variance, not an inflated proxy.
        n_realizations:  Number of Monte Carlo draws
        seed:            RNG seed for reproducibility

    Returns:
        C_mean:     Mean binned Cl over realisations, shape (nbins,)
        Cov_sample: Sample covariance matrix, shape (nbins, nbins)
        estimates:  Per-realisation estimates, shape (n_realizations, nbins)
    """
    import healpy as hp
    import pymaster as nmt

    gl = np.asarray(gl, dtype=float)
    gb = np.asarray(gb, dtype=float)
    w  = np.asarray(w,  dtype=float)
    N  = len(gl)
    lmax = b.lmax  # NaMaster estimator band-limits to this ell

    # Signal is generated up to lmax_signal (can exceed b.lmax to include full spectrum).
    lmax_signal = len(np.asarray(Cl_signal)) - 1
    Cl_padded = np.asarray(Cl_signal, dtype=float)

    # Spherical coordinates in radians: colatitude in [0, pi], longitude in [0, 2pi]
    theta_hp = np.pi / 2.0 - np.deg2rad(gb)
    phi_hp   = np.deg2rad(gl) % (2.0 * np.pi)

    # ducc0 synthesis_general evaluates the alm at arbitrary (theta, phi) without
    # any pixel window function -- this is the same path NaMaster takes internally.
    try:
        from ducc0.sht.experimental import synthesis_general as _synth
        loc = np.column_stack([theta_hp, phi_hp])   # shape (N, 2)
        _use_ducc = True
    except ImportError:
        import warnings
        warnings.warn(
            "ducc0 not available; falling back to hp.alm2map + hp.get_interp_val. "
            "The bilinear interpolation introduces a pixel window function that "
            "suppresses high-ell power and will cause the simulation covariance to "
            "be systematically low at small scales. Install ducc0 for exact results.",
            RuntimeWarning, stacklevel=2,
        )
        _use_ducc = False
        # Choose nside high enough that the pixel window is negligible up to lmax_signal
        _nside = max(lmax_signal, 64)
        _nside = 2 ** int(np.ceil(np.log2(_nside)))

    nbins = b.get_n_bands()
    rng = np.random.default_rng(seed)
    estimates = np.zeros((n_realizations, nbins))

    try:
        from tqdm import tqdm
        loop = tqdm(range(n_realizations), desc='Simulating realisations')
    except ImportError:
        loop = range(n_realizations)

    for r in loop:
        alm = hp.synalm(Cl_padded, lmax=lmax_signal, new=True)

        if _use_ducc:
            # Exact irregular-grid SHT at source positions
            delta = _synth(
                alm=alm[np.newaxis, :].astype(np.complex128),
                spin=0,
                lmax=lmax_signal,
                loc=loc,
                epsilon=1e-8,
            ).real[0]
        else:
            delta = hp.get_interp_val(
                hp.alm2map(alm, nside=_nside, lmax=lmax_signal), theta_hp, phi_hp
            )

        if noise_variance > 0.0:
            delta = delta + rng.standard_normal(N) * np.sqrt(noise_variance)

        f = nmt.NmtFieldCatalog(
            positions=[gl, gb], weights=w, field=delta[None, :],
            lmax=lmax, spin=0, lonlat=True,
        )
        estimates[r] = wasp.decouple_cell(nmt.compute_coupled_cell(f, f))[0]

    C_mean     = np.mean(estimates, axis=0)
    Cov_sample = np.cov(estimates.T)

    return C_mean, Cov_sample, estimates


def batch_matrix_partitions(pos, w, partition=4):
    """
    Batch position and weight data into partitions for parallelized matrix calculations.
    Zero-pads the data so its length is a discrete multiple of `partition`.

    Args:
        pos:       2D array of shape (dims, N)
        w:         1D array of shape (N,)
        partition: number of points to calculate at once (default: 1000)

    Returns:
        posi_list, posj_list, posa_list, posb_list,
        wi_list,   wj_list,   wa_list,   wb_list
    """
    N = pos.shape[1]

    # Pad to the next discrete multiple of partition
    remainder = N % partition
    if remainder != 0:
        pad_size = partition - remainder
        pos = np.concatenate([pos, np.zeros((pos.shape[0], pad_size))], axis=1)
        w   = np.concatenate([w,   np.zeros(pad_size)])

    npartitions = pos.shape[1] // partition
    posi_list, posj_list, posa_list, posb_list = [], [], [], []
    wi_list,   wj_list,   wa_list,   wb_list   = [], [], [], []

    for i in range(npartitions):
        for j in range(npartitions):
            for a in range(npartitions):
                for b in range(npartitions):
                    i_start, i_end = i * partition, (i + 1) * partition
                    j_start, j_end = j * partition, (j + 1) * partition
                    a_start, a_end = a * partition, (a + 1) * partition
                    b_start, b_end = b * partition, (b + 1) * partition

                    posi_list.append(pos[:, i_start:i_end])
                    posj_list.append(pos[:, j_start:j_end])
                    posa_list.append(pos[:, a_start:a_end])
                    posb_list.append(pos[:, b_start:b_end])

                    wi_list.append(w[i_start:i_end])
                    wj_list.append(w[j_start:j_end])
                    wa_list.append(w[a_start:a_end])
                    wb_list.append(w[b_start:b_end])

    return (posi_list, posj_list, posa_list, posb_list,
            wi_list,   wj_list,   wa_list,   wb_list)









