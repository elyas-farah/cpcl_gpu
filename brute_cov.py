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
def mcm_binning( TB, P):
    return jnp.einsum('la, aij->lij', TB, P)

@jit
def corr(s, P):
    return jnp.einsum('l, lij->ij', s, P)


def compute_legendre(cos_theta_ia, lmax):
    def body_fn(carry, l):
        P_lm1, P_l = carry
        P_lp1 = ((2*l + 1)*cos_theta_ia*P_l - l*P_lm1) / (l + 1)
        return (P_l, P_lp1), P_lp1
    
    carry_init = (jnp.ones_like(cos_theta_ia), cos_theta_ia)
    _, P_ia = lax.scan(body_fn, carry_init, jnp.arange(2, lmax))
    P_ia = jnp.concatenate([
        jnp.ones_like(cos_theta_ia)[None, :],   # P_0
        cos_theta_ia[None, :],                  # P_1
        P_ia                        # P_2 ... P_Lmax
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





def load_pcl_dataset(fits_path):

    with fits.open(fits_path) as hdul:
        
        pcl_names = sorted(hdu.name for hdu in hdul if hdu.name.startswith("PCL_"))

        if not pcl_names:

            raise ValueError(f"No PCL_* extensions found in {fits_path}")

        if "CELL" not in hdul:

            raise ValueError(f"No CELL extension found in {fits_path}")

        pcl_header = hdul[pcl_names[0]].header

        pcl_tables = [hdul[name].data for name in pcl_names]

        cell_table = hdul["CELL"].data
        RA = hdul['CATALOG'].data['RA']
        DEC = hdul['CATALOG'].data['DEC']


    ell_eff = np.asarray(pcl_tables[0]["ell_eff"], dtype=float)

    pcl_dm = np.vstack([np.asarray(tab["pcl_dm"], dtype=float) for tab in pcl_tables])

    pcl_dm_gaussian = np.vstack([np.asarray(tab["pcl_dm_gaussian"], dtype=float) for tab in pcl_tables])

    pcl_dm_lognormal = np.vstack([np.asarray(tab["pcl_dm_lognormal"], dtype=float) for tab in pcl_tables])

    theory_ell = np.asarray(cell_table["ell"], dtype=float)

    theory_cl = np.asarray(cell_table["cell"], dtype=float)

    valid_theory = np.isfinite(theory_ell) & np.isfinite(theory_cl) & (theory_ell >= 2)

    theory_ell = theory_ell[valid_theory]

    theory_cl = theory_cl[valid_theory]



    pcl_lmin = int(pcl_header.get("LMIN", max(2, np.floor(ell_eff.min()))))

    pcl_lmax = int(pcl_header.get("LMAX", np.ceil(ell_eff.max())))

    pcl_nbin = int(pcl_header.get("NBIN", len(ell_eff)))

    pcl_edges = np.rint(np.geomspace(pcl_lmin, pcl_lmax + 1, pcl_nbin + 1)).astype(int)

    pcl_edges[0] = pcl_lmin

    pcl_edges[-1] = pcl_lmax + 1

    pcl_edges = np.unique(pcl_edges)



    if len(pcl_edges) - 1 == len(ell_eff):

        theory_binned = np.full(len(ell_eff), np.nan, dtype=float)

        for index, (ell_lo, ell_hi) in enumerate(zip(pcl_edges[:-1], pcl_edges[1:])):

            in_bin = (theory_ell >= ell_lo) & (theory_ell < ell_hi)

            if np.any(in_bin):

                theory_binned[index] = np.mean(theory_cl[in_bin])

    else:

        theory_binned = np.interp(ell_eff, theory_ell, theory_cl, left=np.nan, right=np.nan)



    return {

        "path": Path(fits_path),

        "label": Path(fits_path).stem,

        "ell_eff": ell_eff,
        "pcl_edges": pcl_edges,

        "theory_ell": theory_ell,

        "theory_cl": theory_cl,

        "theory_binned": theory_binned,

        "series": {

            "DM": pcl_dm,

            "DM + Gaussian noise": pcl_dm_gaussian,

            "DM + lognormal noise": pcl_dm_lognormal,

        },
        'RA': RA,
        'DEC': DEC

    }
