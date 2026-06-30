"""Validate cpcl_cov.compute_covariance against a pure-NumPy O(N^4) brute force."""
import numpy as np
import cpcl_cov


def legendre_all(x, lmax):
    P = np.empty((lmax,) + x.shape)
    P[0] = 1.0
    if lmax > 1:
        P[1] = x
    for l in range(1, lmax - 1):
        P[l + 1] = ((2 * l + 1) * x * P[l] - l * P[l - 1]) / (l + 1)
    return P


def cos_theta(gl_i, gb_i, gl_j, gb_j):
    deg = np.pi / 180.0
    gl_i, gb_i, gl_j, gb_j = gl_i * deg, gb_i * deg, gl_j * deg, gb_j * deg
    return (np.sin(gb_i)[:, None] * np.sin(gb_j)[None, :] +
            np.cos(gb_i)[:, None] * np.cos(gb_j)[None, :] *
            np.cos(gl_i[:, None] - gl_j[None, :]))


def brute_force_cov(pos, w, TB, lmax, full_ells, Sl_unbinned, noise_variance):
    N = pos.shape[1]
    n_bins = TB.shape[0]

    cos_full = cos_theta(pos[0], pos[1], pos[0], pos[1])
    P_full = legendre_all(cos_full, lmax)
    TBP_full = np.einsum('al,l...->a...', TB, P_full)  # (n_bins, N, N)

    w_mat = w[:, None] * w[None, :]
    w_mat = w_mat - np.diag(np.diag(w_mat))

    # Match the original algorithm: xi(cos_theta) comes from an interpolated
    # lookup grid (uniform in theta), not an exact per-pair Legendre sum.
    N_xi = 100 * lmax
    theta_grid = np.linspace(np.pi, 0.0, N_xi)
    cos_grid = np.cos(theta_grid)
    P_grid = legendre_all(cos_grid, lmax)
    xi_grid = np.einsum('l,lk->k', (2 * full_ells + 1) * Sl_unbinned, P_grid) / (4 * np.pi)
    xi = np.interp(cos_full.ravel(), cos_grid, xi_grid).reshape(cos_full.shape)
    Phi = xi + noise_variance * np.eye(N)

    M = w_mat[None, :, :] * TBP_full  # (n_bins, N, N)

    Cov = np.zeros((n_bins, n_bins))
    for a in range(n_bins):
        Za = Phi @ M[a]
        for b in range(n_bins):
            Zb = Phi @ M[b]
            Cov[a, b] = 2.0 * np.trace(Za @ Zb) / (4 * np.pi) ** 2
    return Cov


def main():
    rng = np.random.default_rng(0)
    N = 83  # deliberately not a multiple of any nice block size
    lmax = 40
    n_bins = 5

    gl = rng.uniform(0, 360, N)
    gb = rng.uniform(-80, 80, N)
    pos = np.array([gl, gb])
    w = rng.uniform(0.5, 1.5, N)

    edges = np.linspace(2, lmax, n_bins + 1).astype(int)
    edges = np.unique(edges)
    n_bins = len(edges) - 1
    TB = np.zeros((n_bins, lmax))
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        TB[i, lo:hi] = 1.0 / (hi - lo)

    full_ells = np.arange(lmax, dtype=float)
    Sl_unbinned = 1.0 / (full_ells + 1.0) ** 2
    noise_variance = 0.7

    cov_ref = brute_force_cov(pos, w, TB, lmax, full_ells, Sl_unbinned, noise_variance)
    cov_cpp = cpcl_cov.compute_covariance(pos, w, TB, lmax, full_ells, Sl_unbinned, noise_variance)
    cov_cpp_smallblock = cpcl_cov.compute_covariance(
        pos, w, TB, lmax, full_ells, Sl_unbinned, noise_variance, block_size=7)

    print("max abs diff (auto block):", np.max(np.abs(cov_ref - cov_cpp)))
    print("max abs diff (block=7)   :", np.max(np.abs(cov_ref - cov_cpp_smallblock)))
    print("max abs value            :", np.max(np.abs(cov_ref)))
    print("symmetric (auto)?", np.allclose(cov_cpp, cov_cpp.T))

    assert np.allclose(cov_ref, cov_cpp, rtol=1e-8, atol=1e-8), "auto block size mismatch"
    assert np.allclose(cov_ref, cov_cpp_smallblock, rtol=1e-8, atol=1e-8), "block_size=7 mismatch"
    print("OK: C++ result matches brute-force NumPy reference.")


if __name__ == "__main__":
    main()
