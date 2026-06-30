// Brute-force catalog pseudo-Cl covariance, computed via an exact trace
// reformulation of the naive O(N^4) source-quadruplet sum.
//
// The naive covariance kernel (see run_covariance.py / cpcl_gpu.py,
// `sum_matrices`) sums, over all source quadruplets (p, q, r, s):
//
//   Cov[A,B] = (4 / (4*pi)^2) *
//       sum_{p,q,r,s} w_pq * TBP_A(theta_pq) * Phi_pr * w_rs * TBP_B(theta_rs) * Phi_qs
//
// where w_pq = w_p*w_q for p != q (0 on the diagonal), TBP_A(theta) is the
// binning+mode-coupling-projected Legendre sum sum_l TB[A,l]*P_l(cos theta),
// and Phi_pr = xi(theta_pr) + noise_variance*delta_pr is the full (signal +
// noise) field correlation.
//
// Defining the symmetric N x N matrices
//   M_A[p,q] = w_pq * TBP_A(theta_pq)      (zero diagonal)
//   Phi[p,r] = xi(theta_pr) + noise*delta_pr
// the quadruple sum is exactly
//   Cov[A,B] = (2 / (4*pi)^2) * Tr(Phi @ M_A @ Phi @ M_B)
// (provable via the symmetry of M_A, Phi and the cyclic property of the
// trace). This is mathematically identical to the brute-force sum but
// costs O(N^3 * n_bins) instead of O(N^4 * n_bins^2), which is what makes a
// CPU/OpenMP implementation feasible for catalogs with tens of thousands of
// sources.
//
// We never materialize the full N x N matrices. Sources are split into
// contiguous blocks of size `block_size` (auto-selected to bound the
// per-thread working set). For every pair of blocks (P, Q) with P <= Q we
// compute
//   Y_A[P,Q] = sum_K Phi[P,K] @ M_A[K,Q]     for every bin A
//   Y_A[Q,P] = sum_K Phi[Q,K] @ M_A[K,P]     for every bin A   (skipped if P==Q)
// via a block matrix-multiply accumulation over K, then fold both ordered
// contributions into Cov using Tr(X @ Y) = sum_ij X[i,j] * Y[j,i]. The (P,Q)
// task list is parallelized with OpenMP; BLAS (Accelerate / cblas) handles
// the block GEMMs.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <atomic>
#include <cstdio>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
extern "C"
{
#include <cblas.h>
}
#endif

namespace py = pybind11;

namespace
{

    constexpr double PI = 3.14159265358979323846;

    // ---------------------------------------------------------------------
    // Spherical-trig helper: cosine of the angular separation between two
    // points given in (longitude, latitude), both in radians.
    // ---------------------------------------------------------------------
    inline double cos_theta_rad(double gl_i, double gb_i, double gl_j, double gb_j)
    {
        return std::sin(gb_i) * std::sin(gb_j) +
               std::cos(gb_i) * std::cos(gb_j) * std::cos(gl_i - gl_j);
    }

    // ---------------------------------------------------------------------
    // Signal correlation function xi(cos_theta) on a lookup grid, built once.
    // Mirrors run_covariance.py: uniform-in-theta sampling (dense near
    // theta=0, i.e. large-ell pairs), ascending in cos_theta.
    // ---------------------------------------------------------------------
    struct XiGrid
    {
        std::vector<double> cos_grid; // ascending, size N_xi
        std::vector<double> xi;       // xi(cos_grid[k])

        double interp(double x) const
        {
            // Mirrors jnp.interp: clamp outside the grid range.
            if (x <= cos_grid.front())
                return xi.front();
            if (x >= cos_grid.back())
                return xi.back();
            auto it = std::upper_bound(cos_grid.begin(), cos_grid.end(), x);
            size_t hi = static_cast<size_t>(it - cos_grid.begin());
            size_t lo = hi - 1;
            double x0 = cos_grid[lo], x1 = cos_grid[hi];
            double t = (x - x0) / (x1 - x0);
            return xi[lo] + t * (xi[hi] - xi[lo]);
        }
    };

    XiGrid build_xi_grid(const double *full_ells, const double *Sl, int lmax)
    {
        const int N_xi = 100 * lmax;
        XiGrid grid;
        grid.cos_grid.resize(N_xi);
        grid.xi.resize(N_xi);

        for (int k = 0; k < N_xi; ++k)
        {
            double theta = PI - static_cast<double>(k) * (PI / static_cast<double>(N_xi - 1));
            grid.cos_grid[k] = std::cos(theta);
        }

#pragma omp parallel for schedule(static)
        for (int k = 0; k < N_xi; ++k)
        {
            double x = grid.cos_grid[k];
            double acc = (2.0 * full_ells[0] + 1.0) * Sl[0]; // P_0 = 1
            double P_lm1 = 1.0, P_l = x;
            if (lmax > 1)
                acc += (2.0 * full_ells[1] + 1.0) * Sl[1] * x; // P_1 = x
            for (int l = 1; l <= lmax - 2; ++l)
            {
                double P_lp1 = ((2.0 * l + 1.0) * x * P_l - static_cast<double>(l) * P_lm1) / (l + 1.0);
                acc += (2.0 * full_ells[l + 1] + 1.0) * Sl[l + 1] * P_lp1;
                P_lm1 = P_l;
                P_l = P_lp1;
            }
            grid.xi[k] = acc / (4.0 * PI);
        }
        return grid;
    }

    // ---------------------------------------------------------------------
    // Per-thread scratch buffers, sized once for the chosen block size.
    // ---------------------------------------------------------------------
    struct ThreadBuffers
    {
        int C;
        int lmax;
        int n_bins;

        std::vector<double> legendre; // (C*C) x lmax, row-major: one Legendre series per pair
        std::vector<double> M_raw;    // n_bins x (C*C), raw TBP (before weighting)
        std::vector<double> Phi_blk;  // C x C
        std::vector<double> M_blk;    // C x C (single bin's weighted M block, scratch)
        std::vector<double> Y_PQ;     // n_bins x (C*C)
        std::vector<double> Y_QP;     // n_bins x (C*C)
        std::vector<double> cos_blk;  // C*C scratch for cos(theta) values

        void resize(int C_, int lmax_, int n_bins_)
        {
            C = C_;
            lmax = lmax_;
            n_bins = n_bins_;
            legendre.resize(static_cast<size_t>(C) * C * lmax);
            M_raw.resize(static_cast<size_t>(n_bins) * C * C);
            Phi_blk.resize(static_cast<size_t>(C) * C);
            M_blk.resize(static_cast<size_t>(C) * C);
            Y_PQ.resize(static_cast<size_t>(n_bins) * C * C);
            Y_QP.resize(static_cast<size_t>(n_bins) * C * C);
            cos_blk.resize(static_cast<size_t>(C) * C);
        }
    };

    // ---------------------------------------------------------------------
    // Compute TBP_a(theta) for every (k,q) pair in a block, for all bins a
    // simultaneously: M_raw[a, k*mQ+q] = sum_l TB[a,l] * P_l(cos_blk[k*mQ+q]).
    //
    // legendre_buf is laid out (lmax) x (n_pairs), row-major -- i.e. row l
    // holds P_l(x) for every pair. The recurrence is therefore evaluated with
    // l on the outside and pairs on the inside: each step is a flat,
    // dependency-free elementwise loop over n_pairs that reads the two
    // previous rows and writes the next one directly into legendre_buf, no
    // scratch carry variables needed. This is what actually vectorizes (the
    // old per-pair-row layout nested a 300-iteration scalar recurrence
    // inside the "vectorized" loop, which the compiler could not unroll
    // across lanes). The resulting (lmax x n_pairs) layout also means the
    // TB contraction below needs no transpose.
    // ---------------------------------------------------------------------
    void compute_TBP_block(const double *TB, int n_bins, int lmax,
                           const double *cos_blk, int mK, int mQ,
                           std::vector<double> &legendre_buf,
                           std::vector<double> &M_raw_out)
    {
        const int n_pairs = mK * mQ;

        double *row0 = &legendre_buf[0];
#pragma omp simd
        for (int idx = 0; idx < n_pairs; ++idx)
            row0[idx] = 1.0;

        if (lmax > 1)
        {
            double *row1 = &legendre_buf[static_cast<size_t>(n_pairs)];
#pragma omp simd
            for (int idx = 0; idx < n_pairs; ++idx)
                row1[idx] = cos_blk[idx];
        }

        for (int l = 1; l <= lmax - 2; ++l)
        {
            const double *row_lm1 = &legendre_buf[static_cast<size_t>(l - 1) * n_pairs];
            const double *row_l = &legendre_buf[static_cast<size_t>(l) * n_pairs];
            double *row_lp1 = &legendre_buf[static_cast<size_t>(l + 1) * n_pairs];
            const double coef_a = 2.0 * l + 1.0;
            const double coef_b = static_cast<double>(l);
            const double inv_lp1 = 1.0 / (l + 1.0);
#pragma omp simd
            for (int idx = 0; idx < n_pairs; ++idx)
                row_lp1[idx] = (coef_a * cos_blk[idx] * row_l[idx] - coef_b * row_lm1[idx]) * inv_lp1;
        }

        // M_raw_out (n_bins x n_pairs) = TB (n_bins x lmax) @ legendre_buf (lmax x n_pairs)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    n_bins, n_pairs, lmax,
                    1.0, TB, lmax,
                    legendre_buf.data(), n_pairs,
                    0.0, M_raw_out.data(), n_pairs);
    }

    // ---------------------------------------------------------------------
    // Accumulate Y[a][P,Q] = sum_K Phi[P,K] @ M_a[K,Q] for all bins a, by
    // looping the inner block index K over the full source range.
    // ---------------------------------------------------------------------
    void accumulate_Y_block(
        const double *gl, const double *gb, const double *w,
        const double *TB, int n_bins, int lmax,
        double noise_variance, const XiGrid &xi_grid,
        int P_start, int mP, int Q_start, int mQ,
        int N, int block_size,
        ThreadBuffers &buf, double *Y_out /* n_bins x (mP*mQ), zero-initialized */)
    {

        const int n_chunks = (N + block_size - 1) / block_size;

        for (int Kc = 0; Kc < n_chunks; ++Kc)
        {
            int K_start = Kc * block_size;
            int mK = std::min(block_size, N - K_start);

            // Phi[P,K] block (mP x mK)
            for (int i = 0; i < mP; ++i)
            {
                int p = P_start + i;
                for (int k = 0; k < mK; ++k)
                {
                    int gidx = K_start + k;
                    double c = cos_theta_rad(gl[p], gb[p], gl[gidx], gb[gidx]);
                    double val = xi_grid.interp(c);
                    if (p == gidx)
                        val += noise_variance;
                    buf.Phi_blk[static_cast<size_t>(i) * mK + k] = val;
                }
            }

            // cos(theta) for the (K,Q) block, then TBP for all bins at once
            for (int k = 0; k < mK; ++k)
            {
                int gk = K_start + k;
                for (int q = 0; q < mQ; ++q)
                {
                    int gq = Q_start + q;
                    buf.cos_blk[static_cast<size_t>(k) * mQ + q] =
                        cos_theta_rad(gl[gk], gb[gk], gl[gq], gb[gq]);
                }
            }
            compute_TBP_block(TB, n_bins, lmax, buf.cos_blk.data(), mK, mQ,
                              buf.legendre, buf.M_raw);

            // For each bin: weight M_raw -> M_blk (zero diag where global k==q),
            // then GEMM-accumulate into Y_out[a].
            for (int a = 0; a < n_bins; ++a)
            {
                const double *M_raw_a = &buf.M_raw[static_cast<size_t>(a) * mK * mQ];
                double *M_blk = buf.M_blk.data();
                for (int k = 0; k < mK; ++k)
                {
                    int gk = K_start + k;
                    double wk = w[gk];
                    for (int q = 0; q < mQ; ++q)
                    {
                        int gq = Q_start + q;
                        double val = wk * w[gq] * M_raw_a[static_cast<size_t>(k) * mQ + q];
                        if (gk == gq)
                            val = 0.0;
                        M_blk[static_cast<size_t>(k) * mQ + q] = val;
                    }
                }
                double *Y_a = Y_out + static_cast<size_t>(a) * mP * mQ;
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            mP, mQ, mK,
                            1.0, buf.Phi_blk.data(), mK,
                            M_blk, mQ,
                            1.0, Y_a, mQ); // beta=1: accumulate over K
            }
        }
    }

    // ---------------------------------------------------------------------
    // Tr(X @ Y) for X: (m x n), Y: (n x m), both row-major.
    // ---------------------------------------------------------------------
    inline double trace_product(const double *X, const double *Y, int m, int n)
    {
        double s = 0.0;
        for (int i = 0; i < m; ++i)
        {
            const double *Xi = X + static_cast<size_t>(i) * n;
            for (int j = 0; j < n; ++j)
            {
                s += Xi[j] * Y[static_cast<size_t>(j) * m + i];
            }
        }
        return s;
    }

} // namespace

// ---------------------------------------------------------------------
// Python-facing entry point.
// ---------------------------------------------------------------------
py::array_t<double> compute_covariance(
    py::array_t<double, py::array::c_style | py::array::forcecast> pos,
    py::array_t<double, py::array::c_style | py::array::forcecast> w,
    py::array_t<double, py::array::c_style | py::array::forcecast> TB,
    int lmax,
    py::array_t<double, py::array::c_style | py::array::forcecast> full_ells,
    py::array_t<double, py::array::c_style | py::array::forcecast> Sl_unbinned,
    double noise_variance,
    int block_size = -1)
{

    auto pos_buf = pos.request();
    auto w_buf = w.request();
    auto TB_buf = TB.request();
    auto ells_buf = full_ells.request();
    auto Sl_buf = Sl_unbinned.request();

    if (pos_buf.ndim != 2 || pos_buf.shape[0] != 2)
        throw std::invalid_argument("pos must have shape (2, N)");
    const int N = static_cast<int>(pos_buf.shape[1]);
    if (w_buf.ndim != 1 || w_buf.shape[0] != N)
        throw std::invalid_argument("w must have shape (N,) matching pos");
    if (TB_buf.ndim != 2 || TB_buf.shape[1] != lmax)
        throw std::invalid_argument("TB must have shape (n_bins, lmax)");
    if (ells_buf.shape[0] != lmax || Sl_buf.shape[0] != lmax)
        throw std::invalid_argument("full_ells and Sl_unbinned must have length lmax");

    const int n_bins = static_cast<int>(TB_buf.shape[0]);
    const double *pos_ptr = static_cast<const double *>(pos_buf.ptr);
    const double *gl_deg = pos_ptr;
    const double *gb_deg = pos_ptr + N;
    const double *w_ptr = static_cast<const double *>(w_buf.ptr);
    const double *TB_ptr = static_cast<const double *>(TB_buf.ptr);
    const double *ells_ptr = static_cast<const double *>(ells_buf.ptr);
    const double *Sl_ptr = static_cast<const double *>(Sl_buf.ptr);

    // Degrees -> radians, once.
    std::vector<double> gl(N), gb(N);
    const double deg2rad = PI / 180.0;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i)
    {
        gl[i] = gl_deg[i] * deg2rad;
        gb[i] = gb_deg[i] * deg2rad;
    }

    XiGrid xi_grid = build_xi_grid(ells_ptr, Sl_ptr, lmax);

    // Auto-select block size from a per-thread memory budget. ThreadBuffers
    // holds 7 arrays of C*C elements: legendre is C*C*lmax, and M_raw/Y_PQ/
    // Y_QP are each C*C*n_bins, plus Phi_blk/M_blk/cos_blk at C*C each, so
    // total per-thread bytes = C^2 * (lmax + 3*n_bins + 3) * 8.
    // Benchmarked sweet spot (N=3000, lmax=301, n_bins=12, 8 threads):
    // C=81 -> 77.7s, C=200 -> 32.8s, C=400 -> 22.4s, C=800 -> 42.5s -- small
    // blocks waste time on tiny GEMM calls and repeated Legendre-buffer
    // traffic, very large blocks blow the working set out of cache; a ~400MB
    // per-thread budget reproduces the empirical optimum (C~400) and scales
    // sensibly to other lmax/n_bins.
    int C = block_size;
    if (C <= 0)
    {
        constexpr double target_bytes_per_thread = 400.0 * 1024.0 * 1024.0;
        const double per_pair_doubles = lmax + 3.0 * n_bins + 3.0;
        C = static_cast<int>(std::floor(std::sqrt(target_bytes_per_thread / (8.0 * per_pair_doubles))));
        C = std::max(C, 16);
        C = std::min(C, 2048);
    }
    C = std::min(C, N);
    C = std::max(C, 1);

    const int n_chunks = (N + C - 1) / C;
    const double scale = 0.5 / (4.0 * PI * PI); // 2 / (4*pi)^2

    // Flatten (P, Q) tasks with P <= Q.
    std::vector<std::pair<int, int>> tasks;
    tasks.reserve(static_cast<size_t>(n_chunks) * (n_chunks + 1) / 2);
    for (int P = 0; P < n_chunks; ++P)
        for (int Q = P; Q < n_chunks; ++Q)
            tasks.emplace_back(P, Q);

    std::vector<double> Cov(static_cast<size_t>(n_bins) * n_bins, 0.0);

    // Progress tracking: atomic counter incremented by each thread with
    // relaxed ordering (no synchronisation cost). fetch_add returns unique
    // values, so exactly one thread hits each print milestone -- no mutex.
    const size_t n_tasks = tasks.size();
    const size_t print_step = std::max<size_t>(1, n_tasks / 200); // ~0.5% steps
    std::atomic<size_t> completed{0};
    std::fprintf(stderr, "\rcpcl_cov:   0%%");
    std::fflush(stderr);

#pragma omp parallel
    {
        ThreadBuffers buf;
        buf.resize(C, lmax, n_bins);
        std::vector<double> Cov_local(static_cast<size_t>(n_bins) * n_bins, 0.0);

#pragma omp for schedule(dynamic) nowait
        for (size_t t = 0; t < tasks.size(); ++t)
        {
            int P = tasks[t].first;
            int Q = tasks[t].second;
            int P_start = P * C, mP = std::min(C, N - P_start);
            int Q_start = Q * C, mQ = std::min(C, N - Q_start);

            std::fill(buf.Y_PQ.begin(), buf.Y_PQ.begin() + static_cast<size_t>(n_bins) * mP * mQ, 0.0);
            accumulate_Y_block(gl.data(), gb.data(), w_ptr, TB_ptr, n_bins, lmax,
                               noise_variance, xi_grid, P_start, mP, Q_start, mQ,
                               N, C, buf, buf.Y_PQ.data());

            const double *YPQ = buf.Y_PQ.data();
            const double *YQP;
            if (P == Q)
            {
                YQP = YPQ;
            }
            else
            {
                std::fill(buf.Y_QP.begin(), buf.Y_QP.begin() + static_cast<size_t>(n_bins) * mQ * mP, 0.0);
                accumulate_Y_block(gl.data(), gb.data(), w_ptr, TB_ptr, n_bins, lmax,
                                   noise_variance, xi_grid, Q_start, mQ, P_start, mP,
                                   N, C, buf, buf.Y_QP.data());
                YQP = buf.Y_QP.data();
            }

            for (int a = 0; a < n_bins; ++a)
            {
                const double *YPQ_a = YPQ + static_cast<size_t>(a) * mP * mQ;
                const double *YQP_a = YQP + static_cast<size_t>(a) * mQ * mP;
                for (int b = 0; b < n_bins; ++b)
                {
                    const double *YPQ_b = YPQ + static_cast<size_t>(b) * mP * mQ;
                    const double *YQP_b = YQP + static_cast<size_t>(b) * mQ * mP;

                    double contrib = scale * trace_product(YPQ_a, YQP_b, mP, mQ);
                    if (P != Q)
                    {
                        contrib += scale * trace_product(YQP_a, YPQ_b, mQ, mP);
                    }
                    Cov_local[static_cast<size_t>(a) * n_bins + b] += contrib;
                }
            }

            // Exactly one thread will see each multiple of print_step.
            size_t done = completed.fetch_add(1, std::memory_order_relaxed) + 1;
            if (done % print_step == 0 || done == n_tasks)
            {
                std::fprintf(stderr, "\rcpcl_cov: %3.0f%%", 100.0 * done / static_cast<double>(n_tasks));
                std::fflush(stderr);
            }
        }

#pragma omp critical
        {
            for (size_t idx = 0; idx < Cov.size(); ++idx)
                Cov[idx] += Cov_local[idx];
        }
    }
    std::fprintf(stderr, "\n");
    std::fflush(stderr);

    py::array_t<double> result({n_bins, n_bins});
    auto res_buf = result.request();
    double *res_ptr = static_cast<double *>(res_buf.ptr);
    std::copy(Cov.begin(), Cov.end(), res_ptr);
    return result;
}

PYBIND11_MODULE(cpcl_cov, m)
{
    m.doc() = "OpenMP-parallel C++ pseudo-Cl covariance kernel "
              "(exact trace reformulation of the brute-force GPU kernel).";
    m.def("compute_covariance", &compute_covariance,
          py::arg("pos"), py::arg("w"), py::arg("TB"), py::arg("lmax"),
          py::arg("full_ells"), py::arg("Sl_unbinned"), py::arg("noise_variance"),
          py::arg("block_size") = -1,
          "Compute the brute-force pseudo-Cl covariance matrix.\n\n"
          "Same inputs as sum_matrices_jitted(pos, w, TB, lmax, full_ells, "
          "Sl_unbinned, noise_variance, n_chunks) except chunk_size is dropped "
          "(auto-selected internally); pass block_size to override.");
}
