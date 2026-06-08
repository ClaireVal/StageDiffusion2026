
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>
 
namespace py = pybind11;


void process_i_loop(
    const py::array_t<double>& x,
    const py::array_t<double>& t,
    double log_min,
    double log_max,
    int nbBins,
    double max_dt,
    double inv_step,
    std::vector<double>& sums,       
    std::vector<int64_t>& counts     
) {
    auto x_buf = x.request();
    auto t_buf = t.request();
    const double* x_ptr = static_cast<const double*>(x_buf.ptr);
    const double* t_ptr = static_cast<const double*>(t_buf.ptr);
 
    const size_t Nt  = static_cast<size_t>(t_buf.shape[0]);
    const size_t dim = static_cast<size_t>(x_buf.shape[1]);
 
    // Région parallèle
    #pragma omp parallel
    {
        #pragma omp single
        {
            std::cout << "OpenMP actif : " << omp_get_num_threads()
                      << " threads." << std::endl;
        }
 
        // Accumulateurs thread-locaux — zéro contention, zéro atomic
        std::vector<double>  local_sums(nbBins, 0.0);
        std::vector<int64_t> local_counts(nbBins, 0);
 
        #pragma omp for schedule(dynamic, 64)
        for (size_t i = 0; i < Nt - 1; ++i) {
            // Lecture de xi sur la stack — pas d'allocation heap dans la boucle
            const double xi0 = x_ptr[i * dim];
            const double xi1 = x_ptr[i * dim + 1];
            const double ti  = t_ptr[i];
 
            for (size_t j = i + 1; j < Nt; ++j) {
                const double dt_ij = t_ptr[j] - ti;
                if (dt_ij > max_dt) break;
 
                const double dx0 = x_ptr[j * dim]     - xi0;
                const double dx1 = x_ptr[j * dim + 1] - xi1;
                const double dx  = dx0*dx0 + dx1*dx1;
 
                const double log_dt = std::log10(dt_ij);
                const int k = static_cast<int>((log_dt - log_min) * inv_step);
 
                if (k >= 0 && k < nbBins) {
                    local_sums[k]   += dx;
                    local_counts[k] += 1;
                }
            }
        }
 
        // Réduction : une section critique par thread (pas par paire)
        #pragma omp critical
        for (int k = 0; k < nbBins; ++k) {
            sums[k]   += local_sums[k];
            counts[k] += local_counts[k];
        }
    }
}
 
// ─────────────────────────────────────────────────────────────────────────── //
 
py::tuple MSDLongueVar_openmp(
    const py::array_t<double>& x,
    const py::array_t<double>& t
) {
    auto x_buf = x.request();
    auto t_buf = t.request();
 
    const size_t Nt  = static_cast<size_t>(t_buf.shape[0]);
    const double* t_ptr = static_cast<const double*>(t_buf.ptr);
 
    // dtmin = pas de temps moyen
    double dtmin = 0.0;
    for (size_t i = 0; i < Nt - 1; ++i)
        dtmin += t_ptr[i+1] - t_ptr[i];
    dtmin /= static_cast<double>(Nt - 1);
    if (dtmin <= 0.0) dtmin = 1e-15;
 
    // dtmax = moitié de la durée totale
    double dtmax = t_ptr[Nt - 1] * 0.5;
    if (dtmax <= 0.0) dtmax = 1e-15;
 
    // Grille log-uniforme
    const int nbBins  = 100;
    const double log_min  = std::log10(dtmin);
    const double log_max  = std::log10(dtmax);
    const double step     = (log_max - log_min) / (nbBins - 1);
    const double inv_step = 1.0 / step;
 
    std::vector<double> dt_bins(nbBins);
    for (int i = 0; i < nbBins; ++i)
        dt_bins[i] = std::pow(10.0, log_min + i * step);
 
    std::vector<double> dt_centers(nbBins - 1);
    for (int i = 0; i < nbBins - 1; ++i)
        dt_centers[i] = std::sqrt(dt_bins[i] * dt_bins[i+1]);
 
    std::vector<double>  sums(nbBins, 0.0);
    std::vector<int64_t> counts(nbBins, 0);
    const double max_dt = dt_bins.back();
 
    // Appel avec sums et counts passés par référence
    process_i_loop(x, t, log_min, log_max, nbBins, max_dt, inv_step, sums, counts);
 
    // Filtrage
    int n_ok = 0;
    for (int k = 0; k < nbBins; ++k)
        if (counts[k] > 5) ++n_ok;
 
    std::vector<double> msd(n_ok), dmsd(n_ok), dt_out(n_ok);
 
    int p = 0;
    for (int k = 0; k < nbBins - 1; ++k) {
        if (counts[k] > 5) {
            const double val = sums[k] / static_cast<double>(counts[k]);
            msd[p]    = val;
            dmsd[p]   = val / std::sqrt(static_cast<double>(counts[k]));
            dt_out[p] = dt_centers[k];
            ++p;
        }
    }
 
    py::array_t<double> dt_out_np(n_ok, dt_out.data());
    py::array_t<double> msd_np(n_ok, msd.data());
    py::array_t<double> dmsd_np(n_ok, dmsd.data());
 
    return py::make_tuple(dt_out_np, msd_np, dmsd_np);
}
 
PYBIND11_MODULE(msd_module_openmpBIS, m) {
    m.def("MSDLongueVar_openmp", &MSDLongueVar_openmp,
          "Compute MSD with log-uniform bins (OpenMP, thread-local accumulators)");
}

