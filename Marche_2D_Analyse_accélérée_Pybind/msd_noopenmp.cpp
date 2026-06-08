#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Fonction pour traiter la boucle en i (parallélisable)
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
    double *x_ptr = static_cast<double *>(x_buf.ptr);
    double *t_ptr = static_cast<double *>(t_buf.ptr);

    size_t Nt = t_buf.shape[0];
    size_t dim = x_buf.shape[1];

    for (size_t i = 0; i < Nt - 1; ++i) {
        std::vector<double> xi(dim);
        for (size_t d = 0; d < dim; ++d) {
            xi[d] = x_ptr[i * dim + d];
        }
        double ti = t_ptr[i];

        // Boucle en j (appel à une autre fonction)
        for (size_t j = i + 1; j < Nt; ++j) {
            double dt_ij = t_ptr[j] - ti;
            if (dt_ij > max_dt) break;

            // Calcul de dx
            double dx = 0.0;
            for (size_t d = 0; d < dim; ++d) {
                double tmp = x_ptr[j * dim + d] - xi[d];
                dx += tmp * tmp;
            }

            // Binning
            double log_dt = std::log10(dt_ij);
            int64_t k = static_cast<int64_t>((log_dt - log_min) * inv_step);

            if (k >= 0 && k < nbBins) {
                sums[k] += dx;
                counts[k] += 1;
            }
        }
    }
}

// Fonction principale
py::tuple MSDLongueVar_numba_opt(
    const py::array_t<double>& x,
    const py::array_t<double>& t
) {
    auto x_buf = x.request();
    auto t_buf = t.request();
    size_t Nt = t_buf.shape[0];
    size_t dim = x_buf.shape[1];

    // Calcul de dtmin
    double dtmin = 0.0;
    double *t_ptr = static_cast<double *>(t_buf.ptr);
    for (size_t i = 0; i < Nt - 1; ++i) {
        dtmin += (t_ptr[i+1] - t_ptr[i]);
    }
    dtmin /= (Nt - 1);
    if (dtmin <= 0.0) dtmin = 1e-15;

    // Calcul de dtmax
    double dtmax = t_ptr[Nt-1] * 0.5;
    if (dtmax <= 0.0) dtmax = 1e-15;

    // Bins log-uniformes
    int nbBins = 100;
    double log_min = std::log10(dtmin);
    double log_max = std::log10(dtmax);
    double step = (log_max - log_min) / (nbBins - 1);
    double inv_step = 1.0 / step;

    std::vector<double> dt_bins(nbBins);
    for (int i = 0; i < nbBins; ++i) {
        dt_bins[i] = std::pow(10.0, log_min + i * step);
    }

    std::vector<double> dt_centers(nbBins - 1);
    for (int i = 0; i < nbBins - 1; ++i) {
        dt_centers[i] = std::sqrt(dt_bins[i] * dt_bins[i+1]);
    }

    // Initialisation des sommes et comptes
    std::vector<double> sums(nbBins, 0.0);
    std::vector<int64_t> counts(nbBins, 0);
    double max_dt = dt_bins.back();

    // Appel à la fonction pour la boucle en i
    process_i_loop(x, t, log_min, log_max, nbBins, max_dt, inv_step, sums, counts);

    // Filtrage et préparation des résultats
    int n_ok = 0;
    for (int k = 0; k < nbBins; ++k) {
        if (counts[k] > 5) n_ok++;
    }

    std::vector<double> msd(n_ok);
    std::vector<double> dmsd(n_ok);
    std::vector<double> dt_out(n_ok);

    int p = 0;
    for (int k = 0; k < nbBins - 1; ++k) {
        if (counts[k] > 5) {
            double val = sums[k] / counts[k];
            msd[p] = val;
            dmsd[p] = val / std::sqrt(counts[k]);
            dt_out[p] = dt_centers[k];
            p++;
        }
    }

    // Conversion en arrays NumPy pour le retour
    py::array_t<double> dt_out_np = py::array_t<double>(n_ok, dt_out.data());
    py::array_t<double> msd_np = py::array_t<double>(n_ok, msd.data());
    py::array_t<double> dmsd_np = py::array_t<double>(n_ok, dmsd.data());

    return py::make_tuple(dt_out_np, msd_np, dmsd_np);
}

// Binding pour pybind11
PYBIND11_MODULE(msd_module_noopenmp, m) {
    m.def("MSDLongueVar_numba_opt", &MSDLongueVar_numba_opt, "Compute MSD with log-uniform bins");
}
