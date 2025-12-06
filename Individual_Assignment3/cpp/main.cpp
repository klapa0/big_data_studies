#include <iostream>
#include <vector>
#include <functional>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <omp.h>

#include "spmv.hpp"

using namespace std;

/* =========================
   BENCHMARK UTILS
   ========================= */

inline double benchmark(function<void()> func, int iters = 50) {
    auto t0 = chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++)
        func();
    auto t1 = chrono::high_resolution_clock::now();
    return chrono::duration<double>(t1 - t0).count() / iters;
}

void verify_results(const vector<double>& y1, const vector<double>& y2) {
    double max_diff = 0.0;
    for (size_t i = 0; i < y1.size(); i++)
        max_diff = max(max_diff, abs(y1[i] - y2[i]));

    if (max_diff > 1e-9)
        cout << "WARNING (max diff = " << max_diff << ")";
    else
        cout << "Results match";
}

/* =========================
   MAIN
   ========================= */

int main() {
    const string filename = "/home/kac/Pobrane/mc2depi/mc2depi.mtx";
    const int IT = 50;
    const int CB = 4096;

    cout << "Loading matrix: " << filename << endl;
    CSR A = load_mtx(filename);

    cout << "Matrix size: " << A.nrows << " x " << A.ncols
         << ", NNZ = " << A.val.size() << endl;

    vector<double> x(A.ncols, 1.0);
    vector<double> y_naive(A.nrows);
    vector<double> y_blocked(A.nrows);
    BucketCSR_Compact B = build_blocked_csr_compact(A, CB);
    
    cout << "\n--- PARALLEL SPMV BENCHMARK ---\n";

    for (int threads : {1, 2, 4, 8, 16}) {
        omp_set_num_threads(threads);
    
        cout << "\nThreads = " << threads << endl;
    
        double t_naive = benchmark([&]() {
            spmv_naive_csr(A, x, y_naive);
        }, IT);
    
        cout << "1. Naive CSR (parallel): "
             << fixed << setprecision(8) << t_naive << " s\n";
    
        double t_blocked = benchmark([&]() {
            spmv_compact_run_atomic(B, x, y_blocked);
        }, IT);
    
        cout << "2. Blocked CSR (parallel + atomic): "
             << fixed << setprecision(8) << t_blocked << " s | ";
    
        verify_results(y_naive, y_blocked);
        cout << endl;
    }

}
