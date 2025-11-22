#include <iostream>
#include <vector>
#include <functional>
#include <chrono>
#include <cmath>
#include <iomanip>
#include "spmv.hpp"

using namespace std;

inline double benchmark(function<void()> func, int iters=50){
    auto t0 = chrono::high_resolution_clock::now();
    for(int it=0; it<iters; it++)
        func();
    auto t1 = chrono::high_resolution_clock::now();
    return chrono::duration<double>(t1-t0).count() / iters;
}

void verify_results(const vector<double> &y1, const vector<double> &y2) {
    if (y1.size() != y2.size()) {
        cerr << "ERROR: Result vectors have different sizes." << endl;
        return;
    }
    double max_diff = 0.0;
    for (size_t i = 0; i < y1.size(); ++i) {
        max_diff = max(max_diff, abs(y1[i] - y2[i]));
    }
    if (max_diff > 1e-9) { 
        cout << "WARNING: Verification failed. Max difference: " << max_diff << " | ";
    } else {
        cout << "Results match | ";
    }
}

int main() {
    const string filename = "/home/kac/Pobrane/mc2depi/mc2depi.mtx"; 
    
    cout << "Loading matrix from: " << filename << endl;

    CSR A = load_mtx(filename); 
    
    cout << "Dimensions: " << A.nrows << " x " << A.ncols << ", NNZ: " << A.val.size() << endl;

    vector<double> x(A.ncols, 1.0);
    vector<double> y_naive(A.nrows);
    vector<double> y_blocked(A.nrows);
    
    const int IT = 50; 
    const int CB = 4096;
    
    cout << "\n--- SINGLE-THREADED (SIMD/Cache) TEST ---\n";

    double time_naive = benchmark([&](){
        spmv_naive_csr(A, x, y_naive);
    }, IT);
    cout << "1. Naive CSR time: " << fixed << setprecision(8) << time_naive << " s" << endl;

    cout << "   Building BucketCSR_Compact (CB=" << CB << ")...";
    auto t_build_start = chrono::high_resolution_clock::now();
    BucketCSR_Compact B = build_blocked_csr_compact(A, CB); 
    auto t_build_end = chrono::high_resolution_clock::now();
    double time_build = chrono::duration<double>(t_build_end - t_build_start).count();
    cout << " Done (Time: " << fixed << setprecision(8) << time_build << " s)" << endl;
    
    double time_blocked_run = benchmark([&](){
        spmv_compact_run(B, x, y_blocked);
    }, IT);
    cout << "2. Blocked CSR time: " << fixed << setprecision(8) << time_blocked_run << " s";
    verify_results(y_naive, y_blocked); 
    cout << endl;
    
    cout << "\n--- SUMMARY ---" << endl;
    cout << "1. Naive CSR:  " << fixed << setprecision(8) << time_naive << " s" << endl;
    cout << "2. Blocked CSR: " << fixed << setprecision(8) << time_blocked_run << " s" << endl;
    
    return 0;
}
