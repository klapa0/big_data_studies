#include "spmv.hpp"


inline double benchmark(function<void()> func, int iters=50){
    auto t0 = chrono::high_resolution_clock::now();
    for(int it=0; it<iters; it++)
        func();
    auto t1 = chrono::high_resolution_clock::now();
    return chrono::duration<double>(t1-t0).count() / iters;
}

// Przykład użycia (do usunięcia w finalnym pliku nagłówkowym)

int main() {
    // 1. Wczytanie macierzy A (przykład, wymagany plik .mtx)
    CSR A = load_mtx("/home/kac/Pobrane/mc2depi/mc2depi.mtx"); 
    
    // Inicjalizacja wektorów x i y
    vector<double> x(A.ncols, 1.0);
    vector<double> y_naive(A.nrows);
    vector<double> y_blocked(A.nrows);

    // 2. Porównanie naive vs. preprocesowana wersja run:

    // Pomiar SPMV NAIVE
    double time_naive = benchmark([&](){
        spmv_naive_csr(A, x, y_naive);
    }, 50);

    // 3. Budowanie struktury zablokowanej (koszt jednorazowy)
    BucketCSR B = build_blocked_csr(A, 4096); 

    // Pomiar SPMV BLOCKED RUN
    double time_blocked_run = benchmark([&](){
        spmv_blocked_run(B, x, y_blocked);
    }, 50);

    // W tym porównaniu (time_blocked_run vs time_naive)
    // powinny być widoczne zyski z cachowania dla dużej macierzy.
    
    cout << "Naive time: " << time_naive << endl;
    cout << "Blocked RUN time: " << time_blocked_run << endl;

    return 0;
}

