#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <omp.h>

using namespace std;

/* =========================
   DATA STRUCTURES
   ========================= */

struct CSR {
    int nrows, ncols;
    vector<int> ptr;
    vector<int> col;
    vector<double> val;
};

struct BucketCSR_Compact {
    int nrows, ncols;
    int cb;
    int nblocks;

    vector<int> b_ptr;
    vector<int> row_indices;
    vector<int> col_indices;
    vector<double> val_data;
};

/* =========================
   MATRIX LOADING
   ========================= */

inline CSR load_mtx(const string& filename) {
    ifstream f(filename);
    if (!f.good()) {
        cerr << "Cannot open file\n";
        exit(1);
    }

    string line;
    do { getline(f, line); } while (line[0] == '%');

    int M, N, NNZ;
    stringstream ss(line);
    ss >> M >> N >> NNZ;

    vector<int> rows(NNZ), cols(NNZ);
    vector<double> vals(NNZ);

    for (int i = 0; i < NNZ; i++) {
        f >> rows[i] >> cols[i] >> vals[i];
        rows[i]--; cols[i]--;
    }

    CSR A;
    A.nrows = M;
    A.ncols = N;
    A.ptr.assign(M + 1, 0);

    for (int r : rows) A.ptr[r]++;
    for (int i = 1; i <= M; i++) A.ptr[i] += A.ptr[i - 1];

    A.col.resize(NNZ);
    A.val.resize(NNZ);

    vector<int> cnt(M, 0);
    for (int i = 0; i < NNZ; i++) {
        int r = rows[i];
        int dst = A.ptr[r] - (++cnt[r]);
        A.col[dst] = cols[i];
        A.val[dst] = vals[i];
    }

    return A;
}

/* =========================
   NAIVE CSR (PARALLEL)
   ========================= */

inline void spmv_naive_csr(const CSR& A,
                           const vector<double>& x,
                           vector<double>& y)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < A.nrows; i++) {
        double sum = 0.0;
        for (int k = A.ptr[i]; k < A.ptr[i + 1]; k++)
            sum += A.val[k] * x[A.col[k]];
        y[i] = sum;
    }
}

/* =========================
   BLOCKED CSR BUILD
   ========================= */

inline BucketCSR_Compact build_blocked_csr_compact(const CSR& A, int cb = 4096)
{
    BucketCSR_Compact B;
    B.nrows = A.nrows;
    B.ncols = A.ncols;
    B.cb = cb;
    B.nblocks = (A.ncols + cb - 1) / cb;

    vector<int> block_sizes(B.nblocks, 0);
    for (int k = 0; k < A.val.size(); k++)
        block_sizes[A.col[k] / cb]++;

    B.b_ptr.assign(B.nblocks + 1, 0);
    for (int b = 0; b < B.nblocks; b++)
        B.b_ptr[b + 1] = B.b_ptr[b] + block_sizes[b];

    int nnz = A.val.size();
    B.row_indices.resize(nnz);
    B.col_indices.resize(nnz);
    B.val_data.resize(nnz);

    vector<int> current(B.nblocks, 0);

    for (int i = 0; i < A.nrows; i++) {
        for (int k = A.ptr[i]; k < A.ptr[i + 1]; k++) {
            int c = A.col[k];
            int b = c / cb;
            int dst = B.b_ptr[b] + current[b]++;
            B.row_indices[dst] = i;
            B.col_indices[dst] = c;
            B.val_data[dst] = A.val[k];
        }
    }

    return B;
}

/* =========================
   BLOCKED CSR RUN (PARALLEL + ATOMIC)
   ========================= */

inline void spmv_compact_run_atomic(const BucketCSR_Compact& B,
                                    const vector<double>& x,
                                    vector<double>& y)
{
    fill(y.begin(), y.end(), 0.0);

    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B.nblocks; b++) {
        for (int j = B.b_ptr[b]; j < B.b_ptr[b + 1]; j++) {
            int r = B.row_indices[j];
            double v = B.val_data[j] * x[B.col_indices[j]];

            #pragma omp atomic
            y[r] += v;
        }
    }
}
