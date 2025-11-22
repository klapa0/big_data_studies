#include <bits/stdc++.h>
using namespace std;

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

inline CSR load_mtx(const string &filename)
{
    ifstream f(filename);
    if(!f.good()){
        cerr<<"cannot open file\n";
        exit(1);
    }

    string line;
    do { getline(f,line); } while(line[0]=='%');

    int M, N, NNZ;
    stringstream ss(line);
    ss >> M >> N >> NNZ;

    vector<int> rows(NNZ), cols(NNZ);
    vector<double> vals(NNZ);
    for(int i=0; i<NNZ; i++){
        int r, c; double v;
        f >> r >> c >> v;
        rows[i] = r-1;
        cols[i] = c-1;
        vals[i] = v;
    }

    CSR A;
    A.nrows = M; A.ncols = N;
    A.ptr.assign(M+1,0);
    for(int r : rows) A.ptr[r]++;
    for(int i=1;i<=M;i++) A.ptr[i] += A.ptr[i-1];

    int nnz = NNZ;
    A.col.assign(nnz,0);
    A.val.assign(nnz,0);
    vector<int> cnt(M,0);
    for(int i=0;i<nnz;i++){
        int r = rows[i];
        int dst = A.ptr[r] - (++cnt[r]);
        A.col[dst] = cols[i];
        A.val[dst] = vals[i];
    }

    return A;
}

inline void spmv_naive_csr(const CSR &A, const vector<double> &x, vector<double> &y)
{
    for(int i=0;i<A.nrows;i++){
        double sum = 0.0;
        for(int k=A.ptr[i]; k<A.ptr[i+1]; k++)
            sum += A.val[k] * x[A.col[k]];
        y[i] = sum;
    }
}

inline BucketCSR_Compact build_blocked_csr_compact(const CSR &A, int cb = 128)
{
    BucketCSR_Compact B;
    B.nrows = A.nrows;
    B.ncols = A.ncols;
    B.cb = cb;
    B.nblocks = (A.ncols + cb - 1) / cb;

    vector<int> block_sizes(B.nblocks, 0);
    for(int k=0; k < A.val.size(); k++){
        int c = A.col[k];
        int b = c / cb;
        block_sizes[b]++;
    }

    B.b_ptr.assign(B.nblocks + 1, 0);
    for(int b=0; b < B.nblocks; b++){
        B.b_ptr[b+1] = B.b_ptr[b] + block_sizes[b];
    }

    size_t nnz = A.val.size();
    B.row_indices.resize(nnz);
    B.col_indices.resize(nnz);
    B.val_data.resize(nnz);
    
    vector<int> current_idx(B.nblocks, 0); 
    
    for(int i=0;i<A.nrows;i++){
        for(int k=A.ptr[i]; k<A.ptr[i+1]; k++){
            int c = A.col[k];
            int b = c / cb;
            
            int dst = B.b_ptr[b] + current_idx[b]++;
            
            B.row_indices[dst] = i;
            B.col_indices[dst] = c;
            B.val_data[dst] = A.val[k];
        }
    }

    return B;
}

inline void spmv_compact_run(const BucketCSR_Compact &B, const vector<double> &x, vector<double> &y)
{
    fill(y.begin(), y.end(), 0.0);

    for(int b=0; b < B.nblocks; b++){
        int start = B.b_ptr[b];
        int end = B.b_ptr[b+1];
        
        for(int j=start; j < end; j++) {
            y[B.row_indices[j]] += B.val_data[j] * x[B.col_indices[j]];
        }
    }
}