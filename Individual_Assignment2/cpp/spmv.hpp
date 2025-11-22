
#include <bits/stdc++.h>
using namespace std;

struct CSR {
    int nrows, ncols;
    vector<int> ptr;
    vector<int> col;
    vector<double> val;
};

struct BucketCSR {
    int nrows, ncols;
    int cb;
    int nblocks;
    vector<vector<pair<int,int>>> buckets;
    vector<vector<double>> vals;
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

inline BucketCSR build_blocked_csr(const CSR &A, int cb = 128)
{
    BucketCSR B;
    B.nrows = A.nrows;
    B.ncols = A.ncols;
    B.cb = cb;
    B.nblocks = (A.ncols + cb - 1) / cb;

    B.buckets.resize(B.nblocks);
    B.vals.resize(B.nblocks);

    for(int i=0;i<A.nrows;i++){
        for(int k=A.ptr[i]; k<A.ptr[i+1]; k++){
            int c = A.col[k];
            int b = c / cb;
            B.buckets[b].push_back({i,c});
            B.vals[b].push_back(A.val[k]);
        }
    }
    return B;
}

inline void spmv_blocked_run(const BucketCSR &B, const vector<double> &x, vector<double> &y)
{
    fill(y.begin(), y.end(), 0.0);

    for(int b=0; b < B.nblocks; b++){
        const auto &Buck = B.buckets[b];
        const auto &V = B.vals[b];
        
        for(size_t j=0; j < Buck.size(); j++) {
            y[Buck[j].first] += V[j] * x[Buck[j].second];
        }
    }
}
