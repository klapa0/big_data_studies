from mrjob.job import MRJob
from mrjob.step import MRStep

class MRMatrixMultiplication(MRJob):

    def configure_args(self):
        super().configure_args()
        self.add_passthru_arg(
            '--grid-dim',
            type=int,
            default=2,
            help='Number of blocks per dimension'
        )
    def mapper(self, _, line):
        # Input: Matrix, BlockRow, BlockCol, InnerRow, InnerCol, Value
        parts = line.strip().split(',')
        matrix = parts[0]
        block_row = int(parts[1])
        block_col = int(parts[2])
        inner_row = int(parts[3])
        inner_col = int(parts[4])
        value = float(parts[5])

        grid_dim = self.options.grid_dim

        # Logic: If I am block A(i,k), I need to be sent to all C(i, j) for j in 0..grid_dim
        if matrix == 'A':
            for j in range(grid_dim):
                yield (block_row, j), ('A', block_col, inner_row, inner_col, value)
        
        # Logic: If I am block B(k,j), I need to be sent to all C(i, j) for i in 0..grid_dim
        else:
            for i in range(grid_dim):
                yield (i, block_col), ('B', block_row, inner_row, inner_col, value)

    def reducer(self, key, values):
        # Key: (BlockRow_C, BlockCol_C)
        # Separate entries for A and B blocks
        blocks_A = {} # k -> list of (row, col, val)
        blocks_B = {} # k -> list of (row, col, val)

        for val in values:
            m_type, k, r, c, v = val[0], val[1], val[2], val[3], val[4]
            if m_type == 'A':
                if k not in blocks_A: blocks_A[k] = []
                blocks_A[k].append((r, c, v))
            else:
                if k not in blocks_B: blocks_B[k] = []
                blocks_B[k].append((r, c, v))

        # Result storage for the block
        # We use a dictionary to keep it sparse and avoid huge memory allocation
        res = {}

        # Optimization: Only iterate over k that exist in both matrices
        common_ks = set(blocks_A.keys()) & set(blocks_B.keys())

        for k in common_ks:
            # Group block B entries by row to avoid O(N^2) search
            # B_grouped: row_index -> list of (col_index, value)
            B_grouped = {}
            for rb, cb, vb in blocks_B[k]:
                if rb not in B_grouped: B_grouped[rb] = []
                B_grouped[rb].append((cb, vb))

            # Multiply A and B for this k
            for ra, ca, va in blocks_A[k]:
                if ca in B_grouped: # This is the "match" (column A == row B)
                    for cb, vb in B_grouped[ca]:
                        pos = (ra, cb)
                        res[pos] = res.get(pos, 0.0) + (va * vb)

        for (r, c), v in res.items():
            yield key, (r, c, v)

if __name__ == '__main__':
    MRMatrixMultiplication.run()