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
        # Key is (BlockRow_C, BlockCol_C)
        # We need to reconstruct the blocks and multiply
        
        list_A = []
        list_B = []

        for val in values:
            if val[0] == 'A':
                list_A.append(val)
            else:
                list_B.append(val)

        # Naive local multiplication of the collected entries
        # In a real block algorithm, we would build dense numpy arrays here
        
        block_res = {} # (row, col) -> sum

        for a in list_A:
            # a = ('A', k, row, col, val)
            k_a, r_a, c_a, v_a = a[1], a[2], a[3], a[4]
            
            for b in list_B:
                # b = ('B', k, row, col, val)
                k_b, r_b, c_b, v_b = b[1], b[2], b[3], b[4]

                if k_a == k_b and c_a == r_b: # If column of A matches row of B
                    if (r_a, c_b) not in block_res:
                        block_res[(r_a, c_b)] = 0.0
                    block_res[(r_a, c_b)] += v_a * v_b

        # Emit the non-zero cells of the resulting block C
        for (r, c), v in block_res.items():
            yield key, (r, c, v)

if __name__ == '__main__':
    MRMatrixMultiplication.run()