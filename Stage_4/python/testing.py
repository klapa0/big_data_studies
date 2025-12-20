import time
import os
import random
from matrix_mul import MRMatrixMultiplication

def generate_data(filename, N, block_size):
    with open(filename, 'w') as f:
        for i in range(N):
            for j in range(N):
                # 20% sparsity
                if random.random() > 0.8:
                    b_row = i // block_size
                    b_col = j // block_size
                    val = round(random.uniform(1, 10), 2)
                    # Write Matrix A and B (simulated same dimensions)
                    f.write(f"A,{b_row},{b_col},{i%block_size},{j%block_size},{val}\n")
                    f.write(f"B,{b_row},{b_col},{i%block_size},{j%block_size},{val}\n")

def run_evaluation():
    input_file = 'input_data.txt'
    N = 100
    block_size = 50
    grid_dim = N // block_size

    print(f"--- Python Distributed Matrix Mul Evaluation ---")
    print(f"Generating {N}x{N} matrix data...")
    generate_data(input_file, N, block_size)

    args = [input_file, '--grid-dim', str(grid_dim)]
    
    # Run MapReduce Job
    start_time = time.time()
    
    # Run the mrjob programmatically
    job = MRMatrixMultiplication(args=args)
    with job.make_runner() as runner:
        runner.run()
        # Iterate output to actually force execution
        count = 0
        for key, value in job.parse_output(runner.cat_output()):
            count += 1
            
    end_time = time.time()
    
    print(f"Processed {count} output elements.")
    print(f"Execution Time: {end_time - start_time:.4f} seconds")
    
    # Cleanup
    if os.path.exists(input_file):
        os.remove(input_file)

if __name__ == '__main__':
    run_evaluation()