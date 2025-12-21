import time
import os
import random
from matrix_mul import MRMatrixMultiplication

def generate_data(filename, N, block_size):
    """Generates test data with 20% density."""
    with open(filename, 'w') as f:
        for i in range(N):
            for j in range(N):
                if random.random() > 0.8:
                    b_row = i // block_size
                    b_col = j // block_size
                    val = round(random.uniform(1, 10), 2)
                    # Matrix A and B
                    f.write(f"A,{b_row},{b_col},{i%block_size},{j%block_size},{val}\n")
                    f.write(f"B,{b_row},{b_col},{i%block_size},{j%block_size},{val}\n")

def run_evaluation():
    # Test configurations matching your Java setup
    sizes = [100,500 ,1000]
    repetitions = 3

    for N in sizes:
        # Match Java logic: block size 50 for N=100, 500 for N=1000
        block_size = 50 if N <= 100 else 500
        grid_dim = N // block_size
        
        print(f"\n--- Performance Evaluation for N = {N} ---")
        print(f"Setup: Grid {grid_dim}x{grid_dim} (Block Size: {block_size})")

        total_time = 0

        for rep in range(1, repetitions + 1):
            input_file = f'input_{N}_{rep}.txt'
            generate_data(input_file, N, block_size)

            args = [input_file, '--grid-dim', str(grid_dim)]
            
            # Start timing
            start_time = time.time()
            
            job = MRMatrixMultiplication(args=args)
            # Use 'inline' runner for local testing (fastest in Python)
            # Use 'local' to simulate separate processes (closer to Hadoop)
            with job.make_runner() as runner:
                runner.run()
                # Drain the generator to ensure full execution
                count = sum(1 for _ in job.parse_output(runner.cat_output()))
            
            end_time = time.time()
            elapsed = (end_time - start_time) * 1000  # convert to ms
            total_time += elapsed
            
            print(f"Run #{rep} finished in: {elapsed:.2f} ms (Output elements: {count})")
            
            # Cleanup input file
            if os.path.exists(input_file):
                os.remove(input_file)

        avg_time = total_time / repetitions
        print(f">>> Average Execution Time for {N}x{N}: {avg_time:.2f} ms")

if __name__ == '__main__':
    run_evaluation()