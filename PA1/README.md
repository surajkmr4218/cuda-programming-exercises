## Assignment Details

Question 1 for this assignment is split into multiple files: 

The naive-1, naive-2, and blocked matrix multiplication implementations are written in kernels.cuh
The main.cu program runs and compares these three MM kernels using the run.slurm batch job.
The output from this slurm script (output_log.o) was used for python data processing. 
As you can see in the output_log.o file, there were some experiments that were skipped due to 
limits on shared memory size. Also since each thread does N³/p² multiply-adds in sequence, the
skip check drops any experiments that would take too long as Professor K mentioned in class.

The tensor core kernel is written in the kernels_tc.cuh and ran in main_tc.cu using run_tc.slurm
The output of this slurm script lives in output_log_tc.o file.

The cuBLAS implementations are done in main_blas.cu and run using run_blas.slurm.
The output of this slurm script lives in output_blas.o file.

The modified_naive_1_kernel.cuh file answers the 2nd question of this program assignment

