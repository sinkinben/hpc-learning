FLGAS=-O3 -Iinclude/
CC=g++ -std=c++17 $(FLGAS)
ISPC=~/ispc-v1.18.0-linux/bin/ispc -O3
NVCC=nvcc $(FLGAS) -D CUDA=1

cpu_naive:
	$(CC) cpu_naive.cpp main.cpp -o cpu_naive

cpu_opt_loop:
	$(CC) cpu_opt_loop.cpp main.cpp -o cpu_opt_loop

cpu_tile:
	$(CC) cpu_tile.cpp main.cpp -o cpu_tile

spmd:
	$(ISPC) spmd.ispc -h spmd.h -o spmd.o
	$(CC) spmd.o spmd.cpp -o spmd

cuda_mat_mul:
	$(NVCC) cuda_mat_mul.cu main.cpp -o cuda_mat_mul

cuda_mat_mul_shm:
	$(NVCC) cuda_mat_mul_shm.cu main.cpp -o cuda_mat_mul_shm

cuda_mat_mul_mmap:
	$(NVCC) -D CUDA_MMAP=1 cuda_mat_mul_mmap.cu main.cpp -o cuda_mat_mul_mmap

test: cpu_naive cpu_opt_loop cpu_tile spmd cuda_mat_mul cuda_mat_mul_shm cuda_mat_mul_mmap
	@echo -e 'Name\t\t    AvgTime(ns)\t\tAvgCycles'
	@./cpu_naive
	@./cpu_opt_loop
	@./cpu_tile
	@./spmd
	@./cuda_mat_mul
	@./cuda_mat_mul_shm
	@./cuda_mat_mul_mmap

clean:
	rm ./cpu_naive ./cpu_opt_loop ./cpu_tile 
	rm ./spmd ./spmd.h ./spmd.o
	rm ./cuda_mat_mul ./cuda_mat_mul_shm ./cuda_mat_mul_mmap