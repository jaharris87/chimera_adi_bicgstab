This is a reproducer of a bug in the Chimera code for INCITE project AST137.
Running with 1 rank per GCD gives the expected output, but if the GCD is shared by as few as 2 MPI ranks per GCD the solution contains unexpected NaNs.

The structure of the reproducer is as follows (see `driver.F90`):

1. Initialize MPI
2. Initialize GPU libraries and OpenMP runtime (see `gpu_module.F90` and `chimera_gpu.c`)
3. On rank 0, Read initial data from `input_adi_bicgstab`, which was generated from valid data from a `CHIMERA` simulation
4. Broadcast initial data to all ranks
5. Allocate GPU memory (see `pre_adi_init` in `adi_solver_GPU.c`)
6. Repeatedly (30 times) try to solve the system using `adi_bicgstab`, re-initializing the data for each attempt

---

Build instructions for Frontier:

```
module load PrgEnv-cray cray-hdf5-parallel craype-accel-amd-gfx90a rocm hipfort magma
make
```

Running with 1 rank per GCD (expected output):

```
> srun -u -A STF006 -t 00:05:00 -N 1 -n 8 -c 1 --ntasks-per-gpu=1 --gpu-bind=closest ./test > output
```

```
> cat output
 MPI initialized
 GPU initialized
Reading input file: input_adi_bicgstab
 begin iterations
it =   1, ||rhs|| =   1.686912334884165E-05
it =   2, ||rhs|| =   1.686912334884165E-05
it =   3, ||rhs|| =   1.686912334884165E-05
it =   4, ||rhs|| =   1.686912334884165E-05
it =   5, ||rhs|| =   1.686912334884165E-05
it =   6, ||rhs|| =   1.686912334884165E-05
it =   7, ||rhs|| =   1.686912334884165E-05
it =   8, ||rhs|| =   1.686912334884165E-05
it =   9, ||rhs|| =   1.686912334884165E-05
it =  10, ||rhs|| =   1.686912334884165E-05
it =  11, ||rhs|| =   1.686912334884165E-05
it =  12, ||rhs|| =   1.686912334884165E-05
it =  13, ||rhs|| =   1.686912334884165E-05
it =  14, ||rhs|| =   1.686912334884165E-05
it =  15, ||rhs|| =   1.686912334884165E-05
it =  16, ||rhs|| =   1.686912334884165E-05
it =  17, ||rhs|| =   1.686912334884165E-05
it =  18, ||rhs|| =   1.686912334884165E-05
it =  19, ||rhs|| =   1.686912334884165E-05
it =  20, ||rhs|| =   1.686912334884165E-05
it =  21, ||rhs|| =   1.686912334884165E-05
it =  22, ||rhs|| =   1.686912334884165E-05
it =  23, ||rhs|| =   1.686912334884165E-05
it =  24, ||rhs|| =   1.686912334884165E-05
it =  25, ||rhs|| =   1.686912334884165E-05
it =  26, ||rhs|| =   1.686912334884165E-05
it =  27, ||rhs|| =   1.686912334884165E-05
it =  28, ||rhs|| =   1.686912334884165E-05
it =  29, ||rhs|| =   1.686912334884165E-05
it =  30, ||rhs|| =   1.686912334884165E-05
```

Running with multiple ranks per GCD (unexepcted NaNs):

```
> srun -u -A STF006 -t 00:05:00 -N 1 -n 56 -c 1 --ntasks-per-gpu=7 --gpu-bind=closest ./test > output
```

```
> cat output
 MPI initialized
 GPU initialized
Reading input file: input_adi_bicgstab
 begin iterations
it =   1, ||rhs|| =   1.947459292435350+136
it =   2, ||rhs|| =                     NaN
it =   3, ||rhs|| =   1.686912334884165E-05
it =   4, ||rhs|| =                     NaN
it =   5, ||rhs|| =                     NaN
it =   6, ||rhs|| =                     NaN
it =   7, ||rhs|| =   1.686912334884165E-05
it =   8, ||rhs|| =                     NaN
it =   9, ||rhs|| =   1.686912334884165E-05
it =  10, ||rhs|| =   1.686912334884165E-05
it =  11, ||rhs|| =   1.686912334884165E-05
it =  12, ||rhs|| =   1.686912334884165E-05
it =  13, ||rhs|| =                     NaN
it =  14, ||rhs|| =                     NaN
it =  15, ||rhs|| =                     NaN
it =  16, ||rhs|| =   1.686912334884165E-05
it =  17, ||rhs|| =   1.686912334884165E-05
it =  18, ||rhs|| =                     NaN
it =  19, ||rhs|| =                     NaN
it =  20, ||rhs|| =                     NaN
it =  21, ||rhs|| =                     NaN
it =  22, ||rhs|| =                     NaN
it =  23, ||rhs|| =   1.686912334884165E-05
it =  24, ||rhs|| =                     NaN
it =  25, ||rhs|| =   1.686912334884165E-05
it =  26, ||rhs|| =                     NaN
it =  27, ||rhs|| =   1.686912334884165E-05
it =  28, ||rhs|| =   1.686912334884165E-05
it =  29, ||rhs|| =   1.686912334884165E-05
it =  30, ||rhs|| =   1.686912334884165E-05
```
