
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

my_N = 4
N = my_N * comm.size
if comm.rank == 0:
    A = np.arange(N, dtype=np.float64)
else:
    A = np.empty(N, dtype=np.float64)
my_A = np.empty(my_N, dtype=np.float64)

comm.Scatter( [A, MPI.DOUBLE], [my_A, MPI.DOUBLE] )

if comm.rank == 0:
    print(f"total array: {A}")

if comm.rank == 0:
    print("After Scatter:")
for r in range(comm.size):
    if comm.rank == r:
        print(f"thread: {comm.rank} array: {my_A}")
    comm.Barrier()

my_A *= 2

comm.Allgather( [my_A, MPI.DOUBLE], [A, MPI.DOUBLE] )

if comm.rank == 0:
    print("After Allgather:")
for r in range(comm.size):
    if comm.rank == r:
        print(f"thread: {comm.rank} array{A}")
    comm.Barrier()
