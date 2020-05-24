
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

print(f"thread {rank} of {size} running on {name}.")

comm.Barrier()

N = 5 

if rank == 0:
    A = np.arange(N, dtype=np.float64)
else:
    A = np.empty(N, dtype=np.float64)

comm.Bcast([A, MPI.DOUBLE])

print(f"array on {rank} is {A}")
