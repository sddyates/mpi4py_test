
import numpy as np
from mpi4py import MPI
import sys


class Grid:

    def __init__(self, parameters):
        

        self.res = parameters['res']
        self.gz = parameters['gz']
        self.ndims = 2
        self.mpi_x1 = parameters['mpix1']
        self.mpi_x2 = parameters['mpix2']
        self.mpi_x3 = parameters['mpix3']

        if self.ndims == 1:
            self.mpi_dims = [self.mpi_x1]
        if self.ndims == 2:
            self.mpi_dims = [self.mpi_x2, self.mpi_x1]
        if self.ndims == 3:
            self.mpi_dims = [self.mpi_x3, self.mpi_x2, self.mpi_x1]

        self.num_dim = len(self.mpi_dims)
        self.periods = [True for i in range(self.num_dim)]

    def create_mpi_comm(self, comm):
        return comm.Create_cart(
            dims = self.mpi_dims,
            periods = self.periods,
            reorder = True
        )


    def ObtainBoundaries(self, A, decomp):

        rank = decomp.Get_rank()
        coord = np.array(decomp.Get_coords(rank))

        gz = self.gz

        for dim in range(self.num_dim):

            if dim == 0:
                dis = np.array([[-1, 0, 0], [1, 0, 0]])
                sendbuf_up = A[gz:2*gz, :, :].copy()
                sendbuf_lo = A[-2*gz:-gz, :, :].copy()
            if dim == 1:
                dis = np.array([[0, -1, 0], [0, 1, 0]])
                sendbuf_up = A[:, gz:2*gz, :].copy()
                sendbuf_lo = A[:, -2*gz:-gz, :].copy()
            if dim == 2:
                dis = np.array([[0, 0, -1], [0, 0, 1]])
                sendbuf_up = A[:, :, gz:2*gz].copy()
                sendbuf_lo = A[:, :, -2*gz:-gz].copy()

            recvbuf_up = np.empty_like(sendbuf_up)
            recvbuf_lo = np.empty_like(sendbuf_lo)

            # Upper edge case (periodic).
            if self.periods[dim] and (coord[dim] == self.mpi_dims[dim] - 1):
                dis_coord = coord.copy()
                dis_coord[dim] = 0
                dist_rank = decomp.Get_cart_rank(dis_coord)
                decomp.Send([sendbuf_up, MPI.INT], dest=dist_rank, tag=4)
                decomp.Recv([recvbuf_up, MPI.INT], source=dist_rank, tag=3)

                if dim == 0:
                    A[-gz:, :, :] = recvbuf_up
                if dim == 1:
                    A[:, -gz:, :] = recvbuf_up
                if dim == 2:
                    A[:, :, -gz:] = recvbuf_up

            # lower edge case (periodic)
            if self.periods[dim] and (coord[dim] == 0):
                dis_coord = coord.copy()
                dis_coord[dim] = self.mpi_dims[dim]-1
                dist_rank = decomp.Get_cart_rank(dis_coord)
                decomp.Send([sendbuf_lo, MPI.INT], dest=dist_rank, tag=3)
                decomp.Recv([recvbuf_lo, MPI.INT], source=dist_rank, tag=4)

                if dim == 0:
                    A[:gz, :, :] = recvbuf_lo
                if dim == 1:
                    A[:, :gz, :] = recvbuf_lo
                if dim == 2:
                    A[:, :, :gz] = recvbuf_lo

            # Do not send lower BC if at start of grid.
            # Do not recive lower BC if at start of grid.
            if coord[dim] > 0:
                dist_rank = decomp.Get_cart_rank(coord + dis[0, :self.num_dim])
                decomp.Send([sendbuf_up, MPI.INT], dest=dist_rank, tag=1)
                decomp.Recv([recvbuf_lo, MPI.INT], source=dist_rank, tag=2)

                if dim == 0:
                    A[:gz, :, :] = recvbuf_lo
                if dim == 1:
                    A[:, :gz, :] = recvbuf_lo
                if dim == 2:
                    A[:, :, :gz] = recvbuf_lo

            # Do not send upper BC if at end of grid.
            # Do not recive upper BC if at end of grid.
            if coord[dim] < self.mpi_dims[dim] - 1:
                dist_rank = decomp.Get_cart_rank(coord + dis[1, :self.num_dim])
                decomp.Send([sendbuf_lo, MPI.INT], dest=dist_rank, tag=2)
                decomp.Recv([recvbuf_up, MPI.INT], source=dist_rank, tag=1)

                if dim == 0:
                    A[-gz:, :, :] = recvbuf_up
                if dim == 1:
                    A[:, -gz:, :] = recvbuf_up
                if dim == 2:
                    A[:, :, -gz:] = recvbuf_up

            if ~self.periods[dim] and (coord[dim] == self.mpi_dims[dim] - 1):

                if dim == 0:
                    A[-gz:, :, :] = A[-gz-1:-gz, :, :]
                if dim == 1:
                    A[:, -gz:, :] = A[:, -gz-1:-gz, :]
                if dim == 2:
                    A[:, :, -gz:] = A[:, :, -gz-1:-gz]

            if ~self.periods[dim] and (coord[dim] == 0):

                for o in range(gz):
                    if dim == 0:
                        A[o, :, :] = A[gz, :, :]
                    if dim == 1:
                        A[:, o, :] = A[:, gz, :]
                    if dim == 2:
                        A[:, :, o] = A[:, :, gz]


        return


    # def _ObtainExternalBoundaries(self, A, decomp):
    #
    #     gz = self.gz
    #
    #     for dim in range(self.num_dim):
    #         # Lower BC
    #         # reciprical
    #         if coord[dim] == 0:
    #             rank = decomp.Get_cart_rank(coord + dis[0, :self.num_dim])
    #             decomp.Send([sendbuf_up, MPI.INT], dest=rank, tag=1)
    #             decomp.Recv([recvbuf_lo, MPI.INT], source=rank, tag=2)
    #
    #         if dim == 0:
    #             A[:self.gz] = A[self.nx1:self.nx1 + self.gz]
    #         if dim == 1:
    #             A[:, :self.gz] = A[:, self.nx1:self.nx1 + self.gz]
    #         if dim == 2:
    #             A[:, :, :self.gz] = A[:, :, self.nx1:self.nx1 + self.gz]
    #
    #
    #         if coord[dim] == self.mpi_dims[dim] - 1:
    #             rank = decomp.Get_cart_rank(coord + dis[1, :self.num_dim])
    #             decomp.Send([sendbuf_lo, MPI.INT], dest=rank, tag=2)
    #             decomp.Recv([recvbuf_up, MPI.INT], source=rank, tag=1)
    #
    #         # Upper BC
    #         # reciprical
    #         if dim == 0:
    #             A[-gz:] = A[self.gz:self.gz + 1]
    #         if dim == 1:
    #             A[:, -gz:] = A[:, self.gz:self.gz + 1]
    #         if dim == 2:
    #             A[:, :, -gz:] = A[:, :, self.gz:self.gz + 1]
    #
    #     return



class Problem:

    def __init__(self):
        self.parameters = {
            'res': 3,
            'gz': 1,
            'mpix1': 2,
            'mpix2': 2,
            'mpix3': 1,
        }

    def initalise(self, grid, coord, rank):
        A =  np.ones([grid.res, grid.res, grid.res], dtype='i')*rank

        #for dim in range(grid.num_dim):

            #if (coord[dim] == 0) and (dim == 0):
        A[:grid.gz, :, :] = -1
            #if (coord[dim] == grid.mpi_dims[dim] - 1) and (dim == 0):
        A[-grid.gz:, :, :] = -1

            #if (coord[dim] == 0) and (dim == 1):
        A[:, :grid.gz, :] = -1
            #if (coord[dim] == grid.mpi_dims[dim] - 1) and (dim == 1):
        A[:, -grid.gz:, :] = -1

            #if (coord[dim] == 0) and (dim == 2):
        A[:, :, :grid.gz] = -1
            #if (coord[dim] == grid.mpi_dims[dim] - 1) and (dim == 2):
        A[:, :, -grid.gz:] = -1



        return A


def main():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    problem = Problem()
    grid = Grid(problem.parameters)

    decomp = grid.create_mpi_comm(comm)
    rank = decomp.Get_rank()
    coord = decomp.Get_coords(rank)

    A = problem.initalise(grid, coord, rank)

    print(f'rank: {rank}, coords: {coord}')
    print(A)
    print()
    sys.stdout.flush()

    grid.ObtainBoundaries(A, decomp)

    print('')
    print('After boundary')
    print('')
    print(f'rank: {rank}, coords: {coord}')
    print(A)
    print('')
    sys.stdout.flush()

if __name__ == "__main__" :
    main()
