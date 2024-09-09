program driver
  use, intrinsic :: iso_fortran_env, only: dp=>real64, lu_stdout=>output_unit
  use gpu_module, only: initialize_gpu, finalize_gpu
  use mpi

  use adi_c_module, only: adi_bicgstab, pre_adi_init
  implicit none

  integer :: ierr, istat
  integer :: lu_input
  character(128) :: input_file = "input_adi_bicgstab"

  integer :: myrank, nranks

  integer :: i, j, k
  integer :: m, nb, it, iters
  real(dp), allocatable :: A(:,:,:), B(:,:), C(:,:), rhs(:,:)
  real(dp), allocatable :: A0(:,:,:), B0(:,:), C0(:,:), rhs0(:,:)

  integer, parameter :: iter_nu=30, iter_adi=10
  real(dp), parameter :: tol=1.0e-10_dp

  ! Initialize MPI
  call mpi_init( ierr )
  call mpi_comm_rank( MPI_COMM_WORLD, myrank, ierr )
  call mpi_comm_size( MPI_COMM_WORLD, nranks, ierr )
  if ( myrank == 0 ) write(lu_stdout,*) "MPI initialized"

  ! Initialize GPU
  call initialize_gpu
  if ( myrank == 0 ) write(lu_stdout,*) "GPU initialized"

  call mpi_barrier(MPI_COMM_WORLD,ierr)
  if ( myrank == 0 ) then
    write(lu_stdout,'(a)') "Reading input file: "//trim(input_file)
    open(newunit=lu_input, file=trim(input_file), form='unformatted', status='old', action='read')
    read(lu_input) m
    read(lu_input) nb
    allocate(A0(m,m,nb))
    allocate(B0(m,nb))
    allocate(C0(m,nb))
    allocate(rhs0(m,nb))
    read(lu_input) A0
    read(lu_input) B0
    read(lu_input) C0
    read(lu_input) rhs0
    close(lu_input)
    !write(lu_stdout,*) "A: ", A0
    !write(lu_stdout,*) "B: ", B0
    !write(lu_stdout,*) "C: ", C0
    !write(lu_stdout,*) "rhs: ", rhs0
  end if
  call mpi_bcast(m,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
  call mpi_bcast(nb,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)

  if ( myrank /= 0 ) then
    allocate(A0(m,m,nb))
    allocate(B0(m,nb))
    allocate(C0(m,nb))
    allocate(rhs0(m,nb))
  end if
  allocate(A(m,m,nb))
  allocate(B(m,nb))
  allocate(C(m,nb))
  allocate(rhs(m,nb))

  call mpi_bcast(A0,m*m*nb,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
  call mpi_bcast(B0,m*nb,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
  call mpi_bcast(C0,m*nb,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
  call mpi_bcast(rhs0,m*nb,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
  call mpi_barrier(MPI_COMM_WORLD,ierr)

  call pre_adi_init(nb,m)

  if ( myrank == 0 ) write(lu_stdout,*) "begin iterations"

  !$omp target enter data &
  !$omp map(to:A0,B0,C0,rhs0) &
  !$omp map(alloc:A,B,C,rhs)

  do it = 1, iter_nu

    !$omp target teams distribute parallel do simd collapse(2)
    do k = 1, nb
      do j = 1, m
        do i = 1, m
          A(i,j,k) = A0(i,j,k)
        end do
        B(j,k) = B0(j,k)
        C(j,k) = C0(j,k)
        rhs(j,k) = rhs0(j,k)
      end do
    end do

    call adi_bicgstab( m, nb, A, B, C, rhs, 10, 1.d-10, iters )
    if ( myrank == 0 ) write(lu_stdout,'(a,i3,a,es23.15)') "it = ", it, ", ||rhs|| = ",NORM2(rhs)

  end do

  !$omp target exit data &
  !$omp map(release:A0,B0,C0,rhs0,A,B,C,rhs)

  call finalize_gpu

  call mpi_finalize( ierr )

end program driver