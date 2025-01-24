MODULE gpu_module
!-----------------------------------------------------------------------
!
!    Author:       J. A. Harris, U. Tennessee, Knoxville
!
!    Date:         12/18/18
!
!    Purpose:
!      GPU initialization module for Chimera
!
!-----------------------------------------------------------------------

  USE, INTRINSIC :: ISO_C_BINDING

  IMPLICIT none

  INTEGER :: mydevice
  INTEGER :: deviceCount

!-----------------------------------------------------------------------
! Function Interfaces
!-----------------------------------------------------------------------

  INTERFACE

    SUBROUTINE initialize_gpu_c( mydevice, deviceCount, &
        & nhipblas_handle, nhipsparse_handle, nmagma_queue, nstream, nevent,  &
        & hipblas_handle_array_cptr, hipsparse_handle_array_cptr, magma_queue_array_cptr, &
        & streamArray_cptr, eventArray_cptr ) &
        & BIND(C, NAME="initialize_gpu_c")
      USE, INTRINSIC :: ISO_C_BINDING
      INTEGER(C_INT) :: mydevice
      INTEGER(C_INT) :: deviceCount
      INTEGER(C_INT) :: nhipblas_handle
      INTEGER(C_INT) :: nhipsparse_handle
      INTEGER(C_INT) :: nmagma_queue
      INTEGER(C_INT) :: nstream
      INTEGER(C_INT) :: nevent
      TYPE(C_PTR) :: hipblas_handle_array_cptr
      TYPE(C_PTR) :: hipsparse_handle_array_cptr
      TYPE(C_PTR) :: magma_queue_array_cptr
      TYPE(C_PTR) :: streamArray_cptr
      TYPE(C_PTR) :: eventArray_cptr
    END SUBROUTINE initialize_gpu_c

    SUBROUTINE finalize_gpu_c() &
        & BIND(C, NAME="finalize_gpu_c")
      USE, INTRINSIC :: ISO_C_BINDING
    END SUBROUTINE finalize_gpu_c

  END INTERFACE

CONTAINS

  SUBROUTINE initialize_gpu
#if defined( USE_CUDA )
    USE cublas_module, ONLY: &
      ngpublas_handle=>ncublas_handle, &
      gpublas_handle_array_cptr=>cublas_handle_array_cptr, &
      gpublas_handle_array=>cublas_handle_array, &
      mygpublas_handle=>mycublas_handle, &
      gpublas_handle=>cublas_handle
    USE cuda_module
    USE cusparse_module, ONLY: &
      ngpusparse_handle=>ncusparse_handle, &
      gpusparse_handle_array_cptr=>cusparse_handle_array_cptr, &
      gpusparse_handle_array=>cusparse_handle_array, &
      mygpusparse_handle=>mycusparse_handle, &
      gpusparse_handle=>cusparse_handle
#elif defined( USE_HIP )
    USE hipblas_module, ONLY: &
      ngpublas_handle=>nhipblas_handle, &
      gpublas_handle_array_cptr=>hipblas_handle_array_cptr, &
      gpublas_handle_array=>hipblas_handle_array, &
      mygpublas_handle=>myhipblas_handle, &
      gpublas_handle=>hipblas_handle
    USE hip_module
    USE hipsparse_module, ONLY: &
      ngpusparse_handle=>nhipsparse_handle, &
      gpusparse_handle_array_cptr=>hipsparse_handle_array_cptr, &
      gpusparse_handle_array=>hipsparse_handle_array, &
      mygpusparse_handle=>myhipsparse_handle, &
      gpusparse_handle=>hipsparse_handle
#endif
#if defined( USE_OACC )
    USE openacc_module
#endif
#if defined( USE_OMP ) || defined( USE_OMP_OL )
    USE omp_lib
#endif

    IMPLICIT none

    ! Local variables
    INTEGER :: mythread
    INTEGER :: nmagma_queue
    TYPE(C_PTR) :: magma_queue_array_cptr
    TYPE(C_PTR), POINTER :: magma_queue_array(:)

    CALL initialize_gpu_c( mydevice, deviceCount, &
      & ngpublas_handle, ngpusparse_handle, nmagma_queue, nstream, nevent,  &
      & gpublas_handle_array_cptr, gpusparse_handle_array_cptr, magma_queue_array_cptr, &
      & streamArray_cptr, eventArray_cptr )

    CALL C_F_POINTER( gpublas_handle_array_cptr, gpublas_handle_array, (/ ngpublas_handle /) )
    CALL C_F_POINTER( gpusparse_handle_array_cptr, gpusparse_handle_array, (/ ngpusparse_handle /) )
    CALL C_F_POINTER( streamArray_cptr, streamArray, (/ nstream /) )
    CALL C_F_POINTER( eventArray_cptr, eventArray, (/ nevent /) )

!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(mythread)

    mythread = 0
    !$ mythread = omp_get_thread_num()

    mygpublas_handle = MOD( mythread, ngpublas_handle ) + 1
    gpublas_handle = gpublas_handle_array(mygpublas_handle)

    mygpusparse_handle = MOD( mythread, ngpusparse_handle ) + 1
    gpusparse_handle = gpusparse_handle_array(mygpusparse_handle)

    mystream = MOD( mythread, nstream ) + 1
    stream = streamArray(mystream)

    event = eventArray(1)

!$OMP END PARALLEL

    RETURN
  END SUBROUTINE initialize_gpu

  SUBROUTINE finalize_gpu
    IMPLICIT none

    CALL finalize_gpu_c()

    RETURN
  END SUBROUTINE finalize_gpu

END MODULE gpu_module
