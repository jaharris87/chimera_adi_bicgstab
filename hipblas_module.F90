!***************************************************************************************************
! hipblas_module.f90 11/1/23
! This file contains the module defining Fortran interfaces for the hipblas library
!***************************************************************************************************

module hipblas_module
  !-------------------------------------------------------------------------------------------------
  ! Interface to hipblas routines
  !-------------------------------------------------------------------------------------------------
  use, intrinsic :: iso_c_binding
  use hipfort_hipblas_enums, only: &
    HIPBLAS_STATUS_SUCCESS
  use hipfort_hipblas, only: &
    hipblasCreate, &
    hipblasDestroy

  integer :: myhipblas_handle
  type(c_ptr) :: hipblas_handle
  !$omp threadprivate(hipblas_handle,myhipblas_handle)

  integer :: nhipblas_handle
  type(c_ptr) :: hipblas_handle_array_cptr
  type(c_ptr), pointer :: hipblas_handle_array(:)

end module hipblas_module
