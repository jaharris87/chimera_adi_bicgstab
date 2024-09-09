!***************************************************************************************************
! hipsparse_module.f90 11/1/23
! This file contains the module defining Fortran interfaces for the hipsparse library

!***************************************************************************************************
module hipsparse_module
  !-------------------------------------------------------------------------------------------------
  ! Interface to hipsparse routines
  !-------------------------------------------------------------------------------------------------
  use, intrinsic :: iso_c_binding
  use hipfort_hipsparse_enums, only: &
    HIPSPARSE_STATUS_SUCCESS
  use hipfort_hipsparse, only: &
    hipsparseCreate, &
    hipsparseDestroy

  integer :: myhipsparse_handle
  type(c_ptr) :: hipsparse_handle
  !$omp threadprivate(hipsparse_handle,myhipsparse_handle)

  integer :: nhipsparse_handle
  type(c_ptr) :: hipsparse_handle_array_cptr
  type(c_ptr), pointer :: hipsparse_handle_array(:)

end module hipsparse_module
