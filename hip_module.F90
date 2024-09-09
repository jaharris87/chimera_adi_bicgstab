!***************************************************************************************************
! hip_module.f90 11/1/23
! This file contains the module defining Fortran interfaces for the HIP Runtime API
!***************************************************************************************************

module hip_module
  !-------------------------------------------------------------------------------------------------
  ! Interface to HIP Runtime API
  !-------------------------------------------------------------------------------------------------
  use, intrinsic :: iso_c_binding
  use hipfort_check, only: &
    hipCheck, &
    hipblasCheck, &
    hipsparseCheck
  use hipfort, only: &
    hipGetDeviceCount, &
    hipSetDevice, &
    hipStreamCreate, &
    hipStreamSynchronize, &
    hipDeviceSynchronize

  integer :: mystream
  type(c_ptr) :: stream, event
  !$omp threadprivate(stream, mystream, event)

  integer :: nstream, nevent
  type(c_ptr) :: streamArray_cptr
  type(c_ptr) :: eventArray_cptr
  type(c_ptr), pointer :: streamArray(:)
  type(c_ptr), pointer :: eventArray(:)

end module hip_module
