MODULE adi_c_module
!-----------------------------------------------------------------------
!
!    Author:       J. A. Harris, U. Tennessee, Knoxville
!
!    Date:         6/4/14
!
!    Purpose:
!      To interface Fortran with ADI solver module written in C
!
!-----------------------------------------------------------------------

  USE, INTRINSIC :: ISO_C_BINDING

  PUBLIC :: pre_adi_init
  PUBLIC :: adi_cleanup
  PUBLIC :: adi_bicgstab

!-----------------------------------------------------------------------
! Function Interfaces
!-----------------------------------------------------------------------

  INTERFACE

    SUBROUTINE pre_adi_init(msize, nblocks) &
    & BIND(C, NAME="pre_adi_init")
    USE, INTRINSIC :: ISO_C_BINDING
    INTEGER(C_INT), VALUE :: nblocks
    INTEGER(C_INT), VALUE :: msize
    END SUBROUTINE pre_adi_init

    SUBROUTINE adi_cleanup() &
    & BIND(C, NAME="adi_cleanup")
    USE, INTRINSIC :: ISO_C_BINDING
    END SUBROUTINE adi_cleanup

    SUBROUTINE adi_bicgstab(msize, nblocks, A, B, C, rhs, niter, tol, iters) &
    & BIND(C, NAME="adi_bicgstab")
    USE, INTRINSIC :: ISO_C_BINDING
    INTEGER(C_INT), VALUE :: nblocks
    INTEGER(C_INT), VALUE :: msize
    REAL(C_DOUBLE) :: A(*)
    REAL(C_DOUBLE) :: B(*)
    REAL(C_DOUBLE) :: C(*)
    REAL(C_DOUBLE) :: rhs(*)
    INTEGER(C_INT), VALUE :: niter
    REAL(C_DOUBLE), VALUE :: tol
    INTEGER(C_INT) :: iters
    END SUBROUTINE adi_bicgstab

  END INTERFACE

END MODULE adi_c_module
