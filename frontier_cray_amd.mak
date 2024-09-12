# makefile definitions for Frontier


ifeq ($(findstring $(PE_ENV),GNU CRAY),)
$(error Your environment "$(PE_ENV)" is invalid.  It must be "CRAY" or "GNU")
else
$(warning You are using the "$(PE_ENV)" environment)
endif

LC_PE_ENV = $(shell echo ${PE_ENV} | tr A-Z a-z)

#----------------------------------------------------------------------------
# Set the library paths -- these need to be updated for your system
#----------------------------------------------------------------------------

ifdef LAPACK_PATH
  INC_LAPACK = -I${LAPACK_PATH}/include
  LIB_LAPACK = -L${LAPACK_PATH}/lib64 -llapack -lblas
else
  LAPACK_PATH = ${CRAY_LIBSCI_PREFIX_DIR}
  INC_LAPACK =# -I${LAPACK_PATH}/include
  LIB_LAPACK =# -L${LAPACK_PATH}/lib -lsci_cray
endif

ROCM_PATH ?= ${OLCF_ROCM_ROOT}
ifneq (${ROCM_PATH},)
  INC_ROCM = -I$(ROCM_PATH)/include/hip -I$(ROCM_PATH)/include/hipblas -I$(ROCM_PATH)/include/hipsparse -I$(OLCF_HIPFORT_ROOT)/include/hipfort/amdgcn -I$(ROCM_PATH)/include -I$(OLCF_HIPFORT_ROOT)/include/hipfort/amdgcn 
  LIB_ROCM = -L$(OLCF_HIPFORT_ROOT)/lib -lhipfort-amdgcn -L$(ROCM_PATH)/lib -lrocsparse -lrocsolver -lrocblas -lhipblas -lhipsparse -lamdhip64
else
  ifeq (${USE_ROCM},TRUE)
    $(error Cannot resolve ROCM path. \
            Load the ROCM module---e.g. "module load rocm"--- \
            or define ROCM_PATH variable for a valid ROCM build.)
  endif
endif

MAGMA_PATH ?= ${OLCF_MAGMA_ROOT}
ifneq (${MAGMA_PATH},)
  INC_MAGMA = -I${MAGMA_PATH}/include
  LIB_MAGMA = -L$(MAGMA_PATH)/lib -lmagma
else
  ifeq (${USE_MAGMA},TRUE)
    $(error Cannot resolve MAGMA path. \
            Load the MAGMA module---e.g. "module load magma"--- \
            or define MAGMA_PATH variable for a valid MAGMA build.)
  endif
endif

#----------------------------------------------------------------------------
# Compiler and linker commands
#----------------------------------------------------------------------------

FCOMP   = ftn
CCOMP   = cc -x c++
CPPCOMP = CC -std=c++11
LINK    = ftn

#----------------------------------------------------------------------------
# Compilation flags
#----------------------------------------------------------------------------

## Compiler-specific flags
ifeq ($(PE_ENV),GNU)

    # pre-processor flag
    MDEFS         =
    PP            = -D

    # generic flags
    OPENMP        = -fopenmp

    OPT_FLAGS     = -g -O3
    TEST_FLAGS    = -g -O2
    DEBUG_FLAGS   = -g -Og

    # Fortran-specific flags
    OPT_FFLAGS    =
    TEST_FFLAGS   =
    DEBUG_FFLAGS  = -fcheck=bounds,do,mem,pointer -ffpe-trap=invalid,zero,overflow -fbacktrace

    F90FLAGS      = -fdefault-real-8 -fdefault-double-8 -fimplicit-none -ffree-line-length-none -fallow-argument-mismatch -cpp
    f90FLAGS      = ${F90FLAGS}
    F77FLAGS      = -fdefault-real-8 -fdefault-double-8 -fimplicit-none -fallow-argument-mismatch -cpp
    f77FLAGS      = ${F77FLAGS}

    FFLAGS_OACC   = -fopenacc
    FFLAGS_OMP_OL = -fopenmp

    # C-specific flags
    OPT_CFLAGS    =
    TEST_CFLAGS   =
    DEBUG_CFLAGS  =

    CFLAGS_OACC   = -fopenacc
    CFLAGS_OMP_OL = -fopenmp

    # Linker flags
    LIB_OPT       =
    LIB_TEST      =
    LIB_DEBUG     =

    LIB_OACC      =
    LIB_OMP_OL    =

else ifeq ($(PE_ENV),CRAY)

    # pre-processor flag
    MDEFS         =
    PP            = -D

    # generic flags
    OPENMP        = -fopenmp

    OPT_FLAGS     = -O2
    TEST_FLAGS    = -O1
    DEBUG_FLAGS   = -O0 -g

    # Fortran-specific flags
    OPT_FFLAGS    = -G2 -hlist=adm
    TEST_FFLAGS   =
    DEBUG_FFLAGS  = -Ktrap=fp

    F90FLAGS      = -s real64 -s integer32 -eI -eZ -ef -f free
    f90FLAGS      = ${F90FLAGS}
    F77FLAGS      = -s real64 -s integer32 -eI -eZ -ef -f fixed
    f77FLAGS      = ${F77FLAGS}

    FFLAGS_OACC   = -hacc
    FFLAGS_OMP_OL = -fopenmp

    # C-specific flags
    OPT_CFLAGS    = -g -Wno-error=int-conversion
    TEST_CFLAGS   = -Wno-error=int-conversion
    DEBUG_CFLAGS  = -ffpe-trap=fp -Wno-error=int-conversion

    CFLAGS_OACC   = -hacc
    CFLAGS_OMP_OL = -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a #-fno-cray

    # Linker flags
    LIB_OPT       =
    LIB_TEST      =
    LIB_DEBUG     =

    LIB_OACC      = -hacc
    LIB_OMP_OL    = -fopenmp

endif

## Language-specific flags
FFLAGS_OPT   = ${OPT_FLAGS} ${OPT_FFLAGS}
FFLAGS_TEST  = ${TEST_FLAGS} ${TEST_FFLAGS}
FFLAGS_DEBUG = ${DEBUG_FLAGS} ${DEBUG_FFLAGS}

CFLAGS_OPT   = ${OPT_FLAGS} ${OPT_CFLAGS}
CFLAGS_TEST  = ${TEST_FLAGS} ${TEST_CFLAGS}
CFLAGS_DEBUG = ${DEBUG_FLAGS} ${DEBUG_CFLAGS}

#----------------------------------------------------------------------------
# Linker flags
#----------------------------------------------------------------------------

LFLAGS_OPT   = ${OPT_FLAGS}
LFLAGS_TEST  = ${TEST_FLAGS}
LFLAGS_DEBUG = ${DEBUG_FLAGS}

