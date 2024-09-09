#CMODE = DEBUG
CMODE ?= OPT

USE_CUDA   ?= FALSE
USE_HIP    ?= TRUE

USE_OMP    ?= FALSE
USE_OMP_OL ?= TRUE
USE_OACC   ?= FALSE

MACHINE = frontier
-include ${MACHINE}.mak

FFLAGS = ${FFLAGS_${CMODE}}
CFLAGS = ${CFLAGS_${CMODE}}
LFLAGS = ${LFLAGS_${CMODE}}

INCLUDE = ${INC_LAPACK}
LIB     = ${LIB_LAPACK}
ifeq (${USE_CUDA},TRUE)
  INCLUDE += ${INC_CUDA}
  LIB     += ${LIB_CUDA}
  DEFINES += ${PP}USE_CUDA
endif
ifeq (${USE_HIP},TRUE)
  INCLUDE += ${INC_ROCM} ${INC_MAGMA}
  LIB     += ${LIB_ROCM} ${LIB_MAGMA}
  DEFINES += ${PP}USE_HIP
endif

ifeq (${USE_OMP},TRUE)
  FFLAGS  += ${FFLAGS_OMP}
  CFLAGS  += ${CFLAGS_OMP}
  LFLAGS  += ${LFLAGS_OMP}
  DEFINES += ${PP}USE_OMP
endif
ifeq (${USE_OMP_OL},TRUE)
  FFLAGS  += ${FFLAGS_OMP_OL}
  CFLAGS  += ${CFLAGS_OMP_OL}
  LFLAGS  += ${LFLAGS_OMP_OL}
  DEFINES += ${PP}USE_OMP_OL
endif
ifeq (${USE_OACC},TRUE)
  FFLAGS  += ${FFLAGS_OACC}
  CFLAGS  += ${CFLAGS_OACC}
  LFLAGS  += ${LFLAGS_OACC}
  DEFINES += ${PP}USE_OACC
endif

CXXFLAGS  = ${CFLAGS} ${CXXSTD}

OBJS = hip_module.o \
       hipblas_module.o \
       hipsparse_module.o \
       magma_module.o \
       gpu_module.o \
       chimera_gpu.o \
       adi_c_module.o \
       adi_solver_GPU.o \
       driver.o

EXE = test
.DEFAULT_GOAL := ${EXE}

.SUFFIXES: .c .cpp .f .F .f90 .F90

%.o : %.c
	${CCOMP} ${CFLAGS} ${INCLUDE} ${DEFINES} -c $< -o $@

%.o : %.cpp
	${CXXCOMP} ${CXXFLAGS} ${INCLUDE} ${DEFINES} -c $< -o $@

%.o : %.f
	${FCOMP} ${FFLAGS} ${f77FLAGS} ${INCLUDE} ${DEFINES} -c $< -o $@

%.o : %.F
	${FCOMP} ${FFLAGS} ${F77FLAGS} ${INCLUDE} ${DEFINES} -c $< -o $@

%.o : %.f90
	${FCOMP} ${FFLAGS} ${f90FLAGS} ${INCLUDE} ${DEFINES} -c $< -o $@

%.o : %.F90
	${FCOMP} ${FFLAGS} ${F90FLAGS} ${INCLUDE} ${DEFINES} -c $< -o $@

${EXE} : ${OBJS}
	${LINK} ${FFLAGS} ${F90FLAGS} ${INCLUDE} ${DEFINES} ${LFLAGS} ${LIB} ${OBJS} -o ${EXE}

.PHONY: clean

clean :
	rm -f *.o *.mod *.lst *.i *.cg *.opt

gpu_module.o : magma_module.o hipsparse_module.o hip_module.o hipblas_module.o chimera_gpu.o
driver.o : adi_c_module.o gpu_module.o