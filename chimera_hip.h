#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#if !defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC)
#define __HIP_PLATFORM_HCC__
#endif
#include <hip/hip_runtime.h>
#include <hipsparse/hipsparse.h>
#include <hipblas/hipblas.h>

#define HIP_CALL(x) \
    do { \
        hipError_t err = x; \
        if((err) != hipSuccess ) { \
            fflush( stdout ); \
            printf("HIP Error (%d) at %s:%d\n",err,__FILE__,__LINE__); \
            exit(err); \
        } \
    } while(0)

#define HIPSPARSE_CALL(x) \
    do { \
        hipsparseStatus_t err = x; \
        if((err) != HIPSPARSE_STATUS_SUCCESS ) { \
            fflush( stdout ); \
            printf("HIPSPARSE Error (%d) at %s:%d\n",err,__FILE__,__LINE__); \
            exit(err); \
        } \
    } while(0)

#define HIPBLAS_CALL(x) \
    do { \
        hipblasStatus_t err = x; \
        if((err) != HIPBLAS_STATUS_SUCCESS ) { \
            fflush( stdout ); \
            printf("HIPBLAS Error (%d) at %s:%d\n",err,__FILE__,__LINE__); \
            exit(err); \
        } \
    } while(0)
