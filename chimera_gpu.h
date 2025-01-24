#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include "mpi.h"
#include <fenv.h>

#if defined( USE_OMP )
#include "omp.h"
#endif

#if defined( USE_OACC )
#include "openacc.h"
#elif defined( USE_OMP_OL )
#include "omp.h"
#endif

#if defined( USE_CUDA )
#include "chimera_cuda.h"
#elif defined( USE_HIP )
#include "chimera_hip.h"
#endif

#if !defined ( USE_OACC ) && !defined( USE_OMP_OL )
#include <magma_v2.h>
//#include <magma_dbatched.h>
//#include <magmablas_d.h>
#else
#define MAGMA_SUCCESS 0

typedef int magma_int_t;
typedef magma_int_t magma_device_t;

struct magma_queue;
typedef struct magma_queue* magma_queue_t;
#endif

#define MAX(a, b) ((a >= b) ? a : b)
#define MIN(a, b) ((a <= b) ? a : b)

#define MAGMA_CALL(x) \
    do { \
        magma_int_t err = x; \
        if((err) != MAGMA_SUCCESS ) { \
            fflush( stdout ); \
            printf("MAGMA Error at %s:%d\n",__FILE__,__LINE__); \
            exit(err); \
        } \
    } while(0)

#if defined( USE_CUDA )
#define GPU_CALL(x)       CUDA_CALL(x)
#define GPUBLAS_CALL(x)   CUBLAS_CALL(x)
#define GPUSPARSE_CALL(x) CUSPARSE_CALL(x)

#define gpuHostAllocDefault   cudaHostAllocDefault
#define gpuEventDisableTiming cudaEventDisableTiming

#define magma_queue_create_from_gpu \
        magma_queue_create_from_cuda

typedef cublasHandle_t    gpublasHandle_t;
typedef cublasStatus_t    gpublasStatus_t;
typedef cublasOperation_t gpublasOperation_t;
#define GPUBLAS_OP_N         CUBLAS_OP_N
#define GPUBLAS_OP_T         CUBLAS_OP_T
#define GPUBLAS_OP_C         CUBLAS_OP_C
#define GPUBLAS_OP_HERMITAN  CUBLAS_OP_HERMITAN
#define GPUBLAS_OP_CONJG     CUBLAS_OP_CONJG

typedef cusparseHandle_t  gpusparseHandle_t;
typedef cusparseStatus_t  gpusparseStatus_t;

typedef cudaStream_t      gpuStream_t;
typedef cudaEvent_t       gpuEvent_t;
typedef cudaError_t       gpuError_t;
typedef cudaDeviceProp    gpuDeviceProp;
#elif defined( USE_HIP )
#define GPU_CALL(x)       HIP_CALL(x)
#define GPUBLAS_CALL(x)   HIPBLAS_CALL(x)
#define GPUSPARSE_CALL(x) HIPSPARSE_CALL(x)

#define gpuHostAllocDefault   hipHostMallocDefault
#define gpuEventDisableTiming hipEventDisableTiming

#define magma_queue_create_from_gpu \
        magma_queue_create_from_hip

typedef hipblasHandle_t    gpublasHandle_t;
typedef hipblasStatus_t    gpublasStatus_t;
typedef hipblasOperation_t gpublasOperation_t;
#define GPUBLAS_OP_N         HIPBLAS_OP_N
#define GPUBLAS_OP_T         HIPBLAS_OP_T
#define GPUBLAS_OP_C         HIPBLAS_OP_C
#define GPUBLAS_OP_HERMITAN  HIPBLAS_OP_HERMITAN
#define GPUBLAS_OP_CONJG     HIPBLAS_OP_CONJG

typedef hipsparseHandle_t  gpusparseHandle_t;
typedef hipsparseStatus_t  gpusparseStatus_t;

typedef hipStream_t        gpuStream_t;
typedef hipEvent_t         gpuEvent_t;
typedef hipError_t         gpuError_t;
typedef hipDeviceProp_t    gpuDeviceProp;
#endif

extern gpublasHandle_t *gpublas_handle_array;
extern gpublasHandle_t gpublas_handle;
extern int mygpublas_handle;
extern int ngpublas_handle;
//#pragma omp threadprivate( gpublas_handle, mygpublas_handle )

extern gpusparseHandle_t *gpusparse_handle_array;
extern gpusparseHandle_t gpusparse_handle;
extern int mygpusparse_handle;
extern int ngpusparse_handle;
//#pragma omp threadprivate( gpusparse_handle, mygpusparse_handle )

extern gpuStream_t *streamArray;
extern gpuStream_t stream;
extern int nstream;
extern int mystream;
//#pragma omp threadprivate( stream, mystream )

extern gpuEvent_t *eventArray;
extern int nevent;

extern magma_queue_t *magma_queue_array;
extern magma_queue_t magma_queue;
extern int mymagma_queue;
extern int nmagma_queue;
//#pragma omp threadprivate( magma_queue, mymagma_queue )

extern int deviceCount;
extern int mydevice;
extern int myid;
extern magma_device_t magma_device;

#ifdef __cplusplus
extern "C" {
#endif

void initialize_gpu_c( int *mydevice_f, int *deviceCount_f,
                       int *ngpublas_handle_f, int *ngpusparse_handle_f, int *nmagma_queue_f, 
                       int *nstream_f, int* nevent_f,
                       gpublasHandle_t **gpublas_handle_array_f,
                       gpusparseHandle_t **gpusparse_handle_array_f,
                       magma_queue_t **magma_queue_array_f,
                       gpuStream_t **streamArray_f,
                       gpuEvent_t **eventArray_f );

void finalize_gpu_c();

gpuError_t gpuMalloc
  ( void **devPtr, size_t size );

gpuError_t gpuHostAlloc
  ( void **ptr, size_t size, unsigned int flags );

gpuError_t gpuFree
  ( void *devPtr );

gpuError_t gpuFreeHost
  ( void *ptr );

gpuError_t gpuStreamCreate
  ( gpuStream_t *stream );

gpuError_t gpuStreamDestroy
  ( gpuStream_t stream );

gpuError_t gpuStreamSynchronize
  ( gpuStream_t stream );

gpuError_t gpuGetDeviceCount
  ( int *count );

gpuError_t gpuSetDevice
  ( int device );

gpuError_t gpuEventCreateWithFlags
  ( gpuEvent_t *event, unsigned int flags );

gpuError_t gpuEventDestroy
  ( gpuEvent_t event );

gpuError_t gpuGetDeviceProperties
  ( gpuDeviceProp *prop, int deviceId );

// --------------

gpublasStatus_t gpublasCreate
  ( gpublasHandle_t *handle );

gpublasStatus_t gpublasDestroy
  ( gpublasHandle_t handle );

gpublasStatus_t gpublasSetStream
  ( gpublasHandle_t handle, gpuStream_t streamId ); 

gpublasStatus_t gpublasGetStream
  ( gpublasHandle_t handle, gpuStream_t *streamId ); 

gpublasStatus_t gpublasSetMatrixAsync
  ( int rows, int cols, int elemSize,
    const void *A, int lda, void *B,
    int ldb, gpuStream_t stream );

gpublasStatus_t gpublasGetMatrixAsync
  ( int rows, int cols, int elemSize,
    const void *A, int lda, void *B,
    int ldb, gpuStream_t stream );

gpublasStatus_t gpublasSetVectorAsync
  ( int n, int elemSize, 
    const void *x, int incx, 
    void *y, int incy,
    gpuStream_t stream );

gpublasStatus_t gpublasGetVectorAsync
  ( int n, int elemSize,
    const void *x, int incx,
    void *y, int incy,
    gpuStream_t stream );

gpublasStatus_t gpublasDcopy
  ( gpublasHandle_t handle,
    int n, 
    const double *x, 
    int incx, 
    double *y, 
    int incy );

gpublasStatus_t gpublasDgetrfBatched
  ( gpublasHandle_t handle,
    const int n,
    double *const A[],
    const int lda,
    int *P,
    int *info,
    const int batchSize );

gpublasStatus_t gpublasDgetrsBatched
  ( gpublasHandle_t handle, 
    gpublasOperation_t trans, 
    const int n,
    const int nrhs,
    double *const A[],
    const int lda,
    const int *devIpiv, 
    double *const B[], 
    const int ldb,
    int *info,
    const int batchSize );
    
gpublasStatus_t gpublasDgeam 
  ( gpublasHandle_t handle,
    gpublasOperation_t transa, 
    gpublasOperation_t transb,
    int m, 
    int n,
    const double *alpha,
    const double *A, 
    int lda,
    const double *beta,
    const double *B, 
    int ldb,
    double *C, 
    int ldc );

gpublasStatus_t gpublasDgemmStridedBatched
  ( gpublasHandle_t handle,
    gpublasOperation_t transa,
    gpublasOperation_t transb, 
    int m,
    int n,
    int k,
    const double *alpha,
    const double *A, 
    int lda,
    long long int strideA,
    const double *B,
    int ldb, 
    long long int strideB,
    const double *beta,
    double *C,
    int ldc,
    long long int strideC,
    int batchCount );

gpublasStatus_t gpublasDaxpy
  ( gpublasHandle_t handle,
    int n, 
    const double *alpha,
    const double *x, 
    int incx, 
    double *y, 
    int incy );

gpublasStatus_t gpublasDscal
  ( gpublasHandle_t handle, 
    int n, 
    const double *alpha,
    double *x, 
    int incx );

gpublasStatus_t gpublasDdot
  ( gpublasHandle_t handle,
    int n, 
    const double *x, 
    int incx, 
    const double *y,
    int incy,
    double *result );

gpublasStatus_t gpublasDnrm2
  ( gpublasHandle_t handle, 
    int n, 
    const double *x, 
    int incx, 
    double *result );

// --------------

gpusparseStatus_t gpusparseCreate
  ( gpusparseHandle_t* handle );

gpusparseStatus_t gpusparseDestroy
  ( gpusparseHandle_t handle );

gpusparseStatus_t gpusparseSetStream
  ( gpusparseHandle_t handle,
    gpuStream_t       streamId );

gpusparseStatus_t gpusparseDgtsv2StridedBatch_bufferSizeExt 
  ( gpusparseHandle_t handle,
    int               m,
    const double*     dl,
    const double*     d,
    const double*     du,
    const double*     x,
    int               batchCount,
    int               batchStride,
    size_t*           bufferSizeInBytes );

gpusparseStatus_t gpusparseDgtsv2StridedBatch
  ( gpusparseHandle_t handle,
    int              m,
    const double*    dl,
    const double*    d,
    const double*    du,
    double*          x,
    int              batchCount,
    int              batchStride,
    void*            pBuffer);

// --------------

#ifdef __cplusplus
}
#endif
