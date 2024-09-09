#include "chimera_gpu.h"

gpublasHandle_t *gpublas_handle_array;
gpublasHandle_t gpublas_handle;
int mygpublas_handle;
int ngpublas_handle;

gpusparseHandle_t *gpusparse_handle_array;
gpusparseHandle_t gpusparse_handle;
int mygpusparse_handle;
int ngpusparse_handle;

magma_queue_t *magma_queue_array;
magma_queue_t magma_queue;
int mymagma_queue;
int nmagma_queue;

int deviceCount;
int mydevice;
int myid;
int my_gpu_mpi_rank;  // HPE
int *gpu_win_data;    // HPE
MPI_Comm gpu_comm;    // HPE
MPI_Win gpu_win;      // HPE
magma_device_t magma_device;

gpuStream_t *streamArray;
gpuStream_t stream;
int nstream;
int mystream;

gpuEvent_t *eventArray;
int nevent;

void initialize_gpu_c( int *mydevice_f, int *deviceCount_f,
                       int *ngpublas_handle_f, int *ngpusparse_handle_f, int *nmagma_queue_f, 
                       int *nstream_f, int* nevent_f,
                       int *my_gpu_mpi_rank_f, MPI_Win *gpu_win_f,
                       gpublasHandle_t **gpublas_handle_array_f,
                       gpusparseHandle_t **gpusparse_handle_array_f,
                       magma_queue_t **magma_queue_array_f,
                       gpuStream_t **streamArray_f,
                       gpuEvent_t **eventArray_f )
{
   int i;
   int ierr = MPI_Comm_rank( MPI_COMM_WORLD, &myid );
// HPE: begin section 1 of 2 of changes to allow for serialization of MPI calls per GPU
   char my_host_name[MPI_MAX_PROCESSOR_NAME];
   char *me = "initialize_gpu_c", *err_str = NULL;
   const int IC0 = (int) '0', MIN_GPU_ID = 0, MAX_GPU_ID = 7;
   int err_str_len = -1, found = -1, ic = -1, len = -1, my_commu_id = -1, my_gpu_id = -1
     , my_host_num = -1, num_gpu_mpi_ranks = -1, rc = -1, rc2 = -1;
// HPE: end section 1 of 2 of changes to allow for serialization of MPI calls per GPU

   GPU_CALL( gpuGetDeviceCount( &deviceCount ) );
   mydevice = myid % deviceCount;
   GPU_CALL( gpuSetDevice( mydevice ) );
   magma_getdevice( &magma_device );

// HPE: begin section 2 of 2 of changes to allow for serialization of MPI calls per GPU
   err_str = (char*) malloc( MPI_MAX_ERROR_STRING );
   if ( ! err_str )
   {  fprintf( stderr, "%s error: 'malloc( MPI_MAX_ERROR_STRING ) failed.\n", me );
      MPI_Abort( MPI_COMM_WORLD, -1 );
   }

   rc = MPI_Get_processor_name( my_host_name, &len );
   if ( rc != MPI_SUCCESS )
   {  rc2 = MPI_Error_string( rc , err_str, &err_str_len );
      if ( rc2 == MPI_SUCCESS )
      {  fprintf( stderr, "%s error: MPI_Get_processor_name failed with rc = %d = '%s'.\n", me, rc, err_str ); }
      else
      {  fprintf( stderr, "%s error: MPI_Get_processor_name failed with rc = %d (no MPI_Error_string).\n", me, rc ); }
      MPI_Abort( MPI_COMM_WORLD, -1 );
   }

   found = 0;  // number of decimal digits found in my_host_name
   my_host_num = 0;
   for( i = 0; i < len; i++ )  // len = length( my_host_name[] )
   {  ic = ((int) my_host_name[i]) - IC0;
      if ( ic >= 0 && ic <= 9 )
      {  my_host_num *= 10;       // shift 1 decimal digit to the left
         my_host_num += ic;
         found++;
      }
   }
   if ( found <= 0 )
   {  fprintf( stderr, "%s error: found no digits in my_host_name = '%s'\n", me, my_host_name );
      MPI_Abort( MPI_COMM_WORLD, -1 );
   }

   const char* VIS_DEVS = getenv( "ROCR_VISIBLE_DEVICES" );
   if ( ! VIS_DEVS )
   {  fprintf( stderr, "%s error: getenv(ROCR_VISIBLE_DEVICES) failed on MPI-rank %d.\n"
             , me, myid );
      MPI_Abort( MPI_COMM_WORLD, -1 );
   }
   len = (int) strlen( VIS_DEVS );
   if ( len != 1 )
   {  fprintf( stderr, "%s error: ROCR_VISIBLE_DEVICES='%s' is != 1 char long on MPI-rank %d.\n"
             , me, VIS_DEVS, myid );
      MPI_Abort( MPI_COMM_WORLD, -1 );
   }
   my_gpu_id = ((int) VIS_DEVS[0]) - IC0;
   if ( my_gpu_id < MIN_GPU_ID || my_gpu_id > MAX_GPU_ID )
   {  fprintf( stderr, "%s error: my_gpu_id = %d is out of range [%d,%d] on MPI-rank %d in ROCR_VISIBLE_DEVICES='%s'.\n"
             , me, my_gpu_id, MIN_GPU_ID, MAX_GPU_ID, myid, VIS_DEVS );
      MPI_Abort( MPI_COMM_WORLD, -1 );
   }

   // Q: Should we use mydevice or my_gpu_id for my_commu_id?
   // my_commu_id = my_host_num * 10 + mydevice;
      my_commu_id = my_host_num * 10 + my_gpu_id;  // my_gpu_id = atoi( getenv ( "ROCR_VISIBLE_DEVICES" ) )
   fprintf( stderr
          , "%s info: my_commu_id=%d on MPI-rank=%d based on HOSTNAME='%s' and ROCR_VISIBLE_DEVICES='%s'.\n"
          , me, my_commu_id, myid, my_host_name, VIS_DEVS );

   // color = my_commu_id is shared by all MPI-ranks using the same GPU;
   // key = myid (= 0, 1, 2, ... in MPI_COMM_WORLD) determines the rank order
   //       in gpu_comm, regardless of what MPI-ranks of MPI_COMM_WORLD are
   //       on the GPU which is system-wide uniquely identified by my_commu_id;
   rc = MPI_Comm_split( MPI_COMM_WORLD, my_commu_id, myid, &gpu_comm );
   if ( rc != MPI_SUCCESS )
   {  rc2 = MPI_Error_string( rc , err_str, &err_str_len );
      if ( rc2 == MPI_SUCCESS )
      {  fprintf( stderr, "%s error: MPI_Comm_split failed for gpu_comm with rc = %d = '%s'.\n", me, rc, err_str ); }
      else
      {  fprintf( stderr, "%s error: MPI_Comm_split failed for gpu_comm with rc = %d (no MPI_Error_string).\n", me, rc ); }
      MPI_Abort( MPI_COMM_WORLD, -1 );
   }

   rc = MPI_Comm_rank( gpu_comm, &my_gpu_mpi_rank );  // if (7 MPI-ranks/GPU) then my_gpu_mpi_rank in [0,6];
   if ( rc != MPI_SUCCESS )
   {  rc2 = MPI_Error_string( rc , err_str, &err_str_len );
      if ( rc2 == MPI_SUCCESS )
      {  fprintf( stderr, "%s error: MPI_Comm_rank failed for gpu_comm with rc = %d = '%s'.\n", me, rc, err_str ); }
      else
      {  fprintf( stderr, "%s error: MPI_Comm_rank failed for gpu_comm with rc = %d (no MPI_Error_string).\n", me, rc ); }
      MPI_Abort( MPI_COMM_WORLD, -1 );
   }

   rc = MPI_Comm_size( gpu_comm , &num_gpu_mpi_ranks );
   if ( rc != MPI_SUCCESS )
   {  rc2 = MPI_Error_string( rc , err_str, &err_str_len );
      if ( rc2 == MPI_SUCCESS )
      {  fprintf( stderr, "%s error: MPI_Comm_size failed for gpu_comm with rc = %d = '%s'.\n", me, rc, err_str ); }
      else
      {  fprintf( stderr, "%s error: MPI_Comm_size failed for gpu_comm with rc = %d (no MPI_Error_string).\n", me, rc ); }
      MPI_Abort( MPI_COMM_WORLD, -1 );
   }

   printf( "%s info: my_commu_id = %d , myid = %d , my_gpu_mpi_rank = %d , num_gpu_mpi_ranks = %d\n"
         ,  me,      my_commu_id      , myid      , my_gpu_mpi_rank      , num_gpu_mpi_ranks );

   gpu_win_data = (int*) malloc( sizeof(int) * num_gpu_mpi_ranks );
   if ( ! gpu_win_data )
   {  fprintf( stderr, "%s error: MPI-rank %d failed to malloc %ld bytes for gpu_win_data.\n"
             , me, myid, sizeof(int) * num_gpu_mpi_ranks );
      MPI_Abort( MPI_COMM_WORLD, -1 );
   }
   for( i = 0; i < num_gpu_mpi_ranks; i++ ) { gpu_win_data[i] = 0; }
   
   // MPI_Win_create(void *base, MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, MPI_Win *win)
   rc = MPI_Win_create( (void*) gpu_win_data, (MPI_Aint) (sizeof(int) * num_gpu_mpi_ranks)
                      , sizeof(int), MPI_INFO_NULL, gpu_comm, &gpu_win );
   if ( rc != MPI_SUCCESS )
   {  rc2 = MPI_Error_string( rc , err_str, &err_str_len );
      if ( rc2 == MPI_SUCCESS )
      {  fprintf( stderr, "%s error: MPI_Win_create failed for gpu_comm with rc = %d = '%s'.\n", me, rc, err_str ); }
      else
      {  fprintf( stderr, "%s error: MPI_Win_create failed for gpu_comm with rc = %d (no MPI_Error_string).\n", me, rc ); }
      MPI_Abort( MPI_COMM_WORLD, -1 );
   }

   if ( err_str ) { free( err_str ); err_str = NULL; }
// HPE: end section 2 of 2 of changes to allow for serialization of MPI calls per GPU

   nstream = 1;
   ngpublas_handle = nstream;
   ngpusparse_handle = nstream;
   nmagma_queue = nstream;

   gpublas_handle_array = (gpublasHandle_t *)malloc( ngpublas_handle*sizeof(gpublasHandle_t *) );
   for( i=0; i<ngpublas_handle; i++ ) {
      GPUBLAS_CALL( gpublasCreate( &gpublas_handle_array[i] ) );
   }

   gpusparse_handle_array = (gpusparseHandle_t *)malloc( ngpusparse_handle*sizeof(gpusparseHandle_t *) );
   for( i=0; i<ngpusparse_handle; i++ ) {
      GPUSPARSE_CALL( gpusparseCreate( &gpusparse_handle_array[i] ) );
   }

#if defined( USE_OACC )
   stream = (gpuStream_t)acc_get_cuda_stream( acc_async_noval );
   acc_set_cuda_stream( acc_async_sync, stream );
#else
   GPU_CALL( gpuStreamCreate( &stream ) );
#endif

   streamArray = (gpuStream_t *)malloc( nstream*sizeof(gpuStream_t *) );
   for( i=0; i<nstream; i++ ) {
      streamArray[i] = stream;
      //GPU_CALL( gpuStreamCreate( &streamArray[i] ) );
      //GPU_CALL( gpuStreamCreateWithFlags( &streamArray[i], gpuStreamDefault ) );
      //GPU_CALL( gpuStreamCreateWithFlags( &streamArray[i], gpuStreamNonBlocking ) );
      //GPUBLAS_CALL( gpublasGetStream( gpublas_handle, &streamArray[i] ) );
   }

   nevent = 2;
   eventArray = (gpuEvent_t *)malloc( nevent*sizeof(gpuEvent_t *) );
   for( i=0; i<nevent; i++ ) {
      GPU_CALL( gpuEventCreateWithFlags( &eventArray[i], gpuEventDisableTiming ) );
   }

   for( i=0; i<ngpublas_handle; i++ ) {
      GPUBLAS_CALL( gpublasSetStream( gpublas_handle_array[i], streamArray[i] ) );
   }

   for( i=0; i<ngpusparse_handle; i++ ) {
      GPUSPARSE_CALL( gpusparseSetStream( gpusparse_handle_array[i], streamArray[i] ) );
   }

   MAGMA_CALL( magma_init() );

   magma_queue_array = (magma_queue_t *)malloc( nmagma_queue*sizeof(magma_queue_t *) );
   for( i=0; i<nmagma_queue; i++ ) {
      magma_queue_create_from_gpu( magma_device, streamArray[i], 
                                   gpublas_handle_array[i], gpusparse_handle_array[i], 
                                   &magma_queue_array[i] );
   }

#pragma omp parallel default( shared )
   {
#if defined( USE_OMP )
      mystream = omp_get_thread_num() % nstream;
      mygpublas_handle = omp_get_thread_num() % ngpublas_handle;
      mygpusparse_handle = omp_get_thread_num() % ngpusparse_handle;
      mymagma_queue = omp_get_thread_num() % nmagma_queue;
#else
      mystream = 0;
      mygpublas_handle = 0;
      mygpusparse_handle = 0;
      mymagma_queue = 0;
#endif
      stream = streamArray[mystream];
      gpublas_handle = gpublas_handle_array[mygpublas_handle];
      gpusparse_handle = gpusparse_handle_array[mygpusparse_handle];
      magma_queue = magma_queue_array[mymagma_queue];
   }

   // copy pointers to fortran copies
   *mydevice_f = mydevice;
   *deviceCount_f = deviceCount;
   *ngpublas_handle_f = ngpublas_handle;
   *ngpusparse_handle_f = ngpusparse_handle;
   *nmagma_queue_f = nmagma_queue;
   *nstream_f = nstream;
   *nevent_f = nevent;
   *my_gpu_mpi_rank_f = my_gpu_mpi_rank;
   *gpu_win_f = gpu_win;
   *gpublas_handle_array_f = gpublas_handle_array;
   *gpusparse_handle_array_f = gpusparse_handle_array;
   *magma_queue_array_f = magma_queue_array;
   *streamArray_f = streamArray;
   *eventArray_f = eventArray;
}

void finalize_gpu_c()
{
   int i;

   for( i=0; i<ngpublas_handle; i++ ) {
      GPUBLAS_CALL( gpublasDestroy( gpublas_handle_array[i] ) );
   }
   free( gpublas_handle_array );

   for( i=0; i<ngpusparse_handle; i++ ) {
      GPUSPARSE_CALL( gpusparseDestroy( gpusparse_handle_array[i] ) );
   }
   free( gpusparse_handle_array );

   for( i=0; i<nstream; i++ ) {
      GPU_CALL( gpuStreamDestroy( streamArray[i] ) );
   }
   free( streamArray );

   for( i=0; i<nevent; i++ ) {
      GPU_CALL( gpuEventDestroy( eventArray[i] ) );
   }
   free( eventArray );

   for( i=0; i<nmagma_queue; i++ ) {
      magma_queue_destroy( magma_queue_array[i] );
   }
   free( magma_queue_array );

   MAGMA_CALL( magma_finalize() );

// HPE: begin section of changes to allow for serialization of MPI calls per GPU
   MPI_Win_free( &gpu_win );
   if ( gpu_win_data )
   {  free( gpu_win_data );
      gpu_win_data = NULL;
   }
// HPE: end section of changes to allow for serialization of MPI calls per GPU
}

gpuError_t gpuMalloc
  ( void **devPtr, size_t size )
{
#if defined( USE_CUDA )
  return cudaMalloc( devPtr, size );
#elif defined( USE_HIP )
  return hipMalloc( devPtr, size );
#else
  return 0;
#endif
}

gpuError_t gpuHostAlloc
  ( void **ptr, size_t size, unsigned int flags )
{
#if defined( USE_CUDA )
  return cudaHostAlloc( ptr, size, flags );
#elif defined( USE_HIP )
  return hipHostMalloc( ptr, size, flags );
#else
  return 0;
#endif
}

gpuError_t gpuFree
  ( void *devPtr )
{
#if defined( USE_CUDA )
  return cudaFree( devPtr );
#elif defined( USE_HIP )
  return hipFree( devPtr );
#else
  return 0;
#endif
}

gpuError_t gpuFreeHost
  ( void *ptr )
{
#if defined( USE_CUDA )
  return cudaFreeHost( ptr );
#elif defined( USE_HIP )
  return hipHostFree( ptr );
#else
  return 0;
#endif
}

gpuError_t gpuStreamCreate
  ( gpuStream_t *stream )
{
#if defined( USE_CUDA )
  return cudaStreamCreate( stream );
#elif defined( USE_HIP )
  return hipStreamCreate( stream );
#else
  return 0;
#endif
}

gpuError_t gpuStreamDestroy
  ( gpuStream_t stream )
{
#if defined( USE_CUDA )
  return cudaStreamDestroy( stream );
#elif defined( USE_HIP )
  return hipStreamDestroy( stream );
#else
  return 0;
#endif
}

gpuError_t gpuStreamSynchronize
  ( gpuStream_t stream )
{
#if defined( USE_CUDA )
  return cudaStreamSynchronize( stream );
#elif defined( USE_HIP )
  return hipStreamSynchronize( stream );
#else
  return 0;
#endif
}

gpuError_t gpuGetDeviceCount
  ( int *count )
{
#if defined( USE_CUDA )
  return cudaGetDeviceCount( count );
#elif defined( USE_HIP )
  return hipGetDeviceCount( count );
#else
  return 0;
#endif
}

gpuError_t gpuSetDevice
  ( int device )
{
#if defined( USE_CUDA )
  return cudaSetDevice( device );
#elif defined( USE_HIP )
  return hipSetDevice( device );
#else
  return 0;
#endif
}

gpuError_t gpuEventCreateWithFlags
  ( gpuEvent_t *event, unsigned int flags )
{
#if defined( USE_CUDA )
  return cudaEventCreateWithFlags( event, flags );
#elif defined( USE_HIP )
  return hipEventCreateWithFlags( event, flags );
#else
  return 0;
#endif
}

gpuError_t gpuEventDestroy
  ( gpuEvent_t event )
{
#if defined( USE_CUDA )
  return cudaEventDestroy( event );
#elif defined( USE_HIP )
  return hipEventDestroy( event );
#else
  return 0;
#endif
}

gpuError_t gpuGetDeviceProperties
  ( gpuDeviceProp *prop, int deviceId )
{
#if defined( USE_CUDA )
  return cudaGetDeviceProperties( prop, deviceId );
#elif defined( USE_HIP )
  return hipGetDeviceProperties( prop, deviceId );
#else
  return 0;
#endif
}

gpublasStatus_t gpublasCreate
  ( gpublasHandle_t *handle )
{
#if defined( USE_CUDA )
  return cublasCreate( handle );
#elif defined( USE_HIP )
  return hipblasCreate( handle );
#else
  return 0;
#endif
}

gpublasStatus_t gpublasDestroy
  ( gpublasHandle_t handle )
{
#if defined( USE_CUDA )
  return cublasDestroy( handle );
#elif defined( USE_HIP )
  return hipblasDestroy( handle );
#else
  return 0;
#endif
}

gpublasStatus_t gpublasSetStream
  ( gpublasHandle_t handle, gpuStream_t streamId )
{
#if defined( USE_CUDA )
  return cublasSetStream_v2( handle, streamId );
#elif defined( USE_HIP )
  return hipblasSetStream( handle, streamId );
#else
  return 0;
#endif
}

gpublasStatus_t gpublasGetStream
  ( gpublasHandle_t handle, gpuStream_t *streamId )
{
#if defined( USE_CUDA )
  return cublasGetStream_v2( handle, streamId );
#elif defined( USE_HIP )
  return hipblasGetStream( handle, streamId );
#else
  return 0;
#endif
}

gpublasStatus_t gpublasSetMatrixAsync
  ( int rows, int cols, int elemSize,
    const void *A, int lda, void *B,
    int ldb, gpuStream_t stream )
{
#if defined( USE_CUDA )
  return cublasSetMatrixAsync( rows, cols, elemSize, A, lda, B, ldb, stream );
#elif defined( USE_HIP )
  gpublasStatus_t err = hipblasSetMatrixAsync( rows, cols, elemSize, A, lda, B, ldb, stream );
  GPU_CALL( gpuStreamSynchronize( streamArray[0] ) );
  return err;
#else
  return 0;
#endif
}

gpublasStatus_t gpublasGetMatrixAsync
  ( int rows, int cols, int elemSize,
    const void *A, int lda, void *B,
    int ldb, gpuStream_t stream )
{
#if defined( USE_CUDA )
  return cublasGetMatrixAsync( rows, cols, elemSize, A, lda, B, ldb, stream );
#elif defined( USE_HIP )
  gpublasStatus_t err = hipblasGetMatrixAsync( rows, cols, elemSize, A, lda, B, ldb, stream );
  GPU_CALL( gpuStreamSynchronize( streamArray[0] ) );
  return err;
#else
  return 0;
#endif
}

gpublasStatus_t gpublasSetVectorAsync
  ( int n, int elemSize, 
    const void *x, int incx, 
    void *y, int incy,
    gpuStream_t stream )
{
#if defined( USE_CUDA )
  return cublasSetVectorAsync( n, elemSize, x, incx, y, incy, stream );
#elif defined( USE_HIP )
  gpublasStatus_t err = hipblasSetVectorAsync( n, elemSize, x, incx, y, incy, stream );
  GPU_CALL( gpuStreamSynchronize( streamArray[0] ) );
  return err;
#else
  return 0;
#endif
}

gpublasStatus_t gpublasGetVectorAsync
  ( int n, int elemSize, 
    const void *x, int incx, 
    void *y, int incy,
    gpuStream_t stream )
{
#if defined( USE_CUDA )
  return cublasGetVectorAsync( n, elemSize, x, incx, y, incy, stream );
#elif defined( USE_HIP )
  gpublasStatus_t err = hipblasGetVectorAsync( n, elemSize, x, incx, y, incy, stream );
  GPU_CALL( gpuStreamSynchronize( streamArray[0] ) );
  return err;
#else
  return 0;
#endif
}

gpublasStatus_t gpublasDcopy
  ( gpublasHandle_t handle,
    int n, 
    const double *x, 
    int incx, 
    double *y, 
    int incy )
{
#if defined( USE_CUDA )
  return cublasDcopy_v2( handle, n, x, incx, y, incy );
#elif defined( USE_HIP )
  gpublasStatus_t err = hipblasDcopy( handle, n, x, incx, y, incy );
  GPU_CALL( gpuStreamSynchronize( streamArray[0] ) );
  return err;
#else
  return 0;
#endif
}

gpublasStatus_t gpublasDgetrfBatched
  ( gpublasHandle_t handle,
    int n, 
    double *const A[],
    int lda,
    int *P,
    int *info,
    int batchSize )
{
#if defined( USE_CUDA )
  return cublasDgetrfBatched( handle, n, A, lda, P, info, batchSize );
#elif defined( USE_HIP )
  gpublasStatus_t err = hipblasDgetrfBatched( handle, n, A, lda, P, info, batchSize );
  GPU_CALL( gpuStreamSynchronize( streamArray[0] ) );
  return err;
#else
  return 0;
#endif
}

gpublasStatus_t gpublasDgetrsBatched
  ( gpublasHandle_t handle, 
    const gpublasOperation_t trans, 
    const int n, 
    const int nrhs, 
    double *const A[], 
    const int lda, 
    const int *devIpiv, 
    double *const B[], 
    const int ldb, 
    int *info,
    const int batchSize )
{
#if defined( USE_CUDA )
  return cublasDgetrsBatched( handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, info, batchSize );
#elif defined( USE_HIP )
  gpublasStatus_t err = hipblasDgetrsBatched( handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, info, batchSize );
  GPU_CALL( gpuStreamSynchronize( streamArray[0] ) );
  return err;
#else
  return 0;
#endif
}

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
    int ldc )
{
#if defined( USE_CUDA )
  return cublasDgeam( handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc );
#elif defined( USE_HIP )
  gpublasStatus_t err = hipblasDgeam( handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc );
  GPU_CALL( gpuStreamSynchronize( streamArray[0] ) );
  return err;
#else
  return 0;
#endif
}

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
    int batchCount )
{
#if defined( USE_CUDA )
  return cublasDgemmStridedBatched( handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount );
#elif defined( USE_HIP )
  gpublasStatus_t err = hipblasDgemmStridedBatched( handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount );
  GPU_CALL( gpuStreamSynchronize( streamArray[0] ) );
  return err;
#else
  return 0;
#endif
}

gpublasStatus_t gpublasDaxpy
  ( gpublasHandle_t handle,
    int n, 
    const double *alpha,
    const double *x, 
    int incx, 
    double *y, 
    int incy )
{
#if defined( USE_CUDA )
  return cublasDaxpy_v2( handle, n, alpha, x, incx, y, incy );
#elif defined( USE_HIP )
  gpublasStatus_t err = hipblasDaxpy( handle, n, alpha, x, incx, y, incy );
  GPU_CALL( gpuStreamSynchronize( streamArray[0] ) );
  return err;
#else
  return 0;
#endif
}

gpublasStatus_t gpublasDscal
  ( gpublasHandle_t handle, 
    int n, 
    const double *alpha,
    double *x, 
    int incx )
{
#if defined( USE_CUDA )
  return cublasDscal_v2( handle, n, alpha, x, incx );
#elif defined( USE_HIP )
  gpublasStatus_t err = hipblasDscal( handle, n, alpha, x, incx );
  GPU_CALL( gpuStreamSynchronize( streamArray[0] ) );
  return err;
#else
  return 0;
#endif
}

gpublasStatus_t gpublasDdot
  ( gpublasHandle_t handle,
    int n, 
    const double *x, 
    int incx, 
    const double *y,
    int incy,
    double *result )
{
#if defined( USE_CUDA )
  return cublasDdot_v2( handle, n, x, incx, y, incy, result );
#elif defined( USE_HIP )
  gpublasStatus_t err = hipblasDdot( handle, n, x, incx, y, incy, result );
  GPU_CALL( gpuStreamSynchronize( streamArray[0] ) );
  return err;
#else
  return 0;
#endif
}

gpublasStatus_t gpublasDnrm2
  ( gpublasHandle_t handle, 
    int n, 
    const double *x, 
    int incx, 
    double *result )
{
#if defined( USE_CUDA )
  return cublasDnrm2_v2( handle, n, x, incx, result );
#elif defined( USE_HIP )
  gpublasStatus_t err = hipblasDnrm2( handle, n, x, incx, result );
  GPU_CALL( gpuStreamSynchronize( streamArray[0] ) );
  return err;
#else
  return 0;
#endif
}

gpusparseStatus_t gpusparseCreate
  ( gpusparseHandle_t *handle )
{
#if defined( USE_CUDA )
  return cusparseCreate( handle );
#elif defined( USE_HIP )
  return hipsparseCreate( handle );
#else
  return 0;
#endif
}

gpusparseStatus_t gpusparseDestroy
  ( gpusparseHandle_t handle )
{
#if defined( USE_CUDA )
  return cusparseDestroy( handle );
#elif defined( USE_HIP )
  return hipsparseDestroy( handle );
#else
  return 0;
#endif
}

gpusparseStatus_t gpusparseSetStream
  ( gpusparseHandle_t handle, gpuStream_t streamId )
{
#if defined( USE_CUDA )
  return cusparseSetStream( handle, streamId );
#elif defined( USE_HIP )
  return hipsparseSetStream( handle, streamId );
#else
  return 0;
#endif
}

gpusparseStatus_t gpusparseDgtsv2StridedBatch_bufferSizeExt 
  ( gpusparseHandle_t handle,
    int               m,
    const double*     dl,
    const double*     d,
    const double*     du,
    const double*     x,
    int               batchCount,
    int               batchStride,
    size_t*           bufferSizeInBytes )
{
#if defined( USE_CUDA )
  return cusparseDgtsv2StridedBatch_bufferSizeExt
    ( handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes );
#elif defined( USE_HIP )
  gpusparseStatus_t err = hipsparseDgtsv2StridedBatch_bufferSizeExt
    ( handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes );
  GPU_CALL( gpuStreamSynchronize( streamArray[0] ) );
  return err;
#else
  return 0;
#endif
}

gpusparseStatus_t gpusparseDgtsv2StridedBatch
  ( gpusparseHandle_t handle,
    int               m,
    const double*     dl,
    const double*     d,
    const double*     du,
    double*           x,
    int               batchCount,
    int               batchStride,
    void*             pBuffer )
{
#if defined( USE_CUDA )
  return cusparseDgtsv2StridedBatch
    ( handle, m, dl, d, du, x, batchCount, batchStride, pBuffer );
#elif defined( USE_HIP )
  gpusparseStatus_t err = hipsparseDgtsv2StridedBatch
    ( handle, m, dl, d, du, x, batchCount, batchStride, pBuffer );
  GPU_CALL( gpuStreamSynchronize( streamArray[0] ) );
  return err;
#else
  return 0;
#endif
}
