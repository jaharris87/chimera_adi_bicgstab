#include "chimera_gpu.h"
#include "adi_gpu.h"

//prototypes for partition
// void setupPtrVecs(int nblocks, int msize, double **device_ALU_d, double **device_x1_d, double *device_ALU, double *device_x1);
// void residual(int nblocks, int msize, double *device_dx, double *device_rtilde, double *device_pvec, double *device_rvec, double *device_xvec, double *rhs);
// double adi_vnorm( int msize, int nblocks, double *d_xvec );
double adi_dotprod_remote( int msize, int nblocks, double *d_xvec, double *d_yvec );

void pre_adi_init( int nblocks, int msize )
{
    int i, iblock;

    GPU_CALL( gpuMalloc( (void **)&device_ALU_d,      nblocks*sizeof(*device_ALU_d) ) );
    GPU_CALL( gpuMalloc( (void **)&device_x1_d,       nblocks*sizeof(*device_x1_d) ) );

    GPU_CALL( gpuMalloc( (void **)&device_A,        msize*msize*nblocks*sizeof(double) ) );
    GPU_CALL( gpuMalloc( (void **)&device_ALU,      msize*msize*nblocks*sizeof(double) ) );
    GPU_CALL( gpuMalloc( (void **)&device_ALU_ipiv, msize*nblocks*sizeof(int) ) );
    GPU_CALL( gpuMalloc( (void **)&device_x1,       msize*nblocks*sizeof(double) ) );
    GPU_CALL( gpuMalloc( (void **)&device_dx,       msize*nblocks*sizeof(double) ) );
    GPU_CALL( gpuMalloc( (void **)&device_dx_B,     msize*nblocks*sizeof(double) ) );
    GPU_CALL( gpuMalloc( (void **)&device_dx_C,     msize*nblocks*sizeof(double) ) );
    GPU_CALL( gpuMalloc( (void **)&device_linfo,    nblocks*sizeof(int) ) );
    host_linfo = (int *)malloc( nblocks*sizeof(int) );

//    magma_dset_pointer( device_ALU_d, device_ALU, msize, 0, 0, msize*msize, nblocks, magma_queue );
//    magma_dset_pointer( device_x1_d,  device_x1,  msize, 0, 0, msize,       nblocks, magma_queue );

    //get rid of magma
    #pragma omp target teams distribute parallel for \
                        is_device_ptr( device_ALU_d, device_x1_d,device_ALU,device_x1)
    for (int i=0; i<nblocks; i++){ device_ALU_d[i]=&device_ALU[i*msize*msize];
                             device_x1_d[i]=&device_x1[i*msize]; }

    //partition
    // setupPtrVecs(nblocks,msize, device_ALU_d, device_x1_d, device_ALU, device_x1);


    GPU_CALL( gpuMalloc( (void **)&device_Bband,    msize*nblocks*sizeof(double) ) );
    GPU_CALL( gpuMalloc( (void **)&device_Cband,    msize*nblocks*sizeof(double) ) );

    GPU_CALL( gpuMalloc( (void **)&device_dl,       msize*nblocks*sizeof(double) ) );
    GPU_CALL( gpuMalloc( (void **)&device_d,        msize*nblocks*sizeof(double) ) );
    GPU_CALL( gpuMalloc( (void **)&device_du,       msize*nblocks*sizeof(double) ) );
    GPU_CALL( gpuMalloc( (void **)&device_grhs,     msize*nblocks*sizeof(double) ) );

    GPU_CALL( gpuMalloc( (void **)&device_xvec,     msize*nblocks*sizeof(double) ) );
    GPU_CALL( gpuMalloc( (void **)&device_svec,     msize*nblocks*sizeof(double) ) );
    GPU_CALL( gpuMalloc( (void **)&device_shat,     msize*nblocks*sizeof(double) ) );
    GPU_CALL( gpuMalloc( (void **)&device_pvec,     msize*nblocks*sizeof(double) ) );
    GPU_CALL( gpuMalloc( (void **)&device_phat,     msize*nblocks*sizeof(double) ) );
    GPU_CALL( gpuMalloc( (void **)&device_rvec,     msize*nblocks*sizeof(double) ) );
    GPU_CALL( gpuMalloc( (void **)&device_rtilde,   msize*nblocks*sizeof(double) ) );
    GPU_CALL( gpuMalloc( (void **)&device_tvec,     msize*nblocks*sizeof(double) ) );
    GPU_CALL( gpuMalloc( (void **)&device_vvec,     msize*nblocks*sizeof(double) ) );

#if !defined ( USE_OACC ) && !defined( USE_OMP_OL )
    GPU_CALL( gpuHostAlloc((void **)&xvec,     msize*nblocks*sizeof(double),       gpuHostAllocDefault ) );
    GPU_CALL( gpuHostAlloc((void **)&rvec,     msize*nblocks*sizeof(double),       gpuHostAllocDefault ) );

    GPU_CALL( gpuHostAlloc((void **)&ALU,      msize*msize*nblocks*sizeof(double), gpuHostAllocDefault ) );
    GPU_CALL( gpuHostAlloc((void **)&Bband,    msize*nblocks*sizeof(double),       gpuHostAllocDefault ) );
    GPU_CALL( gpuHostAlloc((void **)&Cband,    msize*nblocks*sizeof(double),       gpuHostAllocDefault ) );
    GPU_CALL( gpuHostAlloc((void **)&dl,       msize*nblocks*sizeof(double),       gpuHostAllocDefault ) );
    GPU_CALL( gpuHostAlloc((void **)&d,        msize*nblocks*sizeof(double),       gpuHostAllocDefault ) );
    GPU_CALL( gpuHostAlloc((void **)&du,       msize*nblocks*sizeof(double),       gpuHostAllocDefault ) );
#endif

    /*-------------------------------------------------*
     * Pre-allocate work buffers for tridiagonal solve *
     *------------------------------------------------*/

    GPUSPARSE_CALL( gpusparseDgtsv2StridedBatch_bufferSizeExt( gpusparse_handle, nblocks, device_dl, device_d, device_du, 
                                                               device_grhs, msize, nblocks, &bufferSize ) );
    GPU_CALL( gpuMalloc( (void **)&device_buffer, bufferSize ) );
}

void adi_cleanup()
{
    GPU_CALL( gpuFree( device_ALU_d ) );
    GPU_CALL( gpuFree( device_x1_d ) );

    GPU_CALL( gpuFree( device_A ) );
    GPU_CALL( gpuFree( device_ALU ) );
    GPU_CALL( gpuFree( device_ALU_ipiv ) );
    GPU_CALL( gpuFree( device_x1 ) );
    GPU_CALL( gpuFree( device_dx ) );
    GPU_CALL( gpuFree( device_dx_B ) );
    GPU_CALL( gpuFree( device_dx_C ) );
    GPU_CALL( gpuFree( device_linfo ) );
    free( host_linfo );

    GPU_CALL( gpuFree( device_Bband ) );
    GPU_CALL( gpuFree( device_Cband ) );

    GPU_CALL( gpuFree( device_dl ) );
    GPU_CALL( gpuFree( device_d ) );
    GPU_CALL( gpuFree( device_du ) );
    GPU_CALL( gpuFree( device_grhs ) );

    GPU_CALL( gpuFree( device_xvec ) );
    GPU_CALL( gpuFree( device_svec ) );
    GPU_CALL( gpuFree( device_shat ) );
    GPU_CALL( gpuFree( device_pvec ) );
    GPU_CALL( gpuFree( device_phat ) );
    GPU_CALL( gpuFree( device_rvec ) );
    GPU_CALL( gpuFree( device_rtilde ) );
    GPU_CALL( gpuFree( device_tvec ) );
    GPU_CALL( gpuFree( device_vvec ) );

#if !defined ( USE_OACC ) && !defined( USE_OMP_OL )
    GPU_CALL( gpuFreeHost( ALU ) );
    GPU_CALL( gpuFreeHost( Bband ) );
    GPU_CALL( gpuFreeHost( Cband ) );
    GPU_CALL( gpuFreeHost( dl ) );
    GPU_CALL( gpuFreeHost( d ) );
    GPU_CALL( gpuFreeHost( du ) );

    GPU_CALL( gpuFreeHost( xvec ) );
    GPU_CALL( gpuFreeHost( rvec ) );
#endif

    GPU_CALL( gpuFree( device_buffer ) );
}

void adi_init( int msize, int nblocks, double *A, double *B, double *C )
{
    int matsize = msize*msize;
    int iblock, m, n, linfo;

#if defined ( USE_OACC ) || defined( USE_OMP_OL )

#if defined( USE_OACC )
    #pragma acc parallel loop gang vector collapse(3) \
                         present( A[0:msize*msize*nblocks] ) \
                         deviceptr( device_ALU, device_A )
#elif defined( USE_OMP_OL )
    #pragma omp target teams distribute parallel for simd collapse(3) \
                         is_device_ptr( device_ALU, device_A )
#endif
    for( iblock=0; iblock<nblocks; iblock++ ) {
        for( n=0; n<msize; n++ ) {
            for( m=0; m<msize; m++ ) {
                device_ALU[m+n*msize+iblock*msize*msize] = A[m+n*msize+iblock*msize*msize];
                device_A[m+n*msize+iblock*msize*msize]   = A[m+n*msize+iblock*msize*msize];
            }
        }
    }
#if defined( USE_OACC )
    #pragma acc parallel loop gang vector collapse(2) \
                         present( B[0:msize*nblocks], C[0:msize*nblocks]) \
                         deviceptr( device_Bband, device_Cband )
#elif defined( USE_OMP_OL )
    #pragma omp target teams distribute parallel for simd collapse(2) \
                         is_device_ptr( device_Bband, device_Cband )
#endif
    for( iblock=0; iblock<nblocks; iblock++ ) {
        for( m=0; m<msize; m++ ) {
            device_Bband[m+iblock*msize] = B[m+iblock*msize];
            device_Cband[m+iblock*msize] = C[m+iblock*msize];
        }
    }
#if defined( USE_OACC )
    #pragma acc parallel loop gang \
                         present( A[0:msize*msize*nblocks], B[0:msize*nblocks], C[0:msize*nblocks] ) \
                         deviceptr( device_d, device_dl, device_du )
#elif defined( USE_OMP_OL )
    #pragma omp target teams distribute \
                         is_device_ptr( device_d, device_dl, device_du )
#endif
    for( m=0; m<msize; m++ ) {
#if defined( USE_OACC )
        #pragma acc loop vector
#elif defined( USE_OMP_OL )
        #pragma omp parallel for simd
#endif
        for( iblock=0; iblock<nblocks; iblock++ ) {
            device_d[iblock+m*nblocks]  = A[m+m*msize+iblock*msize*msize];
            device_dl[iblock+m*nblocks] = B[m+iblock*msize];
            device_du[iblock+m*nblocks] = C[m+iblock*msize];
        }
        device_dl[m*nblocks]       = d_zero;
        device_du[(m+1)*nblocks-1] = d_zero;
    }
#else
    #pragma omp parallel default( shared ) \
                         private( iblock, m )
    {
        #pragma omp for schedule( static, 1 )
        for( iblock=0; iblock<nblocks; iblock++ ) {
            blasf77_dcopy( &matsize, &A[iblock*matsize], &i_one, &ALU[iblock*matsize], &i_one );
            blasf77_dcopy( &msize,   &B[iblock*msize],   &i_one, &Bband[iblock*msize], &i_one );
            blasf77_dcopy( &msize,   &C[iblock*msize],   &i_one, &Cband[iblock*msize], &i_one );
        }
    }
    GPUBLAS_CALL( gpublasSetMatrixAsync( msize, msize*nblocks, sizeof(double),
                                         ALU, msize,
                                         device_ALU, msize,
                                         streamArray[0] ) );
    GPUBLAS_CALL( gpublasSetVectorAsync( msize*nblocks, sizeof(double), Bband, 1, 
                                         device_Bband, 1, streamArray[0] ) );
    GPUBLAS_CALL( gpublasSetVectorAsync( msize*nblocks, sizeof(double), Cband, 1,
                                         device_Cband, 1, streamArray[0] ) );

    GPUBLAS_CALL( gpublasDcopy( gpublas_handle, msize*msize*nblocks, device_ALU, 1, device_A, 1 ) );

    #pragma omp parallel default( shared ) \
                         private( iblock, m, linfo )
    {
        #pragma omp for schedule( static, 1 )
        for( m=0; m<msize; m++ ) {
            for( iblock=0; iblock<nblocks; iblock++ ) {
                d[iblock+m*nblocks]  = A[m+m*msize+iblock*msize*msize];
                dl[iblock+m*nblocks] = B[m+iblock*msize];
                du[iblock+m*nblocks] = C[m+iblock*msize];
            }
            dl[m*nblocks]       = d_zero;
            du[(m+1)*nblocks-1] = d_zero;
        }
    }

    /*----------------------------------------*
     * send the tridiagonal system to the GPU *
     *----------------------------------------*/

    GPUBLAS_CALL( gpublasSetVectorAsync( msize*nblocks, sizeof(double), dl, 1,
                                         device_dl, 1, streamArray[0] ) );
    GPUBLAS_CALL( gpublasSetVectorAsync( msize*nblocks, sizeof(double), d, 1,
                                         device_d, 1, streamArray[0] ) );
    GPUBLAS_CALL( gpublasSetVectorAsync( msize*nblocks, sizeof(double), du, 1,
                                         device_du, 1, streamArray[0] ) );
#endif

    /*------------------------------------------*
     * perform LU factorization of dense blocks *
     *------------------------------------------*/

    GPUBLAS_CALL( gpublasDgetrfBatched( gpublas_handle, msize,
                                        device_ALU_d, msize,
                                        device_ALU_ipiv,
                                        device_linfo,
                                        nblocks ) );
} // adi_init

void adi_dblock( int msize, int nblocks )
{
    int iblock, linfo;

#if defined( USE_CUDA )
    GPUBLAS_CALL( cublasDgetrsBatched( gpublas_handle, GPUBLAS_OP_N, 
                                       msize, 1, device_ALU_d, msize, device_ALU_ipiv, 
                                       device_x1_d, msize, host_linfo, nblocks ) );
#elif defined( USE_HIP )
    GPUBLAS_CALL( hipblasDgetrsBatched( gpublas_handle, GPUBLAS_OP_N, 
                                        msize, 1, device_ALU_d, msize, device_ALU_ipiv, 
                                        device_x1_d, msize, host_linfo, nblocks ) );
#endif

} // adi_dblock

void adi_tridiag( int msize, int nblocks )
{
    int iblock, m, linfo;
    
    // Transpose dx = x-dx
    GPUBLAS_CALL( gpublasDgeam( gpublas_handle, GPUBLAS_OP_T, GPUBLAS_OP_T,
                                nblocks, msize,
                                &d_one,
                                device_dx, msize,
                                &d_zero,
                                device_dx, msize,
                                device_grhs, nblocks ) );

    GPUSPARSE_CALL( gpusparseDgtsv2StridedBatch( gpusparse_handle, nblocks,
                                                 device_dl, device_d, device_du,
                                                 device_grhs, msize,
                                                 nblocks, device_buffer ) );

    // Transpose solution
    GPUBLAS_CALL( gpublasDgeam( gpublas_handle, GPUBLAS_OP_T, GPUBLAS_OP_T,
                                msize, nblocks,
                                &d_one,
                                device_grhs, nblocks,
                                &d_zero,
                                device_grhs, nblocks,
                                device_dx, msize ) );
} // adi_tridiag

void adi_matvec( int msize, int nblocks, double *A, double *B, double *C, double *d_xhat, double *d_y )
{
    int n = msize*(nblocks-1);
    int iblock, m, linfo;

    // A*x1
    GPUBLAS_CALL( gpublasDgemmStridedBatched( gpublas_handle, GPUBLAS_OP_N, GPUBLAS_OP_N,
                                              msize, 1, msize,
                                              &d_one,
                                              device_A, msize, msize*msize,
                                              d_xhat, msize, msize,
                                              &d_zero,
                                              d_y, msize, msize,
                                              nblocks ) );

    // Do the B*x1 + C*x1 part
    GPUBLAS_CALL( gpublasDcopy( gpublas_handle, msize*nblocks, d_xhat, 1, device_dx_B, 1 ) );
    GPUBLAS_CALL( gpublasDcopy( gpublas_handle, msize*nblocks, d_xhat, 1, device_dx_C, 1 ) );
    //magmablas_dlascl2( MagmaFull, n, 1, &device_Bband[msize], device_dx_B, n, magma_queue, &linfo );
    // for (int i=0; i<n; i++)device_dx_B[i] *= device_Bband[msize + i];
    // #pragma omp target teams distribute parallel for \
                        //  is_device_ptr( device_dx_B, device_Bband)
    //magmablas_dlascl2( MagmaFull, n, 1, device_Cband, &device_dx_C[msize], n, magma_queue, &linfo );
    // for (int i=0; i<n; i++)device_dx_C[msize + i] *= device_Cband[i];
#pragma omp target teams distribute parallel for \
                         is_device_ptr( device_dx_B, device_Bband, device_dx_C, device_Cband)
    for (int i=0; i<n; i++){ device_dx_B[i] = device_dx_B[i] * device_Bband[msize + i];
                             device_dx_C[msize + i] = device_dx_C[msize + i] * device_Cband[i]; }


    GPUBLAS_CALL( gpublasDaxpy( gpublas_handle, n, &d_one, device_dx_B, 1, &d_y[msize], 1 ) );
    GPUBLAS_CALL( gpublasDaxpy( gpublas_handle, n, &d_one, &device_dx_C[msize], 1, d_y, 1 ) );
} // adi_matvec

void adi_precond( int msize, int nblocks, double *A, double *B, double *C, double *device_xhat )
{
    int n = msize*nblocks;
    int iblock;

    // A*x1 = x
    GPUBLAS_CALL( gpublasDcopy( gpublas_handle, msize*nblocks, device_xhat, 1, device_x1, 1 ) );
    adi_dblock( msize, nblocks );
    
    // dx = (A+B+C)*x1
    // for(int i = 0; i<msize*nblocks; i++){if( isnan(device_x1[i]) ){printf("device_x1\n");abort();}}
    adi_matvec( msize, nblocks, A, B, C, device_x1, device_dx );


    // Calculate x-dx on the GPU
    GPUBLAS_CALL( gpublasDscal( gpublas_handle, msize*nblocks, &d_mone, device_dx, 1 ) );
    GPUBLAS_CALL( gpublasDaxpy( gpublas_handle, msize*nblocks, &d_one,  device_xhat, 1, device_dx, 1 ) );

    // (diag(A)+B+C)*x2 = x-dx
    adi_tridiag( msize, nblocks );

    // xhat = x1 + x2
    GPUBLAS_CALL( gpublasDaxpy( gpublas_handle, msize*nblocks, &d_one, device_dx, 1, device_x1, 1 ) );
    GPUBLAS_CALL( gpublasDcopy( gpublas_handle, msize*nblocks, device_x1, 1, device_xhat, 1 ) );
} // adi_precond















double adi_dotprod( int msize, int nblocks, double *d_xvec, double *d_yvec )
{
    double ddotprod = 0.0;
    int iblock, m;

#if defined ( USE_OACC ) || defined( USE_OMP_OL )

#if defined( USE_OACC )
    #pragma acc parallel loop gang vector collapse(2) \
                         reduction( + : ddotprod ) \
                         copy( ddotprod ) \
                         deviceptr( d_xvec, d_yvec )
#elif defined( USE_OMP_OL )
    #pragma omp target teams distribute parallel for simd collapse(2) \
                         reduction( + : ddotprod ) \
                         map( tofrom: ddotprod ) \
                         is_device_ptr( d_xvec, d_yvec )
#endif
    for( iblock=0; iblock<nblocks; iblock++ ) {
        for( m=0; m<msize; m++ ) {
            ddotprod += d_xvec[m+iblock*msize] * d_yvec[m+iblock*msize];
        }
    }
#else
    GPUBLAS_CALL( gpublasDdot( gpublas_handle, msize*nblocks, d_xvec, 1, d_yvec, 1, &ddotprod ) );
#endif

    return ddotprod;
} // adi_dotprod
// double adi_dotprod( int msize, int nblocks, double *d_xvec, double *d_yvec )
// {
//     double ddotprod = 0.0;
//     int iblock, m;

//     #pragma omp target teams distribute parallel for simd collapse(2) \
//                          reduction( + : ddotprod ) \
//                          map( tofrom: ddotprod ) \
//                          is_device_ptr( d_xvec, d_yvec )
//     for( iblock=0; iblock<nblocks; iblock++ ) {
//         for( m=0; m<msize; m++ ) {
//             ddotprod += d_xvec[m+iblock*msize] * d_yvec[m+iblock*msize];
//         }
//     }

//     return ddotprod;
// } // adi_dotprod
















double adi_vnorm( int msize, int nblocks, double *d_xvec )
{
    double vnorm = 0.0;
    int iblock, m;

#if defined ( USE_OACC ) || defined( USE_OMP_OL )

#if defined( USE_OACC )
    #pragma acc parallel loop gang vector collapse(2) \
                         reduction( + : vnorm ) \
                         copy( vnorm ) \
                         deviceptr( d_xvec )
#elif defined( USE_OMP_OL )
    #pragma omp target teams distribute parallel for simd collapse(2) \
                         reduction( + : vnorm ) \
                         map( tofrom: vnorm ) \
                         is_device_ptr( d_xvec )
#endif
    for( iblock=0; iblock<nblocks; iblock++ ) {
        for( m=0; m<msize; m++ ) {
            vnorm += d_xvec[m+iblock*msize] * d_xvec[m+iblock*msize];
        }
    }
    vnorm = sqrt( vnorm );
#else
    GPUBLAS_CALL( gpublasDnrm2( gpublas_handle, msize*nblocks, d_xvec, 1, &vnorm ) );
#endif

    return vnorm;
} // adi_vnorm

//#define ADI_VERBOSE
void adi_bicgstab( int msize, int nblocks, double *A, double *B, double *C, double *rhs, int niter, double tol, int *iters )
{
    double rho, rhop, beta, alpha, omega, m_alpha, m_omega;
    double rtilde_times_vvec, tvec_times_svec, tvec_times_tvec;
    double rvec_norm0, rvec_norm, svec_norm;
    int iblock, k, m, n;

        // for(int i = 0; i<msize*nblocks; i++){if( isnan(device_vvec[i]) ){printf("adi_solver_GPU.c -a\n");break;}}

    adi_init( msize, nblocks, A, B, C );

    alpha = d_zero;
    beta  = d_zero;
    omega = d_zero;
    rho   = d_zero;
    rhop  = d_zero;
    *iters = 0;

    n = msize*nblocks;

#if defined ( USE_OACC ) || defined( USE_OMP_OL )

#if defined( USE_OACC )
    #pragma acc parallel loop gang vector collapse(2) \
                         present( rhs[0:msize*nblocks] ) \
                         deviceptr( device_dx, device_rtilde, \
                                    device_pvec, device_rvec, device_xvec )
#elif defined( USE_OMP_OL )

/*
    #pragma omp target teams distribute parallel for simd
    for(int i=0; i<1; i++)printf("test from gpu section\n");
*/

    #pragma omp target teams distribute parallel for simd collapse(2) \
                       is_device_ptr( device_dx, device_rtilde, \
                                       device_pvec, device_rvec, device_xvec )
// residual(nblocks, msize, device_dx, device_rtilde, device_pvec, device_rvec, device_xvec, rhs);
#endif
    for( iblock=0; iblock<nblocks; iblock++ ) {
        for( m=0; m<msize; m++ ) {

            // 0: initialize x = x_0 = 0
            device_xvec[m+iblock*msize] = 0.0;
            device_dx[m+iblock*msize] = 0.0;

            // 1: compute initial residual r_0 = b - A * x_0 (x_0 = 0)
            device_rvec[m+iblock*msize] = rhs[m+iblock*msize];
// if( isnan(device_rvec[m+iblock*msize]) )printf("adi_solver_GPU.c a");
            // 2: Set p = r and \tilde{r} = r
            device_pvec[m+iblock*msize]   = device_rvec[m+iblock*msize];
// if( isnan(device_pvec[m+iblock*msize]) )printf("adi_solver_GPU.c aa");
            device_rtilde[m+iblock*msize] = device_rvec[m+iblock*msize];
// if( isnan(device_rtilde[m+iblock*msize]) )printf("adi_solver_GPU.c aaa");

        }
    }
#else
    #pragma omp parallel default( shared ) \
                         private( iblock )
    {
        #pragma omp for schedule( static, 1 )
        for( iblock=0; iblock<nblocks; iblock++ ) {
            for( m=0; m<msize; m++ ) {

                // 0: initialize x = x_0 = 0
                xvec[m+iblock*msize] = 0.0;

                // 1: compute initial residual r_0 = b - A * x_0 (x_0 = 0)
                rvec[m+iblock*msize] = rhs[m+iblock*msize];

            }
        }
    }
    GPUBLAS_CALL( gpublasSetVectorAsync( msize*nblocks, sizeof(double), xvec, 1, device_xvec, 1, streamArray[0] ) );
    GPUBLAS_CALL( gpublasSetVectorAsync( msize*nblocks, sizeof(double), xvec, 1, device_dx,   1, streamArray[0] ) );
    GPUBLAS_CALL( gpublasSetVectorAsync( msize*nblocks, sizeof(double), rvec, 1, device_rvec, 1, streamArray[0] ) );

    // 2: Set p = r and \tilde{r} = r
    GPUBLAS_CALL( gpublasDcopy( gpublas_handle, msize*nblocks, device_rvec, 1, device_pvec, 1 ) );
    GPUBLAS_CALL( gpublasDcopy( gpublas_handle, msize*nblocks, device_rvec, 1, device_rtilde, 1 ) );
#endif

    rvec_norm0 = adi_vnorm( msize, nblocks, device_rvec );


#ifdef ADI_VERBOSE
    printf( "rvec_norm0 = %12.5e\n", rvec_norm0 );
#endif

    // 3: repeat until convergence
    for( k=0; k<niter; k++ ) {
 
        // 4: \rho = \tilde{r}^{T} * r
        rhop = rho;
   
        // for(int i = 0; i<msize; i++){if( isnan(device_rtilde[i]) ){printf("adi_solver_GPU.b y\n");break;}}
        // for(int i = 0; i<msize; i++){if( isnan(device_rvec[i]) ){printf("adi_solver_GPU.c bb\n");break;}}
        rho = adi_dotprod_remote( msize, nblocks, device_rtilde, device_rvec );

        if( k > 0 ) { // not the first iteration

            // 12: \beta = ( \rho_{i} / \rho_{i-1} ) ( \alpha / \omega )
            beta = ( rho / rhop ) * ( alpha / omega );
            
            // 13: p = r + \beta * ( p - \omega * v )
	    // for(int i = 0; i<msize*nblocks; i++){if( isnan(device_vvec[i]) ){printf("device_vvec v\n");abort();}}
            GPUBLAS_CALL( gpublasDaxpy( gpublas_handle, msize*nblocks, &m_omega, device_vvec, 1, device_pvec, 1 ) );
            GPUBLAS_CALL( gpublasDscal( gpublas_handle, msize*nblocks, &beta, device_pvec, 1 ) );
            GPUBLAS_CALL( gpublasDaxpy( gpublas_handle, msize*nblocks, &d_one, device_rvec, 1, device_pvec, 1 ) );
        } // k > 0

        // 15: M * \hat{p} = p (apply preconditioners)
        GPUBLAS_CALL( gpublasDcopy( gpublas_handle, msize*nblocks, device_pvec, 1, device_phat, 1 ) );
	// for(int i = 0; i<msize*nblocks; i++){if( isnan(device_phat[i]) ){printf("device_phat w\n");abort();}}    
        adi_precond( msize, nblocks, A, B, C, device_phat );

        // 16: v = A * \hat{p}
	// for(int i = 0; i<msize*nblocks; i++){if( isnan(device_phat[i]) ){printf("device_phat w\n");abort();}}
        adi_matvec( msize, nblocks, A, B, C, device_phat, device_vvec );
        // for(int i = 0; i<msize*nblocks; i++){if( isnan(device_vvec[i]) ){printf("device_vvec ww\n");abort();}}
#ifdef ADI_VERBOSE
        printf( "Iteration %2da, rho = %12.5e, vvec_norm = %12.5e, phat_norm = %12.5e\n",
                k, rho, adi_vnorm( msize, nblocks, device_vvec ), adi_vnorm( msize, nblocks, device_phat ) );
#endif

        // 17: \alpha = \rho_{i} / ( \tilde{r}^{T} * v )
        rtilde_times_vvec = adi_dotprod_remote( msize, nblocks, device_rtilde, device_vvec );
        alpha = rho / rtilde_times_vvec;
        m_alpha = -alpha;

        // 18: s = r - \alpha * v
        GPUBLAS_CALL( gpublasDcopy( gpublas_handle, msize*nblocks, device_rvec, 1, device_svec, 1 ) );
        // for(int i = 0; i<msize*nblocks; i++){if( isnan(device_vvec[i]) ){printf("device_vvec x\n");break;}}
        GPUBLAS_CALL( gpublasDaxpy( gpublas_handle, msize*nblocks, &m_alpha, device_vvec, 1, device_svec, 1 ) );
        // 20: check for convergence (norm of svec is small)
        svec_norm = adi_vnorm( msize, nblocks, device_svec );
#ifdef ADI_VERBOSE
        printf( "Iteration %2db, svec_norm = %12.5e, rtilde_times_vvec = %12.5e\n",
                k, svec_norm, rtilde_times_vvec );
#endif
        if( svec_norm < d_tiny ) {
            // 19: x = x + \alpha * \hat{p}
            GPUBLAS_CALL( gpublasDaxpy( gpublas_handle, msize*nblocks, &alpha, device_phat, 1, device_xvec, 1 ) );
            break;
        }

        // 23: M * \hat{s} = r (apply preconditioners)
        GPUBLAS_CALL( gpublasDcopy( gpublas_handle, msize*nblocks, device_svec, 1, device_shat, 1 ) );
	// for(int i = 0; i<msize*nblocks; i++){if( isnan(device_shat[i]) ){printf("device_shat w\n");abort();}}    
        adi_precond( msize, nblocks, A, B, C, device_shat );

        // 24: t = A * \hat{s}
	// for(int i = 0; i<msize*nblocks; i++){if( isnan(device_shat[i]) ){printf("device_shat -y\n");abort();}}
        adi_matvec( msize, nblocks, A, B, C, device_shat, device_tvec );
        // for(int i = 0; i<msize*nblocks; i++){if( isnan(device_tvec[i]) ){printf("device_tvec -yy\n");abort();}}


	
        // 25: \omega = ( t^{T} * s ) / ( t^{T} * t )
        tvec_times_svec = adi_dotprod_remote( msize, nblocks, device_tvec, device_svec );
        tvec_times_tvec = adi_dotprod_remote( msize, nblocks, device_tvec, device_tvec );
        omega = tvec_times_svec / tvec_times_tvec;
        m_omega = -omega;

        // 26: x = x + \alpha * \hat{p} + \omega * \hat{s}
        // 27: r = s - \omega * t
        GPUBLAS_CALL( gpublasDaxpy( gpublas_handle, msize*nblocks, &alpha, device_phat, 1, device_xvec, 1 ) );
        GPUBLAS_CALL( gpublasDaxpy( gpublas_handle, msize*nblocks, &omega, device_shat, 1, device_xvec, 1 ) );
        // for(int i = 0; i<msize*nblocks; i++){if( isnan(device_svec[i]) ){printf("adi_solver_GPU.c y\n");break;}}
        GPUBLAS_CALL( gpublasDcopy( gpublas_handle, msize*nblocks, device_svec, 1, device_rvec, 1 ) );
        GPUBLAS_CALL( gpublasDaxpy( gpublas_handle, msize*nblocks, &m_omega, device_tvec, 1, device_rvec, 1 ) );

        // 28: check for convergence, continue if necessary
        // for(int i = 0; i<msize; i++){if( isnan(device_rvec[i]) ){printf("adi_solver_GPU.c z\n");break;}}
        rvec_norm = adi_vnorm( msize, nblocks, device_rvec );
#ifdef ADI_VERBOSE
        printf( "Iteration %2dc, rvec_norm = %12.5e, tvec_times_svec = %12.5e, tvec_times_tvec   = %12.5e\n",
                k, rvec_norm, tvec_times_svec, tvec_times_tvec );
#endif
        if( rvec_norm < tol*rvec_norm0 ) {
            *iters = k+1;
            break;
        }

        *iters = k+1;
    } // k = 1, niter // master iteration loop

    /*-------------------------------------*
     * Copy result out and clean up memory *
     *-------------------------------------*/

#if defined ( USE_OACC ) || defined( USE_OMP_OL )

#if defined( USE_OACC )
    #pragma acc parallel loop gang vector collapse(2) \
                         present( rhs[0:msize*nblocks] ) \
                         deviceptr( device_xvec )
#elif defined( USE_OMP_OL )
    #pragma omp target teams distribute parallel for simd collapse(2) \
                         is_device_ptr( device_xvec )
#endif
    for( iblock=0; iblock<nblocks; iblock++ ) {
        for( m=0; m<msize; m++ ) {
            rhs[m+iblock*msize] = device_xvec[m+iblock*msize];
        }
    }
#if defined( USE_OACC )
    #pragma acc update host( rhs[0:msize*nblocks] )
#elif defined( USE_OMP_OL )
    #pragma omp target update from( rhs[0:msize*nblocks] )
#endif
#else
    GPUBLAS_CALL( gpublasGetVectorAsync( msize*nblocks, sizeof(double), device_xvec, 1, xvec, 1, streamArray[0] ) );
    GPU_CALL( gpuStreamSynchronize( streamArray[0] ) );
    blasf77_dcopy( &n, xvec, &i_one, rhs, &i_one );
#endif
    //adi_cleanup( nblocks );
} // adi_bicgstab
