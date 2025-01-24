// #include "chimera_gpu.h"
// #include "adi_gpu.h"
#include <cmath>

// void setupPtrVecs(int nblocks, int msize, double **device_ALU_d, double **device_x1_d, double *device_ALU, double *device_x1){

//     #pragma omp target teams distribute parallel for \
//                         is_device_ptr( device_ALU_d, device_x1_d,device_ALU,device_x1)
//     for (int i=0; i<nblocks; i++){ device_ALU_d[i]=&device_ALU[i*msize*msize];
//                              device_x1_d[i]=&device_x1[i*msize]; }

// }



// void residual(int nblocks, int msize, double *device_dx, double *device_rtilde, double *device_pvec, double *device_rvec, double *device_xvec, double *rhs){

//     int i, iblock, m;
//     #pragma omp target teams distribute parallel for simd collapse(2) \
//                        is_device_ptr( device_dx, device_rtilde, \
//                                        device_pvec, device_rvec, device_xvec )

//     for( iblock=0; iblock<nblocks; iblock++ ) {
//         for( m=0; m<msize; m++ ) {

//             // 0: initialize x = x_0 = 0
//             device_xvec[m+iblock*msize] = 0.0;
//             device_dx[m+iblock*msize] = 0.0;

//             // 1: compute initial residual r_0 = b - A * x_0 (x_0 = 0)
//             device_rvec[m+iblock*msize] = rhs[m+iblock*msize];
// // if( isnan(device_rvec[m+iblock*msize]) )printf("adi_solver_GPU.c a");
//             // 2: Set p = r and \tilde{r} = r
//             device_pvec[m+iblock*msize]   = device_rvec[m+iblock*msize];
// // if( isnan(device_pvec[m+iblock*msize]) )printf("adi_solver_GPU.c aa");
//             device_rtilde[m+iblock*msize] = device_rvec[m+iblock*msize];
// // if( isnan(device_rtilde[m+iblock*msize]) )printf("adi_solver_GPU.c aaa");
        
//         }
//         }
//         }



//     double adi_vnorm( int msize, int nblocks, double *d_xvec )
// {
//     double vnorm = 0.0;
//     int iblock, m;

// #if defined ( USE_OACC ) || defined( USE_OMP_OL )

// #if defined( USE_OACC )
//     #pragma acc parallel loop gang vector collapse(2) \
//                          reduction( + : vnorm ) \
//                          copy( vnorm ) \
//                          deviceptr( d_xvec )
// #elif defined( USE_OMP_OL )
//     #pragma omp target teams distribute parallel for simd collapse(2) \
//                          reduction( + : vnorm ) \
//                          map( tofrom: vnorm ) \
//                          is_device_ptr( d_xvec )
// #endif
//     for( iblock=0; iblock<nblocks; iblock++ ) {
//         for( m=0; m<msize; m++ ) {
//             vnorm += d_xvec[m+iblock*msize] * d_xvec[m+iblock*msize];
//         }
//     }
//     vnorm = sqrt( vnorm );
// #else
//     GPUBLAS_CALL( gpublasDnrm2( gpublas_handle, msize*nblocks, d_xvec, 1, &vnorm ) );
// #endif

//     return vnorm;
// } // adi_vnorm


//this one!??
double adi_dotprod_remote( int msize, int nblocks, double *d_xvec, double *d_yvec )
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

        // rho = adi_dotprod( msize, nblocks, device_rtilde, device_rvec ); local fails
        // rtilde_times_vvec = adi_dotprod( msize, nblocks, device_rtilde, device_vvec ); local worked!?!
        // tvec_times_svec = adi_dotprod( msize, nblocks, device_tvec, device_svec );
        // tvec_times_tvec = adi_dotprod( msize, nblocks, device_tvec, device_tvec );


/*
        L R R R FAIL
        R L R R Good
        R L L R Fail
        R L R L FAIL
        R R L R FAIL
        R R R L FAIL
*/