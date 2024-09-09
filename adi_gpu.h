#include <unistd.h>

#define MAX(a, b) ((a >= b) ? a : b)
#define MIN(a, b) ((a <= b) ? a : b)

double d_tiny =  10.0*DBL_MIN;
double d_zero =  0.0;
double d_mone = -1.0;
double d_one  =  1.0;

int i_one  = 1;
int i_zero = 0;
int nrhs   = 1;

double *ALU;

double *device_ALU;
double **device_ALU_d;

int *device_ALU_ipiv;
int *device_linfo;
int *host_linfo;

double *device_A;
double **device_A_d;

double *Bband;
double *Cband;

double *device_Bband;
double *device_Cband;

double *dl;
double *d;
double *du;

double *device_dl;
double *device_d;
double *device_du;
double *device_grhs;

double *xvec;
double *rvec;

double *device_x1;
double **device_x1_d;

double *device_dx;
double *device_dx_B;
double *device_dx_C;

double *device_xvec;
double *device_svec;
double *device_shat;
double *device_pvec;
double *device_phat;
double *device_rvec;
double *device_rtilde;
double *device_tvec;
double *device_vvec;

size_t bufferSize;
void *device_buffer;

#ifdef __cplusplus
extern "C" {
#endif

void pre_adi_init( int nblocks, int msize );
void adi_cleanup();
void adi_bicgstab( int msize, int nblocks, double *A, double *B, double *C, double *rhs, int niter, double tol, int *iters );

#define FORTRAN_NAME(lcname, UCNAME) lcname##_

#define blasf77_daxpy FORTRAN_NAME( daxpy, DAXPY )
#define blasf77_dcopy FORTRAN_NAME( dcopy, DCOPY )
#define blasf77_dscal FORTRAN_NAME( dscal, DSCAL )
#define blasf77_dnrm2 FORTRAN_NAME( dnrm2, DNRM2 )
#define blasf77_dgemv FORTRAN_NAME( dgemv, DGEMV )
#define blasf77_dgbmv FORTRAN_NAME( dgbmv, DGBMV )
#define blasf77_dsbmv FORTRAN_NAME( dsbmv, DSBMV )
#define blasf77_ddot  FORTRAN_NAME( ddot,  DDOT  )
#define lapackf77_dgetrf FORTRAN_NAME( dgetrf, DGETRF )
#define lapackf77_dgetrs FORTRAN_NAME( dgetrs, DGETRS )
#define lapackf77_dgttrf FORTRAN_NAME( dgttrf, DGTTRF )
#define lapackf77_dgttrs FORTRAN_NAME( dgttrs, DGTTRS )
#define lapackf77_dgtsv  FORTRAN_NAME( dgtsv,  DGTSV )

void blasf77_daxpy( const int *n,
                    const double *alpha,
                    const double *x, const int *incx,
                    double *y, const int *incy );

void blasf77_dcopy( const int *n,
                    const double *x, const int *incx,
                    double *y, const int *incy );

void blasf77_dscal( const int *n,
                    const double *alpha,
                    double *x, const int *incx );

void blasf77_dgemv( const char *transa,
                    const int *m, const int *n,
                    const double *alpha,
                    const double *A, const int *lda,
                    const double *x, const int *incx,
                    const double *beta,
                    double *y, const int *incy );

void blasf77_dgbmv( const char *uplo,
                    const int *m, const int *n, const int *kl, const int *ku,
                    const double *alpha,
                    const double *A, const int *lda,
                    const double *x, const int *incx,
                    const double *beta,
                    double *y, const int *incy );

void blasf77_dsbmv( const char *uplo,
                    const int *n, const int *k,
                    const double *alpha,
                    const double *A, const int *lda,
                    const double *x, const int *incx,
                    const double *beta,
                    double *y, const int *incy );

double blasf77_dnrm2( const int *n,
                      const double *x, const int *incx );

double blasf77_ddot(  const int *n,
                      const double *x, const int *incx,
                      const double *y, const int *incy );

void lapackf77_dgetrf( const int *m, const int *n,
                       double *A, const int *lda,
                       int *ipiv,
                       int *info );

void lapackf77_dgetrs( const char* trans,
                       const int *n, const int *nrhs,
                       const double *A, const int *lda,
                       const int *ipiv,
                       double *B, const int *ldb,
                       int *info );

void lapackf77_dgttrf( const int *n,
                       double *dl, double *d, double *du, double *du2,
                       int *ipiv,
                       int *info );

void lapackf77_dgttrs( const char* trans,
                       const int *n, const int *nrhs,
                       const double *dl, const double *d, const double *du,
                       const double *du2, const int *ipiv,
                       double *B, const int *ldb,
                       int *info );

void lapackf77_dgtsv( const int *n, const int *nrhs,
                      const double *dl, const double *d, const double *du,
                      double *B, const int *ldb,
                      int *info );

#ifdef __cplusplus
}
#endif
