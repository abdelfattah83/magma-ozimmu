/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Mark Gates
*/
// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"

#define COND_THRESHOLD (1)

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgesv_gpu
*/
int main(int argc, char **argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, cpu_perf, cpu_time, gpu_perf, gpu_time;
    double          error, Rnorm, Anorm, Xnorm, *work;
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex *h_A, *h_B, *h_X;
    magmaDoubleComplex_ptr d_A, d_B;
    magma_int_t *ipiv;
    magma_int_t N, nrhs, lda, ldb, ldda, lddb, info, sizeA, sizeB;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;

    magma_opts opts;
    opts.parse_opts( argc, argv );

    double tol = opts.tolerance * lapackf77_dlamch("E");

    nrhs = opts.nrhs;

    printf("%%   N  NRHS   CPU Gflop/s (sec)   GPU Gflop/s (sec)   ||B - AX|| / N*||A||*||X||\n");
    printf("%%===============================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda    = N;
            ldb    = lda;
            ldda   = magma_roundup( N, opts.align );  // multiple of 32 by default
            lddb   = ldda;
            gflops = ( FLOPS_ZGETRF( N, N ) + FLOPS_ZGETRS( N, nrhs ) ) / 1e9;

            TESTING_CHECK( magma_zmalloc_cpu( &h_A, lda*N    ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_B, ldb*nrhs ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_X, ldb*nrhs ));
            TESTING_CHECK( magma_dmalloc_cpu( &work, N ));
            TESTING_CHECK( magma_imalloc_cpu( &ipiv, N ));

            TESTING_CHECK( magma_zmalloc( &d_A, ldda*N    ));
            TESTING_CHECK( magma_zmalloc( &d_B, lddb*nrhs ));

            /* Initialize the matrices */
            sizeA = lda*N;
            sizeB = ldb*nrhs;
            magma_generate_matrix( opts, N, N, h_A, lda );
            lapackf77_zlarnv( &ione, ISEED, &sizeB, h_B );

            /* perform diagonal scaling */
            #ifdef PRECISION_d
            if(opts.cond > COND_THRESHOLD) {
                bool notransA = (opts.transA == MagmaNoTrans) ? true : false;

				// make A between [1,2]
				#pragma omp parallel for
				for(magma_int_t i = 0; i < sizeA; i++) {
					hA[i] += 1.;
				}

                double cond_sqrt = sqrt(opts.cond);
                double* hD = NULL, *hVA = NULL;
                TESTING_CHECK( magma_dmalloc_cpu( &hD,  K ));
                TESTING_CHECK( magma_dmalloc_cpu( &hVA, M ));
                double scalar = pow( opts.cond, 1/double(K-1) );
                hD[0] = 1 / cond_sqrt;
                for(magma_int_t iD = 1; iD < K; iD++) {
                    hD[iD] = hD[iD-1] * scalar;
                }

                if(N == 8) {
                    magma_dprint(Am, An, hA, lda);
                }

                // scale columns/row of A for N/T
                for(magma_int_t ik = 0; ik < K; ik++) {
                    double* hAt      = ( notransA ) ? hA + lda * ik : hA + ik;
                    magma_int_t incA = ( notransA ) ?             1 : lda;
                    blasf77_dscal(&K, &hD[ik], hAt, &incA);
                }

                if( 1 ) {
                    // rotate rows/cols right/down of A for N/T
                    for(magma_int_t i = 0; i < N; i++) {
                        magma_int_t Vm   = ( notransA ) ? 1 : N;
                        magma_int_t Vn   = ( notransA ) ? N : 1;
                        magma_int_t vlda = Vm;

                        double*     hA0 = ( notransA ) ? hA + i : hA + lda * i;
                        double*     hA1 = hA0;
                        magma_int_t Sm1 = ( notransA ) ? 1    : N-i;
                        magma_int_t Sn1 = ( notransA ) ? N-i  : 1;

                        double*     hA2 = ( notransA ) ? hA0 + (N-i) * lda : hA0 + (N-i);
                        magma_int_t Sm2 = ( notransA ) ? 1 : i;
                        magma_int_t Sn2 = ( notransA ) ? i : 1;

                        lapackf77_dlacpy( "F", &Sm1, &Sn1, hA1, &lda,  hVA + i, &ione );
                        lapackf77_dlacpy( "F", &Sm2, &Sn2, hA2, &lda,  hVA + 0, &ione );
                        lapackf77_dlacpy( "F", &Vm,  &Vn,  hVA, &vlda, hA0,     &lda );
                    }
                }

                if(M == 8 && N == 8 && K == 8) {
                    magma_dprint(Am, An, hA, lda);
                }

                magma_free_cpu( hD );
                magma_free_cpu( hVA );
            }
            #endif // end of diagonal scaling of A

            magma_zsetmatrix( N, N,    h_A, lda, d_A, ldda, opts.queue );
            magma_zsetmatrix( N, nrhs, h_B, ldb, d_B, lddb, opts.queue );

            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            if(opts.version == 1) {
                magma_zgesv_gpu( N, nrhs, d_A, ldda, ipiv, d_B, lddb, &info );
            }
            else {
                magma_zgesv_native_oz(N, nrhs, d_A, ldda, ipiv, d_B, lddb, &info, opts.oz_nsplits);
            }
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_zgesv_gpu returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }

            //=====================================================================
            // Residual
            //=====================================================================
            magma_zgetmatrix( N, nrhs, d_B, lddb, h_X, ldb, opts.queue );

            Anorm = lapackf77_zlange("I", &N, &N,    h_A, &lda, work);
            Xnorm = lapackf77_zlange("I", &N, &nrhs, h_X, &ldb, work);

            blasf77_zgemm( MagmaNoTransStr, MagmaNoTransStr, &N, &nrhs, &N,
                           &c_one,     h_A, &lda,
                                       h_X, &ldb,
                           &c_neg_one, h_B, &ldb);

            Rnorm = lapackf77_zlange("I", &N, &nrhs, h_B, &ldb, work);
            error = Rnorm/(N*Anorm*Xnorm);
            status += ! (error < tol);

            /* ====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_zgesv( &N, &nrhs, h_A, &lda, ipiv, h_B, &ldb, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_zgesv returned error %lld: %s.\n",
                           (long long) info, magma_strerror( info ));
                }

                printf( "%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                        (long long) N, (long long) nrhs, cpu_perf, cpu_time, gpu_perf, gpu_time,
                        error, (error < tol ? "ok" : "failed"));
            }
            else {
                printf( "%5lld %5lld     ---   (  ---  )   %7.2f (%7.2f)   %8.2e   %s\n",
                        (long long) N, (long long) nrhs, gpu_perf, gpu_time,
                        error, (error < tol ? "ok" : "failed"));
            }

            magma_free_cpu( h_A );
            magma_free_cpu( h_B );
            magma_free_cpu( h_X );
            magma_free_cpu( work );
            magma_free_cpu( ipiv );

            magma_free( d_A );
            magma_free( d_B );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
