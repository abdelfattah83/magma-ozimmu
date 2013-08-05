/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011
    
       @author Raffaele Solca
    
       @precisions normal z -> c

*/
#include "common_magma.h"

extern "C" magma_int_t
magma_zhegvx(magma_int_t itype, char jobz, char range, char uplo, magma_int_t n,
             magmaDoubleComplex *a, magma_int_t lda, magmaDoubleComplex *b, magma_int_t ldb,
             double vl, double vu, magma_int_t il, magma_int_t iu, double abstol,
             magma_int_t *m, double *w,  magmaDoubleComplex *z, magma_int_t ldz,
             magmaDoubleComplex *work, magma_int_t lwork, double *rwork,
             magma_int_t *iwork, magma_int_t *ifail, magma_int_t *info)
{
/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======
    ZHEGVX computes selected eigenvalues, and optionally, eigenvectors
    of a complex generalized Hermitian-definite eigenproblem, of the form
    A*x=(lambda)*B*x,  A*Bx=(lambda)*x,  or B*A*x=(lambda)*x.  Here A and
    B are assumed to be Hermitian and B is also positive definite.
    Eigenvalues and eigenvectors can be selected by specifying either a
    range of values or a range of indices for the desired eigenvalues.
    
    Arguments
    =========
    ITYPE   (input) INTEGER
            Specifies the problem type to be solved:
            = 1:  A*x = (lambda)*B*x
            = 2:  A*B*x = (lambda)*x
            = 3:  B*A*x = (lambda)*x
    
    JOBZ    (input) CHARACTER*1
            = 'N':  Compute eigenvalues only;
            = 'V':  Compute eigenvalues and eigenvectors.
    
    RANGE   (input) CHARACTER*1
            = 'A': all eigenvalues will be found.
            = 'V': all eigenvalues in the half-open interval (VL,VU]
                   will be found.
            = 'I': the IL-th through IU-th eigenvalues will be found.
    
    UPLO    (input) CHARACTER*1
            = 'U':  Upper triangles of A and B are stored;
            = 'L':  Lower triangles of A and B are stored.
    
    N       (input) INTEGER
            The order of the matrices A and B.  N >= 0.
    
    A       (input/output) COMPLEX_16 array, dimension (LDA, N)
            On entry, the Hermitian matrix A.  If UPLO = 'U', the
            leading N-by-N upper triangular part of A contains the
            upper triangular part of the matrix A.  If UPLO = 'L',
            the leading N-by-N lower triangular part of A contains
            the lower triangular part of the matrix A.
    
            On exit,  the lower triangle (if UPLO='L') or the upper
            triangle (if UPLO='U') of A, including the diagonal, is
            destroyed.
    
    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).
    
    B       (input/output) COMPLEX_16 array, dimension (LDB, N)
            On entry, the Hermitian matrix B.  If UPLO = 'U', the
            leading N-by-N upper triangular part of B contains the
            upper triangular part of the matrix B.  If UPLO = 'L',
            the leading N-by-N lower triangular part of B contains
            the lower triangular part of the matrix B.
    
            On exit, if INFO <= N, the part of B containing the matrix is
            overwritten by the triangular factor U or L from the Cholesky
            factorization B = U**H*U or B = L*L**H.
    
    LDB     (input) INTEGER
            The leading dimension of the array B.  LDB >= max(1,N).
    
    VL      (input) DOUBLE PRECISION
    VU      (input) DOUBLE PRECISION
            If RANGE='V', the lower and upper bounds of the interval to
            be searched for eigenvalues. VL < VU.
            Not referenced if RANGE = 'A' or 'I'.
    
    IL      (input) INTEGER
    IU      (input) INTEGER
            If RANGE='I', the indices (in ascending order) of the
            smallest and largest eigenvalues to be returned.
            1 <= IL <= IU <= N, if N > 0; IL = 1 and IU = 0 if N = 0.
            Not referenced if RANGE = 'A' or 'V'.
    
    ABSTOL  (input) DOUBLE PRECISION
            The absolute error tolerance for the eigenvalues.
            An approximate eigenvalue is accepted as converged
            when it is determined to lie in an interval [a,b]
            of width less than or equal to
    
                    ABSTOL + EPS *   max( |a|,|b| ) ,
    
            where EPS is the machine precision.  If ABSTOL is less than
            or equal to zero, then  EPS*|T|  will be used in its place,
            where |T| is the 1-norm of the tridiagonal matrix obtained
            by reducing A to tridiagonal form.
    
            Eigenvalues will be computed most accurately when ABSTOL is
            set to twice the underflow threshold 2*DLAMCH('S'), not zero.
            If this routine returns with INFO>0, indicating that some
            eigenvectors did not converge, try setting ABSTOL to
            2*DLAMCH('S').
    
    M       (output) INTEGER
            The total number of eigenvalues found.  0 <= M <= N.
            If RANGE = 'A', M = N, and if RANGE = 'I', M = IU-IL+1.
    
    W       (output) DOUBLE PRECISION array, dimension (N)
            The first M elements contain the selected
            eigenvalues in ascending order.
    
    Z       (output) COMPLEX_16 array, dimension (LDZ, max(1,M))
            If JOBZ = 'N', then Z is not referenced.
            If JOBZ = 'V', then if INFO = 0, the first M columns of Z
            contain the orthonormal eigenvectors of the matrix A
            corresponding to the selected eigenvalues, with the i-th
            column of Z holding the eigenvector associated with W(i).
            The eigenvectors are normalized as follows:
            if ITYPE = 1 or 2, Z**T*B*Z = I;
            if ITYPE = 3, Z**T*inv(B)*Z = I.
    
            If an eigenvector fails to converge, then that column of Z
            contains the latest approximation to the eigenvector, and the
            index of the eigenvector is returned in IFAIL.
            Note: the user must ensure that at least max(1,M) columns are
            supplied in the array Z; if RANGE = 'V', the exact value of M
            is not known in advance and an upper bound must be used.
    
    LDZ     (input) INTEGER
            The leading dimension of the array Z.  LDZ >= 1, and if
            JOBZ = 'V', LDZ >= max(1,N).
    
    WORK    (workspace/output) COMPLEX_16 array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
    
    LWORK   (input) INTEGER
            The length of the array WORK.  LWORK >= max(1,2*N).
            For optimal efficiency, LWORK >= (NB+1)*N,
            where NB is the blocksize for ZHETRD returned by ILAENV.
    
            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.
    
    RWORK   (workspace) DOUBLE PRECISION array, dimension (7*N)
    
    IWORK   (workspace) INTEGER array, dimension (5*N)
    
    IFAIL   (output) INTEGER array, dimension (N)
            If JOBZ = 'V', then if INFO = 0, the first M elements of
            IFAIL are zero.  If INFO > 0, then IFAIL contains the
            indices of the eigenvectors that failed to converge.
            If JOBZ = 'N', then IFAIL is not referenced.
    
    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
            > 0:  ZPOTRF or ZHEEVX returned an error code:
            <= N: if INFO = i, ZHEEVX failed to converge;
                  i eigenvectors failed to converge.  Their indices
                  are stored in array IFAIL.
            > N:  if INFO = N + i, for 1 <= i <= N, then the leading
                  minor of order i of B is not positive definite.
                  The factorization of B could not be completed and
                  no eigenvalues or eigenvectors were computed.
    
    Further Details
    ===============
    Based on contributions by
       Mark Fahey, Department of Mathematics, Univ. of Kentucky, USA
    =====================================================================  */
    
    char uplo_[2] = {uplo, 0};
    char jobz_[2] = {jobz, 0};
    char range_[2] = {range, 0};
    
    magmaDoubleComplex c_one = MAGMA_Z_ONE;
    
    magmaDoubleComplex *da;
    magmaDoubleComplex *db;
    magmaDoubleComplex *dz;
    magma_int_t ldda = n;
    magma_int_t lddb = n;
    magma_int_t lddz = n;
    
    magma_int_t lower;
    char trans[1];
    magma_int_t wantz;
    magma_int_t lquery;
    magma_int_t alleig, valeig, indeig;
    
    magma_int_t lwmin;
    
    magma_queue_t stream;
    magma_queue_create( &stream );
    
    wantz = lapackf77_lsame(jobz_, MagmaVecStr);
    lower = lapackf77_lsame(uplo_, MagmaLowerStr);
    alleig = lapackf77_lsame(range_, "A");
    valeig = lapackf77_lsame(range_, "V");
    indeig = lapackf77_lsame(range_, "I");
    lquery = lwork == -1;
    
    *info = 0;
    if (itype < 1 || itype > 3) {
        *info = -1;
    } else if (! (alleig || valeig || indeig)) {
        *info = -2;
    } else if (! (wantz || lapackf77_lsame(jobz_, MagmaNoVecStr))) {
        *info = -3;
    } else if (! (lower || lapackf77_lsame(uplo_, MagmaUpperStr))) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (lda < max(1,n)) {
        *info = -7;
    } else if (ldb < max(1,n)) {
        *info = -9;
    } else if (ldz < 1 || (wantz && ldz < n)) {
        *info = -18;
    } else {
        if (valeig) {
            if (n > 0 && vu <= vl) {
                *info = -11;
            }
        } else if (indeig) {
            if (il < 1 || il > max(1,n)) {
                *info = -12;
            } else if (iu < min(n,il) || iu > n) {
                *info = -13;
            }
        }
    }
    
    magma_int_t nb = magma_get_zhetrd_nb(n);
    
    lwmin = n * (nb + 1);
    
    MAGMA_Z_SET2REAL(work[0],(double)lwmin);
    
    
    if (lwork < lwmin && ! lquery) {
        *info = -20;
    }
    
    if (*info != 0) {
        magma_xerbla( __func__, -(*info));
        return *info;
    } else if (lquery) {
        return *info;
    }
    
    /* Quick return if possible */
    if (n == 0) {
        return *info;
    }
    
    if (MAGMA_SUCCESS != magma_zmalloc( &da, n*ldda ) ||
        MAGMA_SUCCESS != magma_zmalloc( &db, n*lddb ) ||
        MAGMA_SUCCESS != magma_zmalloc( &dz, n*lddz )) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }
    
    /*     Form a Cholesky factorization of B. */
    
    magma_zsetmatrix( n, n, b, ldb, db, lddb );
    
    magma_zsetmatrix_async( n, n,
                            a,  lda,
                            da, ldda, stream );
    
    magma_zpotrf_gpu(uplo_[0], n, db, lddb, info);
    if (*info != 0) {
        *info = n + *info;
        return *info;
    }
    
    magma_queue_sync( stream );
    
    magma_zgetmatrix_async( n, n,
                            db, lddb,
                            b,  ldb, stream );
    
    /* Transform problem to standard eigenvalue problem and solve. */
    magma_zhegst_gpu(itype, uplo, n, da, ldda, db, lddb, info);
    magma_zheevx_gpu(jobz, range, uplo, n, da, ldda, vl, vu, il, iu, abstol, m, w, dz, lddz, a, lda, z, ldz, work, lwork, rwork, iwork, ifail, info);
    
    if (wantz && *info == 0) {
        /* Backtransform eigenvectors to the original problem. */
        if (itype == 1 || itype == 2) {
            /* For A*x=(lambda)*B*x and A*B*x=(lambda)*x;
               backtransform eigenvectors: x = inv(L)'*y or inv(U)*y */
            if (lower) {
                *(unsigned char *)trans = MagmaConjTrans;
            } else {
                *(unsigned char *)trans = MagmaNoTrans;
            }
            magma_ztrsm(MagmaLeft, uplo, *trans, MagmaNonUnit, n, *m, c_one, db, lddb, dz, lddz);
        }
        else if (itype == 3) {
            /* For B*A*x=(lambda)*x;
               backtransform eigenvectors: x = L*y or U'*y */
            if (lower) {
                *(unsigned char *)trans = MagmaNoTrans;
            } else {
                *(unsigned char *)trans = MagmaConjTrans;
            }
            magma_ztrmm(MagmaLeft, uplo, *trans, MagmaNonUnit, n, *m, c_one, db, lddb, dz, lddz);
        }
        
        magma_zgetmatrix( n, *m, dz, lddz, z, ldz );
    }
    
    magma_queue_sync( stream );
    magma_queue_destroy( stream );
    
    magma_free( da );
    magma_free( db );
    magma_free( dz );
    
    return *info;
} /* zhegvx */
