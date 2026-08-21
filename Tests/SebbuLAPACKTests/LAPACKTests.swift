import ComplexModule
import RealModule
import SebbuBLAS
import SebbuLAPACK
import Testing

@Suite("LAPACK tests")
struct LAPACKTests {
    @Test
    func testLAPACKEArgumentIndices() {
        var a = [Double](repeating: 0, count: 4)
        var b = [Double](repeating: 0, count: 2)
        var pivots = [Int](repeating: 0, count: 2)
        #expect(
            LAPACK.dgesv(
                layout: .rowMajor, n: 2, nrhs: 1,
                a: &a, lda: 1, ipiv: &pivots,
                b: &b, ldb: 1
            ) == -5)

        var singularValues = [Double](repeating: 0, count: 1)
        var u = [Double](repeating: 0, count: 2)
        var vt = [Double](repeating: 0, count: 1)
        #expect(
            LAPACK.dgesdd(
                layout: .rowMajor, job: .some,
                m: 2, n: 1, a: &a, lda: 1,
                s: &singularValues, u: &u, ldu: 0,
                vt: &vt, ldvt: 1
            ) == -9)

        var complexA = [Complex<Double>](repeating: .zero, count: 4)
        var complexU = [Complex<Double>](repeating: .zero, count: 4)
        var complexVT = [Complex<Double>](repeating: .zero, count: 4)
        var superb = [Double](repeating: 0, count: 1)
        #expect(
            LAPACK.zgesvd(
                layout: .rowMajor, jobU: .overwrite, jobVT: .overwrite,
                m: 2, n: 2, a: &complexA, lda: 2,
                s: &singularValues, u: &complexU, ldu: 2,
                vt: &complexVT, ldvt: 2, superb: &superb
            ) == -3)

        var floatA = [Float](repeating: 0, count: 4)
        var floatB = [Float](repeating: 0, count: 2)
        #expect(
            LAPACK.sgels(
                layout: .rowMajor, transpose: .conjugateTranspose,
                m: 2, n: 2, nrhs: 1,
                a: &floatA, lda: 2, b: &floatB, ldb: 1
            ) == -2)
    }

    @Test
    func testGeneralSolveUsesSwiftIntPivots() {
        var a = [
            3.0, 1, -1,
            2, 4, 1,
            -1, 2, 5,
        ]
        var b = [4.0, 1, 1]
        var pivots = [Int](repeating: 0, count: 3)
        let info = LAPACK.dgesv(
            layout: .rowMajor, n: 3, nrhs: 1,
            a: &a, lda: 3, ipiv: &pivots,
            b: &b, ldb: 1
        )
        #expect(info == 0)
        #expect(pivots.allSatisfy { 1...3 ~= $0 })
        #expect(b[0].isApproximatelyEqual(to: 2, absoluteTolerance: 1e-12))
        #expect(b[1].isApproximatelyEqual(to: -1, absoluteTolerance: 1e-12))
        #expect(b[2].isApproximatelyEqual(to: 1, absoluteTolerance: 1e-12))
    }

    @Test
    func testComplexColumnMajorGeneralSolve() {
        var a = [
            Complex<Double>(2, 0), Complex<Double>(1, -1),
            Complex<Double>(1, 1), Complex<Double>(3, 0),
        ]
        var b = [Complex<Double>(1, 1), Complex<Double>(2, -1)]
        var pivots = [Int](repeating: 0, count: 2)
        let originalA = a
        let originalB = b
        let info = LAPACK.zgesv(
            layout: .columnMajor, n: 2, nrhs: 1,
            a: &a, lda: 2, ipiv: &pivots,
            b: &b, ldb: 2
        )
        #expect(info == 0)
        for row in 0..<2 {
            let reconstructed =
                originalA[row] * b[0]
                + originalA[2 + row] * b[1]
            #expect(
                reconstructed.real.isApproximatelyEqual(
                    to: originalB[row].real, absoluteTolerance: 1e-11
                ))
            #expect(
                reconstructed.imaginary.isApproximatelyEqual(
                    to: originalB[row].imaginary, absoluteTolerance: 1e-11
                ))
        }
    }

    @Test
    func testLUInverse() {
        let original = [4.0, 7, 2, 6]
        var inverse = original
        var pivots = [Int](repeating: 0, count: 2)
        #expect(
            LAPACK.dgetrf(
                layout: .rowMajor, m: 2, n: 2,
                a: &inverse, lda: 2, ipiv: &pivots
            ) == 0)
        #expect(
            LAPACK.dgetri(
                layout: .rowMajor, n: 2,
                a: &inverse, lda: 2, ipiv: pivots
            ) == 0)

        var product = [Double](repeating: 0, count: 4)
        BLAS.dgemm(
            layout: .rowMajor,
            transposeA: .noTranspose, transposeB: .noTranspose,
            m: 2, n: 2, k: 2, alpha: 1,
            a: original, lda: 2, b: inverse, ldb: 2,
            beta: 0, c: &product, ldc: 2
        )
        #expect(product[0].isApproximatelyEqual(to: 1, absoluteTolerance: 1e-11))
        #expect(product[1].isApproximatelyEqual(to: 0, absoluteTolerance: 1e-11))
        #expect(product[2].isApproximatelyEqual(to: 0, absoluteTolerance: 1e-11))
        #expect(product[3].isApproximatelyEqual(to: 1, absoluteTolerance: 1e-11))
    }

    @Test
    func testCholeskySolve() {
        var a = [
            4.0, 1, 1,
            1, 3, 0,
            1, 0, 2,
        ]
        var b = [9.0, 7, 7]
        let info = LAPACK.dposv(
            layout: .rowMajor, triangle: .lower,
            n: 3, nrhs: 1,
            a: &a, lda: 3, b: &b, ldb: 1
        )
        #expect(info == 0)
        #expect(b[0].isApproximatelyEqual(to: 1, absoluteTolerance: 1e-12))
        #expect(b[1].isApproximatelyEqual(to: 2, absoluteTolerance: 1e-12))
        #expect(b[2].isApproximatelyEqual(to: 3, absoluteTolerance: 1e-12))
    }

    @Test
    func testComplexUpperCholeskySolve() {
        var a = [
            Complex<Double>(4, 0), Complex<Double>(1, 1),
            Complex<Double>.zero, Complex<Double>(3, 0),
        ]
        var b = [Complex<Double>(7, 5), Complex<Double>(8, -3)]
        let info = LAPACK.zposv(
            layout: .rowMajor, triangle: .upper,
            n: 2, nrhs: 1,
            a: &a, lda: 2, b: &b, ldb: 1
        )
        #expect(info == 0)
        #expect(b[0].real.isApproximatelyEqual(to: 1, absoluteTolerance: 1e-11))
        #expect(b[0].imaginary.isApproximatelyEqual(to: 1, absoluteTolerance: 1e-11))
        #expect(b[1].real.isApproximatelyEqual(to: 2, absoluteTolerance: 1e-11))
        #expect(b[1].imaginary.isApproximatelyEqual(to: -1, absoluteTolerance: 1e-11))
    }

    @Test
    func testTriangularInverse() {
        let original = [2.0, 1, 0, 4]
        var inverse = original
        #expect(
            LAPACK.dtrtri(
                layout: .rowMajor, triangle: .upper,
                diagonal: .nonUnit, n: 2,
                a: &inverse, lda: 2
            ) == 0)
        #expect(inverse[0].isApproximatelyEqual(to: 0.5, absoluteTolerance: 1e-12))
        #expect(inverse[1].isApproximatelyEqual(to: -0.125, absoluteTolerance: 1e-12))
        #expect(inverse[3].isApproximatelyEqual(to: 0.25, absoluteTolerance: 1e-12))
    }

    @Test
    func testQRFactorizationAndExplicitQ() {
        let original = [1.0, 2, 3, 4, 5, 6]
        var qr = original
        var tau = [Double](repeating: 0, count: 2)
        #expect(
            LAPACK.dgeqrf(
                layout: .rowMajor, m: 3, n: 2,
                a: &qr, lda: 2, tau: &tau
            ) == 0)
        let r = [qr[0], qr[1], 0, qr[3]]
        #expect(
            LAPACK.dorgqr(
                layout: .rowMajor, m: 3, n: 2, k: 2,
                a: &qr, lda: 2, tau: tau
            ) == 0)

        var reconstructed = [Double](repeating: 0, count: 6)
        BLAS.dgemm(
            layout: .rowMajor,
            transposeA: .noTranspose, transposeB: .noTranspose,
            m: 3, n: 2, k: 2, alpha: 1,
            a: qr, lda: 2, b: r, ldb: 2,
            beta: 0, c: &reconstructed, ldc: 2
        )
        for index in original.indices {
            #expect(
                reconstructed[index].isApproximatelyEqual(
                    to: original[index], absoluteTolerance: 1e-10
                ))
        }
    }

    @Test
    func testComplexQRFactorizationAndExplicitQ() {
        let original = [
            Complex<Double>(1, 1), Complex<Double>(2, -1),
            Complex<Double>(3, 0), Complex<Double>(-1, 2),
            Complex<Double>(0, -2), Complex<Double>(4, 1),
        ]
        var qr = original
        var tau = [Complex<Double>](repeating: .zero, count: 2)
        #expect(
            LAPACK.zgeqrf(
                layout: .rowMajor, m: 3, n: 2,
                a: &qr, lda: 2, tau: &tau
            ) == 0)
        let r = [qr[0], qr[1], .zero, qr[3]]
        #expect(
            LAPACK.zungqr(
                layout: .rowMajor, m: 3, n: 2, k: 2,
                a: &qr, lda: 2, tau: tau
            ) == 0)

        var reconstructed = [Complex<Double>](repeating: .zero, count: 6)
        BLAS.zgemm(
            layout: .rowMajor,
            transposeA: .noTranspose, transposeB: .noTranspose,
            m: 3, n: 2, k: 2, alpha: .one,
            a: qr, lda: 2, b: r, ldb: 2,
            beta: .zero, c: &reconstructed, ldc: 2
        )
        for index in original.indices {
            #expect(
                reconstructed[index].real.isApproximatelyEqual(
                    to: original[index].real, absoluteTolerance: 1e-10
                ))
            #expect(
                reconstructed[index].imaginary.isApproximatelyEqual(
                    to: original[index].imaginary, absoluteTolerance: 1e-10
                ))
        }
    }

    @Test
    func testLeastSquares() {
        var a = [1.0, 0, 1, 1, 1, 2]
        var b = [1.0, 3, 5]
        let info = LAPACK.dgels(
            layout: .rowMajor, transpose: .noTranspose,
            m: 3, n: 2, nrhs: 1,
            a: &a, lda: 2, b: &b, ldb: 1
        )
        #expect(info == 0)
        #expect(b[0].isApproximatelyEqual(to: 1, absoluteTolerance: 1e-10))
        #expect(b[1].isApproximatelyEqual(to: 2, absoluteTolerance: 1e-10))
    }

    @Test
    func testHermitianEigenvectors() {
        let imaginaryUnit = Complex<Double>(0, 1)
        var a = [
            Complex<Double>(2, 0), imaginaryUnit,
            Complex<Double>.zero, Complex<Double>(3, 0),
        ]
        var values = [Double](repeating: 0, count: 2)
        let info = LAPACK.zheev(
            layout: .rowMajor, job: .vectors, triangle: .upper,
            n: 2, a: &a, lda: 2, w: &values
        )
        #expect(info == 0)
        let rootFive = 5.0.squareRoot()
        #expect(
            values[0].isApproximatelyEqual(
                to: (5 - rootFive) / 2, absoluteTolerance: 1e-10
            ))
        #expect(
            values[1].isApproximatelyEqual(
                to: (5 + rootFive) / 2, absoluteTolerance: 1e-10
            ))

        let original = [
            Complex<Double>(2, 0), imaginaryUnit,
            -imaginaryUnit, Complex<Double>(3, 0),
        ]
        for column in 0..<2 {
            for row in 0..<2 {
                let lhs =
                    original[row * 2] * a[column]
                    + original[row * 2 + 1] * a[2 + column]
                let rhs =
                    a[row * 2 + column]
                    * Complex<Double>(values[column], 0)
                #expect(
                    lhs.real.isApproximatelyEqual(
                        to: rhs.real, absoluteTolerance: 1e-9
                    ))
                #expect(
                    lhs.imaginary.isApproximatelyEqual(
                        to: rhs.imaginary, absoluteTolerance: 1e-9
                    ))
            }
        }
    }

    @Test
    func testSingularValueDecomposition() {
        let original = [3.0, 0, 0, 2, 0, 0]
        var a = original
        var singularValues = [Double](repeating: 0, count: 2)
        var u = [Double](repeating: 0, count: 6)
        var vt = [Double](repeating: 0, count: 4)
        var superb = [Double](repeating: 0, count: 1)
        let info = LAPACK.dgesvd(
            layout: .rowMajor, jobU: .some, jobVT: .some,
            m: 3, n: 2, a: &a, lda: 2,
            s: &singularValues, u: &u, ldu: 2,
            vt: &vt, ldvt: 2, superb: &superb
        )
        #expect(info == 0)
        #expect(
            singularValues[0].isApproximatelyEqual(
                to: 3, absoluteTolerance: 1e-10
            ))
        #expect(
            singularValues[1].isApproximatelyEqual(
                to: 2, absoluteTolerance: 1e-10
            ))

        let sigmaVT = [
            singularValues[0] * vt[0], singularValues[0] * vt[1],
            singularValues[1] * vt[2], singularValues[1] * vt[3],
        ]
        var reconstructed = [Double](repeating: 0, count: 6)
        BLAS.dgemm(
            layout: .rowMajor,
            transposeA: .noTranspose, transposeB: .noTranspose,
            m: 3, n: 2, k: 2, alpha: 1,
            a: u, lda: 2, b: sigmaVT, ldb: 2,
            beta: 0, c: &reconstructed, ldc: 2
        )
        for index in original.indices {
            #expect(
                reconstructed[index].isApproximatelyEqual(
                    to: original[index], absoluteTolerance: 1e-9
                ))
        }
    }

    @Test
    func testDivideAndConquerSingularValueDecomposition() {
        let original = [4.0, 0, 0, 0, 2, 0]
        var a = original
        var singularValues = [Double](repeating: 0, count: 2)
        var u = [Double](repeating: 0, count: 4)
        var vt = [Double](repeating: 0, count: 6)
        let info = LAPACK.dgesdd(
            layout: .rowMajor, job: .some,
            m: 2, n: 3, a: &a, lda: 3,
            s: &singularValues, u: &u, ldu: 2,
            vt: &vt, ldvt: 3
        )
        #expect(info == 0)

        let sigmaVT = [
            singularValues[0] * vt[0], singularValues[0] * vt[1],
            singularValues[0] * vt[2], singularValues[1] * vt[3],
            singularValues[1] * vt[4], singularValues[1] * vt[5],
        ]
        var reconstructed = [Double](repeating: 0, count: 6)
        BLAS.dgemm(
            layout: .rowMajor,
            transposeA: .noTranspose, transposeB: .noTranspose,
            m: 2, n: 3, k: 2, alpha: 1,
            a: u, lda: 2, b: sigmaVT, ldb: 3,
            beta: 0, c: &reconstructed, ldc: 3
        )
        for index in original.indices {
            #expect(
                reconstructed[index].isApproximatelyEqual(
                    to: original[index], absoluteTolerance: 1e-9
                ))
        }
    }

    @Test
    func testComplexSingularValueDecomposition() {
        let original = [
            Complex<Double>(1, 1), Complex<Double>(2, 0),
            Complex<Double>(0, -1), Complex<Double>(3, 2),
        ]
        var a = original
        var singularValues = [Double](repeating: 0, count: 2)
        var u = [Complex<Double>](repeating: .zero, count: 4)
        var vt = [Complex<Double>](repeating: .zero, count: 4)
        var superb = [Double](repeating: 0, count: 1)
        #expect(
            LAPACK.zgesvd(
                layout: .rowMajor, jobU: .all, jobVT: .all,
                m: 2, n: 2, a: &a, lda: 2,
                s: &singularValues, u: &u, ldu: 2,
                vt: &vt, ldvt: 2, superb: &superb
            ) == 0)

        let sigmaVT = [
            vt[0] * Complex<Double>(singularValues[0], 0),
            vt[1] * Complex<Double>(singularValues[0], 0),
            vt[2] * Complex<Double>(singularValues[1], 0),
            vt[3] * Complex<Double>(singularValues[1], 0),
        ]
        var reconstructed = [Complex<Double>](repeating: .zero, count: 4)
        BLAS.zgemm(
            layout: .rowMajor,
            transposeA: .noTranspose, transposeB: .noTranspose,
            m: 2, n: 2, k: 2, alpha: .one,
            a: u, lda: 2, b: sigmaVT, ldb: 2,
            beta: .zero, c: &reconstructed, ldc: 2
        )
        for index in original.indices {
            #expect(
                reconstructed[index].real.isApproximatelyEqual(
                    to: original[index].real, absoluteTolerance: 1e-9
                ))
            #expect(
                reconstructed[index].imaginary.isApproximatelyEqual(
                    to: original[index].imaginary, absoluteTolerance: 1e-9
                ))
        }
    }

    @Test
    func testSolveWithTransposedLUFactors() {
        var factors = [2.0, 3, 1, 4]
        var pivots = [Int](repeating: 0, count: 2)
        #expect(
            LAPACK.dgetrf(
                layout: .columnMajor, m: 2, n: 2,
                a: &factors, lda: 2, ipiv: &pivots
            ) == 0)
        // A = [[2, 1], [3, 4]], so A^T * [2, -1] = [1, -2].
        var b = [1.0, -2]
        #expect(
            LAPACK.dgetrs(
                layout: .columnMajor, transpose: .transpose,
                n: 2, nrhs: 1, a: factors, lda: 2,
                ipiv: pivots, b: &b, ldb: 2
            ) == 0)
        #expect(b[0].isApproximatelyEqual(to: 2, absoluteTolerance: 1e-12))
        #expect(b[1].isApproximatelyEqual(to: -1, absoluteTolerance: 1e-12))
    }

    @Test
    func testMatrixNorms() {
        let a = [1.0, -2, 3, 4, -5, 6]
        let oneNorm = LAPACK.dlange(
            layout: .rowMajor, norm: .one,
            m: 2, n: 3, a: a, lda: 3
        )
        let infinityNorm = LAPACK.dlange(
            layout: .rowMajor, norm: .infinity,
            m: 2, n: 3, a: a, lda: 3
        )
        #expect(oneNorm.isApproximatelyEqual(to: 9, absoluteTolerance: 1e-12))
        #expect(infinityNorm.isApproximatelyEqual(to: 15, absoluteTolerance: 1e-12))
    }
}
