import ComplexModule
import RealModule
import SebbuBLAS
import SebbuLAPACK
import Testing

@Suite("LAPACK decompositions and norms")
struct DecompositionAndNormTests {
    @Test
    func singlePrecisionQRReconstructsInput() {
        let original: [Float] = [1, 2, 3, 4, 5, 7]
        var qr = original
        var tau = [Float](repeating: 0, count: 2)
        #expect(
            LAPACK.sgeqrf(
                layout: .rowMajor, m: 3, n: 2,
                a: &qr, lda: 2, tau: &tau
            ) == 0
        )
        let r: [Float] = [qr[0], qr[1], 0, qr[3]]
        #expect(
            LAPACK.sorgqr(
                layout: .rowMajor, m: 3, n: 2, k: 2,
                a: &qr, lda: 2, tau: tau
            ) == 0
        )

        var reconstructed = [Float](repeating: 0, count: 6)
        BLAS.sgemm(
            layout: .rowMajor,
            transposeA: .noTranspose, transposeB: .noTranspose,
            m: 3, n: 2, k: 2,
            alpha: 1, a: qr, lda: 2,
            b: r, ldb: 2,
            beta: 0, c: &reconstructed, ldc: 2
        )
        expectApproximatelyEqual(reconstructed, original, tolerance: 2e-4)
    }

    @Test
    func complexSingleQRReconstructsVector() {
        let original: [Complex<Float>] = [
            Complex(1, 1), Complex(2, -1), Complex(-1, 2),
        ]
        var qr = original
        var tau = [Complex<Float>.zero]
        #expect(
            LAPACK.cgeqrf(
                layout: .columnMajor, m: 3, n: 1,
                a: &qr, lda: 3, tau: &tau
            ) == 0
        )
        let r = qr[0]
        #expect(
            LAPACK.cungqr(
                layout: .columnMajor, m: 3, n: 1, k: 1,
                a: &qr, lda: 3, tau: tau
            ) == 0
        )
        let reconstructed = qr.map { $0 * r }
        expectApproximatelyEqual(reconstructed, original, tolerance: 2e-4)
    }

    @Test
    func complexSingleLeastSquaresRecoversExactSolution() {
        var a: [Complex<Float>] = [
            .one, .zero,
            .zero, .one,
            .one, .one,
        ]
        var b: [Complex<Float>] = [
            Complex(1, 1), Complex(2, -1), Complex(3, 0),
        ]
        #expect(
            LAPACK.cgels(
                layout: .rowMajor, transpose: .noTranspose,
                m: 3, n: 2, nrhs: 1,
                a: &a, lda: 2,
                b: &b, ldb: 1
            ) == 0
        )
        expectApproximatelyEqual(
            Array(b.prefix(2)),
            [Complex(1, 1), Complex(2, -1)],
            tolerance: 2e-4
        )
    }

    @Test
    func realSymmetricEigenvaluesCoverBothPrecisionsAndLayouts() {
        var single: [Float] = [2, 99, 1, 2]
        var singleValues = [Float](repeating: 0, count: 2)
        #expect(
            LAPACK.ssyev(
                layout: .rowMajor, job: .none, triangle: .lower,
                n: 2, a: &single, lda: 2, w: &singleValues
            ) == 0
        )
        expectApproximatelyEqual(singleValues, [1, 3])

        // Column-major upper storage for the same matrix.
        var double = [2.0, 99, 1, 2]
        var doubleValues = [Double](repeating: 0, count: 2)
        #expect(
            LAPACK.dsyev(
                layout: .columnMajor, job: .none, triangle: .upper,
                n: 2, a: &double, lda: 2, w: &doubleValues
            ) == 0
        )
        expectApproximatelyEqual(doubleValues, [1, 3])
    }

    @Test
    func complexSingleHermitianEigenvaluesIgnoreDiagonalImaginaryParts() {
        var a: [Complex<Float>] = [
            Complex(2, 9), Complex(0, 1),
            Complex(99, 99), Complex(3, 8),
        ]
        var values = [Float](repeating: 0, count: 2)
        #expect(
            LAPACK.cheev(
                layout: .rowMajor, job: .none, triangle: .upper,
                n: 2, a: &a, lda: 2, w: &values
            ) == 0
        )
        let rootFive = Float(5).squareRoot()
        expectApproximatelyEqual(
            values,
            [(5 - rootFive) / 2, (5 + rootFive) / 2],
            tolerance: 2e-4
        )
    }

    @Test
    func singlePrecisionDivideAndConquerSVDWithoutVectors() {
        var a: [Float] = [4, 0, 0, -2]
        var singularValues = [Float](repeating: 0, count: 2)
        var unusedU = [Float.zero]
        var unusedVT = [Float.zero]
        #expect(
            LAPACK.sgesdd(
                layout: .rowMajor, job: .none,
                m: 2, n: 2, a: &a, lda: 2,
                s: &singularValues,
                u: &unusedU, ldu: 1,
                vt: &unusedVT, ldvt: 1
            ) == 0
        )
        expectApproximatelyEqual(singularValues, [4, 2])
    }

    @Test
    func complexSingleSVDComputesSingularVectors() {
        var a: [Complex<Float>] = [
            Complex(3, 0), .zero,
            .zero, Complex(0, 2),
        ]
        var singularValues = [Float](repeating: 0, count: 2)
        var u = [Complex<Float>](repeating: .zero, count: 4)
        var vt = [Complex<Float>](repeating: .zero, count: 4)
        var superb = [Float.zero]
        #expect(
            LAPACK.cgesvd(
                layout: .rowMajor,
                jobU: .some, jobVT: .some,
                m: 2, n: 2, a: &a, lda: 2,
                s: &singularValues,
                u: &u, ldu: 2,
                vt: &vt, ldvt: 2,
                superb: &superb
            ) == 0
        )
        expectApproximatelyEqual(singularValues, [3, 2])

        let sigmaVT = [
            vt[0] * Complex<Float>(singularValues[0], 0),
            vt[1] * Complex<Float>(singularValues[0], 0),
            vt[2] * Complex<Float>(singularValues[1], 0),
            vt[3] * Complex<Float>(singularValues[1], 0),
        ]
        var reconstructed = [Complex<Float>](repeating: .zero, count: 4)
        BLAS.cgemm(
            layout: .rowMajor,
            transposeA: .noTranspose, transposeB: .noTranspose,
            m: 2, n: 2, k: 2,
            alpha: .one, a: u, lda: 2,
            b: sigmaVT, ldb: 2,
            beta: .zero, c: &reconstructed, ldc: 2
        )
        expectApproximatelyEqual(
            reconstructed,
            [Complex(3, 0), .zero, .zero, Complex(0, 2)],
            tolerance: 3e-4
        )
    }

    @Test
    func generalMatrixNormsCoverEveryNormKind() {
        let a = [1.0, -2, 3, 4, -5, 6]
        #expect(
            LAPACK.dlange(
                layout: .rowMajor, norm: .maxAbsoluteValue,
                m: 2, n: 3, a: a, lda: 3
            ).isApproximatelyEqual(to: 6, absoluteTolerance: 1e-12)
        )
        #expect(
            LAPACK.dlange(
                layout: .rowMajor, norm: .one,
                m: 2, n: 3, a: a, lda: 3
            ).isApproximatelyEqual(to: 9, absoluteTolerance: 1e-12)
        )
        #expect(
            LAPACK.dlange(
                layout: .rowMajor, norm: .infinity,
                m: 2, n: 3, a: a, lda: 3
            ).isApproximatelyEqual(to: 15, absoluteTolerance: 1e-12)
        )
        #expect(
            LAPACK.dlange(
                layout: .rowMajor, norm: .frobenius,
                m: 2, n: 3, a: a, lda: 3
            ).isApproximatelyEqual(
                to: 91.squareRoot(),
                absoluteTolerance: 1e-11
            )
        )
    }

    @Test
    func symmetricAndHermitianNormsUseOnlyStoredTriangle() {
        let symmetric = [1.0, 2, 99, 3]
        #expect(
            LAPACK.dlansy(
                layout: .rowMajor, norm: .maxAbsoluteValue,
                triangle: .upper, n: 2,
                a: symmetric, lda: 2
            ).isApproximatelyEqual(to: 3, absoluteTolerance: 1e-12)
        )
        #expect(
            LAPACK.dlansy(
                layout: .rowMajor, norm: .one,
                triangle: .upper, n: 2,
                a: symmetric, lda: 2
            ).isApproximatelyEqual(to: 5, absoluteTolerance: 1e-12)
        )
        #expect(
            LAPACK.dlansy(
                layout: .rowMajor, norm: .infinity,
                triangle: .upper, n: 2,
                a: symmetric, lda: 2
            ).isApproximatelyEqual(to: 5, absoluteTolerance: 1e-12)
        )
        #expect(
            LAPACK.dlansy(
                layout: .rowMajor, norm: .frobenius,
                triangle: .upper, n: 2,
                a: symmetric, lda: 2
            ).isApproximatelyEqual(
                to: 18.squareRoot(),
                absoluteTolerance: 1e-11
            )
        )

        let hermitian: [Complex<Double>] = [
            Complex(1, 9), Complex(2, 0),
            Complex(99, 99), Complex(3, 8),
        ]
        #expect(
            LAPACK.zlanhe(
                layout: .rowMajor, norm: .maxAbsoluteValue,
                triangle: .upper, n: 2,
                a: hermitian, lda: 2
            ).isApproximatelyEqual(to: 3, absoluteTolerance: 1e-12)
        )
        #expect(
            LAPACK.zlanhe(
                layout: .rowMajor, norm: .one,
                triangle: .upper, n: 2,
                a: hermitian, lda: 2
            ).isApproximatelyEqual(to: 5, absoluteTolerance: 1e-12)
        )
        #expect(
            LAPACK.zlanhe(
                layout: .rowMajor, norm: .frobenius,
                triangle: .upper, n: 2,
                a: hermitian, lda: 2
            ).isApproximatelyEqual(
                to: 18.squareRoot(),
                absoluteTolerance: 1e-11
            )
        )
    }
}
