import ComplexModule
import RealModule
import SebbuLAPACK
import Testing

@Suite("LAPACK linear equations")
struct LinearEquationTests {
    @Test
    func singlePrecisionGeneralSolve() {
        var a: [Float] = [3, 1, 1, 2]
        var b: [Float] = [5, 5]
        var pivots = [Int](repeating: 0, count: 2)
        let info = LAPACK.sgesv(
            layout: .rowMajor, n: 2, nrhs: 1,
            a: &a, lda: 2, ipiv: &pivots,
            b: &b, ldb: 1
        )
        #expect(info == 0)
        #expect(pivots.allSatisfy { 1...2 ~= $0 })
        expectApproximatelyEqual(b, [1, 2])
    }

    @Test
    func complexSingleColumnMajorGeneralSolve() {
        var a: [Complex<Float>] = [
            Complex(2, 0), Complex(1, -1),
            Complex(1, 1), Complex(3, 0),
        ]
        var b: [Complex<Float>] = [Complex(5, 3), Complex(8, -3)]
        var pivots = [Int](repeating: 0, count: 2)
        let info = LAPACK.cgesv(
            layout: .columnMajor, n: 2, nrhs: 1,
            a: &a, lda: 2, ipiv: &pivots,
            b: &b, ldb: 2
        )
        #expect(info == 0)
        expectApproximatelyEqual(
            b,
            [Complex(1, 1), Complex(2, -1)]
        )
    }

    @Test
    func LUFactorsSolveMultipleRightHandSides() {
        var factors = [4.0, 1, 2, 3]
        var pivots = [Int](repeating: 0, count: 2)
        #expect(
            LAPACK.dgetrf(
                layout: .rowMajor, m: 2, n: 2,
                a: &factors, lda: 2, ipiv: &pivots
            ) == 0
        )

        // B = A * X for X = [[1, 2], [3, 4]].
        var b = [7.0, 12, 11, 16]
        #expect(
            LAPACK.dgetrs(
                layout: .rowMajor, transpose: .noTranspose,
                n: 2, nrhs: 2,
                a: factors, lda: 2, ipiv: pivots,
                b: &b, ldb: 2
            ) == 0
        )
        expectApproximatelyEqual(b, [1, 2, 3, 4])
    }

    @Test
    func LUFactorizationReportsSingularPivot() {
        var singular = [1.0, 2, 2, 4]
        var pivots = [Int](repeating: 0, count: 2)
        #expect(
            LAPACK.dgetrf(
                layout: .rowMajor, m: 2, n: 2,
                a: &singular, lda: 2, ipiv: &pivots
            ) == 2
        )
    }

    @Test
    func CholeskyFactorSolveAndInverseWorkflow() {
        var factor = [4.0, 1, 1, 3]
        #expect(
            LAPACK.dpotrf(
                layout: .rowMajor, triangle: .lower,
                n: 2, a: &factor, lda: 2
            ) == 0
        )

        var b = [6.0, 7]
        #expect(
            LAPACK.dpotrs(
                layout: .rowMajor, triangle: .lower,
                n: 2, nrhs: 1,
                a: factor, lda: 2,
                b: &b, ldb: 1
            ) == 0
        )
        expectApproximatelyEqual(b, [1, 2])

        #expect(
            LAPACK.dpotri(
                layout: .rowMajor, triangle: .lower,
                n: 2, a: &factor, lda: 2
            ) == 0
        )
        #expect(
            factor[0].isApproximatelyEqual(
                to: 3.0 / 11,
                absoluteTolerance: 1e-11
            )
        )
        #expect(
            factor[2].isApproximatelyEqual(
                to: -1.0 / 11,
                absoluteTolerance: 1e-11
            )
        )
        #expect(
            factor[3].isApproximatelyEqual(
                to: 4.0 / 11,
                absoluteTolerance: 1e-11
            )
        )
    }

    @Test
    func complexSingleCholeskyFactorAndInverse() {
        var diagonal: [Complex<Float>] = [
            Complex(4, 0), Complex(99, 99),
            .zero, Complex(9, 0),
        ]
        #expect(
            LAPACK.cpotrf(
                layout: .rowMajor, triangle: .lower,
                n: 2, a: &diagonal, lda: 2
            ) == 0
        )
        #expect(
            LAPACK.cpotri(
                layout: .rowMajor, triangle: .lower,
                n: 2, a: &diagonal, lda: 2
            ) == 0
        )
        expectApproximatelyEqual(diagonal[0], Complex(0.25, 0))
        expectApproximatelyEqual(diagonal[2], .zero)
        expectApproximatelyEqual(diagonal[3], Complex(1.0 / 9, 0))
    }

    @Test
    func CholeskyFactorizationReportsNonPositiveDefiniteMinor() {
        var indefinite = [1.0, 2, 2, 1]
        #expect(
            LAPACK.dpotrf(
                layout: .rowMajor, triangle: .lower,
                n: 2, a: &indefinite, lda: 2
            ) == 2
        )
    }

    @Test
    func triangularSolveSupportsMultipleRightHandSidesAndUnitDiagonal() {
        // Unit-upper A = [[1, 2], [0, 1]].
        let a = [99.0, 2, 0, -17]
        var b = [7.0, 10, 3, 4]
        #expect(
            LAPACK.dtrtrs(
                layout: .rowMajor, triangle: .upper,
                transpose: .noTranspose, diagonal: .unit,
                n: 2, nrhs: 2,
                a: a, lda: 2,
                b: &b, ldb: 2
            ) == 0
        )
        expectApproximatelyEqual(b, [1, 2, 3, 4])
    }

    @Test
    func complexSingleTriangularInverse() {
        var a: [Complex<Float>] = [
            Complex(2, 0), Complex(0, 1),
            .zero, Complex(4, 0),
        ]
        #expect(
            LAPACK.ctrtri(
                layout: .rowMajor, triangle: .upper,
                diagonal: .nonUnit,
                n: 2, a: &a, lda: 2
            ) == 0
        )
        expectApproximatelyEqual(a[0], Complex(0.5, 0))
        expectApproximatelyEqual(a[1], Complex(0, -0.125))
        expectApproximatelyEqual(a[3], Complex(0.25, 0))
    }

    @Test
    func triangularRoutinesReportZeroDiagonal() {
        var inverse = [1.0, 2, 0, 0]
        #expect(
            LAPACK.dtrtri(
                layout: .rowMajor, triangle: .upper,
                diagonal: .nonUnit,
                n: 2, a: &inverse, lda: 2
            ) == 2
        )

        let triangular = [1.0, 2, 0, 0]
        var b = [1.0, 1]
        #expect(
            LAPACK.dtrtrs(
                layout: .rowMajor, triangle: .upper,
                transpose: .noTranspose, diagonal: .nonUnit,
                n: 2, nrhs: 1,
                a: triangular, lda: 2,
                b: &b, ldb: 1
            ) == 2
        )
    }
}
