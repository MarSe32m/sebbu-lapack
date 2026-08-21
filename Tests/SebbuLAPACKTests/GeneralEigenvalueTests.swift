import ComplexModule
import SebbuLAPACK
import Testing

@Suite("LAPACK general eigenproblems")
struct GeneralEigenvalueTests {
    @Test
    func realSingleEigenvaluesWithoutEigenvectors() {
        var a: [Float] = [
            1, 2,
            0, 3,
        ]
        var wr = [Float](repeating: 0, count: 2)
        var wi = [Float](repeating: 0, count: 2)
        var unusedVL = [Float.zero]
        var unusedVR = [Float.zero]

        #expect(
            LAPACK.sgeev(
                layout: .rowMajor, jobVL: .none, jobVR: .none,
                n: 2, a: &a, lda: 2,
                wr: &wr, wi: &wi,
                vl: &unusedVL, ldvl: 1,
                vr: &unusedVR, ldvr: 1
            ) == 0
        )
        expectApproximatelyEqual(wr, [1, 3], tolerance: 2e-4)
        expectApproximatelyEqual(wi, [0, 0], tolerance: 2e-4)
    }

    @Test
    func realDoubleConjugatePairUsesPackedLeftAndRightEigenvectors() {
        // Column-major representation of [0 -1; 1 0].
        let original = [0.0, 1, -1, 0]
        var a = original
        var wr = [Double](repeating: 0, count: 2)
        var wi = [Double](repeating: 0, count: 2)
        var vl = [Double](repeating: 0, count: 4)
        var vr = [Double](repeating: 0, count: 4)

        #expect(
            LAPACK.dgeev(
                layout: .columnMajor, jobVL: .vectors, jobVR: .vectors,
                n: 2, a: &a, lda: 2,
                wr: &wr, wi: &wi,
                vl: &vl, ldvl: 2,
                vr: &vr, ldvr: 2
            ) == 0
        )
        expectApproximatelyEqual(wr, [0, 0], tolerance: 1e-11)
        expectApproximatelyEqual(wi, [1, -1], tolerance: 1e-11)

        let eigenvalue = Complex(wr[0], wi[0])
        let right = [Complex(vr[0], vr[2]), Complex(vr[1], vr[3])]
        let left = [Complex(vl[0], vl[2]), Complex(vl[1], vl[3])]

        let appliedRight = [
            -right[1],
            right[0],
        ]
        for row in 0..<2 {
            expectApproximatelyEqual(
                appliedRight[row], eigenvalue * right[row], tolerance: 1e-11
            )
        }

        // A^H u = conjugate(lambda) u for LAPACK's left eigenvectors.
        let appliedAdjoint = [
            left[1],
            -left[0],
        ]
        for row in 0..<2 {
            expectApproximatelyEqual(
                appliedAdjoint[row], eigenvalue.conjugate * left[row], tolerance: 1e-11
            )
        }
    }

    @Test
    func complexSingleRightEigenvectorsMatchTheirEigenvalues() {
        let original: [Complex<Float>] = [
            Complex(1, 1), Complex(2, -1),
            .zero, Complex(3, -1),
        ]
        var a = original
        var w = [Complex<Float>](repeating: .zero, count: 2)
        var unusedVL = [Complex<Float>.zero]
        var vr = [Complex<Float>](repeating: .zero, count: 4)

        #expect(
            LAPACK.cgeev(
                layout: .rowMajor, jobVL: .none, jobVR: .vectors,
                n: 2, a: &a, lda: 2, w: &w,
                vl: &unusedVL, ldvl: 1,
                vr: &vr, ldvr: 2
            ) == 0
        )

        for column in 0..<2 {
            let vector = [vr[column], vr[2 + column]]
            let applied = [
                original[0] * vector[0] + original[1] * vector[1],
                original[2] * vector[0] + original[3] * vector[1],
            ]
            for row in 0..<2 {
                expectApproximatelyEqual(
                    applied[row], w[column] * vector[row], tolerance: 5e-4
                )
            }
        }
    }

    @Test
    func complexDoubleDenseProblemComputesBothEigenvectorSides() {
        let rowMajorOriginal: [Complex<Double>] = [
            Complex(1, 1), Complex(2, -1),
            Complex(-0.5, 0.25), Complex(3, -0.5),
        ]
        var a: [Complex<Double>] = [
            rowMajorOriginal[0], rowMajorOriginal[2],
            rowMajorOriginal[1], rowMajorOriginal[3],
        ]
        var w = [Complex<Double>](repeating: .zero, count: 2)
        var vl = [Complex<Double>](repeating: .zero, count: 4)
        var vr = [Complex<Double>](repeating: .zero, count: 4)

        #expect(
            LAPACK.zgeev(
                layout: .columnMajor, jobVL: .vectors, jobVR: .vectors,
                n: 2, a: &a, lda: 2, w: &w,
                vl: &vl, ldvl: 2,
                vr: &vr, ldvr: 2
            ) == 0
        )

        for column in 0..<2 {
            let right = [vr[2 * column], vr[2 * column + 1]]
            let appliedRight = [
                rowMajorOriginal[0] * right[0] + rowMajorOriginal[1] * right[1],
                rowMajorOriginal[2] * right[0] + rowMajorOriginal[3] * right[1],
            ]
            for row in 0..<2 {
                expectApproximatelyEqual(
                    appliedRight[row], w[column] * right[row], tolerance: 2e-9
                )
            }

            let left = [vl[2 * column], vl[2 * column + 1]]
            let appliedAdjoint = [
                rowMajorOriginal[0].conjugate * left[0]
                    + rowMajorOriginal[2].conjugate * left[1],
                rowMajorOriginal[1].conjugate * left[0]
                    + rowMajorOriginal[3].conjugate * left[1],
            ]
            for row in 0..<2 {
                expectApproximatelyEqual(
                    appliedAdjoint[row], w[column].conjugate * left[row], tolerance: 2e-9
                )
            }
        }
    }

    @Test
    func generalEigenproblemArgumentIndicesFollowLAPACKE() {
        var realA = [Double](repeating: 0, count: 4)
        var wr = [Double](repeating: 0, count: 2)
        var wi = [Double](repeating: 0, count: 2)
        var realVL = [Double](repeating: 0, count: 4)
        var realVR = [Double](repeating: 0, count: 4)
        #expect(
            LAPACK.dgeev(
                layout: .rowMajor, jobVL: .none, jobVR: .none,
                n: 2, a: &realA, lda: 1,
                wr: &wr, wi: &wi,
                vl: &realVL, ldvl: 1,
                vr: &realVR, ldvr: 1
            ) == -6
        )

        var complexA = [Complex<Double>](repeating: .zero, count: 4)
        var values = [Complex<Double>](repeating: .zero, count: 2)
        var complexVL = [Complex<Double>](repeating: .zero, count: 4)
        var complexVR = [Complex<Double>](repeating: .zero, count: 4)
        #expect(
            LAPACK.zgeev(
                layout: .rowMajor, jobVL: .vectors, jobVR: .none,
                n: 2, a: &complexA, lda: 2, w: &values,
                vl: &complexVL, ldvl: 1,
                vr: &complexVR, ldvr: 1
            ) == -9
        )
        #expect(
            LAPACK.zgeev(
                layout: .rowMajor, jobVL: .none, jobVR: .vectors,
                n: 2, a: &complexA, lda: 2, w: &values,
                vl: &complexVL, ldvl: 1,
                vr: &complexVR, ldvr: 1
            ) == -11
        )
    }
}
