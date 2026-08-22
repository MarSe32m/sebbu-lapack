import ComplexModule
import SebbuLAPACK
import Testing

private let selectPositiveImaginaryDouble: LAPACK.DgeesSelect = { _, imaginary in
    guard let imaginary else { return 0 }
    return imaginary.pointee > 0 ? 1 : 0
}

private let selectNegativeComplexFloat: LAPACK.CgeesSelect = { pointer in
    guard let pointer else { return 0 }
    #if canImport(Accelerate)
    let value = UnsafeRawPointer(pointer).load(as: Complex<Float>.self)
    #else
    let value = pointer.load(as: Complex<Float>.self)
    #endif
    return value.real < 0 ? 1 : 0
}

private func columnMajor<T>(_ rowMajor: [T], n: Int) -> [T] {
    var result: [T] = []
    result.reserveCapacity(n * n)
    for column in 0..<n {
        for row in 0..<n { result.append(rowMajor[row * n + column]) }
    }
    return result
}

private func rowMajor<T>(
    _ matrix: [T],
    layout: LAPACK.Layout,
    n: Int,
    leadingDimension: Int
) -> [T] {
    var result: [T] = []
    result.reserveCapacity(n * n)
    for row in 0..<n {
        for column in 0..<n {
            switch layout {
            case .rowMajor:
                result.append(matrix[row * leadingDimension + column])
            case .columnMajor:
                result.append(matrix[column * leadingDimension + row])
            }
        }
    }
    return result
}

private func multiply(_ lhs: [Double], _ rhs: [Double], n: Int) -> [Double] {
    var result = [Double](repeating: 0, count: n * n)
    for row in 0..<n {
        for column in 0..<n {
            for inner in 0..<n {
                result[row * n + column] +=
                    lhs[row * n + inner] * rhs[inner * n + column]
            }
        }
    }
    return result
}

private func transpose(_ matrix: [Double], n: Int) -> [Double] {
    var result = [Double](repeating: 0, count: n * n)
    for row in 0..<n {
        for column in 0..<n {
            result[row * n + column] = matrix[column * n + row]
        }
    }
    return result
}

private func multiply<RealType>(
    _ lhs: [Complex<RealType>],
    _ rhs: [Complex<RealType>],
    n: Int
) -> [Complex<RealType>] {
    var result = [Complex<RealType>](repeating: .zero, count: n * n)
    for row in 0..<n {
        for column in 0..<n {
            for inner in 0..<n {
                result[row * n + column] +=
                    lhs[row * n + inner] * rhs[inner * n + column]
            }
        }
    }
    return result
}

private func adjoint<RealType>(
    _ matrix: [Complex<RealType>],
    n: Int
) -> [Complex<RealType>] {
    var result = [Complex<RealType>](repeating: .zero, count: n * n)
    for row in 0..<n {
        for column in 0..<n {
            result[row * n + column] = matrix[column * n + row].conjugate
        }
    }
    return result
}

private func expectOrthogonal(_ matrix: [Double], n: Int, tolerance: Double) {
    let product = multiply(transpose(matrix, n: n), matrix, n: n)
    var identity = [Double](repeating: 0, count: n * n)
    for index in 0..<n { identity[index * n + index] = 1 }
    expectApproximatelyEqual(product, identity, tolerance: tolerance)
}

private func expectUnitary(
    _ matrix: [Complex<Double>],
    n: Int,
    tolerance: Double
) {
    let product = multiply(adjoint(matrix, n: n), matrix, n: n)
    var identity = [Complex<Double>](repeating: .zero, count: n * n)
    for index in 0..<n { identity[index * n + index] = .one }
    expectApproximatelyEqual(product, identity, tolerance: tolerance)
}

private func expectUnitary(
    _ matrix: [Complex<Float>],
    n: Int,
    tolerance: Float
) {
    let product = multiply(adjoint(matrix, n: n), matrix, n: n)
    var identity = [Complex<Float>](repeating: .zero, count: n * n)
    for index in 0..<n { identity[index * n + index] = .one }
    expectApproximatelyEqual(product, identity, tolerance: tolerance)
}

@Suite("LAPACK general Schur decompositions")
struct SchurDecompositionTests {
    @Test
    func realDoubleColumnMajorReconstructsDenseMatrix() {
        let originalRowMajor = [
            1.0, -3, 2,
            3, 1, -1,
            1, 2, 4,
        ]
        var a = columnMajor(originalRowMajor, n: 3)
        var wr = [Double](repeating: 0, count: 3)
        var wi = [Double](repeating: 0, count: 3)
        var vs = [Double](repeating: 0, count: 9)
        var sdim = -1

        #expect(
            LAPACK.dgees(
                layout: .columnMajor, jobVS: .vectors, sort: .none,
                n: 3, a: &a, lda: 3, sdim: &sdim,
                wr: &wr, wi: &wi, vs: &vs, ldvs: 3
            ) == 0
        )
        #expect(sdim == 0)

        let t = rowMajor(a, layout: .columnMajor, n: 3, leadingDimension: 3)
        let q = rowMajor(vs, layout: .columnMajor, n: 3, leadingDimension: 3)
        let reconstructed = multiply(multiply(q, t, n: 3), transpose(q, n: 3), n: 3)
        expectApproximatelyEqual(reconstructed, originalRowMajor, tolerance: 2e-9)
        expectOrthogonal(q, n: 3, tolerance: 2e-10)
        #expect(t[6].magnitude < 2e-10)

        let complexCount = wi.count(where: { $0.magnitude > 1e-8 })
        #expect(complexCount == 2)
        #expect(wi.reduce(0, +).magnitude < 2e-9)
        #expect((wr.reduce(0, +) - 6).magnitude < 2e-9)
    }

    @Test
    func realSingleRowMajorWithoutSchurVectors() {
        var a: [Float] = [
            1, 4, -2,
            0, -2, 3,
            0, 0, 5,
        ]
        var wr = [Float](repeating: 0, count: 3)
        var wi = [Float](repeating: 0, count: 3)
        var unusedVS = [Float](repeating: 0, count: 9)
        var sdim = -1

        #expect(
            LAPACK.sgees(
                layout: .rowMajor, jobVS: .none, sort: .none,
                n: 3, a: &a, lda: 3, sdim: &sdim,
                wr: &wr, wi: &wi, vs: &unusedVS, ldvs: 3
            ) == 0
        )
        #expect(sdim == 0)
        expectApproximatelyEqual(wr.sorted(), [-2, 1, 5], tolerance: 5e-4)
        expectApproximatelyEqual(wi, [0, 0, 0], tolerance: 5e-4)
        #expect(a[3].magnitude < 5e-4)
        #expect(a[6].magnitude < 5e-4)
        #expect(a[7].magnitude < 5e-4)
    }

    @Test
    func realSortingSelectsWholeConjugatePair() {
        let original: [Double] = [
            4, 1, -0.5,
            0, -1, -2,
            0, 2, -1,
        ]
        var a = original
        var wr = [Double](repeating: 0, count: 3)
        var wi = [Double](repeating: 0, count: 3)
        var vs = [Double](repeating: 0, count: 9)
        var sdim = 0

        #expect(
            LAPACK.dgees(
                layout: .rowMajor, jobVS: .vectors, sort: .selected,
                select: selectPositiveImaginaryDouble,
                n: 3, a: &a, lda: 3, sdim: &sdim,
                wr: &wr, wi: &wi, vs: &vs, ldvs: 3
            ) == 0
        )
        #expect(sdim == 2)
        expectApproximatelyEqual(Array(wr.prefix(2)), [-1, -1], tolerance: 2e-10)
        #expect(wi[0] > 0)
        #expect(wi[1] < 0)
        #expect((wi[0] + wi[1]).magnitude < 2e-10)
        #expect((wr[2] - 4).magnitude < 2e-10)

        let reconstructed = multiply(multiply(vs, a, n: 3), transpose(vs, n: 3), n: 3)
        expectApproximatelyEqual(reconstructed, original, tolerance: 2e-9)
        expectOrthogonal(vs, n: 3, tolerance: 2e-10)
        #expect(a[6].magnitude < 2e-10)
    }

    @Test
    func complexDoubleColumnMajorReconstructsDenseMatrix() {
        let originalRowMajor: [Complex<Double>] = [
            Complex(1, 1), Complex(2, -1), Complex(-0.5, 0),
            Complex(0.25, 0.5), Complex(-2, 0.25), Complex(1, 1),
            Complex(1, 0), Complex(0.5, -0.5), Complex(3, -2),
        ]
        var a = columnMajor(originalRowMajor, n: 3)
        var w = [Complex<Double>](repeating: .zero, count: 3)
        var vs = [Complex<Double>](repeating: .zero, count: 9)
        var sdim = -1

        #expect(
            LAPACK.zgees(
                layout: .columnMajor, jobVS: .vectors, sort: .none,
                n: 3, a: &a, lda: 3, sdim: &sdim,
                w: &w, vs: &vs, ldvs: 3
            ) == 0
        )
        #expect(sdim == 0)

        let t = rowMajor(a, layout: .columnMajor, n: 3, leadingDimension: 3)
        let q = rowMajor(vs, layout: .columnMajor, n: 3, leadingDimension: 3)
        let reconstructed = multiply(multiply(q, t, n: 3), adjoint(q, n: 3), n: 3)
        expectApproximatelyEqual(reconstructed, originalRowMajor, tolerance: 5e-9)
        expectUnitary(q, n: 3, tolerance: 5e-10)
        for row in 1..<3 {
            for column in 0..<row {
                expectApproximatelyEqual(t[row * 3 + column], .zero, tolerance: 5e-9)
            }
        }
        for index in 0..<3 {
            expectApproximatelyEqual(w[index], t[index * 3 + index], tolerance: 5e-9)
        }
    }

    @Test
    func complexSingleRowMajorSortsSelectedEigenvalue() {
        let original: [Complex<Float>] = [
            Complex(2, 1), Complex(0.75, -0.25), Complex(-0.5, 0.2),
            .zero, Complex(-3, 0.5), Complex(1, 1),
            .zero, .zero, Complex(1, -2),
        ]
        var a = original
        var w = [Complex<Float>](repeating: .zero, count: 3)
        var vs = [Complex<Float>](repeating: .zero, count: 9)
        var sdim = 0

        #expect(
            LAPACK.cgees(
                layout: .rowMajor, jobVS: .vectors, sort: .selected,
                select: selectNegativeComplexFloat,
                n: 3, a: &a, lda: 3, sdim: &sdim,
                w: &w, vs: &vs, ldvs: 3
            ) == 0
        )
        #expect(sdim == 1)
        expectApproximatelyEqual(w[0], Complex(-3, 0.5), tolerance: 2e-4)
        #expect(w[1].real >= 0)
        #expect(w[2].real >= 0)

        let reconstructed = multiply(multiply(vs, a, n: 3), adjoint(vs, n: 3), n: 3)
        expectApproximatelyEqual(reconstructed, original, tolerance: 8e-4)
        expectUnitary(vs, n: 3, tolerance: 5e-4)
    }

    @Test
    func emptySchurDecompositionsSucceed() {
        var realA = [Double.zero]
        var wr = [Double.zero]
        var wi = [Double.zero]
        var realVS = [Double.zero]
        var realSDim = -1
        #expect(
            LAPACK.dgees(
                layout: .rowMajor, jobVS: .vectors, sort: .none,
                n: 0, a: &realA, lda: 1, sdim: &realSDim,
                wr: &wr, wi: &wi, vs: &realVS, ldvs: 1
            ) == 0
        )
        #expect(realSDim == 0)

        var complexA = [Complex<Double>.zero]
        var w = [Complex<Double>.zero]
        var complexVS = [Complex<Double>.zero]
        var complexSDim = -1
        #expect(
            LAPACK.zgees(
                layout: .columnMajor, jobVS: .none, sort: .none,
                n: 0, a: &complexA, lda: 1, sdim: &complexSDim,
                w: &w, vs: &complexVS, ldvs: 1
            ) == 0
        )
        #expect(complexSDim == 0)
    }

    @Test
    func schurArgumentIndicesFollowLAPACKE() {
        var a = [Double](repeating: 0, count: 4)
        var wr = [Double](repeating: 0, count: 2)
        var wi = [Double](repeating: 0, count: 2)
        var vs = [Double](repeating: 0, count: 4)
        var sdim = 0

        #expect(
            LAPACK.dgees(
                layout: .rowMajor, jobVS: .none, sort: .selected,
                select: nil, n: 2, a: &a, lda: 2, sdim: &sdim,
                wr: &wr, wi: &wi, vs: &vs, ldvs: 1
            ) == -4
        )
        #expect(
            LAPACK.dgees(
                layout: .rowMajor, jobVS: .none, sort: .none,
                n: -1, a: &a, lda: 1, sdim: &sdim,
                wr: &wr, wi: &wi, vs: &vs, ldvs: 1
            ) == -5
        )
        #expect(
            LAPACK.dgees(
                layout: .rowMajor, jobVS: .none, sort: .none,
                n: 2, a: &a, lda: 1, sdim: &sdim,
                wr: &wr, wi: &wi, vs: &vs, ldvs: 1
            ) == -7
        )
        #expect(
            LAPACK.dgees(
                layout: .rowMajor, jobVS: .vectors, sort: .none,
                n: 2, a: &a, lda: 2, sdim: &sdim,
                wr: &wr, wi: &wi, vs: &vs, ldvs: 1
            ) == -12
        )

        var complexA = [Complex<Double>](repeating: .zero, count: 4)
        var w = [Complex<Double>](repeating: .zero, count: 2)
        var complexVS = [Complex<Double>](repeating: .zero, count: 4)
        #expect(
            LAPACK.zgees(
                layout: .columnMajor, jobVS: .vectors, sort: .none,
                n: 2, a: &complexA, lda: 2, sdim: &sdim,
                w: &w, vs: &complexVS, ldvs: 1
            ) == -11
        )
    }
}
