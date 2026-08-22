// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: MIT

import ComplexModule
import RealModule

private struct _LAPACKSchurResult<Scalar> {
    var form: [Scalar]
    var vectors: [Scalar]
    var converged: Bool
}

@inline(__always)
private func _lapackIdentity<Scalar: LAPACKScalar>(
    dimension: Int,
    as: Scalar.Type
) -> [Scalar] {
    var result = [Scalar](repeating: .zero, count: dimension * dimension)
    for index in 0..<dimension { result[index * dimension + index] = .one }
    return result
}

/// Returns the orthogonal or unitary factor of a square Householder QR
/// factorization. The explicit QR step is deliberately used by the portable
/// Schur solver: it is slower than an implicit Hessenberg QR iteration, but is
/// compact, stable for rank-deficient shifted matrices, and easy to audit.
private func _lapackHouseholderQ<Scalar: LAPACKScalar>(
    _ matrix: [Scalar],
    n: Int
) -> [Scalar] {
    var r = matrix
    var q = _lapackIdentity(dimension: n, as: Scalar.self)
    guard n > 1 else { return q }

    for column in 0..<(n - 1) {
        let count = n - column
        var reflector = [Scalar](repeating: .zero, count: count)
        var normSquared = Scalar.Magnitude.zero
        for index in 0..<count {
            let value = r[column * n + column + index]
            reflector[index] = value
            normSquared += value.normSquared
        }
        let norm = Scalar.Magnitude.sqrt(normSquared)
        if norm == .zero { continue }

        let leadingMagnitude = reflector[0].normSquared.squareRoot()
        let phase = leadingMagnitude == .zero
            ? Scalar.one
            : reflector[0].scaled(by: 1 / leadingMagnitude)
        reflector[0] -= phase.scaled(by: -norm)

        var reflectorNormSquared = Scalar.Magnitude.zero
        for value in reflector { reflectorNormSquared += value.normSquared }
        if reflectorNormSquared == .zero { continue }
        let factor: Scalar.Magnitude = 2 / reflectorNormSquared

        // R <- H R.
        for targetColumn in column..<n {
            var projection = Scalar.zero
            for index in 0..<count {
                projection +=
                    reflector[index].conjugate
                    * r[targetColumn * n + column + index]
            }
            projection = projection.scaled(by: factor)
            for index in 0..<count {
                r[targetColumn * n + column + index] -= reflector[index] * projection
            }
        }

        // Q <- Q H. Householder reflectors are Hermitian.
        for row in 0..<n {
            var projection = Scalar.zero
            for index in 0..<count {
                projection += q[(column + index) * n + row] * reflector[index]
            }
            projection = projection.scaled(by: factor)
            for index in 0..<count {
                q[(column + index) * n + row] -=
                    projection * reflector[index].conjugate
            }
        }
    }
    return q
}

/// Applies the embedded similarity `A <- Q^H A Q` and accumulates Schur
/// vectors as `Z <- Z Q`. `q` is column-major and acts on `range`.
private func _lapackApplySchurSimilarity<Scalar: LAPACKScalar>(
    matrix: inout [Scalar],
    vectors: inout [Scalar],
    n: Int,
    range: Range<Int>,
    q: [Scalar]
) {
    let count = range.count
    guard count > 0 else { return }
    let start = range.lowerBound

    var left = matrix
    for column in 0..<n {
        for localRow in 0..<count {
            var value = Scalar.zero
            for inner in 0..<count {
                value +=
                    q[localRow * count + inner].conjugate
                    * matrix[column * n + start + inner]
            }
            left[column * n + start + localRow] = value
        }
    }

    var transformed = left
    for localColumn in 0..<count {
        for row in 0..<n {
            var value = Scalar.zero
            for inner in 0..<count {
                value +=
                    left[(start + inner) * n + row]
                    * q[localColumn * count + inner]
            }
            transformed[(start + localColumn) * n + row] = value
        }
    }
    matrix = transformed

    var accumulated = vectors
    for localColumn in 0..<count {
        for row in 0..<n {
            var value = Scalar.zero
            for inner in 0..<count {
                value +=
                    vectors[(start + inner) * n + row]
                    * q[localColumn * count + inner]
            }
            accumulated[(start + localColumn) * n + row] = value
        }
    }
    vectors = accumulated
}

private func _lapackSchurScale<Scalar: LAPACKScalar>(
    _ matrix: [Scalar],
    stride: Int,
    range: Range<Int>
) -> Scalar.Magnitude {
    var scale: Scalar.Magnitude = 1
    for column in range {
        for row in range {
            scale = Swift.max(scale, matrix[column * stride + row].abs1)
        }
    }
    return scale
}

@inline(__always)
private func _lapackSchurComplexSquareRoot<RealType: Real>(
    _ value: Complex<RealType>
) -> Complex<RealType> {
    if value.imaginary == .zero {
        if value.real >= .zero {
            return Complex(RealType.sqrt(value.real), .zero)
        }
        return Complex(.zero, RealType.sqrt(-value.real))
    }
    let magnitude = value.length
    let real = RealType.sqrt(Swift.max(.zero, (magnitude + value.real) / 2))
    let imaginaryMagnitude = RealType.sqrt(
        Swift.max(.zero, (magnitude - value.real) / 2)
    )
    return Complex(
        real,
        value.imaginary < .zero ? -imaginaryMagnitude : imaginaryMagnitude
    )
}

private func _lapackComplexSchur<RealType: Real>(
    matrix: [Complex<RealType>],
    n: Int
) -> _LAPACKSchurResult<Complex<RealType>> {
    var form = matrix
    var vectors = _lapackIdentity(dimension: n, as: Complex<RealType>.self)
    guard n > 1 else {
        return .init(form: form, vectors: vectors, converged: true)
    }

    var activeCount = n
    var iterations = 0
    var iterationsSinceDeflation = 0
    let maximumIterations = Swift.max(512, 2048 * n)
    let relativeTolerance =
        RealType.ulpOfOne * RealType(Swift.max(1, 64 * n))

    while activeCount > 1 && iterations < maximumIterations {
        let last = activeCount - 1
        let activeRange = 0..<activeCount
        let scale = _lapackSchurScale(form, stride: n, range: activeRange)
        var lowerNormSquared = RealType.zero
        for column in 0..<last {
            lowerNormSquared += form[column * n + last].normSquared
        }
        if RealType.sqrt(lowerNormSquared) <= relativeTolerance * scale {
            for column in 0..<last { form[column * n + last] = .zero }
            activeCount -= 1
            iterationsSinceDeflation = 0
            continue
        }

        let first = activeCount - 2
        let a = form[first * n + first]
        let b = form[last * n + first]
        let c = form[first * n + last]
        let d = form[last * n + last]
        let discriminant = _lapackSchurComplexSquareRoot((a - d) * (a - d) + 4 * b * c)
        let firstEigenvalue = (a + d + discriminant) / 2
        let secondEigenvalue = (a + d - discriminant) / 2
        var shift =
            (firstEigenvalue - d).normSquared <= (secondEigenvalue - d).normSquared
            ? firstEigenvalue : secondEigenvalue

        // An occasional exceptional shift breaks rare exact cycles in the
        // deliberately simple explicit QR iteration.
        if iterationsSinceDeflation > 0 && iterationsSinceDeflation % 32 == 0 {
            let realOffset: RealType = 3 / 4
            let imaginaryOffset: RealType = 1 / 4
            shift = d + Complex(scale * realOffset, scale * imaginaryOffset)
        }

        var shifted = [Complex<RealType>](
            repeating: .zero, count: activeCount * activeCount
        )
        for column in 0..<activeCount {
            for row in 0..<activeCount {
                var value = form[column * n + row]
                if row == column { value -= shift }
                shifted[column * activeCount + row] = value
            }
        }
        let q = _lapackHouseholderQ(shifted, n: activeCount)
        _lapackApplySchurSimilarity(
            matrix: &form, vectors: &vectors, n: n,
            range: activeRange, q: q
        )
        iterations += 1
        iterationsSinceDeflation += 1
    }

    return .init(
        form: form,
        vectors: vectors,
        converged: activeCount <= 1
    )
}

private func _lapackRealSchurRotation<RealType: Real & LAPACKScalar>(
    a: RealType,
    b: RealType,
    c: RealType,
    d: RealType
) -> (cosine: RealType, sine: RealType, complexPair: Bool)
where RealType.Magnitude == RealType {
    let difference = a - d
    let discriminant = difference * difference + 4 * b * c
    if discriminant >= .zero {
        let root = RealType.sqrt(discriminant)
        let firstEigenvalue = (a + d + root) / 2
        let secondEigenvalue = (a + d - root) / 2
        let eigenvalue =
            (firstEigenvalue - d).magnitude <= (secondEigenvalue - d).magnitude
            ? firstEigenvalue : secondEigenvalue
        let x: RealType
        let y: RealType
        if b.magnitude >= c.magnitude {
            x = b
            y = eigenvalue - a
        } else {
            x = eigenvalue - d
            y = c
        }
        let norm = RealType.sqrt(x * x + y * y)
        if norm == .zero { return (1, 0, false) }
        return (x / norm, y / norm, false)
    }

    // Choose a plane rotation for which the two diagonal entries of the
    // transformed 2-by-2 block are equal. Its off-diagonal product is then
    // negative, which is LAPACK's standardized real Schur block.
    let sum = b + c
    let radius = RealType.sqrt(difference * difference + sum * sum)
    if radius == .zero { return (1, 0, true) }
    let cosineDoubleAngle = sum / radius
    let sineDoubleAngle = -difference / radius
    let cosine = RealType.sqrt(Swift.max(.zero, (1 + cosineDoubleAngle) / 2))
    var sine = RealType.sqrt(Swift.max(.zero, (1 - cosineDoubleAngle) / 2))
    if sineDoubleAngle < .zero { sine = -sine }
    return (cosine, sine, true)
}

private func _lapackRealSchur<RealType: Real & LAPACKScalar>(
    matrix: [RealType],
    n: Int
) -> _LAPACKSchurResult<RealType>
where RealType.Magnitude == RealType {
    var form = matrix
    var vectors = _lapackIdentity(dimension: n, as: RealType.self)
    guard n > 1 else {
        return .init(form: form, vectors: vectors, converged: true)
    }

    var activeCount = n
    var iterations = 0
    var iterationsSinceDeflation = 0
    let maximumIterations = Swift.max(512, 4096 * n)
    let relativeTolerance =
        RealType.ulpOfOne * RealType(Swift.max(1, 64 * n))

    while activeCount > 0 && iterations < maximumIterations {
        if activeCount == 1 {
            activeCount = 0
            break
        }

        let last = activeCount - 1
        let pairStart = activeCount - 2
        let activeRange = 0..<activeCount
        let scale = _lapackSchurScale(form, stride: n, range: activeRange)

        var lowerNormSquared = RealType.zero
        for column in 0..<last {
            let value = form[column * n + last]
            lowerNormSquared += value * value
        }
        if RealType.sqrt(lowerNormSquared) <= relativeTolerance * scale {
            for column in 0..<last { form[column * n + last] = .zero }
            activeCount -= 1
            iterationsSinceDeflation = 0
            continue
        }

        var pairLowerNormSquared = RealType.zero
        if pairStart > 0 {
            for column in 0..<pairStart {
                let upper = form[column * n + pairStart]
                let lower = form[column * n + last]
                pairLowerNormSquared += upper * upper + lower * lower
            }
        }
        if activeCount == 2
            || RealType.sqrt(pairLowerNormSquared) <= relativeTolerance * scale
        {
            if pairStart > 0 {
                for column in 0..<pairStart {
                    form[column * n + pairStart] = .zero
                    form[column * n + last] = .zero
                }
            }

            let rotation = _lapackRealSchurRotation(
                a: form[pairStart * n + pairStart],
                b: form[last * n + pairStart],
                c: form[pairStart * n + last],
                d: form[last * n + last]
            )
            let q: [RealType] = [
                rotation.cosine, rotation.sine,
                -rotation.sine, rotation.cosine,
            ]
            _lapackApplySchurSimilarity(
                matrix: &form, vectors: &vectors, n: n,
                range: pairStart..<activeCount, q: q
            )
            if !rotation.complexPair
                || form[pairStart * n + last].magnitude <= relativeTolerance * scale
            {
                form[pairStart * n + last] = .zero
            }
            activeCount -= 2
            iterationsSinceDeflation = 0
            continue
        }

        let a = form[pairStart * n + pairStart]
        let b = form[last * n + pairStart]
        let c = form[pairStart * n + last]
        let d = form[last * n + last]
        let difference = a - d
        let discriminant = difference * difference + 4 * b * c
        var shifted = [RealType](repeating: .zero, count: activeCount * activeCount)

        if iterationsSinceDeflation > 0 && iterationsSinceDeflation % 24 == 0 {
            let direction: RealType = (iterationsSinceDeflation / 24).isMultiple(of: 2)
                ? -7 / 16 : 3 / 4
            let shift = d + direction * scale
            for column in 0..<activeCount {
                for row in 0..<activeCount {
                    var value = form[column * n + row]
                    if row == column { value -= shift }
                    shifted[column * activeCount + row] = value
                }
            }
        } else if discriminant >= .zero {
            let root = RealType.sqrt(discriminant)
            let firstEigenvalue = (a + d + root) / 2
            let secondEigenvalue = (a + d - root) / 2
            let shift =
                (firstEigenvalue - d).magnitude <= (secondEigenvalue - d).magnitude
                ? firstEigenvalue : secondEigenvalue
            for column in 0..<activeCount {
                for row in 0..<activeCount {
                    var value = form[column * n + row]
                    if row == column { value -= shift }
                    shifted[column * activeCount + row] = value
                }
            }
        } else {
            // A real Francis double shift is an explicit QR factorization of
            // H^2 - trace(H22) H + det(H22) I.
            let trace = a + d
            let determinant = a * d - b * c
            for column in 0..<activeCount {
                for row in 0..<activeCount {
                    var value = RealType.zero
                    for inner in 0..<activeCount {
                        value +=
                            form[inner * n + row]
                            * form[column * n + inner]
                    }
                    value -= trace * form[column * n + row]
                    if row == column { value += determinant }
                    shifted[column * activeCount + row] = value
                }
            }
        }

        let q = _lapackHouseholderQ(shifted, n: activeCount)
        _lapackApplySchurSimilarity(
            matrix: &form, vectors: &vectors, n: n,
            range: activeRange, q: q
        )
        iterations += 1
        iterationsSinceDeflation += 1
    }

    return .init(
        form: form,
        vectors: vectors,
        converged: activeCount == 0
    )
}

private struct _LAPACKRealSchurBlock<RealType: Real> {
    var range: Range<Int>
    var values: [Complex<RealType>]
}

private func _lapackRealSchurBlocks<RealType: Real & LAPACKScalar>(
    form: [RealType],
    n: Int
) -> [_LAPACKRealSchurBlock<RealType>]
where RealType.Magnitude == RealType {
    let scale = _lapackSchurScale(form, stride: n, range: 0..<n)
    let tolerance =
        RealType.sqrt(RealType.ulpOfOne)
        * RealType(Swift.max(1, 8 * n)) * scale
    var blocks: [_LAPACKRealSchurBlock<RealType>] = []
    var index = 0
    while index < n {
        if index + 1 < n
            && form[index * n + index + 1].magnitude > tolerance
        {
            let a = form[index * n + index]
            let b = form[(index + 1) * n + index]
            let c = form[index * n + index + 1]
            let d = form[(index + 1) * n + index + 1]
            let center = (a + d) / 2
            let halfDifference = (a - d) / 2
            let discriminant = halfDifference * halfDifference + b * c
            let values: [Complex<RealType>]
            if discriminant < .zero {
                let imaginary = RealType.sqrt(-discriminant)
                values = [
                    Complex(center, imaginary),
                    Complex(center, -imaginary),
                ]
            } else {
                let root = RealType.sqrt(discriminant)
                values = [
                    Complex(center + root, .zero),
                    Complex(center - root, .zero),
                ]
            }
            blocks.append(.init(range: index..<(index + 2), values: values))
            index += 2
        } else {
            blocks.append(
                .init(
                    range: index..<(index + 1),
                    values: [Complex(form[index * n + index], .zero)]
                ))
            index += 1
        }
    }
    return blocks
}

private func _lapackAppendOrthonormalColumn<Scalar: LAPACKScalar>(
    _ source: [Scalar],
    to basis: inout [Scalar],
    n: Int,
    tolerance: Scalar.Magnitude
) -> Bool {
    var column = source
    let existingCount = basis.count / Swift.max(1, n)
    for _ in 0..<2 {
        for existing in 0..<existingCount {
            var projection = Scalar.zero
            for row in 0..<n {
                projection += basis[existing * n + row].conjugate * column[row]
            }
            for row in 0..<n {
                column[row] -= basis[existing * n + row] * projection
            }
        }
    }

    var normSquared = Scalar.Magnitude.zero
    for value in column { normSquared += value.normSquared }
    let norm = Scalar.Magnitude.sqrt(normSquared)
    if norm <= tolerance { return false }
    let inverseNorm = 1 / norm
    for row in 0..<n { basis.append(column[row].scaled(by: inverseNorm)) }
    return true
}

private func _lapackCompleteOrthonormalBasis<Scalar: LAPACKScalar>(
    _ basis: inout [Scalar],
    n: Int,
    tolerance: Scalar.Magnitude
) -> Bool {
    for candidate in 0..<n where basis.count / Swift.max(1, n) < n {
        var column = [Scalar](repeating: .zero, count: n)
        column[candidate] = .one
        _ = _lapackAppendOrthonormalColumn(
            column, to: &basis, n: n, tolerance: tolerance
        )
    }
    return basis.count == n * n
}

private func _lapackComplexSelectedBasis<RealType: Real>(
    matrix: [Complex<RealType>],
    n: Int,
    values: [Complex<RealType>],
    selected: [Bool]
) -> [Complex<RealType>]? {
    let tolerance =
        RealType.sqrt(RealType.ulpOfOne)
        * RealType(Swift.max(1, 32 * n))
    var basis: [Complex<RealType>] = []
    basis.reserveCapacity(n * n)
    for index in 0..<n where selected[index] {
        let vector = _lapackNormalizedNullVector(
            matrix: matrix, n: n, eigenvalue: values[index], left: false
        )
        if !_lapackAppendOrthonormalColumn(
            vector, to: &basis, n: n, tolerance: tolerance
        ) {
            return nil
        }
    }
    return _lapackCompleteOrthonormalBasis(
        &basis, n: n, tolerance: tolerance
    ) ? basis : nil
}

private func _lapackRealSelectedBasis<RealType: Real & LAPACKScalar>(
    matrix: [RealType],
    n: Int,
    blocks: [_LAPACKRealSchurBlock<RealType>],
    selected: [Bool]
) -> [RealType]?
where RealType.Magnitude == RealType {
    let tolerance =
        RealType.sqrt(RealType.ulpOfOne)
        * RealType(Swift.max(1, 32 * n))
    var complexMatrix = [Complex<RealType>](repeating: .zero, count: n * n)
    for index in complexMatrix.indices {
        complexMatrix[index] = Complex(matrix[index], .zero)
    }

    var basis: [RealType] = []
    basis.reserveCapacity(n * n)
    for blockIndex in blocks.indices where selected[blockIndex] {
        let block = blocks[blockIndex]
        let eigenvector = _lapackNormalizedNullVector(
            matrix: complexMatrix, n: n,
            eigenvalue: block.values[0], left: false
        )
        if block.range.count == 1 {
            var real = [RealType](repeating: .zero, count: n)
            var imaginary = [RealType](repeating: .zero, count: n)
            var realNormSquared = RealType.zero
            var imaginaryNormSquared = RealType.zero
            for row in 0..<n {
                real[row] = eigenvector[row].real
                imaginary[row] = eigenvector[row].imaginary
                realNormSquared += real[row] * real[row]
                imaginaryNormSquared += imaginary[row] * imaginary[row]
            }
            let candidate = realNormSquared >= imaginaryNormSquared ? real : imaginary
            if !_lapackAppendOrthonormalColumn(
                candidate, to: &basis, n: n, tolerance: tolerance
            ) {
                return nil
            }
        } else {
            var real = [RealType](repeating: .zero, count: n)
            var imaginary = [RealType](repeating: .zero, count: n)
            var realNormSquared = RealType.zero
            var imaginaryNormSquared = RealType.zero
            for row in 0..<n {
                real[row] = eigenvector[row].real
                imaginary[row] = eigenvector[row].imaginary
                realNormSquared += real[row] * real[row]
                imaginaryNormSquared += imaginary[row] * imaginary[row]
            }
            let first = realNormSquared >= imaginaryNormSquared ? real : imaginary
            let second = realNormSquared >= imaginaryNormSquared ? imaginary : real
            if !_lapackAppendOrthonormalColumn(
                first, to: &basis, n: n, tolerance: tolerance
            ) || !_lapackAppendOrthonormalColumn(
                second, to: &basis, n: n, tolerance: tolerance
            ) {
                return nil
            }
        }
    }
    return _lapackCompleteOrthonormalBasis(
        &basis, n: n, tolerance: tolerance
    ) ? basis : nil
}

private func _lapackExtractSquareBlock<Scalar>(
    _ matrix: [Scalar],
    stride: Int,
    range: Range<Int>
) -> [Scalar] {
    var result: [Scalar] = []
    result.reserveCapacity(range.count * range.count)
    for column in range {
        for row in range { result.append(matrix[column * stride + row]) }
    }
    return result
}

private func _lapackOverwriteSquareBlock<Scalar>(
    _ block: [Scalar],
    in matrix: inout [Scalar],
    stride: Int,
    range: Range<Int>
) {
    for localColumn in 0..<range.count {
        for localRow in 0..<range.count {
            matrix[(range.lowerBound + localColumn) * stride + range.lowerBound + localRow] =
                block[localColumn * range.count + localRow]
        }
    }
}

private func _lapackTransformToBasis<Scalar: LAPACKScalar>(
    matrix: [Scalar],
    basis: [Scalar],
    n: Int
) -> (matrix: [Scalar], vectors: [Scalar]) {
    var transformed = matrix
    var vectors = _lapackIdentity(dimension: n, as: Scalar.self)
    _lapackApplySchurSimilarity(
        matrix: &transformed, vectors: &vectors, n: n,
        range: 0..<n, q: basis
    )
    return (transformed, vectors)
}

private func _lapackSelectedSubspaceIsInvariant<Scalar: LAPACKScalar>(
    _ matrix: [Scalar],
    n: Int,
    selectedCount: Int
) -> Bool {
    guard selectedCount > 0 && selectedCount < n else { return true }
    var residualSquared = Scalar.Magnitude.zero
    for column in 0..<selectedCount {
        for row in selectedCount..<n {
            residualSquared += matrix[column * n + row].normSquared
        }
    }
    let scale = _lapackSchurScale(matrix, stride: n, range: 0..<n)
    let tolerance =
        Scalar.Magnitude.sqrt(Scalar.Magnitude.ulpOfOne)
        * Scalar.Magnitude(Swift.max(1, 256 * n)) * scale
    return Scalar.Magnitude.sqrt(residualSquared) <= tolerance
}

private func _lapackZeroSelectedSubspaceResidual<Scalar: LAPACKScalar>(
    _ matrix: inout [Scalar],
    n: Int,
    selectedCount: Int
) {
    guard selectedCount > 0 && selectedCount < n else { return }
    for column in 0..<selectedCount {
        for row in selectedCount..<n { matrix[column * n + row] = .zero }
    }
}

private func _lapackRefineComplexSchurBlocks<RealType: Real>(
    matrix: inout [Complex<RealType>],
    vectors: inout [Complex<RealType>],
    n: Int,
    split: Int
) -> Bool {
    var converged = true
    for range in [0..<split, split..<n] where range.count > 1 {
        let block = _lapackExtractSquareBlock(matrix, stride: n, range: range)
        let result = _lapackComplexSchur(matrix: block, n: range.count)
        _lapackApplySchurSimilarity(
            matrix: &matrix, vectors: &vectors, n: n,
            range: range, q: result.vectors
        )
        _lapackOverwriteSquareBlock(
            result.form, in: &matrix, stride: n, range: range
        )
        converged = converged && result.converged
    }
    return converged
}

private func _lapackRefineRealSchurBlocks<RealType: Real & LAPACKScalar>(
    matrix: inout [RealType],
    vectors: inout [RealType],
    n: Int,
    split: Int
) -> Bool where RealType.Magnitude == RealType {
    var converged = true
    for range in [0..<split, split..<n] where range.count > 1 {
        let block = _lapackExtractSquareBlock(matrix, stride: n, range: range)
        let result = _lapackRealSchur(matrix: block, n: range.count)
        _lapackApplySchurSimilarity(
            matrix: &matrix, vectors: &vectors, n: n,
            range: range, q: result.vectors
        )
        _lapackOverwriteSquareBlock(
            result.form, in: &matrix, stride: n, range: range
        )
        converged = converged && result.converged
    }
    return converged
}

@inline(__always)
private func _lapackGeesArgumentError(
    layout: LAPACK.Layout,
    jobVS: LAPACK.Eigenvectors,
    n: Int,
    lda: Int,
    ldvs: Int,
    ldvsArgument: Int
) -> Int? {
    if n < 0 { return -5 }
    if lda < Swift.max(1, n) { return -7 }
    let minimumLDVS = layout == .rowMajor
        ? Swift.max(1, n)
        : (jobVS == .vectors ? Swift.max(1, n) : 1)
    if ldvs < minimumLDVS { return -ldvsArgument }
    return nil
}

private func _lapackRealGees<RealType: Real & LAPACKScalar>(
    layout: LAPACK.Layout,
    jobVS: LAPACK.Eigenvectors,
    sort: LAPACK.SchurSort,
    selection: ((Complex<RealType>) -> Bool)?,
    n: Int,
    a: UnsafeMutablePointer<RealType>,
    lda: Int,
    sdim: UnsafeMutablePointer<Int>,
    wr: UnsafeMutablePointer<RealType>,
    wi: UnsafeMutablePointer<RealType>,
    vs: UnsafeMutablePointer<RealType>,
    ldvs: Int
) -> Int where RealType.Magnitude == RealType {
    if sort == .selected && selection == nil { return -4 }
    if let error = _lapackGeesArgumentError(
        layout: layout, jobVS: jobVS, n: n, lda: lda,
        ldvs: ldvs, ldvsArgument: 12
    ) { return error }

    var original = [RealType](repeating: .zero, count: n * n)
    for column in 0..<n {
        for row in 0..<n {
            original[column * n + row] = a[_matrixIndex(layout, row, column, lda)]
        }
    }

    var result = _lapackRealSchur(matrix: original, n: n)
    var info = result.converged ? 0 : 1
    var selectedCount = 0
    if sort == .selected, result.converged, let selection {
        let blocks = _lapackRealSchurBlocks(form: result.form, n: n)
        var selected = [Bool](repeating: false, count: blocks.count)
        for index in blocks.indices {
            selected[index] = blocks[index].values.contains(where: selection)
            if selected[index] { selectedCount += blocks[index].range.count }
        }

        if selectedCount > 0 && selectedCount < n {
            if let basis = _lapackRealSelectedBasis(
                matrix: original, n: n, blocks: blocks, selected: selected
            ) {
                var transformed = _lapackTransformToBasis(
                    matrix: original, basis: basis, n: n
                )
                if _lapackSelectedSubspaceIsInvariant(
                    transformed.matrix, n: n, selectedCount: selectedCount
                ) {
                    _lapackZeroSelectedSubspaceResidual(
                        &transformed.matrix, n: n, selectedCount: selectedCount
                    )
                    let converged = _lapackRefineRealSchurBlocks(
                        matrix: &transformed.matrix,
                        vectors: &transformed.vectors,
                        n: n, split: selectedCount
                    )
                    result = .init(
                        form: transformed.matrix,
                        vectors: transformed.vectors,
                        converged: converged
                    )
                    info = converged ? 0 : 1
                } else {
                    info = n + 1
                }
            } else {
                info = n + 1
            }
        }
    }
    sdim.pointee = sort == .selected ? selectedCount : 0

    let blocks = _lapackRealSchurBlocks(form: result.form, n: n)
    let values = blocks.flatMap(\.values)
    for index in 0..<n {
        wr[index] = values[index].real
        wi[index] = values[index].imaginary
    }
    for column in 0..<n {
        for row in 0..<n {
            a[_matrixIndex(layout, row, column, lda)] = result.form[column * n + row]
            if jobVS == .vectors {
                vs[_matrixIndex(layout, row, column, ldvs)] =
                    result.vectors[column * n + row]
            }
        }
    }
    return info
}

private func _lapackComplexGees<RealType: Real>(
    layout: LAPACK.Layout,
    jobVS: LAPACK.Eigenvectors,
    sort: LAPACK.SchurSort,
    selection: ((Complex<RealType>) -> Bool)?,
    n: Int,
    a: UnsafeMutablePointer<Complex<RealType>>,
    lda: Int,
    sdim: UnsafeMutablePointer<Int>,
    w: UnsafeMutablePointer<Complex<RealType>>,
    vs: UnsafeMutablePointer<Complex<RealType>>,
    ldvs: Int
) -> Int {
    if sort == .selected && selection == nil { return -4 }
    if let error = _lapackGeesArgumentError(
        layout: layout, jobVS: jobVS, n: n, lda: lda,
        ldvs: ldvs, ldvsArgument: 11
    ) { return error }

    var original = [Complex<RealType>](repeating: .zero, count: n * n)
    for column in 0..<n {
        for row in 0..<n {
            original[column * n + row] = a[_matrixIndex(layout, row, column, lda)]
        }
    }

    var result = _lapackComplexSchur(matrix: original, n: n)
    var info = result.converged ? 0 : 1
    var selectedCount = 0
    if sort == .selected, result.converged, let selection {
        let values = (0..<n).map { result.form[$0 * n + $0] }
        let selected = values.map(selection)
        selectedCount = selected.reduce(into: 0) { $0 += $1 ? 1 : 0 }
        if selectedCount > 0 && selectedCount < n {
            if let basis = _lapackComplexSelectedBasis(
                matrix: original, n: n, values: values, selected: selected
            ) {
                var transformed = _lapackTransformToBasis(
                    matrix: original, basis: basis, n: n
                )
                if _lapackSelectedSubspaceIsInvariant(
                    transformed.matrix, n: n, selectedCount: selectedCount
                ) {
                    _lapackZeroSelectedSubspaceResidual(
                        &transformed.matrix, n: n, selectedCount: selectedCount
                    )
                    let converged = _lapackRefineComplexSchurBlocks(
                        matrix: &transformed.matrix,
                        vectors: &transformed.vectors,
                        n: n, split: selectedCount
                    )
                    result = .init(
                        form: transformed.matrix,
                        vectors: transformed.vectors,
                        converged: converged
                    )
                    info = converged ? 0 : 1
                } else {
                    info = n + 1
                }
            } else {
                info = n + 1
            }
        }
    }
    sdim.pointee = sort == .selected ? selectedCount : 0

    for index in 0..<n { w[index] = result.form[index * n + index] }
    for column in 0..<n {
        for row in 0..<n {
            a[_matrixIndex(layout, row, column, lda)] = result.form[column * n + row]
            if jobVS == .vectors {
                vs[_matrixIndex(layout, row, column, ldvs)] =
                    result.vectors[column * n + row]
            }
        }
    }
    return info
}

internal func _lapackSgees(
    layout: LAPACK.Layout, jobVS: LAPACK.Eigenvectors, sort: LAPACK.SchurSort,
    select: LAPACK.SgeesSelect?, n: Int,
    a: UnsafeMutablePointer<Float>, lda: Int, sdim: UnsafeMutablePointer<Int>,
    wr: UnsafeMutablePointer<Float>, wi: UnsafeMutablePointer<Float>,
    vs: UnsafeMutablePointer<Float>, ldvs: Int
) -> Int {
    let selection: ((Complex<Float>) -> Bool)? = select.map { select in
        { value in
            var real = value.real
            var imaginary = value.imaginary
            #if canImport(Accelerate)
            return withUnsafeMutablePointer(to: &real) { real in
                withUnsafeMutablePointer(to: &imaginary) { imaginary in
                    select(real, imaginary) != 0
                }
            }
            #else
            return withUnsafePointer(to: &real) { real in
                withUnsafePointer(to: &imaginary) { imaginary in
                    select(real, imaginary) != 0
                }
            }
            #endif
        }
    }
    return _lapackRealGees(
        layout: layout, jobVS: jobVS, sort: sort, selection: selection,
        n: n, a: a, lda: lda, sdim: sdim, wr: wr, wi: wi,
        vs: vs, ldvs: ldvs
    )
}

internal func _lapackDgees(
    layout: LAPACK.Layout, jobVS: LAPACK.Eigenvectors, sort: LAPACK.SchurSort,
    select: LAPACK.DgeesSelect?, n: Int,
    a: UnsafeMutablePointer<Double>, lda: Int, sdim: UnsafeMutablePointer<Int>,
    wr: UnsafeMutablePointer<Double>, wi: UnsafeMutablePointer<Double>,
    vs: UnsafeMutablePointer<Double>, ldvs: Int
) -> Int {
    let selection: ((Complex<Double>) -> Bool)? = select.map { select in
        { value in
            var real = value.real
            var imaginary = value.imaginary
            #if canImport(Accelerate)
            return withUnsafeMutablePointer(to: &real) { real in
                withUnsafeMutablePointer(to: &imaginary) { imaginary in
                    select(real, imaginary) != 0
                }
            }
            #else
            return withUnsafePointer(to: &real) { real in
                withUnsafePointer(to: &imaginary) { imaginary in
                    select(real, imaginary) != 0
                }
            }
            #endif
        }
    }
    return _lapackRealGees(
        layout: layout, jobVS: jobVS, sort: sort, selection: selection,
        n: n, a: a, lda: lda, sdim: sdim, wr: wr, wi: wi,
        vs: vs, ldvs: ldvs
    )
}

internal func _lapackCgees(
    layout: LAPACK.Layout, jobVS: LAPACK.Eigenvectors, sort: LAPACK.SchurSort,
    select: LAPACK.CgeesSelect?, n: Int,
    a: UnsafeMutablePointer<Complex<Float>>, lda: Int,
    sdim: UnsafeMutablePointer<Int>, w: UnsafeMutablePointer<Complex<Float>>,
    vs: UnsafeMutablePointer<Complex<Float>>, ldvs: Int
) -> Int {
    let selection: ((Complex<Float>) -> Bool)? = select.map { select in
        { value in
            var value = value
            return withUnsafePointer(to: &value) { value in
                #if canImport(Accelerate)
                select(OpaquePointer(value)) != 0
                #else
                select(UnsafeRawPointer(value)) != 0
                #endif
            }
        }
    }
    return _lapackComplexGees(
        layout: layout, jobVS: jobVS, sort: sort, selection: selection,
        n: n, a: a, lda: lda, sdim: sdim, w: w, vs: vs, ldvs: ldvs
    )
}

internal func _lapackZgees(
    layout: LAPACK.Layout, jobVS: LAPACK.Eigenvectors, sort: LAPACK.SchurSort,
    select: LAPACK.ZgeesSelect?, n: Int,
    a: UnsafeMutablePointer<Complex<Double>>, lda: Int,
    sdim: UnsafeMutablePointer<Int>, w: UnsafeMutablePointer<Complex<Double>>,
    vs: UnsafeMutablePointer<Complex<Double>>, ldvs: Int
) -> Int {
    let selection: ((Complex<Double>) -> Bool)? = select.map { select in
        { value in
            var value = value
            return withUnsafePointer(to: &value) { value in
                #if canImport(Accelerate)
                select(OpaquePointer(value)) != 0
                #else
                select(UnsafeRawPointer(value)) != 0
                #endif
            }
        }
    }
    return _lapackComplexGees(
        layout: layout, jobVS: jobVS, sort: sort, selection: selection,
        n: n, a: a, lda: lda, sdim: sdim, w: w, vs: vs, ldvs: ldvs
    )
}
