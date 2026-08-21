// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: MIT

import ComplexModule
import RealModule
import SebbuBLAS

@inlinable
internal func _lapackMinimumLeadingDimension(
    layout: LAPACK.Layout,
    rows: Int,
    columns: Int
) -> Int {
    switch layout {
    case .rowMajor: Swift.max(1, columns)
    case .columnMajor: Swift.max(1, rows)
    }
}

@inlinable
internal func _lapackElement<T: LAPACKScalar>(
    _ a: UnsafePointer<T>,
    layout: LAPACK.Layout,
    transpose: LAPACK.Transpose,
    row: Int,
    column: Int,
    leadingDimension: Int
) -> T {
    switch transpose {
    case .noTranspose:
        return a[_matrixIndex(layout, row, column, leadingDimension)]
    case .transpose:
        return a[_matrixIndex(layout, column, row, leadingDimension)]
    case .conjugateTranspose:
        return a[_matrixIndex(layout, column, row, leadingDimension)].conjugate
    }
}

@inlinable
internal func _lapackGetrf<T: LAPACKScalar>(
    layout: LAPACK.Layout,
    m: Int,
    n: Int,
    a: UnsafeMutablePointer<T>,
    lda: Int,
    ipiv: UnsafeMutablePointer<Int>
) -> Int {
    guard m >= 0 else { return -2 }
    guard n >= 0 else { return -3 }
    guard
        lda
            >= _lapackMinimumLeadingDimension(
                layout: layout, rows: m, columns: n
            )
    else { return -5 }

    let factorCount = Swift.min(m, n)
    var firstZeroPivot = 0
    for column in 0..<factorCount {
        var pivot = column
        var pivotMagnitude = a[
            _matrixIndex(layout, column, column, lda)
        ].normSquared
        if column + 1 < m {
            for row in (column + 1)..<m {
                let magnitude = a[
                    _matrixIndex(layout, row, column, lda)
                ].normSquared
                if magnitude > pivotMagnitude {
                    pivotMagnitude = magnitude
                    pivot = row
                }
            }
        }
        ipiv[column] = pivot + 1

        if pivot != column {
            for trailingColumn in 0..<n {
                let first = _matrixIndex(layout, column, trailingColumn, lda)
                let second = _matrixIndex(layout, pivot, trailingColumn, lda)
                let temporary = a[first]
                a[first] = a[second]
                a[second] = temporary
            }
        }

        let diagonalIndex = _matrixIndex(layout, column, column, lda)
        let diagonal = a[diagonalIndex]
        if diagonal == .zero {
            if firstZeroPivot == 0 { firstZeroPivot = column + 1 }
            continue
        }

        if column + 1 < m {
            for row in (column + 1)..<m {
                let index = _matrixIndex(layout, row, column, lda)
                a[index] /= diagonal
            }
        }
        guard column + 1 < m && column + 1 < n else { continue }
        for trailingColumn in (column + 1)..<n {
            let upper = a[
                _matrixIndex(
                    layout, column, trailingColumn, lda
                )]
            if upper == .zero { continue }
            for row in (column + 1)..<m {
                let index = _matrixIndex(layout, row, trailingColumn, lda)
                a[index] -= a[_matrixIndex(layout, row, column, lda)] * upper
            }
        }
    }
    return firstZeroPivot
}

@inlinable
internal func _lapackApplyPivots<T: LAPACKScalar>(
    layout: LAPACK.Layout,
    n: Int,
    nrhs: Int,
    ipiv: UnsafePointer<Int>,
    b: UnsafeMutablePointer<T>,
    ldb: Int,
    inverse: Bool
) {
    guard n > 0 && nrhs > 0 else { return }
    if inverse {
        for row in stride(from: n - 1, through: 0, by: -1) {
            let pivot = ipiv[row] - 1
            if pivot != row {
                for rhs in 0..<nrhs {
                    let first = _matrixIndex(layout, row, rhs, ldb)
                    let second = _matrixIndex(layout, pivot, rhs, ldb)
                    let temporary = b[first]
                    b[first] = b[second]
                    b[second] = temporary
                }
            }
        }
    } else {
        for row in 0..<n {
            let pivot = ipiv[row] - 1
            if pivot != row {
                for rhs in 0..<nrhs {
                    let first = _matrixIndex(layout, row, rhs, ldb)
                    let second = _matrixIndex(layout, pivot, rhs, ldb)
                    let temporary = b[first]
                    b[first] = b[second]
                    b[second] = temporary
                }
            }
        }
    }
}

@inlinable
internal func _lapackTrtrs<T: LAPACKScalar>(
    layout: LAPACK.Layout,
    triangle: LAPACK.Triangle,
    transpose: LAPACK.Transpose,
    diagonal: LAPACK.Diagonal,
    n: Int,
    nrhs: Int,
    a: UnsafePointer<T>,
    lda: Int,
    b: UnsafeMutablePointer<T>,
    ldb: Int
) -> Int {
    guard n >= 0 else { return -5 }
    guard nrhs >= 0 else { return -6 }
    guard
        lda
            >= _lapackMinimumLeadingDimension(
                layout: layout, rows: n, columns: n
            )
    else { return -8 }
    guard
        ldb
            >= _lapackMinimumLeadingDimension(
                layout: layout, rows: n, columns: nrhs
            )
    else { return -10 }

    if diagonal == .nonUnit {
        for index in 0..<n where a[_matrixIndex(layout, index, index, lda)] == .zero {
            return index + 1
        }
    }

    let effectiveUpper: Bool
    switch transpose {
    case .noTranspose: effectiveUpper = triangle == .upper
    case .transpose, .conjugateTranspose: effectiveUpper = triangle == .lower
    }

    for rhs in 0..<nrhs {
        if effectiveUpper {
            guard n > 0 else { continue }
            for row in stride(from: n - 1, through: 0, by: -1) {
                var value = b[_matrixIndex(layout, row, rhs, ldb)]
                if row + 1 < n {
                    for column in (row + 1)..<n {
                        value -=
                            _lapackElement(
                                a, layout: layout, transpose: transpose,
                                row: row, column: column,
                                leadingDimension: lda
                            ) * b[_matrixIndex(layout, column, rhs, ldb)]
                    }
                }
                if diagonal == .nonUnit {
                    value /= _lapackElement(
                        a, layout: layout, transpose: transpose,
                        row: row, column: row, leadingDimension: lda
                    )
                }
                b[_matrixIndex(layout, row, rhs, ldb)] = value
            }
        } else {
            for row in 0..<n {
                var value = b[_matrixIndex(layout, row, rhs, ldb)]
                if row > 0 {
                    for column in 0..<row {
                        value -=
                            _lapackElement(
                                a, layout: layout, transpose: transpose,
                                row: row, column: column,
                                leadingDimension: lda
                            ) * b[_matrixIndex(layout, column, rhs, ldb)]
                    }
                }
                if diagonal == .nonUnit {
                    value /= _lapackElement(
                        a, layout: layout, transpose: transpose,
                        row: row, column: row, leadingDimension: lda
                    )
                }
                b[_matrixIndex(layout, row, rhs, ldb)] = value
            }
        }
    }
    return 0
}

@inlinable
internal func _lapackGetrs<T: LAPACKScalar>(
    layout: LAPACK.Layout,
    transpose: LAPACK.Transpose,
    n: Int,
    nrhs: Int,
    a: UnsafePointer<T>,
    lda: Int,
    ipiv: UnsafePointer<Int>,
    b: UnsafeMutablePointer<T>,
    ldb: Int
) -> Int {
    guard n >= 0 else { return -3 }
    guard nrhs >= 0 else { return -4 }
    guard
        lda
            >= _lapackMinimumLeadingDimension(
                layout: layout, rows: n, columns: n
            )
    else { return -6 }
    guard
        ldb
            >= _lapackMinimumLeadingDimension(
                layout: layout, rows: n, columns: nrhs
            )
    else { return -9 }

    if transpose == .noTranspose {
        _lapackApplyPivots(
            layout: layout, n: n, nrhs: nrhs, ipiv: ipiv,
            b: b, ldb: ldb, inverse: false
        )
        let lowerInfo = _lapackTrtrs(
            layout: layout, triangle: .lower, transpose: .noTranspose,
            diagonal: .unit, n: n, nrhs: nrhs,
            a: a, lda: lda, b: b, ldb: ldb
        )
        if lowerInfo != 0 { return lowerInfo }
        return _lapackTrtrs(
            layout: layout, triangle: .upper, transpose: .noTranspose,
            diagonal: .nonUnit, n: n, nrhs: nrhs,
            a: a, lda: lda, b: b, ldb: ldb
        )
    }

    let upperInfo = _lapackTrtrs(
        layout: layout, triangle: .upper, transpose: transpose,
        diagonal: .nonUnit, n: n, nrhs: nrhs,
        a: a, lda: lda, b: b, ldb: ldb
    )
    if upperInfo != 0 { return upperInfo }
    let lowerInfo = _lapackTrtrs(
        layout: layout, triangle: .lower, transpose: transpose,
        diagonal: .unit, n: n, nrhs: nrhs,
        a: a, lda: lda, b: b, ldb: ldb
    )
    if lowerInfo != 0 { return lowerInfo }
    _lapackApplyPivots(
        layout: layout, n: n, nrhs: nrhs, ipiv: ipiv,
        b: b, ldb: ldb, inverse: true
    )
    return 0
}

@inlinable
internal func _lapackGesv<T: LAPACKScalar>(
    layout: LAPACK.Layout,
    n: Int,
    nrhs: Int,
    a: UnsafeMutablePointer<T>,
    lda: Int,
    ipiv: UnsafeMutablePointer<Int>,
    b: UnsafeMutablePointer<T>,
    ldb: Int
) -> Int {
    guard n >= 0 else { return -2 }
    guard nrhs >= 0 else { return -3 }
    guard
        lda
            >= _lapackMinimumLeadingDimension(
                layout: layout, rows: n, columns: n
            )
    else { return -5 }
    guard
        ldb
            >= _lapackMinimumLeadingDimension(
                layout: layout, rows: n, columns: nrhs
            )
    else { return -8 }

    let factorInfo = _lapackGetrf(
        layout: layout, m: n, n: n, a: a, lda: lda, ipiv: ipiv
    )
    if factorInfo != 0 { return factorInfo }
    return _lapackGetrs(
        layout: layout, transpose: .noTranspose, n: n, nrhs: nrhs,
        a: a, lda: lda, ipiv: ipiv, b: b, ldb: ldb
    )
}

@inlinable
internal func _lapackGetri<T: LAPACKScalar>(
    layout: LAPACK.Layout,
    n: Int,
    a: UnsafeMutablePointer<T>,
    lda: Int,
    ipiv: UnsafePointer<Int>
) -> Int {
    guard n >= 0 else { return -2 }
    guard
        lda
            >= _lapackMinimumLeadingDimension(
                layout: layout, rows: n, columns: n
            )
    else { return -4 }

    var inverse = [T](repeating: .zero, count: n * n)
    for index in 0..<n { inverse[index * n + index] = .one }
    let info = inverse.withUnsafeMutableBufferPointer { inverse in
        _lapackGetrs(
            layout: layout, transpose: .noTranspose,
            n: n, nrhs: n, a: a, lda: lda, ipiv: ipiv,
            b: inverse.baseAddress!, ldb: Swift.max(1, n)
        )
    }
    if info != 0 { return info }
    for row in 0..<n {
        for column in 0..<n {
            a[_matrixIndex(layout, row, column, lda)] =
                inverse[
                    _matrixIndex(layout, row, column, Swift.max(1, n))
                ]
        }
    }
    return 0
}

@inlinable
internal func _lapackPotrf<T: LAPACKScalar>(
    layout: LAPACK.Layout,
    triangle: LAPACK.Triangle,
    n: Int,
    a: UnsafeMutablePointer<T>,
    lda: Int
) -> Int {
    guard n >= 0 else { return -3 }
    guard
        lda
            >= _lapackMinimumLeadingDimension(
                layout: layout, rows: n, columns: n
            )
    else { return -5 }

    switch triangle {
    case .lower:
        for column in 0..<n {
            var diagonal = a[_matrixIndex(layout, column, column, lda)].realComponent
            for inner in 0..<column {
                diagonal -=
                    a[
                        _matrixIndex(layout, column, inner, lda)
                    ].normSquared
            }
            guard diagonal > .zero && diagonal.isFinite else {
                return column + 1
            }
            let root = T.Magnitude.sqrt(diagonal)
            a[_matrixIndex(layout, column, column, lda)] = .one.scaled(by: root)
            if column + 1 < n {
                for row in (column + 1)..<n {
                    var value = a[_matrixIndex(layout, row, column, lda)]
                    for inner in 0..<column {
                        value -=
                            a[_matrixIndex(layout, row, inner, lda)]
                            * a[_matrixIndex(layout, column, inner, lda)].conjugate
                    }
                    a[_matrixIndex(layout, row, column, lda)] = value.scaled(by: 1 / root)
                }
            }
        }
    case .upper:
        for column in 0..<n {
            var diagonal = a[_matrixIndex(layout, column, column, lda)].realComponent
            for inner in 0..<column {
                diagonal -=
                    a[
                        _matrixIndex(layout, inner, column, lda)
                    ].normSquared
            }
            guard diagonal > .zero && diagonal.isFinite else {
                return column + 1
            }
            let root = T.Magnitude.sqrt(diagonal)
            a[_matrixIndex(layout, column, column, lda)] = .one.scaled(by: root)
            if column + 1 < n {
                for trailingColumn in (column + 1)..<n {
                    var value = a[
                        _matrixIndex(
                            layout, column, trailingColumn, lda
                        )]
                    for inner in 0..<column {
                        value -=
                            a[_matrixIndex(layout, inner, column, lda)].conjugate
                            * a[_matrixIndex(layout, inner, trailingColumn, lda)]
                    }
                    a[_matrixIndex(layout, column, trailingColumn, lda)] =
                        value.scaled(by: 1 / root)
                }
            }
        }
    }
    return 0
}

@inlinable
internal func _lapackPotrs<T: LAPACKScalar>(
    layout: LAPACK.Layout,
    triangle: LAPACK.Triangle,
    n: Int,
    nrhs: Int,
    a: UnsafePointer<T>,
    lda: Int,
    b: UnsafeMutablePointer<T>,
    ldb: Int
) -> Int {
    guard n >= 0 else { return -3 }
    guard nrhs >= 0 else { return -4 }
    guard
        lda
            >= _lapackMinimumLeadingDimension(
                layout: layout, rows: n, columns: n
            )
    else { return -6 }
    guard
        ldb
            >= _lapackMinimumLeadingDimension(
                layout: layout, rows: n, columns: nrhs
            )
    else { return -8 }

    let firstTranspose: LAPACK.Transpose =
        triangle == .lower
        ? .noTranspose : .conjugateTranspose
    let secondTranspose: LAPACK.Transpose =
        triangle == .lower
        ? .conjugateTranspose : .noTranspose
    let first = _lapackTrtrs(
        layout: layout, triangle: triangle, transpose: firstTranspose,
        diagonal: .nonUnit, n: n, nrhs: nrhs,
        a: a, lda: lda, b: b, ldb: ldb
    )
    if first != 0 { return first }
    return _lapackTrtrs(
        layout: layout, triangle: triangle, transpose: secondTranspose,
        diagonal: .nonUnit, n: n, nrhs: nrhs,
        a: a, lda: lda, b: b, ldb: ldb
    )
}

@inlinable
internal func _lapackPosv<T: LAPACKScalar>(
    layout: LAPACK.Layout,
    triangle: LAPACK.Triangle,
    n: Int,
    nrhs: Int,
    a: UnsafeMutablePointer<T>,
    lda: Int,
    b: UnsafeMutablePointer<T>,
    ldb: Int
) -> Int {
    guard n >= 0 else { return -3 }
    guard nrhs >= 0 else { return -4 }
    guard
        lda
            >= _lapackMinimumLeadingDimension(
                layout: layout, rows: n, columns: n
            )
    else { return -6 }
    guard
        ldb
            >= _lapackMinimumLeadingDimension(
                layout: layout, rows: n, columns: nrhs
            )
    else { return -8 }
    let factorInfo = _lapackPotrf(
        layout: layout, triangle: triangle, n: n, a: a, lda: lda
    )
    if factorInfo != 0 { return factorInfo }
    return _lapackPotrs(
        layout: layout, triangle: triangle, n: n, nrhs: nrhs,
        a: a, lda: lda, b: b, ldb: ldb
    )
}

@inlinable
internal func _lapackPotri<T: LAPACKScalar>(
    layout: LAPACK.Layout,
    triangle: LAPACK.Triangle,
    n: Int,
    a: UnsafeMutablePointer<T>,
    lda: Int
) -> Int {
    guard n >= 0 else { return -3 }
    guard
        lda
            >= _lapackMinimumLeadingDimension(
                layout: layout, rows: n, columns: n
            )
    else { return -5 }
    var inverse = [T](repeating: .zero, count: n * n)
    for index in 0..<n { inverse[index * n + index] = .one }
    let info = inverse.withUnsafeMutableBufferPointer { inverse in
        _lapackPotrs(
            layout: layout, triangle: triangle,
            n: n, nrhs: n, a: a, lda: lda,
            b: inverse.baseAddress!, ldb: Swift.max(1, n)
        )
    }
    if info != 0 { return info }
    for row in 0..<n {
        for column in 0..<n where triangle == .upper ? row <= column : row >= column {
            a[_matrixIndex(layout, row, column, lda)] =
                inverse[
                    _matrixIndex(layout, row, column, Swift.max(1, n))
                ]
        }
    }
    return 0
}

@inlinable
internal func _lapackTrtri<T: LAPACKScalar>(
    layout: LAPACK.Layout,
    triangle: LAPACK.Triangle,
    diagonal: LAPACK.Diagonal,
    n: Int,
    a: UnsafeMutablePointer<T>,
    lda: Int
) -> Int {
    guard n >= 0 else { return -4 }
    guard
        lda
            >= _lapackMinimumLeadingDimension(
                layout: layout, rows: n, columns: n
            )
    else { return -6 }
    var inverse = [T](repeating: .zero, count: n * n)
    for index in 0..<n { inverse[index * n + index] = .one }
    let info = inverse.withUnsafeMutableBufferPointer { inverse in
        _lapackTrtrs(
            layout: layout, triangle: triangle,
            transpose: .noTranspose, diagonal: diagonal,
            n: n, nrhs: n, a: a, lda: lda,
            b: inverse.baseAddress!, ldb: Swift.max(1, n)
        )
    }
    if info != 0 { return info }
    for row in 0..<n {
        for column in 0..<n where triangle == .upper ? row <= column : row >= column {
            a[_matrixIndex(layout, row, column, lda)] =
                inverse[
                    _matrixIndex(layout, row, column, Swift.max(1, n))
                ]
        }
    }
    return 0
}

@inlinable
internal func _lapackLange<T: LAPACKScalar>(
    layout: LAPACK.Layout,
    norm: LAPACK.Norm,
    m: Int,
    n: Int,
    a: UnsafePointer<T>,
    lda: Int
) -> T.Magnitude {
    guard
        lda
            >= _lapackMinimumLeadingDimension(
                layout: layout, rows: m, columns: n
            )
    else { return -6 }
    guard m > 0 && n > 0 else { return .zero }
    switch norm {
    case .maxAbsoluteValue:
        var result = T.Magnitude.zero
        for row in 0..<m {
            for column in 0..<n {
                result = Swift.max(
                    result,
                    a[_matrixIndex(layout, row, column, lda)].abs1
                )
            }
        }
        return result
    case .one:
        var result = T.Magnitude.zero
        for column in 0..<n {
            var sum = T.Magnitude.zero
            for row in 0..<m {
                sum += a[_matrixIndex(layout, row, column, lda)].abs1
            }
            result = Swift.max(result, sum)
        }
        return result
    case .infinity:
        var result = T.Magnitude.zero
        for row in 0..<m {
            var sum = T.Magnitude.zero
            for column in 0..<n {
                sum += a[_matrixIndex(layout, row, column, lda)].abs1
            }
            result = Swift.max(result, sum)
        }
        return result
    case .frobenius:
        var sum = T.Magnitude.zero
        for row in 0..<m {
            for column in 0..<n {
                sum += a[_matrixIndex(layout, row, column, lda)].normSquared
            }
        }
        return T.Magnitude.sqrt(sum)
    }
}

@inlinable
internal func _lapackLanhe<T: LAPACKScalar>(
    layout: LAPACK.Layout,
    norm: LAPACK.Norm,
    triangle: LAPACK.Triangle,
    n: Int,
    a: UnsafePointer<T>,
    lda: Int
) -> T.Magnitude {
    guard
        lda
            >= _lapackMinimumLeadingDimension(
                layout: layout, rows: n, columns: n
            )
    else { return -6 }
    guard n > 0 else { return .zero }
    switch norm {
    case .maxAbsoluteValue:
        var result = T.Magnitude.zero
        for row in 0..<n {
            for column in 0..<n {
                result = Swift.max(
                    result,
                    _symmetricElement(
                        a, layout: layout, triangle: triangle,
                        row: row, column: column, leadingDimension: lda,
                        hermitian: true
                    ).abs1)
            }
        }
        return result
    case .one, .infinity:
        var result = T.Magnitude.zero
        for column in 0..<n {
            var sum = T.Magnitude.zero
            for row in 0..<n {
                sum +=
                    _symmetricElement(
                        a, layout: layout, triangle: triangle,
                        row: row, column: column, leadingDimension: lda,
                        hermitian: true
                    ).abs1
            }
            result = Swift.max(result, sum)
        }
        return result
    case .frobenius:
        var sum = T.Magnitude.zero
        for row in 0..<n {
            for column in 0..<n {
                sum +=
                    _symmetricElement(
                        a, layout: layout, triangle: triangle,
                        row: row, column: column, leadingDimension: lda,
                        hermitian: true
                    ).normSquared
            }
        }
        return T.Magnitude.sqrt(sum)
    }
}

// MARK: - Orthogonal factorizations and least squares

@inlinable
internal func _lapackGeqrf<T: LAPACKScalar>(
    layout: LAPACK.Layout,
    m: Int,
    n: Int,
    a: UnsafeMutablePointer<T>,
    lda: Int,
    tau: UnsafeMutablePointer<T>
) -> Int {
    guard m >= 0 else { return -2 }
    guard n >= 0 else { return -3 }
    guard
        lda
            >= _lapackMinimumLeadingDimension(
                layout: layout, rows: m, columns: n
            )
    else { return -5 }

    let reflectorCount = Swift.min(m, n)
    for column in 0..<reflectorCount {
        let diagonalIndex = _matrixIndex(layout, column, column, lda)
        let alpha = a[diagonalIndex]
        var tailNormSquared = T.Magnitude.zero
        if column + 1 < m {
            for row in (column + 1)..<m {
                tailNormSquared +=
                    a[
                        _matrixIndex(layout, row, column, lda)
                    ].normSquared
            }
        }
        let realAlphaSquared = alpha.realComponent * alpha.realComponent
        if tailNormSquared == .zero && alpha.normSquared == realAlphaSquared {
            tau[column] = .zero
            continue
        }

        let norm = T.Magnitude.sqrt(alpha.normSquared + tailNormSquared)
        let betaMagnitude = alpha.realComponent >= .zero ? -norm : norm
        let beta = T.one.scaled(by: betaMagnitude)
        let reflectorScale = T.one / (alpha - beta)
        tau[column] = (beta - alpha) / beta
        a[diagonalIndex] = beta
        if column + 1 < m {
            for row in (column + 1)..<m {
                let index = _matrixIndex(layout, row, column, lda)
                a[index] *= reflectorScale
            }
        }

        guard column + 1 < n else { continue }
        for trailingColumn in (column + 1)..<n {
            var innerProduct = a[
                _matrixIndex(layout, column, trailingColumn, lda)
            ]
            if column + 1 < m {
                for row in (column + 1)..<m {
                    innerProduct +=
                        a[
                            _matrixIndex(layout, row, column, lda)
                        ].conjugate
                        * a[
                            _matrixIndex(layout, row, trailingColumn, lda)
                        ]
                }
            }
            let coefficient = tau[column].conjugate * innerProduct
            a[_matrixIndex(layout, column, trailingColumn, lda)] -= coefficient
            if column + 1 < m {
                for row in (column + 1)..<m {
                    a[_matrixIndex(layout, row, trailingColumn, lda)] -=
                        a[_matrixIndex(layout, row, column, lda)] * coefficient
                }
            }
        }
    }
    return 0
}

@inlinable
internal func _lapackOrgqr<T: LAPACKScalar>(
    layout: LAPACK.Layout,
    m: Int,
    n: Int,
    k: Int,
    a: UnsafeMutablePointer<T>,
    lda: Int,
    tau: UnsafePointer<T>
) -> Int {
    guard m >= 0 else { return -2 }
    guard n >= 0 && n <= m else { return -3 }
    guard k >= 0 && k <= n else { return -4 }
    guard
        lda
            >= _lapackMinimumLeadingDimension(
                layout: layout, rows: m, columns: n
            )
    else { return -6 }

    var reflectors = [T](repeating: .zero, count: m * k)
    if k > 0 {
        for column in 0..<k {
            reflectors[column * m + column] = .one
            if column + 1 < m {
                for row in (column + 1)..<m {
                    reflectors[column * m + row] =
                        a[
                            _matrixIndex(layout, row, column, lda)
                        ]
                }
            }
        }
    }

    var q = [T](repeating: .zero, count: m * n)
    for index in 0..<Swift.min(m, n) { q[index * m + index] = .one }
    if k > 0 {
        for reflector in stride(from: k - 1, through: 0, by: -1) {
            for column in 0..<n {
                var innerProduct = q[column * m + reflector]
                if reflector + 1 < m {
                    for row in (reflector + 1)..<m {
                        innerProduct +=
                            reflectors[reflector * m + row].conjugate
                            * q[column * m + row]
                    }
                }
                let coefficient = tau[reflector] * innerProduct
                q[column * m + reflector] -= coefficient
                if reflector + 1 < m {
                    for row in (reflector + 1)..<m {
                        q[column * m + row] -=
                            reflectors[reflector * m + row] * coefficient
                    }
                }
            }
        }
    }

    for row in 0..<m {
        for column in 0..<n {
            a[_matrixIndex(layout, row, column, lda)] = q[column * m + row]
        }
    }
    return 0
}

@inlinable
internal func _lapackGels<T: LAPACKScalar>(
    layout: LAPACK.Layout,
    transpose: LAPACK.Transpose,
    m: Int,
    n: Int,
    nrhs: Int,
    a: UnsafeMutablePointer<T>,
    lda: Int,
    b: UnsafeMutablePointer<T>,
    ldb: Int
) -> Int {
    guard m >= 0 else { return -3 }
    guard n >= 0 else { return -4 }
    guard nrhs >= 0 else { return -5 }
    guard
        lda
            >= _lapackMinimumLeadingDimension(
                layout: layout, rows: m, columns: n
            )
    else { return -7 }
    let bRows = Swift.max(m, n)
    guard
        ldb
            >= _lapackMinimumLeadingDimension(
                layout: layout, rows: bRows, columns: nrhs
            )
    else { return -9 }

    let rows = transpose == .noTranspose ? m : n
    let columns = transpose == .noTranspose ? n : m
    guard rows > 0 && columns > 0 && nrhs > 0 else { return 0 }

    var operation = [T](repeating: .zero, count: rows * columns)
    for row in 0..<rows {
        for column in 0..<columns {
            operation[column * rows + row] = _lapackElement(
                a, layout: layout, transpose: transpose,
                row: row, column: column, leadingDimension: lda
            )
        }
    }

    var solution = [T](repeating: .zero, count: columns * nrhs)
    var info = 0
    if rows >= columns {
        var gram = [T](repeating: .zero, count: columns * columns)
        for column in 0..<columns {
            for row in 0..<columns {
                var sum = T.zero
                for inner in 0..<rows {
                    sum +=
                        operation[row * rows + inner].conjugate
                        * operation[column * rows + inner]
                }
                gram[column * columns + row] = sum
            }
        }
        for rhs in 0..<nrhs {
            for row in 0..<columns {
                var sum = T.zero
                for inner in 0..<rows {
                    sum +=
                        operation[row * rows + inner].conjugate
                        * b[_matrixIndex(layout, inner, rhs, ldb)]
                }
                solution[rhs * columns + row] = sum
            }
        }
        var pivots = [Int](repeating: 0, count: columns)
        info = gram.withUnsafeMutableBufferPointer { gram in
            pivots.withUnsafeMutableBufferPointer { pivots in
                solution.withUnsafeMutableBufferPointer { solution in
                    _lapackGesv(
                        layout: .columnMajor, n: columns, nrhs: nrhs,
                        a: gram.baseAddress!, lda: columns,
                        ipiv: pivots.baseAddress!,
                        b: solution.baseAddress!, ldb: columns
                    )
                }
            }
        }
    } else {
        var gram = [T](repeating: .zero, count: rows * rows)
        for column in 0..<rows {
            for row in 0..<rows {
                var sum = T.zero
                for inner in 0..<columns {
                    sum +=
                        operation[inner * rows + row]
                        * operation[inner * rows + column].conjugate
                }
                gram[column * rows + row] = sum
            }
        }
        var intermediate = [T](repeating: .zero, count: rows * nrhs)
        for rhs in 0..<nrhs {
            for row in 0..<rows {
                intermediate[rhs * rows + row] =
                    b[
                        _matrixIndex(layout, row, rhs, ldb)
                    ]
            }
        }
        var pivots = [Int](repeating: 0, count: rows)
        info = gram.withUnsafeMutableBufferPointer { gram in
            pivots.withUnsafeMutableBufferPointer { pivots in
                intermediate.withUnsafeMutableBufferPointer { intermediate in
                    _lapackGesv(
                        layout: .columnMajor, n: rows, nrhs: nrhs,
                        a: gram.baseAddress!, lda: rows,
                        ipiv: pivots.baseAddress!,
                        b: intermediate.baseAddress!, ldb: rows
                    )
                }
            }
        }
        if info == 0 {
            for rhs in 0..<nrhs {
                for row in 0..<columns {
                    var sum = T.zero
                    for inner in 0..<rows {
                        sum +=
                            operation[row * rows + inner].conjugate
                            * intermediate[rhs * rows + inner]
                    }
                    solution[rhs * columns + row] = sum
                }
            }
        }
    }
    if info != 0 { return info }

    for rhs in 0..<nrhs {
        for row in 0..<columns {
            b[_matrixIndex(layout, row, rhs, ldb)] =
                solution[
                    rhs * columns + row
                ]
        }
        if columns < bRows {
            for row in columns..<bRows {
                b[_matrixIndex(layout, row, rhs, ldb)] = .zero
            }
        }
    }
    return 0
}

// MARK: - Hermitian eigendecomposition

@inlinable
internal func _lapackHermitianJacobi<T: LAPACKScalar>(
    matrix: inout [T],
    n: Int
) -> (values: [T.Magnitude], vectors: [T], converged: Bool) {
    var vectors = [T](repeating: .zero, count: n * n)
    for index in 0..<n { vectors[index * n + index] = .one }
    guard n > 1 else {
        return (
            n == 1 ? [matrix[0].realComponent] : [],
            vectors,
            true
        )
    }

    var scale: T.Magnitude = 1
    for value in matrix {
        scale = Swift.max(scale, T.Magnitude.sqrt(value.normSquared))
    }
    let tolerance = T.Magnitude.ulpOfOne * T.Magnitude(n) * scale
    let maximumRotations = Swift.max(32, 32 * n * n)
    var converged = false

    for _ in 0..<maximumRotations {
        var p = 0
        var q = 1
        var largest = T.Magnitude.zero
        for column in 1..<n {
            for row in 0..<column {
                let magnitude = T.Magnitude.sqrt(
                    matrix[column * n + row].normSquared
                )
                if magnitude > largest {
                    largest = magnitude
                    p = row
                    q = column
                }
            }
        }
        if largest <= tolerance {
            converged = true
            break
        }

        let app = matrix[p * n + p].realComponent
        let aqq = matrix[q * n + q].realComponent
        let tau = (aqq - app) / (2 * largest)
        let root = T.Magnitude.sqrt(1 + tau * tau)
        let tangent: T.Magnitude
        if tau >= .zero {
            tangent = 1 / (tau + root)
        } else {
            tangent = -1 / (-tau + root)
        }
        let cosine = 1 / T.Magnitude.sqrt(1 + tangent * tangent)
        let sine = tangent * cosine
        let phase = matrix[q * n + p].conjugate.scaled(by: 1 / largest)

        var newP = [T](repeating: .zero, count: n)
        var newQ = [T](repeating: .zero, count: n)
        for row in 0..<n {
            let oldP = matrix[p * n + row]
            let oldQ = matrix[q * n + row]
            newP[row] =
                oldP.scaled(by: cosine)
                - oldQ * phase.scaled(by: sine)
            newQ[row] =
                oldP.scaled(by: sine)
                + oldQ * phase.scaled(by: cosine)
        }
        for row in 0..<n where row != p && row != q {
            matrix[p * n + row] = newP[row]
            matrix[row * n + p] = newP[row].conjugate
            matrix[q * n + row] = newQ[row]
            matrix[row * n + q] = newQ[row].conjugate
        }
        matrix[p * n + p] = .one.scaled(by: app - tangent * largest)
        matrix[q * n + q] = .one.scaled(by: aqq + tangent * largest)
        matrix[q * n + p] = .zero
        matrix[p * n + q] = .zero

        for row in 0..<n {
            let oldP = vectors[p * n + row]
            let oldQ = vectors[q * n + row]
            vectors[p * n + row] =
                oldP.scaled(by: cosine)
                - oldQ * phase.scaled(by: sine)
            vectors[q * n + row] =
                oldP.scaled(by: sine)
                + oldQ * phase.scaled(by: cosine)
        }
    }

    var values = [T.Magnitude](repeating: .zero, count: n)
    for index in 0..<n { values[index] = matrix[index * n + index].realComponent }
    return (values, vectors, converged)
}

@inlinable
internal func _lapackSyev<T: LAPACKScalar>(
    layout: LAPACK.Layout,
    job: LAPACK.Eigenvectors,
    triangle: LAPACK.Triangle,
    n: Int,
    a: UnsafeMutablePointer<T>,
    lda: Int,
    w: UnsafeMutablePointer<T.Magnitude>
) -> Int {
    guard n >= 0 else { return -4 }
    guard
        lda
            >= _lapackMinimumLeadingDimension(
                layout: layout, rows: n, columns: n
            )
    else { return -6 }

    var matrix = [T](repeating: .zero, count: n * n)
    for row in 0..<n {
        for column in 0..<n {
            matrix[column * n + row] = _symmetricElement(
                a, layout: layout, triangle: triangle,
                row: row, column: column, leadingDimension: lda,
                hermitian: true
            )
        }
    }
    let result = _lapackHermitianJacobi(matrix: &matrix, n: n)
    let order = (0..<n).sorted { result.values[$0] < result.values[$1] }
    for column in 0..<n {
        w[column] = result.values[order[column]]
        if job == .vectors {
            for row in 0..<n {
                a[_matrixIndex(layout, row, column, lda)] =
                    result.vectors[order[column] * n + row]
            }
        }
    }
    return result.converged ? 0 : 1
}

// MARK: - Singular value decomposition

@inlinable
internal func _lapackGesvdArgumentError(
    layout: LAPACK.Layout,
    jobU: LAPACK.SingularVectors,
    jobVT: LAPACK.SingularVectors,
    m: Int,
    n: Int,
    lda: Int,
    ldu: Int,
    ldvt: Int
) -> Int? {
    if jobU == .overwrite && jobVT == .overwrite { return -3 }
    if m < 0 { return -4 }
    if n < 0 { return -5 }
    if lda
        < _lapackMinimumLeadingDimension(
            layout: layout, rows: m, columns: n
        )
    {
        return -7
    }

    let minimum = Swift.min(m, n)
    switch layout {
    case .rowMajor:
        if jobU == .all && ldu < m { return -10 }
        if jobU == .some && ldu < minimum { return -10 }
        if (jobVT == .all || jobVT == .some) && ldvt < n { return -12 }
    case .columnMajor:
        let minimumLDU =
            jobU == .all || jobU == .some
            ? Swift.max(1, m) : 1
        if ldu < minimumLDU { return -10 }
        let minimumLDVT: Int
        switch jobVT {
        case .all: minimumLDVT = Swift.max(1, n)
        case .some: minimumLDVT = Swift.max(1, minimum)
        case .overwrite, .none: minimumLDVT = 1
        }
        if ldvt < minimumLDVT { return -12 }
    }
    return nil
}

@inlinable
internal func _lapackGesddArgumentError(
    layout: LAPACK.Layout,
    job: LAPACK.SingularVectors,
    m: Int,
    n: Int,
    lda: Int,
    ldu: Int,
    ldvt: Int
) -> Int? {
    if m < 0 { return -3 }
    if n < 0 { return -4 }
    if lda
        < _lapackMinimumLeadingDimension(
            layout: layout, rows: m, columns: n
        )
    {
        return -6
    }

    let minimum = Swift.min(m, n)
    let minimumLDU: Int
    let minimumLDVT: Int
    switch layout {
    case .rowMajor:
        switch job {
        case .all: minimumLDU = m
        case .some: minimumLDU = minimum
        case .overwrite where m < n: minimumLDU = m
        case .overwrite, .none: minimumLDU = 1
        }
        switch job {
        case .all, .some: minimumLDVT = n
        case .overwrite where m >= n: minimumLDVT = n
        case .overwrite, .none: minimumLDVT = 1
        }
    case .columnMajor:
        switch job {
        case .all, .some: minimumLDU = Swift.max(1, m)
        case .overwrite where m < n: minimumLDU = Swift.max(1, m)
        case .overwrite, .none: minimumLDU = 1
        }
        switch job {
        case .all: minimumLDVT = Swift.max(1, n)
        case .some: minimumLDVT = Swift.max(1, minimum)
        case .overwrite where m >= n: minimumLDVT = Swift.max(1, n)
        case .overwrite, .none: minimumLDVT = 1
        }
    }
    if ldu < minimumLDU { return -9 }
    if ldvt < minimumLDVT { return -11 }
    return nil
}

@inlinable
internal func _lapackGesvd<T: LAPACKScalar>(
    layout: LAPACK.Layout,
    jobU: LAPACK.SingularVectors,
    jobVT: LAPACK.SingularVectors,
    m: Int,
    n: Int,
    a: UnsafeMutablePointer<T>,
    lda: Int,
    s: UnsafeMutablePointer<T.Magnitude>,
    u: UnsafeMutablePointer<T>,
    ldu: Int,
    vt: UnsafeMutablePointer<T>,
    ldvt: Int,
    superb: UnsafeMutablePointer<T.Magnitude>?
) -> Int {
    if let error = _lapackGesvdArgumentError(
        layout: layout, jobU: jobU, jobVT: jobVT,
        m: m, n: n, lda: lda, ldu: ldu, ldvt: ldvt
    ) {
        return error
    }

    let minimum = Swift.min(m, n)
    guard m > 0 && n > 0 else { return 0 }
    var gram = [T](repeating: .zero, count: n * n)
    for column in 0..<n {
        for row in 0..<n {
            var sum = T.zero
            for inner in 0..<m {
                sum +=
                    a[_matrixIndex(layout, inner, row, lda)].conjugate
                    * a[_matrixIndex(layout, inner, column, lda)]
            }
            gram[column * n + row] = sum
        }
    }
    let eigen = _lapackHermitianJacobi(matrix: &gram, n: n)
    let order = (0..<n).sorted { eigen.values[$0] > eigen.values[$1] }
    var rightVectors = [T](repeating: .zero, count: n * n)
    for column in 0..<n {
        for row in 0..<n {
            rightVectors[column * n + row] =
                eigen.vectors[
                    order[column] * n + row
                ]
        }
    }
    for index in 0..<minimum {
        s[index] = T.Magnitude.sqrt(Swift.max(.zero, eigen.values[order[index]]))
    }
    if let superb, minimum > 1 {
        for index in 0..<(minimum - 1) { superb[index] = .zero }
    }

    let largestSingularValue = minimum > 0 ? s[0] : .zero
    let dimension = T.Magnitude(Swift.max(m, n))
    let one: T.Magnitude = 1
    let tolerance: T.Magnitude =
        T.Magnitude.ulpOfOne
        * dimension
        * Swift.max(one, largestSingularValue)
    var leftVectors = [T](repeating: .zero, count: m * m)
    for column in 0..<m {
        if column < minimum && s[column] > tolerance {
            for row in 0..<m {
                var sum = T.zero
                for inner in 0..<n {
                    sum +=
                        a[_matrixIndex(layout, row, inner, lda)]
                        * rightVectors[column * n + inner]
                }
                leftVectors[column * m + row] = sum.scaled(by: 1 / s[column])
            }
        } else {
            leftVectors[column * m + (column % m)] = .one
        }

        for previous in 0..<column {
            var projection = T.zero
            for row in 0..<m {
                projection +=
                    leftVectors[previous * m + row].conjugate
                    * leftVectors[column * m + row]
            }
            for row in 0..<m {
                leftVectors[column * m + row] -=
                    leftVectors[previous * m + row] * projection
            }
        }
        var normSquared = T.Magnitude.zero
        for row in 0..<m {
            normSquared += leftVectors[column * m + row].normSquared
        }
        if normSquared <= tolerance * tolerance {
            var found = false
            for basis in 0..<m where !found {
                for row in 0..<m {
                    leftVectors[column * m + row] = row == basis ? .one : .zero
                }
                for previous in 0..<column {
                    var projection = T.zero
                    for row in 0..<m {
                        projection +=
                            leftVectors[previous * m + row].conjugate
                            * leftVectors[column * m + row]
                    }
                    for row in 0..<m {
                        leftVectors[column * m + row] -=
                            leftVectors[previous * m + row] * projection
                    }
                }
                normSquared = .zero
                for row in 0..<m {
                    normSquared += leftVectors[column * m + row].normSquared
                }
                found = normSquared > tolerance * tolerance
            }
        }
        let inverseNorm = 1 / T.Magnitude.sqrt(normSquared)
        for row in 0..<m {
            leftVectors[column * m + row] =
                leftVectors[column * m + row].scaled(by: inverseNorm)
        }
    }

    switch jobU {
    case .all, .some:
        let columns = jobU == .all ? m : minimum
        for row in 0..<m {
            for column in 0..<columns {
                u[_matrixIndex(layout, row, column, ldu)] =
                    leftVectors[column * m + row]
            }
        }
    case .overwrite where m >= n:
        for row in 0..<m {
            for column in 0..<minimum {
                a[_matrixIndex(layout, row, column, lda)] =
                    leftVectors[column * m + row]
            }
        }
    case .overwrite, .none:
        break
    }

    switch jobVT {
    case .all, .some:
        let rows = jobVT == .all ? n : minimum
        for row in 0..<rows {
            for column in 0..<n {
                vt[_matrixIndex(layout, row, column, ldvt)] =
                    rightVectors[row * n + column].conjugate
            }
        }
    case .overwrite where m < n:
        for row in 0..<minimum {
            for column in 0..<n {
                a[_matrixIndex(layout, row, column, lda)] =
                    rightVectors[row * n + column].conjugate
            }
        }
    case .overwrite, .none:
        break
    }
    return eigen.converged ? 0 : 1
}
