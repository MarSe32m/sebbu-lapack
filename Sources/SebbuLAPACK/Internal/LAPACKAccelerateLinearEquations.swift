// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: MIT

#if canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
import Accelerate
import ComplexModule
import RealModule
import SebbuBLAS

@inline(__always)
internal func _lapackeInfo(_ info: Int) -> Int {
    info < 0 ? info - 1 : info
}

// MARK: General systems

internal func _accelerateSgesv(
    layout: LAPACK.Layout, n: Int, nrhs: Int, a: UnsafeMutablePointer<Float>, lda: Int,
    ipiv: UnsafeMutablePointer<Int>, b: UnsafeMutablePointer<Float>, ldb: Int
) -> Int {
    if n < 0 { return -2 }
    if nrhs < 0 { return -3 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: n) { return -5 }
    if ldb < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: nrhs) {
        return -8
    }
    return _withLAPACKColumnMajorMutableMatrix(
        layout: layout, rows: n, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        _withLAPACKColumnMajorMutableMatrix(
            layout: layout, rows: n, columns: nrhs, matrix: b, leadingDimension: ldb
        ) { b, ldb in
            var n = n
            var nrhs = nrhs
            var lda = lda
            var ldb = ldb
            var info = 0
            sgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info)
            return _lapackeInfo(info)
        }
    }
}

internal func _accelerateDgesv(
    layout: LAPACK.Layout, n: Int, nrhs: Int, a: UnsafeMutablePointer<Double>, lda: Int,
    ipiv: UnsafeMutablePointer<Int>, b: UnsafeMutablePointer<Double>, ldb: Int
) -> Int {
    if n < 0 { return -2 }
    if nrhs < 0 { return -3 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: n) { return -5 }
    if ldb < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: nrhs) {
        return -8
    }
    return _withLAPACKColumnMajorMutableMatrix(
        layout: layout, rows: n, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        _withLAPACKColumnMajorMutableMatrix(
            layout: layout, rows: n, columns: nrhs, matrix: b, leadingDimension: ldb
        ) { b, ldb in
            var n = n
            var nrhs = nrhs
            var lda = lda
            var ldb = ldb
            var info = 0
            dgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info)
            return _lapackeInfo(info)
        }
    }
}

internal func _accelerateCgesv(
    layout: LAPACK.Layout, n: Int, nrhs: Int, a: UnsafeMutablePointer<Complex<Float>>, lda: Int,
    ipiv: UnsafeMutablePointer<Int>, b: UnsafeMutablePointer<Complex<Float>>, ldb: Int
) -> Int {
    if n < 0 { return -2 }
    if nrhs < 0 { return -3 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: n) { return -5 }
    if ldb < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: nrhs) {
        return -8
    }
    return _withLAPACKColumnMajorMutableMatrix(
        layout: layout, rows: n, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        _withLAPACKColumnMajorMutableMatrix(
            layout: layout, rows: n, columns: nrhs, matrix: b, leadingDimension: ldb
        ) { b, ldb in
            var n = n
            var nrhs = nrhs
            var lda = lda
            var ldb = ldb
            var info = 0
            cgesv_(
                &n, &nrhs, _complexFloatPointer(a), &lda, ipiv, _complexFloatPointer(b), &ldb,
                &info)
            return _lapackeInfo(info)
        }
    }
}

internal func _accelerateZgesv(
    layout: LAPACK.Layout, n: Int, nrhs: Int, a: UnsafeMutablePointer<Complex<Double>>,
    lda: Int,
    ipiv: UnsafeMutablePointer<Int>, b: UnsafeMutablePointer<Complex<Double>>, ldb: Int
) -> Int {
    if n < 0 { return -2 }
    if nrhs < 0 { return -3 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: n) { return -5 }
    if ldb < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: nrhs) {
        return -8
    }
    return _withLAPACKColumnMajorMutableMatrix(
        layout: layout, rows: n, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        _withLAPACKColumnMajorMutableMatrix(
            layout: layout, rows: n, columns: nrhs, matrix: b, leadingDimension: ldb
        ) { b, ldb in
            var n = n
            var nrhs = nrhs
            var lda = lda
            var ldb = ldb
            var info = 0
            zgesv_(
                &n, &nrhs, _complexDoublePointer(a), &lda, ipiv, _complexDoublePointer(b), &ldb,
                &info)
            return _lapackeInfo(info)
        }
    }
}

internal func _accelerateSgetrf(
    layout: LAPACK.Layout, m: Int, n: Int, a: UnsafeMutablePointer<Float>, lda: Int,
    ipiv: UnsafeMutablePointer<Int>
) -> Int {
    if m < 0 { return -2 }
    if n < 0 { return -3 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: m, columns: n) { return -5 }
    return _withLAPACKColumnMajorMutableMatrix(
        layout: layout, rows: m, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        var m = m
        var n = n
        var lda = lda
        var info = 0
        sgetrf_(&m, &n, a, &lda, ipiv, &info)
        return _lapackeInfo(info)
    }
}

internal func _accelerateDgetrf(
    layout: LAPACK.Layout, m: Int, n: Int, a: UnsafeMutablePointer<Double>, lda: Int,
    ipiv: UnsafeMutablePointer<Int>
) -> Int {
    if m < 0 { return -2 }
    if n < 0 { return -3 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: m, columns: n) { return -5 }
    return _withLAPACKColumnMajorMutableMatrix(
        layout: layout, rows: m, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        var m = m
        var n = n
        var lda = lda
        var info = 0
        dgetrf_(&m, &n, a, &lda, ipiv, &info)
        return _lapackeInfo(info)
    }
}

internal func _accelerateCgetrf(
    layout: LAPACK.Layout, m: Int, n: Int, a: UnsafeMutablePointer<Complex<Float>>, lda: Int,
    ipiv: UnsafeMutablePointer<Int>
) -> Int {
    if m < 0 { return -2 }
    if n < 0 { return -3 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: m, columns: n) { return -5 }
    return _withLAPACKColumnMajorMutableMatrix(
        layout: layout, rows: m, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        var m = m
        var n = n
        var lda = lda
        var info = 0
        cgetrf_(&m, &n, _complexFloatPointer(a), &lda, ipiv, &info)
        return _lapackeInfo(info)
    }
}

internal func _accelerateZgetrf(
    layout: LAPACK.Layout, m: Int, n: Int, a: UnsafeMutablePointer<Complex<Double>>, lda: Int,
    ipiv: UnsafeMutablePointer<Int>
) -> Int {
    if m < 0 { return -2 }
    if n < 0 { return -3 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: m, columns: n) { return -5 }
    return _withLAPACKColumnMajorMutableMatrix(
        layout: layout, rows: m, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        var m = m
        var n = n
        var lda = lda
        var info = 0
        zgetrf_(&m, &n, _complexDoublePointer(a), &lda, ipiv, &info)
        return _lapackeInfo(info)
    }
}

internal func _accelerateSgetrs(
    layout: LAPACK.Layout, transpose: LAPACK.Transpose, n: Int, nrhs: Int,
    a: UnsafePointer<Float>,
    lda: Int, ipiv: UnsafePointer<Int>, b: UnsafeMutablePointer<Float>, ldb: Int
) -> Int {
    if n < 0 { return -3 }
    if nrhs < 0 { return -4 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: n) { return -6 }
    if ldb < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: nrhs) {
        return -9
    }
    return _withLAPACKColumnMajorInputMatrix(
        layout: layout, rows: n, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        _withLAPACKColumnMajorMutableMatrix(
            layout: layout, rows: n, columns: nrhs, matrix: b, leadingDimension: ldb
        ) { b, ldb in
            var trans = transpose._character
            var n = n
            var nrhs = nrhs
            var lda = lda
            var ldb = ldb
            var info = 0
            sgetrs_(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info)
            return _lapackeInfo(info)
        }
    }
}

internal func _accelerateDgetrs(
    layout: LAPACK.Layout, transpose: LAPACK.Transpose, n: Int, nrhs: Int,
    a: UnsafePointer<Double>,
    lda: Int, ipiv: UnsafePointer<Int>, b: UnsafeMutablePointer<Double>, ldb: Int
) -> Int {
    if n < 0 { return -3 }
    if nrhs < 0 { return -4 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: n) { return -6 }
    if ldb < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: nrhs) {
        return -9
    }
    return _withLAPACKColumnMajorInputMatrix(
        layout: layout, rows: n, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        _withLAPACKColumnMajorMutableMatrix(
            layout: layout, rows: n, columns: nrhs, matrix: b, leadingDimension: ldb
        ) { b, ldb in
            var trans = transpose._character
            var n = n
            var nrhs = nrhs
            var lda = lda
            var ldb = ldb
            var info = 0
            dgetrs_(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info)
            return _lapackeInfo(info)
        }
    }
}

internal func _accelerateCgetrs(
    layout: LAPACK.Layout, transpose: LAPACK.Transpose, n: Int, nrhs: Int,
    a: UnsafePointer<Complex<Float>>, lda: Int, ipiv: UnsafePointer<Int>,
    b: UnsafeMutablePointer<Complex<Float>>, ldb: Int
) -> Int {
    if n < 0 { return -3 }
    if nrhs < 0 { return -4 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: n) { return -6 }
    if ldb < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: nrhs) {
        return -9
    }
    return _withLAPACKColumnMajorInputMatrix(
        layout: layout, rows: n, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        _withLAPACKColumnMajorMutableMatrix(
            layout: layout, rows: n, columns: nrhs, matrix: b, leadingDimension: ldb
        ) { b, ldb in
            var trans = transpose._character
            var n = n
            var nrhs = nrhs
            var lda = lda
            var ldb = ldb
            var info = 0
            cgetrs_(
                &trans, &n, &nrhs, _complexFloatPointer(a), &lda, ipiv, _complexFloatPointer(b),
                &ldb,
                &info)
            return _lapackeInfo(info)
        }
    }
}

internal func _accelerateZgetrs(
    layout: LAPACK.Layout, transpose: LAPACK.Transpose, n: Int, nrhs: Int,
    a: UnsafePointer<Complex<Double>>, lda: Int, ipiv: UnsafePointer<Int>,
    b: UnsafeMutablePointer<Complex<Double>>, ldb: Int
) -> Int {
    if n < 0 { return -3 }
    if nrhs < 0 { return -4 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: n) { return -6 }
    if ldb < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: nrhs) {
        return -9
    }
    return _withLAPACKColumnMajorInputMatrix(
        layout: layout, rows: n, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        _withLAPACKColumnMajorMutableMatrix(
            layout: layout, rows: n, columns: nrhs, matrix: b, leadingDimension: ldb
        ) { b, ldb in
            var trans = transpose._character
            var n = n
            var nrhs = nrhs
            var lda = lda
            var ldb = ldb
            var info = 0
            zgetrs_(
                &trans, &n, &nrhs, _complexDoublePointer(a), &lda, ipiv,
                _complexDoublePointer(b), &ldb,
                &info)
            return _lapackeInfo(info)
        }
    }
}

internal func _accelerateSgetri(
    layout: LAPACK.Layout, n: Int, a: UnsafeMutablePointer<Float>, lda: Int,
    ipiv: UnsafePointer<Int>
) -> Int {
    if n < 0 { return -2 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: n) { return -4 }
    return _withLAPACKColumnMajorMutableMatrix(
        layout: layout, rows: n, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        var n = n
        var lda = lda
        var lwork = -1
        var info = 0
        var query: Float = 0
        sgetri_(&n, a, &lda, ipiv, &query, &lwork, &info)
        if info != 0 { return _lapackeInfo(info) }
        lwork = Swift.max(1, Int(query.rounded(.up)))
        var work = [Float](repeating: 0, count: lwork)
        sgetri_(&n, a, &lda, ipiv, &work, &lwork, &info)
        return _lapackeInfo(info)
    }
}

internal func _accelerateDgetri(
    layout: LAPACK.Layout, n: Int, a: UnsafeMutablePointer<Double>, lda: Int,
    ipiv: UnsafePointer<Int>
) -> Int {
    if n < 0 { return -2 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: n) { return -4 }
    return _withLAPACKColumnMajorMutableMatrix(
        layout: layout, rows: n, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        var n = n
        var lda = lda
        var lwork = -1
        var info = 0
        var query: Double = 0
        dgetri_(&n, a, &lda, ipiv, &query, &lwork, &info)
        if info != 0 { return _lapackeInfo(info) }
        lwork = Swift.max(1, Int(query.rounded(.up)))
        var work = [Double](repeating: 0, count: lwork)
        dgetri_(&n, a, &lda, ipiv, &work, &lwork, &info)
        return _lapackeInfo(info)
    }
}

internal func _accelerateCgetri(
    layout: LAPACK.Layout, n: Int, a: UnsafeMutablePointer<Complex<Float>>, lda: Int,
    ipiv: UnsafePointer<Int>
) -> Int {
    if n < 0 { return -2 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: n) { return -4 }
    return _withLAPACKColumnMajorMutableMatrix(
        layout: layout, rows: n, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        var n = n
        var lda = lda
        var lwork = -1
        var info = 0
        var query = Complex<Float>.zero
        withUnsafeMutablePointer(to: &query) { query in
            cgetri_(
                &n, _complexFloatPointer(a), &lda, ipiv, _complexFloatPointer(query), &lwork,
                &info)
        }
        if info != 0 { return _lapackeInfo(info) }
        lwork = Swift.max(1, Int(query.real.rounded(.up)))
        var work = [Complex<Float>](repeating: .zero, count: lwork)
        work.withUnsafeMutableBufferPointer { work in
            cgetri_(
                &n, _complexFloatPointer(a), &lda, ipiv,
                _complexFloatPointer(work.baseAddress!), &lwork,
                &info)
        }
        return _lapackeInfo(info)
    }
}

internal func _accelerateZgetri(
    layout: LAPACK.Layout, n: Int, a: UnsafeMutablePointer<Complex<Double>>, lda: Int,
    ipiv: UnsafePointer<Int>
) -> Int {
    if n < 0 { return -2 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: n) { return -4 }
    return _withLAPACKColumnMajorMutableMatrix(
        layout: layout, rows: n, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        var n = n
        var lda = lda
        var lwork = -1
        var info = 0
        var query = Complex<Double>.zero
        withUnsafeMutablePointer(to: &query) { query in
            zgetri_(
                &n, _complexDoublePointer(a), &lda, ipiv, _complexDoublePointer(query), &lwork,
                &info)
        }
        if info != 0 { return _lapackeInfo(info) }
        lwork = Swift.max(1, Int(query.real.rounded(.up)))
        var work = [Complex<Double>](repeating: .zero, count: lwork)
        work.withUnsafeMutableBufferPointer { work in
            zgetri_(
                &n, _complexDoublePointer(a), &lda, ipiv,
                _complexDoublePointer(work.baseAddress!),
                &lwork, &info)
        }
        return _lapackeInfo(info)
    }
}

// MARK: Cholesky systems

internal func _accelerateSpotrf(
    layout: LAPACK.Layout, triangle: LAPACK.Triangle, n: Int, a: UnsafeMutablePointer<Float>,
    lda: Int
) -> Int {
    if n < 0 { return -3 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: n) { return -5 }
    return _withLAPACKColumnMajorMutableMatrix(
        layout: layout, rows: n, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        var uplo = triangle._lapackCharacter
        var n = n
        var lda = lda
        var info = 0
        spotrf_(&uplo, &n, a, &lda, &info)
        return _lapackeInfo(info)
    }
}

internal func _accelerateDpotrf(
    layout: LAPACK.Layout, triangle: LAPACK.Triangle, n: Int, a: UnsafeMutablePointer<Double>,
    lda: Int
) -> Int {
    if n < 0 { return -3 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: n) { return -5 }
    return _withLAPACKColumnMajorMutableMatrix(
        layout: layout, rows: n, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        var uplo = triangle._lapackCharacter
        var n = n
        var lda = lda
        var info = 0
        dpotrf_(&uplo, &n, a, &lda, &info)
        return _lapackeInfo(info)
    }
}

internal func _accelerateCpotrf(
    layout: LAPACK.Layout, triangle: LAPACK.Triangle, n: Int,
    a: UnsafeMutablePointer<Complex<Float>>, lda: Int
) -> Int {
    if n < 0 { return -3 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: n) { return -5 }
    return _withLAPACKColumnMajorMutableMatrix(
        layout: layout, rows: n, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        var uplo = triangle._lapackCharacter
        var n = n
        var lda = lda
        var info = 0
        cpotrf_(&uplo, &n, _complexFloatPointer(a), &lda, &info)
        return _lapackeInfo(info)
    }
}

internal func _accelerateZpotrf(
    layout: LAPACK.Layout, triangle: LAPACK.Triangle, n: Int,
    a: UnsafeMutablePointer<Complex<Double>>, lda: Int
) -> Int {
    if n < 0 { return -3 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: n) { return -5 }
    return _withLAPACKColumnMajorMutableMatrix(
        layout: layout, rows: n, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        var uplo = triangle._lapackCharacter
        var n = n
        var lda = lda
        var info = 0
        zpotrf_(&uplo, &n, _complexDoublePointer(a), &lda, &info)
        return _lapackeInfo(info)
    }
}

internal func _accelerateSpotrs(
    layout: LAPACK.Layout, triangle: LAPACK.Triangle, n: Int, nrhs: Int,
    a: UnsafePointer<Float>,
    lda: Int, b: UnsafeMutablePointer<Float>, ldb: Int
) -> Int {
    if n < 0 { return -3 }
    if nrhs < 0 { return -4 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: n) { return -6 }
    if ldb < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: nrhs) {
        return -8
    }
    return _withLAPACKColumnMajorInputMatrix(
        layout: layout, rows: n, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        _withLAPACKColumnMajorMutableMatrix(
            layout: layout, rows: n, columns: nrhs, matrix: b, leadingDimension: ldb
        ) { b, ldb in
            var uplo = triangle._lapackCharacter
            var n = n
            var nrhs = nrhs
            var lda = lda
            var ldb = ldb
            var info = 0
            spotrs_(&uplo, &n, &nrhs, a, &lda, b, &ldb, &info)
            return _lapackeInfo(info)
        }
    }
}

internal func _accelerateDpotrs(
    layout: LAPACK.Layout, triangle: LAPACK.Triangle, n: Int, nrhs: Int,
    a: UnsafePointer<Double>,
    lda: Int, b: UnsafeMutablePointer<Double>, ldb: Int
) -> Int {
    if n < 0 { return -3 }
    if nrhs < 0 { return -4 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: n) { return -6 }
    if ldb < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: nrhs) {
        return -8
    }
    return _withLAPACKColumnMajorInputMatrix(
        layout: layout, rows: n, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        _withLAPACKColumnMajorMutableMatrix(
            layout: layout, rows: n, columns: nrhs, matrix: b, leadingDimension: ldb
        ) { b, ldb in
            var uplo = triangle._lapackCharacter
            var n = n
            var nrhs = nrhs
            var lda = lda
            var ldb = ldb
            var info = 0
            dpotrs_(&uplo, &n, &nrhs, a, &lda, b, &ldb, &info)
            return _lapackeInfo(info)
        }
    }
}

internal func _accelerateCpotrs(
    layout: LAPACK.Layout, triangle: LAPACK.Triangle, n: Int, nrhs: Int,
    a: UnsafePointer<Complex<Float>>, lda: Int, b: UnsafeMutablePointer<Complex<Float>>,
    ldb: Int
) -> Int {
    if n < 0 { return -3 }
    if nrhs < 0 { return -4 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: n) { return -6 }
    if ldb < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: nrhs) {
        return -8
    }
    return _withLAPACKColumnMajorInputMatrix(
        layout: layout, rows: n, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        _withLAPACKColumnMajorMutableMatrix(
            layout: layout, rows: n, columns: nrhs, matrix: b, leadingDimension: ldb
        ) { b, ldb in
            var uplo = triangle._lapackCharacter
            var n = n
            var nrhs = nrhs
            var lda = lda
            var ldb = ldb
            var info = 0
            cpotrs_(
                &uplo, &n, &nrhs, _complexFloatPointer(a), &lda, _complexFloatPointer(b), &ldb,
                &info)
            return _lapackeInfo(info)
        }
    }
}

internal func _accelerateZpotrs(
    layout: LAPACK.Layout, triangle: LAPACK.Triangle, n: Int, nrhs: Int,
    a: UnsafePointer<Complex<Double>>, lda: Int, b: UnsafeMutablePointer<Complex<Double>>,
    ldb: Int
) -> Int {
    if n < 0 { return -3 }
    if nrhs < 0 { return -4 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: n) { return -6 }
    if ldb < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: nrhs) {
        return -8
    }
    return _withLAPACKColumnMajorInputMatrix(
        layout: layout, rows: n, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        _withLAPACKColumnMajorMutableMatrix(
            layout: layout, rows: n, columns: nrhs, matrix: b, leadingDimension: ldb
        ) { b, ldb in
            var uplo = triangle._lapackCharacter
            var n = n
            var nrhs = nrhs
            var lda = lda
            var ldb = ldb
            var info = 0
            zpotrs_(
                &uplo, &n, &nrhs, _complexDoublePointer(a), &lda, _complexDoublePointer(b),
                &ldb, &info)
            return _lapackeInfo(info)
        }
    }
}

internal func _accelerateSposv(
    layout: LAPACK.Layout, triangle: LAPACK.Triangle, n: Int, nrhs: Int,
    a: UnsafeMutablePointer<Float>, lda: Int, b: UnsafeMutablePointer<Float>, ldb: Int
) -> Int {
    if n < 0 { return -3 }
    if nrhs < 0 { return -4 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: n) { return -6 }
    if ldb < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: nrhs) {
        return -8
    }
    return _withLAPACKColumnMajorMutableMatrix(
        layout: layout, rows: n, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        _withLAPACKColumnMajorMutableMatrix(
            layout: layout, rows: n, columns: nrhs, matrix: b, leadingDimension: ldb
        ) { b, ldb in
            var uplo = triangle._lapackCharacter
            var n = n
            var nrhs = nrhs
            var lda = lda
            var ldb = ldb
            var info = 0
            sposv_(&uplo, &n, &nrhs, a, &lda, b, &ldb, &info)
            return _lapackeInfo(info)
        }
    }
}

internal func _accelerateDposv(
    layout: LAPACK.Layout, triangle: LAPACK.Triangle, n: Int, nrhs: Int,
    a: UnsafeMutablePointer<Double>, lda: Int, b: UnsafeMutablePointer<Double>, ldb: Int
) -> Int {
    if n < 0 { return -3 }
    if nrhs < 0 { return -4 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: n) { return -6 }
    if ldb < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: nrhs) {
        return -8
    }
    return _withLAPACKColumnMajorMutableMatrix(
        layout: layout, rows: n, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        _withLAPACKColumnMajorMutableMatrix(
            layout: layout, rows: n, columns: nrhs, matrix: b, leadingDimension: ldb
        ) { b, ldb in
            var uplo = triangle._lapackCharacter
            var n = n
            var nrhs = nrhs
            var lda = lda
            var ldb = ldb
            var info = 0
            dposv_(&uplo, &n, &nrhs, a, &lda, b, &ldb, &info)
            return _lapackeInfo(info)
        }
    }
}

internal func _accelerateCposv(
    layout: LAPACK.Layout, triangle: LAPACK.Triangle, n: Int, nrhs: Int,
    a: UnsafeMutablePointer<Complex<Float>>, lda: Int, b: UnsafeMutablePointer<Complex<Float>>,
    ldb: Int
) -> Int {
    if n < 0 { return -3 }
    if nrhs < 0 { return -4 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: n) { return -6 }
    if ldb < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: nrhs) {
        return -8
    }
    return _withLAPACKColumnMajorMutableMatrix(
        layout: layout, rows: n, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        _withLAPACKColumnMajorMutableMatrix(
            layout: layout, rows: n, columns: nrhs, matrix: b, leadingDimension: ldb
        ) { b, ldb in
            var uplo = triangle._lapackCharacter
            var n = n
            var nrhs = nrhs
            var lda = lda
            var ldb = ldb
            var info = 0
            cposv_(
                &uplo, &n, &nrhs, _complexFloatPointer(a), &lda,
                _complexFloatPointer(b), &ldb, &info)
            return _lapackeInfo(info)
        }
    }
}

internal func _accelerateZposv(
    layout: LAPACK.Layout, triangle: LAPACK.Triangle, n: Int, nrhs: Int,
    a: UnsafeMutablePointer<Complex<Double>>, lda: Int,
    b: UnsafeMutablePointer<Complex<Double>>,
    ldb: Int
) -> Int {
    if n < 0 { return -3 }
    if nrhs < 0 { return -4 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: n) { return -6 }
    if ldb < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: nrhs) {
        return -8
    }
    return _withLAPACKColumnMajorMutableMatrix(
        layout: layout, rows: n, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        _withLAPACKColumnMajorMutableMatrix(
            layout: layout, rows: n, columns: nrhs, matrix: b, leadingDimension: ldb
        ) { b, ldb in
            var uplo = triangle._lapackCharacter
            var n = n
            var nrhs = nrhs
            var lda = lda
            var ldb = ldb
            var info = 0
            zposv_(
                &uplo, &n, &nrhs, _complexDoublePointer(a), &lda,
                _complexDoublePointer(b), &ldb, &info)
            return _lapackeInfo(info)
        }
    }
}

internal func _accelerateSpotri(
    layout: LAPACK.Layout, triangle: LAPACK.Triangle, n: Int, a: UnsafeMutablePointer<Float>,
    lda: Int
) -> Int {
    if n < 0 { return -3 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: n) { return -5 }
    return _withLAPACKColumnMajorMutableMatrix(
        layout: layout, rows: n, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        var uplo = triangle._lapackCharacter
        var n = n
        var lda = lda
        var info = 0
        spotri_(&uplo, &n, a, &lda, &info)
        return _lapackeInfo(info)
    }
}

internal func _accelerateDpotri(
    layout: LAPACK.Layout, triangle: LAPACK.Triangle, n: Int, a: UnsafeMutablePointer<Double>,
    lda: Int
) -> Int {
    if n < 0 { return -3 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: n) { return -5 }
    return _withLAPACKColumnMajorMutableMatrix(
        layout: layout, rows: n, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        var uplo = triangle._lapackCharacter
        var n = n
        var lda = lda
        var info = 0
        dpotri_(&uplo, &n, a, &lda, &info)
        return _lapackeInfo(info)
    }
}

internal func _accelerateCpotri(
    layout: LAPACK.Layout, triangle: LAPACK.Triangle, n: Int,
    a: UnsafeMutablePointer<Complex<Float>>, lda: Int
) -> Int {
    if n < 0 { return -3 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: n) { return -5 }
    return _withLAPACKColumnMajorMutableMatrix(
        layout: layout, rows: n, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        var uplo = triangle._lapackCharacter
        var n = n
        var lda = lda
        var info = 0
        cpotri_(&uplo, &n, _complexFloatPointer(a), &lda, &info)
        return _lapackeInfo(info)
    }
}

internal func _accelerateZpotri(
    layout: LAPACK.Layout, triangle: LAPACK.Triangle, n: Int,
    a: UnsafeMutablePointer<Complex<Double>>, lda: Int
) -> Int {
    if n < 0 { return -3 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: n) { return -5 }
    return _withLAPACKColumnMajorMutableMatrix(
        layout: layout, rows: n, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        var uplo = triangle._lapackCharacter
        var n = n
        var lda = lda
        var info = 0
        zpotri_(&uplo, &n, _complexDoublePointer(a), &lda, &info)
        return _lapackeInfo(info)
    }
}

// MARK: Triangular systems

internal func _accelerateStrtrs(
    layout: LAPACK.Layout, triangle: LAPACK.Triangle, transpose: LAPACK.Transpose,
    diagonal: LAPACK.Diagonal, n: Int, nrhs: Int, a: UnsafePointer<Float>, lda: Int,
    b: UnsafeMutablePointer<Float>, ldb: Int
) -> Int {
    if n < 0 { return -5 }
    if nrhs < 0 { return -6 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: n) { return -8 }
    if ldb < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: nrhs) {
        return -10
    }
    return _withLAPACKColumnMajorInputMatrix(
        layout: layout, rows: n, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        _withLAPACKColumnMajorMutableMatrix(
            layout: layout, rows: n, columns: nrhs, matrix: b, leadingDimension: ldb
        ) { b, ldb in
            var uplo = triangle._lapackCharacter
            var trans = transpose._character
            var diag = diagonal._lapackCharacter
            var n = n
            var nrhs = nrhs
            var lda = lda
            var ldb = ldb
            var info = 0
            strtrs_(&uplo, &trans, &diag, &n, &nrhs, a, &lda, b, &ldb, &info)
            return _lapackeInfo(info)
        }
    }
}

internal func _accelerateDtrtrs(
    layout: LAPACK.Layout, triangle: LAPACK.Triangle, transpose: LAPACK.Transpose,
    diagonal: LAPACK.Diagonal, n: Int, nrhs: Int, a: UnsafePointer<Double>, lda: Int,
    b: UnsafeMutablePointer<Double>, ldb: Int
) -> Int {
    if n < 0 { return -5 }
    if nrhs < 0 { return -6 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: n) { return -8 }
    if ldb < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: nrhs) {
        return -10
    }
    return _withLAPACKColumnMajorInputMatrix(
        layout: layout, rows: n, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        _withLAPACKColumnMajorMutableMatrix(
            layout: layout, rows: n, columns: nrhs, matrix: b, leadingDimension: ldb
        ) { b, ldb in
            var uplo = triangle._lapackCharacter
            var trans = transpose._character
            var diag = diagonal._lapackCharacter
            var n = n
            var nrhs = nrhs
            var lda = lda
            var ldb = ldb
            var info = 0
            dtrtrs_(&uplo, &trans, &diag, &n, &nrhs, a, &lda, b, &ldb, &info)
            return _lapackeInfo(info)
        }
    }
}

internal func _accelerateCtrtrs(
    layout: LAPACK.Layout, triangle: LAPACK.Triangle, transpose: LAPACK.Transpose,
    diagonal: LAPACK.Diagonal, n: Int, nrhs: Int, a: UnsafePointer<Complex<Float>>, lda: Int,
    b: UnsafeMutablePointer<Complex<Float>>, ldb: Int
) -> Int {
    if n < 0 { return -5 }
    if nrhs < 0 { return -6 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: n) { return -8 }
    if ldb < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: nrhs) {
        return -10
    }
    return _withLAPACKColumnMajorInputMatrix(
        layout: layout, rows: n, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        _withLAPACKColumnMajorMutableMatrix(
            layout: layout, rows: n, columns: nrhs, matrix: b, leadingDimension: ldb
        ) { b, ldb in
            var uplo = triangle._lapackCharacter
            var trans = transpose._character
            var diag = diagonal._lapackCharacter
            var n = n
            var nrhs = nrhs
            var lda = lda
            var ldb = ldb
            var info = 0
            ctrtrs_(
                &uplo, &trans, &diag, &n, &nrhs, _complexFloatPointer(a), &lda,
                _complexFloatPointer(b),
                &ldb, &info)
            return _lapackeInfo(info)
        }
    }
}

internal func _accelerateZtrtrs(
    layout: LAPACK.Layout, triangle: LAPACK.Triangle, transpose: LAPACK.Transpose,
    diagonal: LAPACK.Diagonal, n: Int, nrhs: Int, a: UnsafePointer<Complex<Double>>, lda: Int,
    b: UnsafeMutablePointer<Complex<Double>>, ldb: Int
) -> Int {
    if n < 0 { return -5 }
    if nrhs < 0 { return -6 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: n) { return -8 }
    if ldb < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: nrhs) {
        return -10
    }
    return _withLAPACKColumnMajorInputMatrix(
        layout: layout, rows: n, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        _withLAPACKColumnMajorMutableMatrix(
            layout: layout, rows: n, columns: nrhs, matrix: b, leadingDimension: ldb
        ) { b, ldb in
            var uplo = triangle._lapackCharacter
            var trans = transpose._character
            var diag = diagonal._lapackCharacter
            var n = n
            var nrhs = nrhs
            var lda = lda
            var ldb = ldb
            var info = 0
            ztrtrs_(
                &uplo, &trans, &diag, &n, &nrhs, _complexDoublePointer(a), &lda,
                _complexDoublePointer(b),
                &ldb, &info)
            return _lapackeInfo(info)
        }
    }
}

internal func _accelerateStrtri(
    layout: LAPACK.Layout, triangle: LAPACK.Triangle, diagonal: LAPACK.Diagonal, n: Int,
    a: UnsafeMutablePointer<Float>, lda: Int
) -> Int {
    if n < 0 { return -4 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: n) { return -6 }
    return _withLAPACKColumnMajorMutableMatrix(
        layout: layout, rows: n, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        var uplo = triangle._lapackCharacter
        var diag = diagonal._lapackCharacter
        var n = n
        var lda = lda
        var info = 0
        strtri_(&uplo, &diag, &n, a, &lda, &info)
        return _lapackeInfo(info)
    }
}

internal func _accelerateDtrtri(
    layout: LAPACK.Layout, triangle: LAPACK.Triangle, diagonal: LAPACK.Diagonal, n: Int,
    a: UnsafeMutablePointer<Double>, lda: Int
) -> Int {
    if n < 0 { return -4 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: n) { return -6 }
    return _withLAPACKColumnMajorMutableMatrix(
        layout: layout, rows: n, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        var uplo = triangle._lapackCharacter
        var diag = diagonal._lapackCharacter
        var n = n
        var lda = lda
        var info = 0
        dtrtri_(&uplo, &diag, &n, a, &lda, &info)
        return _lapackeInfo(info)
    }
}

internal func _accelerateCtrtri(
    layout: LAPACK.Layout, triangle: LAPACK.Triangle, diagonal: LAPACK.Diagonal, n: Int,
    a: UnsafeMutablePointer<Complex<Float>>, lda: Int
) -> Int {
    if n < 0 { return -4 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: n) { return -6 }
    return _withLAPACKColumnMajorMutableMatrix(
        layout: layout, rows: n, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        var uplo = triangle._lapackCharacter
        var diag = diagonal._lapackCharacter
        var n = n
        var lda = lda
        var info = 0
        ctrtri_(&uplo, &diag, &n, _complexFloatPointer(a), &lda, &info)
        return _lapackeInfo(info)
    }
}

internal func _accelerateZtrtri(
    layout: LAPACK.Layout, triangle: LAPACK.Triangle, diagonal: LAPACK.Diagonal, n: Int,
    a: UnsafeMutablePointer<Complex<Double>>, lda: Int
) -> Int {
    if n < 0 { return -4 }
    if lda < _lapackMinimumLeadingDimension(layout: layout, rows: n, columns: n) { return -6 }
    return _withLAPACKColumnMajorMutableMatrix(
        layout: layout, rows: n, columns: n, matrix: a, leadingDimension: lda
    ) { a, lda in
        var uplo = triangle._lapackCharacter
        var diag = diagonal._lapackCharacter
        var n = n
        var lda = lda
        var info = 0
        ztrtri_(&uplo, &diag, &n, _complexDoublePointer(a), &lda, &info)
        return _lapackeInfo(info)
    }
}
#endif
