// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: MIT

#if canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
import Accelerate
import ComplexModule
import RealModule
import SebbuBLAS

private protocol _LAPACKWorkspaceScalar: LAPACKScalar {
    var _lapackWorkspaceCount: Int { get }
}

extension Float: _LAPACKWorkspaceScalar {
    fileprivate var _lapackWorkspaceCount: Int { Int(rounded(.up)) }
}

extension Double: _LAPACKWorkspaceScalar {
    fileprivate var _lapackWorkspaceCount: Int { Int(rounded(.up)) }
}

extension Complex: _LAPACKWorkspaceScalar where RealType: BinaryFloatingPoint {
    fileprivate var _lapackWorkspaceCount: Int { Int(real.rounded(.up)) }
}

@inline(__always)
private func _accelerateWorkspaceCount<Scalar: _LAPACKWorkspaceScalar>(
    _ query: Scalar
) -> Int {
    Swift.max(1, query._lapackWorkspaceCount)
}

// MARK: QR factorization

private func _accelerateGeqrf<Scalar: _LAPACKWorkspaceScalar>(
    layout: LAPACK.Layout,
    m: Int,
    n: Int,
    a: UnsafeMutablePointer<Scalar>,
    lda: Int,
    tau: UnsafeMutablePointer<Scalar>,
    _ call: (
        UnsafeMutablePointer<Int>, UnsafeMutablePointer<Int>,
        UnsafeMutablePointer<Scalar>, UnsafeMutablePointer<Int>,
        UnsafeMutablePointer<Scalar>, UnsafeMutablePointer<Scalar>,
        UnsafeMutablePointer<Int>, UnsafeMutablePointer<Int>
    ) -> Void
) -> Int {
    if m < 0 { return -2 }
    if n < 0 { return -3 }
    if lda
        < _lapackMinimumLeadingDimension(
            layout: layout, rows: m, columns: n
        )
    {
        return -5
    }
    return _withLAPACKColumnMajorMutableMatrix(
        layout: layout, rows: m, columns: n, matrix: a,
        leadingDimension: lda
    ) { a, lda in
        var m = m
        var n = n
        var lda = lda
        var lwork = -1
        var info = 0
        var query = Scalar.zero
        withUnsafeMutablePointer(to: &query) { query in
            call(&m, &n, a, &lda, tau, query, &lwork, &info)
        }
        if info != 0 { return _lapackeInfo(info) }
        lwork = _accelerateWorkspaceCount(query)
        var work = [Scalar](repeating: .zero, count: lwork)
        work.withUnsafeMutableBufferPointer { work in
            call(&m, &n, a, &lda, tau, work.baseAddress!, &lwork, &info)
        }
        return _lapackeInfo(info)
    }
}

internal func _accelerateSgeqrf(
    layout: LAPACK.Layout, m: Int, n: Int, a: UnsafeMutablePointer<Float>, lda: Int,
    tau: UnsafeMutablePointer<Float>
) -> Int {
    _accelerateGeqrf(layout: layout, m: m, n: n, a: a, lda: lda, tau: tau) {
        sgeqrf_($0, $1, $2, $3, $4, $5, $6, $7)
    }
}

internal func _accelerateDgeqrf(
    layout: LAPACK.Layout, m: Int, n: Int, a: UnsafeMutablePointer<Double>, lda: Int,
    tau: UnsafeMutablePointer<Double>
) -> Int {
    _accelerateGeqrf(layout: layout, m: m, n: n, a: a, lda: lda, tau: tau) {
        dgeqrf_($0, $1, $2, $3, $4, $5, $6, $7)
    }
}

internal func _accelerateCgeqrf(
    layout: LAPACK.Layout, m: Int, n: Int, a: UnsafeMutablePointer<Complex<Float>>, lda: Int,
    tau: UnsafeMutablePointer<Complex<Float>>
) -> Int {
    _accelerateGeqrf(layout: layout, m: m, n: n, a: a, lda: lda, tau: tau) {
        cgeqrf_(
            $0, $1, _complexFloatPointer($2), $3, _complexFloatPointer($4),
            _complexFloatPointer($5),
            $6, $7)
    }
}

internal func _accelerateZgeqrf(
    layout: LAPACK.Layout, m: Int, n: Int, a: UnsafeMutablePointer<Complex<Double>>, lda: Int,
    tau: UnsafeMutablePointer<Complex<Double>>
) -> Int {
    _accelerateGeqrf(layout: layout, m: m, n: n, a: a, lda: lda, tau: tau) {
        zgeqrf_(
            $0, $1, _complexDoublePointer($2), $3, _complexDoublePointer($4),
            _complexDoublePointer($5),
            $6, $7)
    }
}

private func _accelerateGenerateQ<Scalar: _LAPACKWorkspaceScalar>(
    layout: LAPACK.Layout,
    m: Int,
    n: Int,
    k: Int,
    a: UnsafeMutablePointer<Scalar>,
    lda: Int,
    tau: UnsafePointer<Scalar>,
    _ call: (
        UnsafeMutablePointer<Int>, UnsafeMutablePointer<Int>,
        UnsafeMutablePointer<Int>, UnsafeMutablePointer<Scalar>,
        UnsafeMutablePointer<Int>, UnsafePointer<Scalar>,
        UnsafeMutablePointer<Scalar>, UnsafeMutablePointer<Int>,
        UnsafeMutablePointer<Int>
    ) -> Void
) -> Int {
    if m < 0 { return -2 }
    if n < 0 || n > m { return -3 }
    if k < 0 || k > n { return -4 }
    if lda
        < _lapackMinimumLeadingDimension(
            layout: layout, rows: m, columns: n
        )
    {
        return -6
    }
    return _withLAPACKColumnMajorMutableMatrix(
        layout: layout, rows: m, columns: n, matrix: a,
        leadingDimension: lda
    ) { a, lda in
        var m = m
        var n = n
        var k = k
        var lda = lda
        var lwork = -1
        var info = 0
        var query = Scalar.zero
        withUnsafeMutablePointer(to: &query) { query in
            call(&m, &n, &k, a, &lda, tau, query, &lwork, &info)
        }
        if info != 0 { return _lapackeInfo(info) }
        lwork = _accelerateWorkspaceCount(query)
        var work = [Scalar](repeating: .zero, count: lwork)
        work.withUnsafeMutableBufferPointer { work in
            call(&m, &n, &k, a, &lda, tau, work.baseAddress!, &lwork, &info)
        }
        return _lapackeInfo(info)
    }
}

internal func _accelerateSorgqr(
    layout: LAPACK.Layout, m: Int, n: Int, k: Int, a: UnsafeMutablePointer<Float>, lda: Int,
    tau: UnsafePointer<Float>
) -> Int {
    _accelerateGenerateQ(layout: layout, m: m, n: n, k: k, a: a, lda: lda, tau: tau) {
        sorgqr_($0, $1, $2, $3, $4, $5, $6, $7, $8)
    }
}

internal func _accelerateDorgqr(
    layout: LAPACK.Layout, m: Int, n: Int, k: Int, a: UnsafeMutablePointer<Double>, lda: Int,
    tau: UnsafePointer<Double>
) -> Int {
    _accelerateGenerateQ(layout: layout, m: m, n: n, k: k, a: a, lda: lda, tau: tau) {
        dorgqr_($0, $1, $2, $3, $4, $5, $6, $7, $8)
    }
}

internal func _accelerateCungqr(
    layout: LAPACK.Layout, m: Int, n: Int, k: Int, a: UnsafeMutablePointer<Complex<Float>>,
    lda: Int, tau: UnsafePointer<Complex<Float>>
) -> Int {
    _accelerateGenerateQ(layout: layout, m: m, n: n, k: k, a: a, lda: lda, tau: tau) {
        cungqr_(
            $0, $1, $2, _complexFloatPointer($3), $4, _complexFloatPointer($5),
            _complexFloatPointer($6), $7, $8)
    }
}

internal func _accelerateZungqr(
    layout: LAPACK.Layout, m: Int, n: Int, k: Int, a: UnsafeMutablePointer<Complex<Double>>,
    lda: Int, tau: UnsafePointer<Complex<Double>>
) -> Int {
    _accelerateGenerateQ(layout: layout, m: m, n: n, k: k, a: a, lda: lda, tau: tau) {
        zungqr_(
            $0, $1, $2, _complexDoublePointer($3), $4, _complexDoublePointer($5),
            _complexDoublePointer($6), $7, $8)
    }
}

// MARK: Least squares

private func _accelerateGels<Scalar: _LAPACKWorkspaceScalar>(
    layout: LAPACK.Layout,
    transpose: LAPACK.Transpose,
    m: Int,
    n: Int,
    nrhs: Int,
    a: UnsafeMutablePointer<Scalar>,
    lda: Int,
    b: UnsafeMutablePointer<Scalar>,
    ldb: Int,
    _ call: (
        UnsafeMutablePointer<CChar>, UnsafeMutablePointer<Int>,
        UnsafeMutablePointer<Int>, UnsafeMutablePointer<Int>,
        UnsafeMutablePointer<Scalar>, UnsafeMutablePointer<Int>,
        UnsafeMutablePointer<Scalar>, UnsafeMutablePointer<Int>,
        UnsafeMutablePointer<Scalar>, UnsafeMutablePointer<Int>,
        UnsafeMutablePointer<Int>
    ) -> Void
) -> Int {
    if m < 0 { return -3 }
    if n < 0 { return -4 }
    if nrhs < 0 { return -5 }
    if lda
        < _lapackMinimumLeadingDimension(
            layout: layout, rows: m, columns: n
        )
    {
        return -7
    }
    if ldb
        < _lapackMinimumLeadingDimension(
            layout: layout, rows: Swift.max(m, n), columns: nrhs
        )
    {
        return -9
    }
    return _withLAPACKColumnMajorMutableMatrix(
        layout: layout, rows: m, columns: n, matrix: a,
        leadingDimension: lda
    ) { a, lda in
        _withLAPACKColumnMajorMutableMatrix(
            layout: layout, rows: Swift.max(m, n), columns: nrhs,
            matrix: b, leadingDimension: ldb
        ) { b, ldb in
            var trans = transpose._character
            var m = m
            var n = n
            var nrhs = nrhs
            var lda = lda
            var ldb = ldb
            var lwork = -1
            var info = 0
            var query = Scalar.zero
            withUnsafeMutablePointer(to: &query) { query in
                call(&trans, &m, &n, &nrhs, a, &lda, b, &ldb, query, &lwork, &info)
            }
            if info != 0 { return _lapackeInfo(info) }
            lwork = _accelerateWorkspaceCount(query)
            var work = [Scalar](repeating: .zero, count: lwork)
            work.withUnsafeMutableBufferPointer { work in
                call(&trans, &m, &n, &nrhs, a, &lda, b, &ldb, work.baseAddress!, &lwork, &info)
            }
            return _lapackeInfo(info)
        }
    }
}

internal func _accelerateSgels(
    layout: LAPACK.Layout, transpose: LAPACK.Transpose, m: Int, n: Int, nrhs: Int,
    a: UnsafeMutablePointer<Float>, lda: Int, b: UnsafeMutablePointer<Float>, ldb: Int
) -> Int {
    _accelerateGels(
        layout: layout, transpose: transpose, m: m, n: n, nrhs: nrhs, a: a, lda: lda, b: b,
        ldb: ldb
    ) {
        sgels_($0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
    }
}

internal func _accelerateDgels(
    layout: LAPACK.Layout, transpose: LAPACK.Transpose, m: Int, n: Int, nrhs: Int,
    a: UnsafeMutablePointer<Double>, lda: Int, b: UnsafeMutablePointer<Double>, ldb: Int
) -> Int {
    _accelerateGels(
        layout: layout, transpose: transpose, m: m, n: n, nrhs: nrhs, a: a, lda: lda, b: b,
        ldb: ldb
    ) {
        dgels_($0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
    }
}

internal func _accelerateCgels(
    layout: LAPACK.Layout, transpose: LAPACK.Transpose, m: Int, n: Int, nrhs: Int,
    a: UnsafeMutablePointer<Complex<Float>>, lda: Int, b: UnsafeMutablePointer<Complex<Float>>,
    ldb: Int
) -> Int {
    _accelerateGels(
        layout: layout, transpose: transpose, m: m, n: n, nrhs: nrhs, a: a, lda: lda, b: b,
        ldb: ldb
    ) {
        cgels_(
            $0, $1, $2, $3, _complexFloatPointer($4), $5, _complexFloatPointer($6), $7,
            _complexFloatPointer($8), $9, $10)
    }
}

internal func _accelerateZgels(
    layout: LAPACK.Layout, transpose: LAPACK.Transpose, m: Int, n: Int, nrhs: Int,
    a: UnsafeMutablePointer<Complex<Double>>, lda: Int,
    b: UnsafeMutablePointer<Complex<Double>>,
    ldb: Int
) -> Int {
    _accelerateGels(
        layout: layout, transpose: transpose, m: m, n: n, nrhs: nrhs, a: a, lda: lda, b: b,
        ldb: ldb
    ) {
        zgels_(
            $0, $1, $2, $3, _complexDoublePointer($4), $5, _complexDoublePointer($6), $7,
            _complexDoublePointer($8), $9, $10)
    }
}

// MARK: Symmetric and Hermitian eigendecomposition

private func _accelerateSyev<Scalar: _LAPACKWorkspaceScalar>(
    layout: LAPACK.Layout,
    job: LAPACK.Eigenvectors,
    triangle: LAPACK.Triangle,
    n: Int,
    a: UnsafeMutablePointer<Scalar>,
    lda: Int,
    w: UnsafeMutablePointer<Scalar.Magnitude>,
    _ call: (
        UnsafeMutablePointer<CChar>, UnsafeMutablePointer<CChar>,
        UnsafeMutablePointer<Int>, UnsafeMutablePointer<Scalar>,
        UnsafeMutablePointer<Int>, UnsafeMutablePointer<Scalar.Magnitude>,
        UnsafeMutablePointer<Scalar>, UnsafeMutablePointer<Int>,
        UnsafeMutablePointer<Scalar.Magnitude>, UnsafeMutablePointer<Int>
    ) -> Void
) -> Int {
    if n < 0 { return -4 }
    if lda
        < _lapackMinimumLeadingDimension(
            layout: layout, rows: n, columns: n
        )
    {
        return -6
    }
    return _withLAPACKColumnMajorMutableMatrix(
        layout: layout, rows: n, columns: n, matrix: a,
        leadingDimension: lda
    ) { a, lda in
        var job = job._character
        var uplo = triangle._lapackCharacter
        var n = n
        var lda = lda
        var lwork = -1
        var info = 0
        var query = Scalar.zero
        var rwork = [Scalar.Magnitude](
            repeating: .zero, count: Swift.max(1, 3 * n - 2)
        )
        rwork.withUnsafeMutableBufferPointer { rwork in
            withUnsafeMutablePointer(to: &query) { query in
                call(&job, &uplo, &n, a, &lda, w, query, &lwork, rwork.baseAddress!, &info)
            }
        }
        if info != 0 { return _lapackeInfo(info) }
        lwork = _accelerateWorkspaceCount(query)
        var work = [Scalar](repeating: .zero, count: lwork)
        work.withUnsafeMutableBufferPointer { work in
            rwork.withUnsafeMutableBufferPointer { rwork in
                call(
                    &job, &uplo, &n, a, &lda, w, work.baseAddress!, &lwork, rwork.baseAddress!,
                    &info)
            }
        }
        return _lapackeInfo(info)
    }
}

internal func _accelerateSsyev(
    layout: LAPACK.Layout, job: LAPACK.Eigenvectors, triangle: LAPACK.Triangle, n: Int,
    a: UnsafeMutablePointer<Float>, lda: Int, w: UnsafeMutablePointer<Float>
) -> Int {
    _accelerateSyev(layout: layout, job: job, triangle: triangle, n: n, a: a, lda: lda, w: w) {
        ssyev_($0, $1, $2, $3, $4, $5, $6, $7, $9)
    }
}

internal func _accelerateDsyev(
    layout: LAPACK.Layout, job: LAPACK.Eigenvectors, triangle: LAPACK.Triangle, n: Int,
    a: UnsafeMutablePointer<Double>, lda: Int, w: UnsafeMutablePointer<Double>
) -> Int {
    _accelerateSyev(layout: layout, job: job, triangle: triangle, n: n, a: a, lda: lda, w: w) {
        dsyev_($0, $1, $2, $3, $4, $5, $6, $7, $9)
    }
}

internal func _accelerateCheev(
    layout: LAPACK.Layout, job: LAPACK.Eigenvectors, triangle: LAPACK.Triangle, n: Int,
    a: UnsafeMutablePointer<Complex<Float>>, lda: Int, w: UnsafeMutablePointer<Float>
) -> Int {
    _accelerateSyev(layout: layout, job: job, triangle: triangle, n: n, a: a, lda: lda, w: w) {
        cheev_(
            $0, $1, $2, _complexFloatPointer($3), $4, $5, _complexFloatPointer($6), $7, $8, $9)
    }
}

internal func _accelerateZheev(
    layout: LAPACK.Layout, job: LAPACK.Eigenvectors, triangle: LAPACK.Triangle, n: Int,
    a: UnsafeMutablePointer<Complex<Double>>, lda: Int, w: UnsafeMutablePointer<Double>
) -> Int {
    _accelerateSyev(layout: layout, job: job, triangle: triangle, n: n, a: a, lda: lda, w: w) {
        zheev_(
            $0, $1, $2, _complexDoublePointer($3), $4, $5, _complexDoublePointer($6), $7, $8, $9
        )
    }
}

// MARK: General eigendecomposition

private func _accelerateRealGeev<Scalar: _LAPACKWorkspaceScalar>(
    layout: LAPACK.Layout,
    jobVL: LAPACK.Eigenvectors,
    jobVR: LAPACK.Eigenvectors,
    n: Int,
    a: UnsafeMutablePointer<Scalar>,
    lda: Int,
    wr: UnsafeMutablePointer<Scalar>,
    wi: UnsafeMutablePointer<Scalar>,
    vl: UnsafeMutablePointer<Scalar>,
    ldvl: Int,
    vr: UnsafeMutablePointer<Scalar>,
    ldvr: Int,
    _ call: (
        UnsafeMutablePointer<CChar>, UnsafeMutablePointer<CChar>,
        UnsafeMutablePointer<Int>, UnsafeMutablePointer<Scalar>,
        UnsafeMutablePointer<Int>, UnsafeMutablePointer<Scalar>,
        UnsafeMutablePointer<Scalar>, UnsafeMutablePointer<Scalar>,
        UnsafeMutablePointer<Int>, UnsafeMutablePointer<Scalar>,
        UnsafeMutablePointer<Int>, UnsafeMutablePointer<Scalar>,
        UnsafeMutablePointer<Int>, UnsafeMutablePointer<Int>
    ) -> Void
) -> Int {
    if n < 0 { return -4 }
    if lda
        < _lapackMinimumLeadingDimension(
            layout: layout, rows: n, columns: n
        )
    {
        return -6
    }
    if ldvl < (jobVL == .vectors ? Swift.max(1, n) : 1) { return -10 }
    if ldvr < (jobVR == .vectors ? Swift.max(1, n) : 1) { return -12 }

    let vlColumns = jobVL == .vectors ? n : 0
    let vrColumns = jobVR == .vectors ? n : 0
    return _withLAPACKColumnMajorMutableMatrix(
        layout: layout, rows: n, columns: n, matrix: a,
        leadingDimension: lda
    ) { a, lda in
        _withLAPACKColumnMajorOutputMatrix(
            layout: layout, rows: n, columns: vlColumns,
            matrix: vl, leadingDimension: ldvl, initialValue: Scalar.zero
        ) { vl, ldvl in
            _withLAPACKColumnMajorOutputMatrix(
                layout: layout, rows: n, columns: vrColumns,
                matrix: vr, leadingDimension: ldvr, initialValue: Scalar.zero
            ) { vr, ldvr in
                var jobVL = jobVL._character
                var jobVR = jobVR._character
                var n = n
                var lda = lda
                var ldvl = ldvl
                var ldvr = ldvr
                var lwork = -1
                var info = 0
                var query = Scalar.zero
                withUnsafeMutablePointer(to: &query) { query in
                    call(
                        &jobVL, &jobVR, &n, a, &lda, wr, wi,
                        vl, &ldvl, vr, &ldvr, query, &lwork, &info)
                }
                if info != 0 { return _lapackeInfo(info) }
                lwork = _accelerateWorkspaceCount(query)
                var work = [Scalar](repeating: .zero, count: lwork)
                work.withUnsafeMutableBufferPointer { work in
                    call(
                        &jobVL, &jobVR, &n, a, &lda, wr, wi,
                        vl, &ldvl, vr, &ldvr, work.baseAddress!, &lwork, &info)
                }
                return _lapackeInfo(info)
            }
        }
    }
}

internal func _accelerateSgeev(
    layout: LAPACK.Layout, jobVL: LAPACK.Eigenvectors, jobVR: LAPACK.Eigenvectors, n: Int,
    a: UnsafeMutablePointer<Float>, lda: Int,
    wr: UnsafeMutablePointer<Float>, wi: UnsafeMutablePointer<Float>,
    vl: UnsafeMutablePointer<Float>, ldvl: Int,
    vr: UnsafeMutablePointer<Float>, ldvr: Int
) -> Int {
    _accelerateRealGeev(
        layout: layout, jobVL: jobVL, jobVR: jobVR, n: n,
        a: a, lda: lda, wr: wr, wi: wi, vl: vl, ldvl: ldvl, vr: vr, ldvr: ldvr
    ) {
        sgeev_($0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
    }
}

internal func _accelerateDgeev(
    layout: LAPACK.Layout, jobVL: LAPACK.Eigenvectors, jobVR: LAPACK.Eigenvectors, n: Int,
    a: UnsafeMutablePointer<Double>, lda: Int,
    wr: UnsafeMutablePointer<Double>, wi: UnsafeMutablePointer<Double>,
    vl: UnsafeMutablePointer<Double>, ldvl: Int,
    vr: UnsafeMutablePointer<Double>, ldvr: Int
) -> Int {
    _accelerateRealGeev(
        layout: layout, jobVL: jobVL, jobVR: jobVR, n: n,
        a: a, lda: lda, wr: wr, wi: wi, vl: vl, ldvl: ldvl, vr: vr, ldvr: ldvr
    ) {
        dgeev_($0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
    }
}

private func _accelerateComplexGeev<Scalar: _LAPACKWorkspaceScalar>(
    layout: LAPACK.Layout,
    jobVL: LAPACK.Eigenvectors,
    jobVR: LAPACK.Eigenvectors,
    n: Int,
    a: UnsafeMutablePointer<Scalar>,
    lda: Int,
    w: UnsafeMutablePointer<Scalar>,
    vl: UnsafeMutablePointer<Scalar>,
    ldvl: Int,
    vr: UnsafeMutablePointer<Scalar>,
    ldvr: Int,
    _ call: (
        UnsafeMutablePointer<CChar>, UnsafeMutablePointer<CChar>,
        UnsafeMutablePointer<Int>, UnsafeMutablePointer<Scalar>,
        UnsafeMutablePointer<Int>, UnsafeMutablePointer<Scalar>,
        UnsafeMutablePointer<Scalar>, UnsafeMutablePointer<Int>,
        UnsafeMutablePointer<Scalar>, UnsafeMutablePointer<Int>,
        UnsafeMutablePointer<Scalar>, UnsafeMutablePointer<Int>,
        UnsafeMutablePointer<Scalar.Magnitude>, UnsafeMutablePointer<Int>
    ) -> Void
) -> Int {
    if n < 0 { return -4 }
    if lda
        < _lapackMinimumLeadingDimension(
            layout: layout, rows: n, columns: n
        )
    {
        return -6
    }
    if ldvl < (jobVL == .vectors ? Swift.max(1, n) : 1) { return -9 }
    if ldvr < (jobVR == .vectors ? Swift.max(1, n) : 1) { return -11 }

    let vlColumns = jobVL == .vectors ? n : 0
    let vrColumns = jobVR == .vectors ? n : 0
    return _withLAPACKColumnMajorMutableMatrix(
        layout: layout, rows: n, columns: n, matrix: a,
        leadingDimension: lda
    ) { a, lda in
        _withLAPACKColumnMajorOutputMatrix(
            layout: layout, rows: n, columns: vlColumns,
            matrix: vl, leadingDimension: ldvl, initialValue: Scalar.zero
        ) { vl, ldvl in
            _withLAPACKColumnMajorOutputMatrix(
                layout: layout, rows: n, columns: vrColumns,
                matrix: vr, leadingDimension: ldvr, initialValue: Scalar.zero
            ) { vr, ldvr in
                var jobVL = jobVL._character
                var jobVR = jobVR._character
                var n = n
                var lda = lda
                var ldvl = ldvl
                var ldvr = ldvr
                var lwork = -1
                var info = 0
                var query = Scalar.zero
                var rwork = [Scalar.Magnitude](
                    repeating: .zero, count: Swift.max(1, 2 * n)
                )
                rwork.withUnsafeMutableBufferPointer { rwork in
                    withUnsafeMutablePointer(to: &query) { query in
                        call(
                            &jobVL, &jobVR, &n, a, &lda, w,
                            vl, &ldvl, vr, &ldvr, query, &lwork,
                            rwork.baseAddress!, &info)
                    }
                }
                if info != 0 { return _lapackeInfo(info) }
                lwork = _accelerateWorkspaceCount(query)
                var work = [Scalar](repeating: .zero, count: lwork)
                work.withUnsafeMutableBufferPointer { work in
                    rwork.withUnsafeMutableBufferPointer { rwork in
                        call(
                            &jobVL, &jobVR, &n, a, &lda, w,
                            vl, &ldvl, vr, &ldvr, work.baseAddress!, &lwork,
                            rwork.baseAddress!, &info)
                    }
                }
                return _lapackeInfo(info)
            }
        }
    }
}

internal func _accelerateCgeev(
    layout: LAPACK.Layout, jobVL: LAPACK.Eigenvectors, jobVR: LAPACK.Eigenvectors, n: Int,
    a: UnsafeMutablePointer<Complex<Float>>, lda: Int,
    w: UnsafeMutablePointer<Complex<Float>>,
    vl: UnsafeMutablePointer<Complex<Float>>, ldvl: Int,
    vr: UnsafeMutablePointer<Complex<Float>>, ldvr: Int
) -> Int {
    _accelerateComplexGeev(
        layout: layout, jobVL: jobVL, jobVR: jobVR, n: n,
        a: a, lda: lda, w: w, vl: vl, ldvl: ldvl, vr: vr, ldvr: ldvr
    ) {
        cgeev_(
            $0, $1, $2, _complexFloatPointer($3), $4, _complexFloatPointer($5),
            _complexFloatPointer($6), $7, _complexFloatPointer($8), $9,
            _complexFloatPointer($10), $11, $12, $13)
    }
}

internal func _accelerateZgeev(
    layout: LAPACK.Layout, jobVL: LAPACK.Eigenvectors, jobVR: LAPACK.Eigenvectors, n: Int,
    a: UnsafeMutablePointer<Complex<Double>>, lda: Int,
    w: UnsafeMutablePointer<Complex<Double>>,
    vl: UnsafeMutablePointer<Complex<Double>>, ldvl: Int,
    vr: UnsafeMutablePointer<Complex<Double>>, ldvr: Int
) -> Int {
    _accelerateComplexGeev(
        layout: layout, jobVL: jobVL, jobVR: jobVR, n: n,
        a: a, lda: lda, w: w, vl: vl, ldvl: ldvl, vr: vr, ldvr: ldvr
    ) {
        zgeev_(
            $0, $1, $2, _complexDoublePointer($3), $4, _complexDoublePointer($5),
            _complexDoublePointer($6), $7, _complexDoublePointer($8), $9,
            _complexDoublePointer($10), $11, $12, $13)
    }
}

// MARK: General Schur decomposition

private let _accelerateSgeesSelectNone: LAPACK.SgeesSelect = { _, _ in 0 }
private let _accelerateDgeesSelectNone: LAPACK.DgeesSelect = { _, _ in 0 }
private let _accelerateCgeesSelectNone: LAPACK.CgeesSelect = { _ in 0 }
private let _accelerateZgeesSelectNone: LAPACK.ZgeesSelect = { _ in 0 }

@inline(__always)
private func _accelerateGeesArgumentError(
    layout: LAPACK.Layout,
    jobVS: LAPACK.Eigenvectors,
    n: Int,
    lda: Int,
    ldvs: Int,
    ldvsArgument: Int
) -> Int? {
    if n < 0 { return -5 }
    if lda
        < _lapackMinimumLeadingDimension(
            layout: layout, rows: n, columns: n
        )
    {
        return -7
    }
    let minimumLDVS: Int
    if layout == .rowMajor {
        minimumLDVS = Swift.max(1, n)
    } else {
        minimumLDVS = jobVS == .vectors ? Swift.max(1, n) : 1
    }
    if ldvs < minimumLDVS { return -ldvsArgument }
    return nil
}

private func _accelerateRealGees<Scalar: _LAPACKWorkspaceScalar>(
    layout: LAPACK.Layout,
    jobVS: LAPACK.Eigenvectors,
    sort: LAPACK.SchurSort,
    n: Int,
    a: UnsafeMutablePointer<Scalar>,
    lda: Int,
    sdim: UnsafeMutablePointer<Int>,
    wr: UnsafeMutablePointer<Scalar>,
    wi: UnsafeMutablePointer<Scalar>,
    vs: UnsafeMutablePointer<Scalar>,
    ldvs: Int,
    _ call: (
        UnsafeMutablePointer<CChar>, UnsafeMutablePointer<CChar>,
        UnsafeMutablePointer<Int>, UnsafeMutablePointer<Scalar>,
        UnsafeMutablePointer<Int>, UnsafeMutablePointer<Int>,
        UnsafeMutablePointer<Scalar>, UnsafeMutablePointer<Scalar>,
        UnsafeMutablePointer<Scalar>, UnsafeMutablePointer<Int>,
        UnsafeMutablePointer<Scalar>, UnsafeMutablePointer<Int>,
        UnsafeMutablePointer<Int>, UnsafeMutablePointer<Int>
    ) -> Void
) -> Int {
    if let error = _accelerateGeesArgumentError(
        layout: layout, jobVS: jobVS, n: n, lda: lda, ldvs: ldvs,
        ldvsArgument: 12
    ) {
        return error
    }

    let vsColumns = jobVS == .vectors ? n : 0
    return _withLAPACKColumnMajorMutableMatrix(
        layout: layout, rows: n, columns: n, matrix: a,
        leadingDimension: lda
    ) { a, lda in
        _withLAPACKColumnMajorOutputMatrix(
            layout: layout, rows: n, columns: vsColumns, matrix: vs,
            leadingDimension: ldvs, initialValue: Scalar.zero
        ) { vs, ldvs in
            var jobVS = jobVS._character
            var sort = sort._character
            var n = n
            var lda = lda
            var ldvs = ldvs
            var lwork = -1
            var info = 0
            var query = Scalar.zero
            var bwork = [Int](repeating: 0, count: Swift.max(1, n))
            bwork.withUnsafeMutableBufferPointer { bwork in
                withUnsafeMutablePointer(to: &query) { query in
                    call(
                        &jobVS, &sort, &n, a, &lda, sdim, wr, wi, vs,
                        &ldvs, query, &lwork, bwork.baseAddress!, &info)
                }
            }
            if info != 0 { return _lapackeInfo(info) }
            lwork = _accelerateWorkspaceCount(query)
            var work = [Scalar](repeating: .zero, count: lwork)
            work.withUnsafeMutableBufferPointer { work in
                bwork.withUnsafeMutableBufferPointer { bwork in
                    call(
                        &jobVS, &sort, &n, a, &lda, sdim, wr, wi, vs,
                        &ldvs, work.baseAddress!, &lwork,
                        bwork.baseAddress!, &info)
                }
            }
            return _lapackeInfo(info)
        }
    }
}

internal func _accelerateSgees(
    layout: LAPACK.Layout, jobVS: LAPACK.Eigenvectors,
    sort: LAPACK.SchurSort, select: LAPACK.SgeesSelect?, n: Int,
    a: UnsafeMutablePointer<Float>, lda: Int,
    sdim: UnsafeMutablePointer<Int>, wr: UnsafeMutablePointer<Float>,
    wi: UnsafeMutablePointer<Float>, vs: UnsafeMutablePointer<Float>,
    ldvs: Int
) -> Int {
    if sort == .selected && select == nil { return -4 }
    let select = select ?? _accelerateSgeesSelectNone
    return _accelerateRealGees(
        layout: layout, jobVS: jobVS, sort: sort, n: n, a: a, lda: lda,
        sdim: sdim, wr: wr, wi: wi, vs: vs, ldvs: ldvs
    ) {
        sgees_(
            $0, $1, select, $2, $3, $4, $5, $6, $7, $8, $9, $10,
            $11, $12, $13)
    }
}

internal func _accelerateDgees(
    layout: LAPACK.Layout, jobVS: LAPACK.Eigenvectors,
    sort: LAPACK.SchurSort, select: LAPACK.DgeesSelect?, n: Int,
    a: UnsafeMutablePointer<Double>, lda: Int,
    sdim: UnsafeMutablePointer<Int>, wr: UnsafeMutablePointer<Double>,
    wi: UnsafeMutablePointer<Double>, vs: UnsafeMutablePointer<Double>,
    ldvs: Int
) -> Int {
    if sort == .selected && select == nil { return -4 }
    let select = select ?? _accelerateDgeesSelectNone
    return _accelerateRealGees(
        layout: layout, jobVS: jobVS, sort: sort, n: n, a: a, lda: lda,
        sdim: sdim, wr: wr, wi: wi, vs: vs, ldvs: ldvs
    ) {
        dgees_(
            $0, $1, select, $2, $3, $4, $5, $6, $7, $8, $9, $10,
            $11, $12, $13)
    }
}

private func _accelerateComplexGees<Scalar: _LAPACKWorkspaceScalar>(
    layout: LAPACK.Layout,
    jobVS: LAPACK.Eigenvectors,
    sort: LAPACK.SchurSort,
    n: Int,
    a: UnsafeMutablePointer<Scalar>,
    lda: Int,
    sdim: UnsafeMutablePointer<Int>,
    w: UnsafeMutablePointer<Scalar>,
    vs: UnsafeMutablePointer<Scalar>,
    ldvs: Int,
    _ call: (
        UnsafeMutablePointer<CChar>, UnsafeMutablePointer<CChar>,
        UnsafeMutablePointer<Int>, UnsafeMutablePointer<Scalar>,
        UnsafeMutablePointer<Int>, UnsafeMutablePointer<Int>,
        UnsafeMutablePointer<Scalar>, UnsafeMutablePointer<Scalar>,
        UnsafeMutablePointer<Int>, UnsafeMutablePointer<Scalar>,
        UnsafeMutablePointer<Int>, UnsafeMutablePointer<Scalar.Magnitude>,
        UnsafeMutablePointer<Int>, UnsafeMutablePointer<Int>
    ) -> Void
) -> Int {
    if let error = _accelerateGeesArgumentError(
        layout: layout, jobVS: jobVS, n: n, lda: lda, ldvs: ldvs,
        ldvsArgument: 11
    ) {
        return error
    }

    let vsColumns = jobVS == .vectors ? n : 0
    return _withLAPACKColumnMajorMutableMatrix(
        layout: layout, rows: n, columns: n, matrix: a,
        leadingDimension: lda
    ) { a, lda in
        _withLAPACKColumnMajorOutputMatrix(
            layout: layout, rows: n, columns: vsColumns, matrix: vs,
            leadingDimension: ldvs, initialValue: Scalar.zero
        ) { vs, ldvs in
            var jobVS = jobVS._character
            var sort = sort._character
            var n = n
            var lda = lda
            var ldvs = ldvs
            var lwork = -1
            var info = 0
            var query = Scalar.zero
            var rwork = [Scalar.Magnitude](
                repeating: .zero, count: Swift.max(1, n)
            )
            var bwork = [Int](repeating: 0, count: Swift.max(1, n))
            rwork.withUnsafeMutableBufferPointer { rwork in
                bwork.withUnsafeMutableBufferPointer { bwork in
                    withUnsafeMutablePointer(to: &query) { query in
                        call(
                            &jobVS, &sort, &n, a, &lda, sdim, w, vs,
                            &ldvs, query, &lwork, rwork.baseAddress!,
                            bwork.baseAddress!, &info)
                    }
                }
            }
            if info != 0 { return _lapackeInfo(info) }
            lwork = _accelerateWorkspaceCount(query)
            var work = [Scalar](repeating: .zero, count: lwork)
            work.withUnsafeMutableBufferPointer { work in
                rwork.withUnsafeMutableBufferPointer { rwork in
                    bwork.withUnsafeMutableBufferPointer { bwork in
                        call(
                            &jobVS, &sort, &n, a, &lda, sdim, w, vs,
                            &ldvs, work.baseAddress!, &lwork,
                            rwork.baseAddress!, bwork.baseAddress!, &info)
                    }
                }
            }
            return _lapackeInfo(info)
        }
    }
}

internal func _accelerateCgees(
    layout: LAPACK.Layout, jobVS: LAPACK.Eigenvectors,
    sort: LAPACK.SchurSort, select: LAPACK.CgeesSelect?, n: Int,
    a: UnsafeMutablePointer<Complex<Float>>, lda: Int,
    sdim: UnsafeMutablePointer<Int>, w: UnsafeMutablePointer<Complex<Float>>,
    vs: UnsafeMutablePointer<Complex<Float>>, ldvs: Int
) -> Int {
    if sort == .selected && select == nil { return -4 }
    let select = select ?? _accelerateCgeesSelectNone
    return _accelerateComplexGees(
        layout: layout, jobVS: jobVS, sort: sort, n: n, a: a, lda: lda,
        sdim: sdim, w: w, vs: vs, ldvs: ldvs
    ) {
        cgees_(
            $0, $1, select, $2, _complexFloatPointer($3), $4, $5,
            _complexFloatPointer($6), _complexFloatPointer($7), $8,
            _complexFloatPointer($9), $10, $11, $12, $13)
    }
}

internal func _accelerateZgees(
    layout: LAPACK.Layout, jobVS: LAPACK.Eigenvectors,
    sort: LAPACK.SchurSort, select: LAPACK.ZgeesSelect?, n: Int,
    a: UnsafeMutablePointer<Complex<Double>>, lda: Int,
    sdim: UnsafeMutablePointer<Int>, w: UnsafeMutablePointer<Complex<Double>>,
    vs: UnsafeMutablePointer<Complex<Double>>, ldvs: Int
) -> Int {
    if sort == .selected && select == nil { return -4 }
    let select = select ?? _accelerateZgeesSelectNone
    return _accelerateComplexGees(
        layout: layout, jobVS: jobVS, sort: sort, n: n, a: a, lda: lda,
        sdim: sdim, w: w, vs: vs, ldvs: ldvs
    ) {
        zgees_(
            $0, $1, select, $2, _complexDoublePointer($3), $4, $5,
            _complexDoublePointer($6), _complexDoublePointer($7), $8,
            _complexDoublePointer($9), $10, $11, $12, $13)
    }
}

// MARK: Singular value decomposition

@inline(__always)
private func _accelerateSVDOutputDimensions(
    jobU: LAPACK.SingularVectors,
    jobVT: LAPACK.SingularVectors,
    m: Int,
    n: Int
) -> (uRows: Int, uColumns: Int, vtRows: Int, vtColumns: Int) {
    let minimum = Swift.min(m, n)
    let uColumns: Int
    switch jobU {
    case .all: uColumns = m
    case .some: uColumns = minimum
    case .overwrite, .none: uColumns = 0
    }
    let vtRows: Int
    switch jobVT {
    case .all: vtRows = n
    case .some: vtRows = minimum
    case .overwrite, .none: vtRows = 0
    }
    return (m, uColumns, vtRows, n)
}

private func _accelerateGesvd<Scalar: _LAPACKWorkspaceScalar>(
    layout: LAPACK.Layout,
    jobU: LAPACK.SingularVectors,
    jobVT: LAPACK.SingularVectors,
    m: Int,
    n: Int,
    a: UnsafeMutablePointer<Scalar>,
    lda: Int,
    s: UnsafeMutablePointer<Scalar.Magnitude>,
    u: UnsafeMutablePointer<Scalar>,
    ldu: Int,
    vt: UnsafeMutablePointer<Scalar>,
    ldvt: Int,
    superb: UnsafeMutablePointer<Scalar.Magnitude>?,
    superbIsInRealWork: Bool,
    _ call: (
        UnsafeMutablePointer<CChar>, UnsafeMutablePointer<CChar>,
        UnsafeMutablePointer<Int>, UnsafeMutablePointer<Int>,
        UnsafeMutablePointer<Scalar>, UnsafeMutablePointer<Int>,
        UnsafeMutablePointer<Scalar.Magnitude>, UnsafeMutablePointer<Scalar>,
        UnsafeMutablePointer<Int>, UnsafeMutablePointer<Scalar>,
        UnsafeMutablePointer<Int>, UnsafeMutablePointer<Scalar>,
        UnsafeMutablePointer<Int>, UnsafeMutablePointer<Scalar.Magnitude>,
        UnsafeMutablePointer<Int>
    ) -> Void
) -> Int {
    if let error = _lapackGesvdArgumentError(
        layout: layout, jobU: jobU, jobVT: jobVT,
        m: m, n: n, lda: lda, ldu: ldu, ldvt: ldvt
    ) {
        return error
    }
    let dimensions = _accelerateSVDOutputDimensions(
        jobU: jobU, jobVT: jobVT, m: m, n: n
    )
    return _withLAPACKColumnMajorMutableMatrix(
        layout: layout, rows: m, columns: n, matrix: a,
        leadingDimension: lda
    ) { a, lda in
        _withLAPACKColumnMajorOutputMatrix(
            layout: layout, rows: dimensions.uRows,
            columns: dimensions.uColumns, matrix: u,
            leadingDimension: ldu, initialValue: .zero
        ) { u, ldu in
            _withLAPACKColumnMajorOutputMatrix(
                layout: layout, rows: dimensions.vtRows,
                columns: dimensions.vtColumns, matrix: vt,
                leadingDimension: ldvt, initialValue: .zero
            ) { vt, ldvt in
                var jobU = jobU._character
                var jobVT = jobVT._character
                var m = m
                var n = n
                var lda = lda
                var ldu = ldu
                var ldvt = ldvt
                var lwork = -1
                var info = 0
                var query = Scalar.zero
                let minimum = Swift.max(0, Swift.min(m, n))
                var rwork = [Scalar.Magnitude](
                    repeating: .zero, count: Swift.max(1, 5 * minimum)
                )
                rwork.withUnsafeMutableBufferPointer { rwork in
                    withUnsafeMutablePointer(to: &query) { query in
                        call(
                            &jobU, &jobVT, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, query,
                            &lwork,
                            rwork.baseAddress!, &info)
                    }
                }
                if info != 0 { return _lapackeInfo(info) }
                lwork = _accelerateWorkspaceCount(query)
                var work = [Scalar](repeating: .zero, count: lwork)
                work.withUnsafeMutableBufferPointer { work in
                    rwork.withUnsafeMutableBufferPointer { rwork in
                        call(
                            &jobU, &jobVT, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt,
                            work.baseAddress!, &lwork,
                            rwork.baseAddress!, &info)
                    }
                }
                if let superb {
                    for index in 0..<Swift.max(0, minimum - 1) {
                        superb[index] =
                            superbIsInRealWork
                            ? work[index + 1].realComponent
                            : rwork[index]
                    }
                }
                return _lapackeInfo(info)
            }
        }
    }
}

internal func _accelerateSgesvd(
    layout: LAPACK.Layout, jobU: LAPACK.SingularVectors, jobVT: LAPACK.SingularVectors, m: Int,
    n: Int, a: UnsafeMutablePointer<Float>, lda: Int, s: UnsafeMutablePointer<Float>,
    u: UnsafeMutablePointer<Float>, ldu: Int, vt: UnsafeMutablePointer<Float>, ldvt: Int,
    superb: UnsafeMutablePointer<Float>
) -> Int {
    _accelerateGesvd(
        layout: layout, jobU: jobU, jobVT: jobVT, m: m, n: n, a: a, lda: lda, s: s, u: u,
        ldu: ldu,
        vt: vt, ldvt: ldvt, superb: superb, superbIsInRealWork: true
    ) {
        sgesvd_($0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $14)
    }
}

internal func _accelerateDgesvd(
    layout: LAPACK.Layout, jobU: LAPACK.SingularVectors, jobVT: LAPACK.SingularVectors, m: Int,
    n: Int, a: UnsafeMutablePointer<Double>, lda: Int, s: UnsafeMutablePointer<Double>,
    u: UnsafeMutablePointer<Double>, ldu: Int, vt: UnsafeMutablePointer<Double>, ldvt: Int,
    superb: UnsafeMutablePointer<Double>
) -> Int {
    _accelerateGesvd(
        layout: layout, jobU: jobU, jobVT: jobVT, m: m, n: n, a: a, lda: lda, s: s, u: u,
        ldu: ldu,
        vt: vt, ldvt: ldvt, superb: superb, superbIsInRealWork: true
    ) {
        dgesvd_($0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $14)
    }
}

internal func _accelerateCgesvd(
    layout: LAPACK.Layout, jobU: LAPACK.SingularVectors, jobVT: LAPACK.SingularVectors, m: Int,
    n: Int, a: UnsafeMutablePointer<Complex<Float>>, lda: Int, s: UnsafeMutablePointer<Float>,
    u: UnsafeMutablePointer<Complex<Float>>, ldu: Int, vt: UnsafeMutablePointer<Complex<Float>>,
    ldvt: Int, superb: UnsafeMutablePointer<Float>
) -> Int {
    _accelerateGesvd(
        layout: layout, jobU: jobU, jobVT: jobVT, m: m, n: n, a: a, lda: lda, s: s, u: u,
        ldu: ldu,
        vt: vt, ldvt: ldvt, superb: superb, superbIsInRealWork: false
    ) {
        cgesvd_(
            $0, $1, $2, $3, _complexFloatPointer($4), $5, $6, _complexFloatPointer($7), $8,
            _complexFloatPointer($9), $10, _complexFloatPointer($11), $12, $13, $14)
    }
}

internal func _accelerateZgesvd(
    layout: LAPACK.Layout, jobU: LAPACK.SingularVectors, jobVT: LAPACK.SingularVectors, m: Int,
    n: Int, a: UnsafeMutablePointer<Complex<Double>>, lda: Int, s: UnsafeMutablePointer<Double>,
    u: UnsafeMutablePointer<Complex<Double>>, ldu: Int,
    vt: UnsafeMutablePointer<Complex<Double>>,
    ldvt: Int, superb: UnsafeMutablePointer<Double>
) -> Int {
    _accelerateGesvd(
        layout: layout, jobU: jobU, jobVT: jobVT, m: m, n: n, a: a, lda: lda, s: s, u: u,
        ldu: ldu,
        vt: vt, ldvt: ldvt, superb: superb, superbIsInRealWork: false
    ) {
        zgesvd_(
            $0, $1, $2, $3, _complexDoublePointer($4), $5, $6, _complexDoublePointer($7), $8,
            _complexDoublePointer($9), $10, _complexDoublePointer($11), $12, $13, $14)
    }
}

private func _accelerateGesdd<Scalar: _LAPACKWorkspaceScalar>(
    layout: LAPACK.Layout,
    job: LAPACK.SingularVectors,
    m: Int,
    n: Int,
    a: UnsafeMutablePointer<Scalar>,
    lda: Int,
    s: UnsafeMutablePointer<Scalar.Magnitude>,
    u: UnsafeMutablePointer<Scalar>,
    ldu: Int,
    vt: UnsafeMutablePointer<Scalar>,
    ldvt: Int,
    usesRealWorkspace: Bool,
    _ call: (
        UnsafeMutablePointer<CChar>, UnsafeMutablePointer<Int>,
        UnsafeMutablePointer<Int>, UnsafeMutablePointer<Scalar>,
        UnsafeMutablePointer<Int>, UnsafeMutablePointer<Scalar.Magnitude>,
        UnsafeMutablePointer<Scalar>, UnsafeMutablePointer<Int>,
        UnsafeMutablePointer<Scalar>, UnsafeMutablePointer<Int>,
        UnsafeMutablePointer<Scalar>, UnsafeMutablePointer<Int>,
        UnsafeMutablePointer<Scalar.Magnitude>, UnsafeMutablePointer<Int>,
        UnsafeMutablePointer<Int>
    ) -> Void
) -> Int {
    if let error = _lapackGesddArgumentError(
        layout: layout, job: job, m: m, n: n,
        lda: lda, ldu: ldu, ldvt: ldvt
    ) {
        return error
    }

    let jobs: (LAPACK.SingularVectors, LAPACK.SingularVectors)
    switch job {
    case .all: jobs = (.all, .all)
    case .some: jobs = (.some, .some)
    case .none: jobs = (.none, .none)
    case .overwrite where m >= n: jobs = (.overwrite, .all)
    case .overwrite: jobs = (.all, .overwrite)
    }
    let dimensions = _accelerateSVDOutputDimensions(
        jobU: jobs.0, jobVT: jobs.1, m: m, n: n
    )
    return _withLAPACKColumnMajorMutableMatrix(
        layout: layout, rows: m, columns: n, matrix: a,
        leadingDimension: lda
    ) { a, lda in
        _withLAPACKColumnMajorOutputMatrix(
            layout: layout, rows: dimensions.uRows,
            columns: dimensions.uColumns, matrix: u,
            leadingDimension: ldu, initialValue: .zero
        ) { u, ldu in
            _withLAPACKColumnMajorOutputMatrix(
                layout: layout, rows: dimensions.vtRows,
                columns: dimensions.vtColumns, matrix: vt,
                leadingDimension: ldvt, initialValue: .zero
            ) { vt, ldvt in
                var job = job._character
                var m = m
                var n = n
                var lda = lda
                var ldu = ldu
                var ldvt = ldvt
                var lwork = -1
                var info = 0
                var query = Scalar.zero
                let minimum = Swift.max(0, Swift.min(m, n))
                let maximum = Swift.max(0, Swift.max(m, n))
                let realWorkspaceCount: Int
                if usesRealWorkspace {
                    if job == LAPACK.SingularVectors.none._character {
                        realWorkspaceCount = Swift.max(1, 7 * minimum)
                    } else {
                        realWorkspaceCount = Swift.max(
                            1,
                            minimum
                                * Swift.max(
                                    5 * minimum + 7,
                                    2 * maximum + 2 * minimum + 1
                                )
                        )
                    }
                } else {
                    realWorkspaceCount = 1
                }
                var rwork = [Scalar.Magnitude](
                    repeating: .zero, count: realWorkspaceCount
                )
                var iwork = [Int](
                    repeating: 0, count: Swift.max(1, 8 * minimum)
                )
                rwork.withUnsafeMutableBufferPointer { rwork in
                    iwork.withUnsafeMutableBufferPointer { iwork in
                        withUnsafeMutablePointer(to: &query) { query in
                            call(
                                &job, &m, &n, a, &lda, s, u, &ldu, vt,
                                &ldvt, query, &lwork, rwork.baseAddress!,
                                iwork.baseAddress!, &info
                            )
                        }
                    }
                }
                if info != 0 { return _lapackeInfo(info) }
                lwork = _accelerateWorkspaceCount(query)
                var work = [Scalar](repeating: .zero, count: lwork)
                work.withUnsafeMutableBufferPointer { work in
                    rwork.withUnsafeMutableBufferPointer { rwork in
                        iwork.withUnsafeMutableBufferPointer { iwork in
                            call(
                                &job, &m, &n, a, &lda, s, u, &ldu, vt,
                                &ldvt, work.baseAddress!, &lwork,
                                rwork.baseAddress!, iwork.baseAddress!,
                                &info
                            )
                        }
                    }
                }
                return _lapackeInfo(info)
            }
        }
    }
}

internal func _accelerateSgesdd(
    layout: LAPACK.Layout, job: LAPACK.SingularVectors, m: Int, n: Int,
    a: UnsafeMutablePointer<Float>, lda: Int, s: UnsafeMutablePointer<Float>,
    u: UnsafeMutablePointer<Float>, ldu: Int, vt: UnsafeMutablePointer<Float>, ldvt: Int
) -> Int {
    _accelerateGesdd(
        layout: layout, job: job, m: m, n: n, a: a, lda: lda, s: s,
        u: u, ldu: ldu, vt: vt, ldvt: ldvt, usesRealWorkspace: false
    ) {
        sgesdd_($0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $13, $14)
    }
}

internal func _accelerateDgesdd(
    layout: LAPACK.Layout, job: LAPACK.SingularVectors, m: Int, n: Int,
    a: UnsafeMutablePointer<Double>, lda: Int, s: UnsafeMutablePointer<Double>,
    u: UnsafeMutablePointer<Double>, ldu: Int, vt: UnsafeMutablePointer<Double>, ldvt: Int
) -> Int {
    _accelerateGesdd(
        layout: layout, job: job, m: m, n: n, a: a, lda: lda, s: s,
        u: u, ldu: ldu, vt: vt, ldvt: ldvt, usesRealWorkspace: false
    ) {
        dgesdd_($0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $13, $14)
    }
}

internal func _accelerateCgesdd(
    layout: LAPACK.Layout, job: LAPACK.SingularVectors, m: Int, n: Int,
    a: UnsafeMutablePointer<Complex<Float>>, lda: Int, s: UnsafeMutablePointer<Float>,
    u: UnsafeMutablePointer<Complex<Float>>, ldu: Int, vt: UnsafeMutablePointer<Complex<Float>>,
    ldvt: Int
) -> Int {
    _accelerateGesdd(
        layout: layout, job: job, m: m, n: n, a: a, lda: lda, s: s,
        u: u, ldu: ldu, vt: vt, ldvt: ldvt, usesRealWorkspace: true
    ) {
        cgesdd_(
            $0, $1, $2, _complexFloatPointer($3), $4, $5,
            _complexFloatPointer($6), $7, _complexFloatPointer($8), $9,
            _complexFloatPointer($10), $11, $12, $13, $14)
    }
}

internal func _accelerateZgesdd(
    layout: LAPACK.Layout, job: LAPACK.SingularVectors, m: Int, n: Int,
    a: UnsafeMutablePointer<Complex<Double>>, lda: Int, s: UnsafeMutablePointer<Double>,
    u: UnsafeMutablePointer<Complex<Double>>, ldu: Int,
    vt: UnsafeMutablePointer<Complex<Double>>,
    ldvt: Int
) -> Int {
    _accelerateGesdd(
        layout: layout, job: job, m: m, n: n, a: a, lda: lda, s: s,
        u: u, ldu: ldu, vt: vt, ldvt: ldvt, usesRealWorkspace: true
    ) {
        zgesdd_(
            $0, $1, $2, _complexDoublePointer($3), $4, $5,
            _complexDoublePointer($6), $7, _complexDoublePointer($8), $9,
            _complexDoublePointer($10), $11, $12, $13, $14)
    }
}

// MARK: Matrix norms

private func _accelerateLange<Scalar: LAPACKScalar>(
    layout: LAPACK.Layout,
    norm: LAPACK.Norm,
    m: Int,
    n: Int,
    a: UnsafePointer<Scalar>,
    lda: Int,
    _ call: (
        UnsafeMutablePointer<CChar>, UnsafeMutablePointer<Int>,
        UnsafeMutablePointer<Int>, UnsafePointer<Scalar>,
        UnsafeMutablePointer<Int>, UnsafeMutablePointer<Scalar.Magnitude>
    ) -> Scalar.Magnitude
) -> Scalar.Magnitude {
    if lda
        < _lapackMinimumLeadingDimension(
            layout: layout, rows: m, columns: n
        )
    {
        return -6
    }
    return _withLAPACKColumnMajorInputMatrix(
        layout: layout, rows: m, columns: n, matrix: a,
        leadingDimension: lda
    ) { a, lda in
        var norm = norm._character
        var m = m
        var n = n
        var lda = lda
        var work = [Scalar.Magnitude](
            repeating: .zero, count: Swift.max(1, m)
        )
        return work.withUnsafeMutableBufferPointer { work in
            call(&norm, &m, &n, a, &lda, work.baseAddress!)
        }
    }
}

internal func _accelerateSlange(
    layout: LAPACK.Layout, norm: LAPACK.Norm, m: Int, n: Int, a: UnsafePointer<Float>, lda: Int
) -> Float {
    _accelerateLange(layout: layout, norm: norm, m: m, n: n, a: a, lda: lda) {
        slange_($0, $1, $2, $3, $4, $5)
    }
}

internal func _accelerateDlange(
    layout: LAPACK.Layout, norm: LAPACK.Norm, m: Int, n: Int, a: UnsafePointer<Double>, lda: Int
) -> Double {
    _accelerateLange(layout: layout, norm: norm, m: m, n: n, a: a, lda: lda) {
        dlange_($0, $1, $2, $3, $4, $5)
    }
}

internal func _accelerateClange(
    layout: LAPACK.Layout, norm: LAPACK.Norm, m: Int, n: Int, a: UnsafePointer<Complex<Float>>,
    lda: Int
) -> Float {
    _accelerateLange(layout: layout, norm: norm, m: m, n: n, a: a, lda: lda) {
        clange_($0, $1, $2, _complexFloatPointer($3), $4, $5)
    }
}

internal func _accelerateZlange(
    layout: LAPACK.Layout, norm: LAPACK.Norm, m: Int, n: Int, a: UnsafePointer<Complex<Double>>,
    lda: Int
) -> Double {
    _accelerateLange(layout: layout, norm: norm, m: m, n: n, a: a, lda: lda) {
        zlange_($0, $1, $2, _complexDoublePointer($3), $4, $5)
    }
}

private func _accelerateLanhe<Scalar: LAPACKScalar>(
    layout: LAPACK.Layout,
    norm: LAPACK.Norm,
    triangle: LAPACK.Triangle,
    n: Int,
    a: UnsafePointer<Scalar>,
    lda: Int,
    _ call: (
        UnsafeMutablePointer<CChar>, UnsafeMutablePointer<CChar>,
        UnsafeMutablePointer<Int>, UnsafePointer<Scalar>,
        UnsafeMutablePointer<Int>, UnsafeMutablePointer<Scalar.Magnitude>
    ) -> Scalar.Magnitude
) -> Scalar.Magnitude {
    if lda
        < _lapackMinimumLeadingDimension(
            layout: layout, rows: n, columns: n
        )
    {
        return -6
    }
    return _withLAPACKColumnMajorInputMatrix(
        layout: layout, rows: n, columns: n, matrix: a,
        leadingDimension: lda
    ) { a, lda in
        var norm = norm._character
        var uplo = triangle._lapackCharacter
        var n = n
        var lda = lda
        var work = [Scalar.Magnitude](
            repeating: .zero, count: Swift.max(1, n)
        )
        return work.withUnsafeMutableBufferPointer { work in
            call(&norm, &uplo, &n, a, &lda, work.baseAddress!)
        }
    }
}

internal func _accelerateSlansy(
    layout: LAPACK.Layout, norm: LAPACK.Norm, triangle: LAPACK.Triangle, n: Int,
    a: UnsafePointer<Float>, lda: Int
) -> Float {
    _accelerateLanhe(layout: layout, norm: norm, triangle: triangle, n: n, a: a, lda: lda) {
        slansy_($0, $1, $2, $3, $4, $5)
    }
}

internal func _accelerateDlansy(
    layout: LAPACK.Layout, norm: LAPACK.Norm, triangle: LAPACK.Triangle, n: Int,
    a: UnsafePointer<Double>, lda: Int
) -> Double {
    _accelerateLanhe(layout: layout, norm: norm, triangle: triangle, n: n, a: a, lda: lda) {
        dlansy_($0, $1, $2, $3, $4, $5)
    }
}

internal func _accelerateClanhe(
    layout: LAPACK.Layout, norm: LAPACK.Norm, triangle: LAPACK.Triangle, n: Int,
    a: UnsafePointer<Complex<Float>>, lda: Int
) -> Float {
    _accelerateLanhe(layout: layout, norm: norm, triangle: triangle, n: n, a: a, lda: lda) {
        clanhe_($0, $1, $2, _complexFloatPointer($3), $4, $5)
    }
}

internal func _accelerateZlanhe(
    layout: LAPACK.Layout, norm: LAPACK.Norm, triangle: LAPACK.Triangle, n: Int,
    a: UnsafePointer<Complex<Double>>, lda: Int
) -> Double {
    _accelerateLanhe(layout: layout, norm: norm, triangle: triangle, n: n, a: a, lda: lda) {
        zlanhe_($0, $1, $2, _complexDoublePointer($3), $4, $5)
    }
}
#endif
