// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: MIT

import ComplexModule
import RealModule

#if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
import COpenBLAS
#elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
import Accelerate
#endif

// MARK: - QR factorization

extension LAPACK {
    @discardableResult
    public static func sgeqrf(
        layout: Layout, m: Int, n: Int, a: UnsafeMutablePointer<Float>, lda: Int,
        tau: UnsafeMutablePointer<Float>
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_sgeqrf(
                _backendIndex(layout.rawValue), _backendIndex(m), _backendIndex(n), a,
                _backendIndex(lda),
                tau))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateSgeqrf(layout: layout, m: m, n: n, a: a, lda: lda, tau: tau)
        #else
        return _lapackGeqrf(layout: layout, m: m, n: n, a: a, lda: lda, tau: tau)
        #endif
    }

    @discardableResult
    public static func dgeqrf(
        layout: Layout, m: Int, n: Int, a: UnsafeMutablePointer<Double>, lda: Int,
        tau: UnsafeMutablePointer<Double>
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_dgeqrf(
                _backendIndex(layout.rawValue), _backendIndex(m), _backendIndex(n), a,
                _backendIndex(lda),
                tau))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateDgeqrf(layout: layout, m: m, n: n, a: a, lda: lda, tau: tau)
        #else
        return _lapackGeqrf(layout: layout, m: m, n: n, a: a, lda: lda, tau: tau)
        #endif
    }

    @discardableResult
    public static func cgeqrf(
        layout: Layout, m: Int, n: Int, a: UnsafeMutablePointer<Complex<Float>>, lda: Int,
        tau: UnsafeMutablePointer<Complex<Float>>
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_cgeqrf(
                _backendIndex(layout.rawValue), _backendIndex(m), _backendIndex(n),
                _complexFloatPointer(a), _backendIndex(lda), _complexFloatPointer(tau)))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateCgeqrf(layout: layout, m: m, n: n, a: a, lda: lda, tau: tau)
        #else
        return _lapackGeqrf(layout: layout, m: m, n: n, a: a, lda: lda, tau: tau)
        #endif
    }

    @discardableResult
    public static func zgeqrf(
        layout: Layout, m: Int, n: Int, a: UnsafeMutablePointer<Complex<Double>>, lda: Int,
        tau: UnsafeMutablePointer<Complex<Double>>
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_zgeqrf(
                _backendIndex(layout.rawValue), _backendIndex(m), _backendIndex(n),
                _complexDoublePointer(a), _backendIndex(lda), _complexDoublePointer(tau)))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateZgeqrf(layout: layout, m: m, n: n, a: a, lda: lda, tau: tau)
        #else
        return _lapackGeqrf(layout: layout, m: m, n: n, a: a, lda: lda, tau: tau)
        #endif
    }

    @discardableResult
    public static func sorgqr(
        layout: Layout, m: Int, n: Int, k: Int, a: UnsafeMutablePointer<Float>, lda: Int,
        tau: UnsafePointer<Float>
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_sorgqr(
                _backendIndex(layout.rawValue), _backendIndex(m), _backendIndex(n),
                _backendIndex(k), a,
                _backendIndex(lda), tau))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateSorgqr(layout: layout, m: m, n: n, k: k, a: a, lda: lda, tau: tau)
        #else
        return _lapackOrgqr(layout: layout, m: m, n: n, k: k, a: a, lda: lda, tau: tau)
        #endif
    }

    @discardableResult
    public static func dorgqr(
        layout: Layout, m: Int, n: Int, k: Int, a: UnsafeMutablePointer<Double>, lda: Int,
        tau: UnsafePointer<Double>
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_dorgqr(
                _backendIndex(layout.rawValue), _backendIndex(m), _backendIndex(n),
                _backendIndex(k), a,
                _backendIndex(lda), tau))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateDorgqr(layout: layout, m: m, n: n, k: k, a: a, lda: lda, tau: tau)
        #else
        return _lapackOrgqr(layout: layout, m: m, n: n, k: k, a: a, lda: lda, tau: tau)
        #endif
    }

    @discardableResult
    public static func cungqr(
        layout: Layout, m: Int, n: Int, k: Int, a: UnsafeMutablePointer<Complex<Float>>, lda: Int,
        tau: UnsafePointer<Complex<Float>>
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_cungqr(
                _backendIndex(layout.rawValue), _backendIndex(m), _backendIndex(n),
                _backendIndex(k),
                _complexFloatPointer(a), _backendIndex(lda), _complexFloatPointer(tau)))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateCungqr(layout: layout, m: m, n: n, k: k, a: a, lda: lda, tau: tau)
        #else
        return _lapackOrgqr(layout: layout, m: m, n: n, k: k, a: a, lda: lda, tau: tau)
        #endif
    }

    @discardableResult
    public static func zungqr(
        layout: Layout, m: Int, n: Int, k: Int, a: UnsafeMutablePointer<Complex<Double>>, lda: Int,
        tau: UnsafePointer<Complex<Double>>
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_zungqr(
                _backendIndex(layout.rawValue), _backendIndex(m), _backendIndex(n),
                _backendIndex(k),
                _complexDoublePointer(a), _backendIndex(lda), _complexDoublePointer(tau)))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateZungqr(layout: layout, m: m, n: n, k: k, a: a, lda: lda, tau: tau)
        #else
        return _lapackOrgqr(layout: layout, m: m, n: n, k: k, a: a, lda: lda, tau: tau)
        #endif
    }
}

// MARK: - Least squares

extension LAPACK {
    @discardableResult
    public static func sgels(
        layout: Layout, transpose: Transpose, m: Int, n: Int, nrhs: Int,
        a: UnsafeMutablePointer<Float>,
        lda: Int, b: UnsafeMutablePointer<Float>, ldb: Int
    ) -> Int {
        if transpose == .conjugateTranspose { return -2 }
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_sgels(
                _backendIndex(layout.rawValue), transpose._character, _backendIndex(m),
                _backendIndex(n),
                _backendIndex(nrhs), a, _backendIndex(lda), b, _backendIndex(ldb)))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateSgels(
            layout: layout, transpose: transpose, m: m, n: n, nrhs: nrhs, a: a, lda: lda, b: b,
            ldb: ldb
        )
        #else
        return _lapackGels(
            layout: layout, transpose: transpose, m: m, n: n, nrhs: nrhs, a: a, lda: lda, b: b,
            ldb: ldb
        )
        #endif
    }

    @discardableResult
    public static func dgels(
        layout: Layout, transpose: Transpose, m: Int, n: Int, nrhs: Int,
        a: UnsafeMutablePointer<Double>, lda: Int, b: UnsafeMutablePointer<Double>, ldb: Int
    ) -> Int {
        if transpose == .conjugateTranspose { return -2 }
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_dgels(
                _backendIndex(layout.rawValue), transpose._character, _backendIndex(m),
                _backendIndex(n),
                _backendIndex(nrhs), a, _backendIndex(lda), b, _backendIndex(ldb)))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateDgels(
            layout: layout, transpose: transpose, m: m, n: n, nrhs: nrhs, a: a, lda: lda, b: b,
            ldb: ldb
        )
        #else
        return _lapackGels(
            layout: layout, transpose: transpose, m: m, n: n, nrhs: nrhs, a: a, lda: lda, b: b,
            ldb: ldb
        )
        #endif
    }

    @discardableResult
    public static func cgels(
        layout: Layout, transpose: Transpose, m: Int, n: Int, nrhs: Int,
        a: UnsafeMutablePointer<Complex<Float>>, lda: Int, b: UnsafeMutablePointer<Complex<Float>>,
        ldb: Int
    ) -> Int {
        if transpose == .transpose { return -2 }
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_cgels(
                _backendIndex(layout.rawValue), transpose._character, _backendIndex(m),
                _backendIndex(n),
                _backendIndex(nrhs), _complexFloatPointer(a), _backendIndex(lda),
                _complexFloatPointer(b),
                _backendIndex(ldb)))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateCgels(
            layout: layout, transpose: transpose, m: m, n: n, nrhs: nrhs, a: a, lda: lda, b: b,
            ldb: ldb
        )
        #else
        return _lapackGels(
            layout: layout, transpose: transpose, m: m, n: n, nrhs: nrhs, a: a, lda: lda, b: b,
            ldb: ldb
        )
        #endif
    }

    @discardableResult
    public static func zgels(
        layout: Layout, transpose: Transpose, m: Int, n: Int, nrhs: Int,
        a: UnsafeMutablePointer<Complex<Double>>, lda: Int,
        b: UnsafeMutablePointer<Complex<Double>>,
        ldb: Int
    ) -> Int {
        if transpose == .transpose { return -2 }
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_zgels(
                _backendIndex(layout.rawValue), transpose._character, _backendIndex(m),
                _backendIndex(n),
                _backendIndex(nrhs), _complexDoublePointer(a), _backendIndex(lda),
                _complexDoublePointer(b), _backendIndex(ldb)))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateZgels(
            layout: layout, transpose: transpose, m: m, n: n, nrhs: nrhs, a: a, lda: lda, b: b,
            ldb: ldb
        )
        #else
        return _lapackGels(
            layout: layout, transpose: transpose, m: m, n: n, nrhs: nrhs, a: a, lda: lda, b: b,
            ldb: ldb
        )
        #endif
    }
}

// MARK: - Symmetric and Hermitian eigendecomposition

extension LAPACK {
    @discardableResult
    public static func ssyev(
        layout: Layout, job: Eigenvectors, triangle: Triangle, n: Int,
        a: UnsafeMutablePointer<Float>,
        lda: Int, w: UnsafeMutablePointer<Float>
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_ssyev(
                _backendIndex(layout.rawValue), job._character, triangle._lapackCharacter,
                _backendIndex(n), a, _backendIndex(lda), w))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateSsyev(
            layout: layout, job: job, triangle: triangle, n: n, a: a, lda: lda, w: w)
        #else
        return _lapackSyev(
            layout: layout, job: job, triangle: triangle, n: n, a: a, lda: lda, w: w)
        #endif
    }

    @discardableResult
    public static func dsyev(
        layout: Layout, job: Eigenvectors, triangle: Triangle, n: Int,
        a: UnsafeMutablePointer<Double>,
        lda: Int, w: UnsafeMutablePointer<Double>
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_dsyev(
                _backendIndex(layout.rawValue), job._character, triangle._lapackCharacter,
                _backendIndex(n), a, _backendIndex(lda), w))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateDsyev(
            layout: layout, job: job, triangle: triangle, n: n, a: a, lda: lda, w: w)
        #else
        return _lapackSyev(
            layout: layout, job: job, triangle: triangle, n: n, a: a, lda: lda, w: w)
        #endif
    }

    @discardableResult
    public static func cheev(
        layout: Layout, job: Eigenvectors, triangle: Triangle, n: Int,
        a: UnsafeMutablePointer<Complex<Float>>, lda: Int, w: UnsafeMutablePointer<Float>
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_cheev(
                _backendIndex(layout.rawValue), job._character, triangle._lapackCharacter,
                _backendIndex(n), _complexFloatPointer(a), _backendIndex(lda), w))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateCheev(
            layout: layout, job: job, triangle: triangle, n: n, a: a, lda: lda, w: w)
        #else
        return _lapackSyev(
            layout: layout, job: job, triangle: triangle, n: n, a: a, lda: lda, w: w)
        #endif
    }

    @discardableResult
    public static func zheev(
        layout: Layout, job: Eigenvectors, triangle: Triangle, n: Int,
        a: UnsafeMutablePointer<Complex<Double>>, lda: Int, w: UnsafeMutablePointer<Double>
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_zheev(
                _backendIndex(layout.rawValue), job._character, triangle._lapackCharacter,
                _backendIndex(n), _complexDoublePointer(a), _backendIndex(lda), w))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateZheev(
            layout: layout, job: job, triangle: triangle, n: n, a: a, lda: lda, w: w)
        #else
        return _lapackSyev(
            layout: layout, job: job, triangle: triangle, n: n, a: a, lda: lda, w: w)
        #endif
    }
}

// MARK: - General eigendecomposition

extension LAPACK {
    /// Computes the eigenvalues and, optionally, the left and/or right
    /// eigenvectors of a real single-precision general matrix.
    ///
    /// For a complex conjugate pair `wr[j] +/- i * wi[j]`, where `wi[j] > 0`,
    /// the corresponding eigenvectors are stored in consecutive real columns:
    /// column `j` contains the real part and column `j + 1` the imaginary part.
    @discardableResult
    public static func sgeev(
        layout: Layout, jobVL: Eigenvectors, jobVR: Eigenvectors, n: Int,
        a: UnsafeMutablePointer<Float>, lda: Int,
        wr: UnsafeMutablePointer<Float>, wi: UnsafeMutablePointer<Float>,
        vl: UnsafeMutablePointer<Float>, ldvl: Int,
        vr: UnsafeMutablePointer<Float>, ldvr: Int
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_sgeev(
                _backendIndex(layout.rawValue), jobVL._character, jobVR._character,
                _backendIndex(n), a, _backendIndex(lda), wr, wi,
                vl, _backendIndex(ldvl), vr, _backendIndex(ldvr)))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateSgeev(
            layout: layout, jobVL: jobVL, jobVR: jobVR, n: n,
            a: a, lda: lda, wr: wr, wi: wi,
            vl: vl, ldvl: ldvl, vr: vr, ldvr: ldvr)
        #else
        return _lapackRealGeev(
            layout: layout, jobVL: jobVL, jobVR: jobVR, n: n,
            a: a, lda: lda, wr: wr, wi: wi,
            vl: vl, ldvl: ldvl, vr: vr, ldvr: ldvr)
        #endif
    }

    /// Computes the eigenvalues and, optionally, the left and/or right
    /// eigenvectors of a real double-precision general matrix.
    ///
    /// For a complex conjugate pair `wr[j] +/- i * wi[j]`, where `wi[j] > 0`,
    /// the corresponding eigenvectors are stored in consecutive real columns:
    /// column `j` contains the real part and column `j + 1` the imaginary part.
    @discardableResult
    public static func dgeev(
        layout: Layout, jobVL: Eigenvectors, jobVR: Eigenvectors, n: Int,
        a: UnsafeMutablePointer<Double>, lda: Int,
        wr: UnsafeMutablePointer<Double>, wi: UnsafeMutablePointer<Double>,
        vl: UnsafeMutablePointer<Double>, ldvl: Int,
        vr: UnsafeMutablePointer<Double>, ldvr: Int
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_dgeev(
                _backendIndex(layout.rawValue), jobVL._character, jobVR._character,
                _backendIndex(n), a, _backendIndex(lda), wr, wi,
                vl, _backendIndex(ldvl), vr, _backendIndex(ldvr)))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateDgeev(
            layout: layout, jobVL: jobVL, jobVR: jobVR, n: n,
            a: a, lda: lda, wr: wr, wi: wi,
            vl: vl, ldvl: ldvl, vr: vr, ldvr: ldvr)
        #else
        return _lapackRealGeev(
            layout: layout, jobVL: jobVL, jobVR: jobVR, n: n,
            a: a, lda: lda, wr: wr, wi: wi,
            vl: vl, ldvl: ldvl, vr: vr, ldvr: ldvr)
        #endif
    }

    /// Computes the eigenvalues and, optionally, the left and/or right
    /// eigenvectors of a complex single-precision general matrix.
    @discardableResult
    public static func cgeev(
        layout: Layout, jobVL: Eigenvectors, jobVR: Eigenvectors, n: Int,
        a: UnsafeMutablePointer<Complex<Float>>, lda: Int,
        w: UnsafeMutablePointer<Complex<Float>>,
        vl: UnsafeMutablePointer<Complex<Float>>, ldvl: Int,
        vr: UnsafeMutablePointer<Complex<Float>>, ldvr: Int
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_cgeev(
                _backendIndex(layout.rawValue), jobVL._character, jobVR._character,
                _backendIndex(n), _complexFloatPointer(a), _backendIndex(lda),
                _complexFloatPointer(w), _complexFloatPointer(vl), _backendIndex(ldvl),
                _complexFloatPointer(vr), _backendIndex(ldvr)))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateCgeev(
            layout: layout, jobVL: jobVL, jobVR: jobVR, n: n,
            a: a, lda: lda, w: w,
            vl: vl, ldvl: ldvl, vr: vr, ldvr: ldvr)
        #else
        return _lapackComplexGeev(
            layout: layout, jobVL: jobVL, jobVR: jobVR, n: n,
            a: a, lda: lda, w: w,
            vl: vl, ldvl: ldvl, vr: vr, ldvr: ldvr)
        #endif
    }

    /// Computes the eigenvalues and, optionally, the left and/or right
    /// eigenvectors of a complex double-precision general matrix.
    @discardableResult
    public static func zgeev(
        layout: Layout, jobVL: Eigenvectors, jobVR: Eigenvectors, n: Int,
        a: UnsafeMutablePointer<Complex<Double>>, lda: Int,
        w: UnsafeMutablePointer<Complex<Double>>,
        vl: UnsafeMutablePointer<Complex<Double>>, ldvl: Int,
        vr: UnsafeMutablePointer<Complex<Double>>, ldvr: Int
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_zgeev(
                _backendIndex(layout.rawValue), jobVL._character, jobVR._character,
                _backendIndex(n), _complexDoublePointer(a), _backendIndex(lda),
                _complexDoublePointer(w), _complexDoublePointer(vl), _backendIndex(ldvl),
                _complexDoublePointer(vr), _backendIndex(ldvr)))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateZgeev(
            layout: layout, jobVL: jobVL, jobVR: jobVR, n: n,
            a: a, lda: lda, w: w,
            vl: vl, ldvl: ldvl, vr: vr, ldvr: ldvr)
        #else
        return _lapackComplexGeev(
            layout: layout, jobVL: jobVL, jobVR: jobVR, n: n,
            a: a, lda: lda, w: w,
            vl: vl, ldvl: ldvl, vr: vr, ldvr: ldvr)
        #endif
    }
}

// MARK: - General Schur decomposition

extension LAPACK {
    /// Computes the real Schur form and, optionally, the Schur vectors of a
    /// real single-precision general matrix.
    ///
    /// When `sort` is `.selected`, `select` is called with pointers to the
    /// real and imaginary parts of each eigenvalue. A nonzero result selects
    /// the eigenvalue. Selecting either member of a complex conjugate pair
    /// selects both members. On success, `sdim` is the number of selected
    /// eigenvalues. `a` is overwritten by the upper quasi-triangular Schur
    /// form and, when requested, `vs` contains the orthogonal Schur vectors.
    @discardableResult
    public static func sgees(
        layout: Layout, jobVS: Eigenvectors, sort: SchurSort,
        select: SgeesSelect? = nil, n: Int,
        a: UnsafeMutablePointer<Float>, lda: Int,
        sdim: UnsafeMutablePointer<Int>,
        wr: UnsafeMutablePointer<Float>, wi: UnsafeMutablePointer<Float>,
        vs: UnsafeMutablePointer<Float>, ldvs: Int
    ) -> Int {
        if sort == .selected && select == nil { return -4 }
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        var backendSDim: Int32 = 0
        let info = LAPACKE_sgees(
            _backendIndex(layout.rawValue), jobVS._character, sort._character, select,
            _backendIndex(n), a, _backendIndex(lda), &backendSDim, wr, wi,
            vs, _backendIndex(ldvs)
        )
        sdim.pointee = Int(backendSDim)
        return Int(info)
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateSgees(
            layout: layout, jobVS: jobVS, sort: sort, select: select,
            n: n, a: a, lda: lda, sdim: sdim, wr: wr, wi: wi,
            vs: vs, ldvs: ldvs
        )
        #else
        return _lapackSgees(
            layout: layout, jobVS: jobVS, sort: sort, select: select,
            n: n, a: a, lda: lda, sdim: sdim, wr: wr, wi: wi,
            vs: vs, ldvs: ldvs
        )
        #endif
    }

    /// Computes the real Schur form and, optionally, the Schur vectors of a
    /// real double-precision general matrix.
    ///
    /// When `sort` is `.selected`, `select` is called with pointers to the
    /// real and imaginary parts of each eigenvalue. A nonzero result selects
    /// the eigenvalue. Selecting either member of a complex conjugate pair
    /// selects both members. On success, `sdim` is the number of selected
    /// eigenvalues. `a` is overwritten by the upper quasi-triangular Schur
    /// form and, when requested, `vs` contains the orthogonal Schur vectors.
    @discardableResult
    public static func dgees(
        layout: Layout, jobVS: Eigenvectors, sort: SchurSort,
        select: DgeesSelect? = nil, n: Int,
        a: UnsafeMutablePointer<Double>, lda: Int,
        sdim: UnsafeMutablePointer<Int>,
        wr: UnsafeMutablePointer<Double>, wi: UnsafeMutablePointer<Double>,
        vs: UnsafeMutablePointer<Double>, ldvs: Int
    ) -> Int {
        if sort == .selected && select == nil { return -4 }
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        var backendSDim: Int32 = 0
        let info = LAPACKE_dgees(
            _backendIndex(layout.rawValue), jobVS._character, sort._character, select,
            _backendIndex(n), a, _backendIndex(lda), &backendSDim, wr, wi,
            vs, _backendIndex(ldvs)
        )
        sdim.pointee = Int(backendSDim)
        return Int(info)
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateDgees(
            layout: layout, jobVS: jobVS, sort: sort, select: select,
            n: n, a: a, lda: lda, sdim: sdim, wr: wr, wi: wi,
            vs: vs, ldvs: ldvs
        )
        #else
        return _lapackDgees(
            layout: layout, jobVS: jobVS, sort: sort, select: select,
            n: n, a: a, lda: lda, sdim: sdim, wr: wr, wi: wi,
            vs: vs, ldvs: ldvs
        )
        #endif
    }

    /// Computes the complex Schur form and, optionally, the Schur vectors of
    /// a complex single-precision general matrix.
    ///
    /// The selection function receives a raw pointer to one `Complex<Float>`
    /// value. When sorting is disabled, it is not referenced. On success,
    /// `sdim` is the number of selected eigenvalues, `a` contains the upper
    /// triangular Schur form, and `vs` contains the requested unitary Schur
    /// vectors.
    @discardableResult
    public static func cgees(
        layout: Layout, jobVS: Eigenvectors, sort: SchurSort,
        select: CgeesSelect? = nil, n: Int,
        a: UnsafeMutablePointer<Complex<Float>>, lda: Int,
        sdim: UnsafeMutablePointer<Int>,
        w: UnsafeMutablePointer<Complex<Float>>,
        vs: UnsafeMutablePointer<Complex<Float>>, ldvs: Int
    ) -> Int {
        if sort == .selected && select == nil { return -4 }
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        let backendSelect: LAPACK_C_SELECT1? = select.map {
            unsafeBitCast($0, to: LAPACK_C_SELECT1.self)
        }
        var backendSDim: Int32 = 0
        let info = LAPACKE_cgees(
            _backendIndex(layout.rawValue), jobVS._character, sort._character, backendSelect,
            _backendIndex(n), _complexFloatPointer(a), _backendIndex(lda), &backendSDim,
            _complexFloatPointer(w), _complexFloatPointer(vs), _backendIndex(ldvs)
        )
        sdim.pointee = Int(backendSDim)
        return Int(info)
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateCgees(
            layout: layout, jobVS: jobVS, sort: sort, select: select,
            n: n, a: a, lda: lda, sdim: sdim, w: w, vs: vs, ldvs: ldvs
        )
        #else
        return _lapackCgees(
            layout: layout, jobVS: jobVS, sort: sort, select: select,
            n: n, a: a, lda: lda, sdim: sdim, w: w, vs: vs, ldvs: ldvs
        )
        #endif
    }

    /// Computes the complex Schur form and, optionally, the Schur vectors of
    /// a complex double-precision general matrix.
    ///
    /// The selection function receives a raw pointer to one `Complex<Double>`
    /// value. When sorting is disabled, it is not referenced. On success,
    /// `sdim` is the number of selected eigenvalues, `a` contains the upper
    /// triangular Schur form, and `vs` contains the requested unitary Schur
    /// vectors.
    @discardableResult
    public static func zgees(
        layout: Layout, jobVS: Eigenvectors, sort: SchurSort,
        select: ZgeesSelect? = nil, n: Int,
        a: UnsafeMutablePointer<Complex<Double>>, lda: Int,
        sdim: UnsafeMutablePointer<Int>,
        w: UnsafeMutablePointer<Complex<Double>>,
        vs: UnsafeMutablePointer<Complex<Double>>, ldvs: Int
    ) -> Int {
        if sort == .selected && select == nil { return -4 }
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        let backendSelect: LAPACK_Z_SELECT1? = select.map {
            unsafeBitCast($0, to: LAPACK_Z_SELECT1.self)
        }
        var backendSDim: Int32 = 0
        let info = LAPACKE_zgees(
            _backendIndex(layout.rawValue), jobVS._character, sort._character, backendSelect,
            _backendIndex(n), _complexDoublePointer(a), _backendIndex(lda), &backendSDim,
            _complexDoublePointer(w), _complexDoublePointer(vs), _backendIndex(ldvs)
        )
        sdim.pointee = Int(backendSDim)
        return Int(info)
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateZgees(
            layout: layout, jobVS: jobVS, sort: sort, select: select,
            n: n, a: a, lda: lda, sdim: sdim, w: w, vs: vs, ldvs: ldvs
        )
        #else
        return _lapackZgees(
            layout: layout, jobVS: jobVS, sort: sort, select: select,
            n: n, a: a, lda: lda, sdim: sdim, w: w, vs: vs, ldvs: ldvs
        )
        #endif
    }
}

// MARK: - Singular value decomposition

extension LAPACK {
    @discardableResult
    public static func sgesvd(
        layout: Layout, jobU: SingularVectors, jobVT: SingularVectors, m: Int, n: Int,
        a: UnsafeMutablePointer<Float>, lda: Int, s: UnsafeMutablePointer<Float>,
        u: UnsafeMutablePointer<Float>, ldu: Int, vt: UnsafeMutablePointer<Float>, ldvt: Int,
        superb: UnsafeMutablePointer<Float>
    ) -> Int {
        if let error = _lapackGesvdArgumentError(
            layout: layout, jobU: jobU, jobVT: jobVT,
            m: m, n: n, lda: lda, ldu: ldu, ldvt: ldvt
        ) {
            return error
        }
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_sgesvd(
                _backendIndex(layout.rawValue), jobU._character, jobVT._character,
                _backendIndex(m),
                _backendIndex(n), a, _backendIndex(lda), s, u, _backendIndex(ldu), vt,
                _backendIndex(ldvt), superb))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateSgesvd(
            layout: layout, jobU: jobU, jobVT: jobVT, m: m, n: n, a: a, lda: lda, s: s, u: u,
            ldu: ldu,
            vt: vt, ldvt: ldvt, superb: superb)
        #else
        return _lapackGesvd(
            layout: layout, jobU: jobU, jobVT: jobVT, m: m, n: n, a: a, lda: lda, s: s, u: u,
            ldu: ldu,
            vt: vt, ldvt: ldvt, superb: superb)
        #endif
    }

    @discardableResult
    public static func dgesvd(
        layout: Layout, jobU: SingularVectors, jobVT: SingularVectors, m: Int, n: Int,
        a: UnsafeMutablePointer<Double>, lda: Int, s: UnsafeMutablePointer<Double>,
        u: UnsafeMutablePointer<Double>, ldu: Int, vt: UnsafeMutablePointer<Double>, ldvt: Int,
        superb: UnsafeMutablePointer<Double>
    ) -> Int {
        if let error = _lapackGesvdArgumentError(
            layout: layout, jobU: jobU, jobVT: jobVT,
            m: m, n: n, lda: lda, ldu: ldu, ldvt: ldvt
        ) {
            return error
        }
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_dgesvd(
                _backendIndex(layout.rawValue), jobU._character, jobVT._character,
                _backendIndex(m),
                _backendIndex(n), a, _backendIndex(lda), s, u, _backendIndex(ldu), vt,
                _backendIndex(ldvt), superb))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateDgesvd(
            layout: layout, jobU: jobU, jobVT: jobVT, m: m, n: n, a: a, lda: lda, s: s, u: u,
            ldu: ldu,
            vt: vt, ldvt: ldvt, superb: superb)
        #else
        return _lapackGesvd(
            layout: layout, jobU: jobU, jobVT: jobVT, m: m, n: n, a: a, lda: lda, s: s, u: u,
            ldu: ldu,
            vt: vt, ldvt: ldvt, superb: superb)
        #endif
    }

    @discardableResult
    public static func cgesvd(
        layout: Layout, jobU: SingularVectors, jobVT: SingularVectors, m: Int, n: Int,
        a: UnsafeMutablePointer<Complex<Float>>, lda: Int, s: UnsafeMutablePointer<Float>,
        u: UnsafeMutablePointer<Complex<Float>>, ldu: Int, vt: UnsafeMutablePointer<Complex<Float>>,
        ldvt: Int, superb: UnsafeMutablePointer<Float>
    ) -> Int {
        if let error = _lapackGesvdArgumentError(
            layout: layout, jobU: jobU, jobVT: jobVT,
            m: m, n: n, lda: lda, ldu: ldu, ldvt: ldvt
        ) {
            return error
        }
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_cgesvd(
                _backendIndex(layout.rawValue), jobU._character, jobVT._character,
                _backendIndex(m),
                _backendIndex(n), _complexFloatPointer(a), _backendIndex(lda), s,
                _complexFloatPointer(u),
                _backendIndex(ldu), _complexFloatPointer(vt), _backendIndex(ldvt), superb))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateCgesvd(
            layout: layout, jobU: jobU, jobVT: jobVT, m: m, n: n, a: a, lda: lda, s: s, u: u,
            ldu: ldu,
            vt: vt, ldvt: ldvt, superb: superb)
        #else
        return _lapackGesvd(
            layout: layout, jobU: jobU, jobVT: jobVT, m: m, n: n, a: a, lda: lda, s: s, u: u,
            ldu: ldu,
            vt: vt, ldvt: ldvt, superb: superb)
        #endif
    }

    @discardableResult
    public static func zgesvd(
        layout: Layout, jobU: SingularVectors, jobVT: SingularVectors, m: Int, n: Int,
        a: UnsafeMutablePointer<Complex<Double>>, lda: Int, s: UnsafeMutablePointer<Double>,
        u: UnsafeMutablePointer<Complex<Double>>, ldu: Int,
        vt: UnsafeMutablePointer<Complex<Double>>,
        ldvt: Int, superb: UnsafeMutablePointer<Double>
    ) -> Int {
        if let error = _lapackGesvdArgumentError(
            layout: layout, jobU: jobU, jobVT: jobVT,
            m: m, n: n, lda: lda, ldu: ldu, ldvt: ldvt
        ) {
            return error
        }
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_zgesvd(
                _backendIndex(layout.rawValue), jobU._character, jobVT._character,
                _backendIndex(m),
                _backendIndex(n), _complexDoublePointer(a), _backendIndex(lda), s,
                _complexDoublePointer(u), _backendIndex(ldu), _complexDoublePointer(vt),
                _backendIndex(ldvt), superb))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateZgesvd(
            layout: layout, jobU: jobU, jobVT: jobVT, m: m, n: n, a: a, lda: lda, s: s, u: u,
            ldu: ldu,
            vt: vt, ldvt: ldvt, superb: superb)
        #else
        return _lapackGesvd(
            layout: layout, jobU: jobU, jobVT: jobVT, m: m, n: n, a: a, lda: lda, s: s, u: u,
            ldu: ldu,
            vt: vt, ldvt: ldvt, superb: superb)
        #endif
    }

    @discardableResult
    public static func sgesdd(
        layout: Layout, job: SingularVectors, m: Int, n: Int, a: UnsafeMutablePointer<Float>,
        lda: Int,
        s: UnsafeMutablePointer<Float>, u: UnsafeMutablePointer<Float>, ldu: Int,
        vt: UnsafeMutablePointer<Float>, ldvt: Int
    ) -> Int {
        if let error = _lapackGesddArgumentError(
            layout: layout, job: job, m: m, n: n,
            lda: lda, ldu: ldu, ldvt: ldvt
        ) {
            return error
        }
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_sgesdd(
                _backendIndex(layout.rawValue), job._character, _backendIndex(m),
                _backendIndex(n), a,
                _backendIndex(lda), s, u, _backendIndex(ldu), vt, _backendIndex(ldvt)))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateSgesdd(
            layout: layout, job: job, m: m, n: n, a: a, lda: lda, s: s, u: u, ldu: ldu, vt: vt,
            ldvt: ldvt)
        #else
        let jobs = _gesddJobs(job: job, m: m, n: n)
        return _lapackGesvd(
            layout: layout, jobU: jobs.0, jobVT: jobs.1, m: m, n: n, a: a, lda: lda, s: s, u: u,
            ldu: ldu, vt: vt, ldvt: ldvt, superb: nil)
        #endif
    }

    @discardableResult
    public static func dgesdd(
        layout: Layout, job: SingularVectors, m: Int, n: Int, a: UnsafeMutablePointer<Double>,
        lda: Int,
        s: UnsafeMutablePointer<Double>, u: UnsafeMutablePointer<Double>, ldu: Int,
        vt: UnsafeMutablePointer<Double>, ldvt: Int
    ) -> Int {
        if let error = _lapackGesddArgumentError(
            layout: layout, job: job, m: m, n: n,
            lda: lda, ldu: ldu, ldvt: ldvt
        ) {
            return error
        }
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_dgesdd(
                _backendIndex(layout.rawValue), job._character, _backendIndex(m),
                _backendIndex(n), a,
                _backendIndex(lda), s, u, _backendIndex(ldu), vt, _backendIndex(ldvt)))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateDgesdd(
            layout: layout, job: job, m: m, n: n, a: a, lda: lda, s: s, u: u, ldu: ldu, vt: vt,
            ldvt: ldvt)
        #else
        let jobs = _gesddJobs(job: job, m: m, n: n)
        return _lapackGesvd(
            layout: layout, jobU: jobs.0, jobVT: jobs.1, m: m, n: n, a: a, lda: lda, s: s, u: u,
            ldu: ldu, vt: vt, ldvt: ldvt, superb: nil)
        #endif
    }

    @discardableResult
    public static func cgesdd(
        layout: Layout, job: SingularVectors, m: Int, n: Int,
        a: UnsafeMutablePointer<Complex<Float>>,
        lda: Int, s: UnsafeMutablePointer<Float>, u: UnsafeMutablePointer<Complex<Float>>, ldu: Int,
        vt: UnsafeMutablePointer<Complex<Float>>, ldvt: Int
    ) -> Int {
        if let error = _lapackGesddArgumentError(
            layout: layout, job: job, m: m, n: n,
            lda: lda, ldu: ldu, ldvt: ldvt
        ) {
            return error
        }
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_cgesdd(
                _backendIndex(layout.rawValue), job._character, _backendIndex(m),
                _backendIndex(n),
                _complexFloatPointer(a), _backendIndex(lda), s, _complexFloatPointer(u),
                _backendIndex(ldu), _complexFloatPointer(vt), _backendIndex(ldvt)))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateCgesdd(
            layout: layout, job: job, m: m, n: n, a: a, lda: lda, s: s, u: u, ldu: ldu, vt: vt,
            ldvt: ldvt)
        #else
        let jobs = _gesddJobs(job: job, m: m, n: n)
        return _lapackGesvd(
            layout: layout, jobU: jobs.0, jobVT: jobs.1, m: m, n: n, a: a, lda: lda, s: s, u: u,
            ldu: ldu, vt: vt, ldvt: ldvt, superb: nil)
        #endif
    }

    @discardableResult
    public static func zgesdd(
        layout: Layout, job: SingularVectors, m: Int, n: Int,
        a: UnsafeMutablePointer<Complex<Double>>,
        lda: Int, s: UnsafeMutablePointer<Double>, u: UnsafeMutablePointer<Complex<Double>>,
        ldu: Int,
        vt: UnsafeMutablePointer<Complex<Double>>, ldvt: Int
    ) -> Int {
        if let error = _lapackGesddArgumentError(
            layout: layout, job: job, m: m, n: n,
            lda: lda, ldu: ldu, ldvt: ldvt
        ) {
            return error
        }
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_zgesdd(
                _backendIndex(layout.rawValue), job._character, _backendIndex(m),
                _backendIndex(n),
                _complexDoublePointer(a), _backendIndex(lda), s, _complexDoublePointer(u),
                _backendIndex(ldu), _complexDoublePointer(vt), _backendIndex(ldvt)))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateZgesdd(
            layout: layout, job: job, m: m, n: n, a: a, lda: lda, s: s, u: u, ldu: ldu, vt: vt,
            ldvt: ldvt)
        #else
        let jobs = _gesddJobs(job: job, m: m, n: n)
        return _lapackGesvd(
            layout: layout, jobU: jobs.0, jobVT: jobs.1, m: m, n: n, a: a, lda: lda, s: s, u: u,
            ldu: ldu, vt: vt, ldvt: ldvt, superb: nil)
        #endif
    }
}

@inline(__always)
private func _gesddJobs(job: LAPACK.SingularVectors, m: Int, n: Int) -> (
    LAPACK.SingularVectors, LAPACK.SingularVectors
) {
    switch job {
    case .all: return (.all, .all)
    case .some: return (.some, .some)
    case .none: return (.none, .none)
    case .overwrite where m >= n: return (.overwrite, .all)
    case .overwrite: return (.all, .overwrite)
    }
}

// MARK: - Matrix norms

extension LAPACK {
    public static func slange(
        layout: Layout, norm: Norm, m: Int, n: Int, a: UnsafePointer<Float>, lda: Int
    ) -> Float {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return LAPACKE_slange(
            _backendIndex(layout.rawValue), norm._character, _backendIndex(m), _backendIndex(n),
            a,
            _backendIndex(lda))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateSlange(layout: layout, norm: norm, m: m, n: n, a: a, lda: lda)
        #else
        return _lapackLange(layout: layout, norm: norm, m: m, n: n, a: a, lda: lda)
        #endif
    }

    public static func dlange(
        layout: Layout, norm: Norm, m: Int, n: Int, a: UnsafePointer<Double>, lda: Int
    ) -> Double {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return LAPACKE_dlange(
            _backendIndex(layout.rawValue), norm._character, _backendIndex(m), _backendIndex(n),
            a,
            _backendIndex(lda))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateDlange(layout: layout, norm: norm, m: m, n: n, a: a, lda: lda)
        #else
        return _lapackLange(layout: layout, norm: norm, m: m, n: n, a: a, lda: lda)
        #endif
    }

    public static func clange(
        layout: Layout, norm: Norm, m: Int, n: Int, a: UnsafePointer<Complex<Float>>, lda: Int
    ) -> Float {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return LAPACKE_clange(
            _backendIndex(layout.rawValue), norm._character, _backendIndex(m), _backendIndex(n),
            _complexFloatPointer(a), _backendIndex(lda))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateClange(layout: layout, norm: norm, m: m, n: n, a: a, lda: lda)
        #else
        return _lapackLange(layout: layout, norm: norm, m: m, n: n, a: a, lda: lda)
        #endif
    }

    public static func zlange(
        layout: Layout, norm: Norm, m: Int, n: Int, a: UnsafePointer<Complex<Double>>, lda: Int
    ) -> Double {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return LAPACKE_zlange(
            _backendIndex(layout.rawValue), norm._character, _backendIndex(m), _backendIndex(n),
            _complexDoublePointer(a), _backendIndex(lda))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateZlange(layout: layout, norm: norm, m: m, n: n, a: a, lda: lda)
        #else
        return _lapackLange(layout: layout, norm: norm, m: m, n: n, a: a, lda: lda)
        #endif
    }

    public static func slansy(
        layout: Layout, norm: Norm, triangle: Triangle, n: Int, a: UnsafePointer<Float>, lda: Int
    ) -> Float {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return LAPACKE_slansy(
            _backendIndex(layout.rawValue), norm._character, triangle._lapackCharacter,
            _backendIndex(n), a, _backendIndex(lda))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateSlansy(
            layout: layout, norm: norm, triangle: triangle, n: n, a: a, lda: lda)
        #else
        return _lapackLanhe(
            layout: layout, norm: norm, triangle: triangle, n: n, a: a, lda: lda)
        #endif
    }

    public static func dlansy(
        layout: Layout, norm: Norm, triangle: Triangle, n: Int, a: UnsafePointer<Double>, lda: Int
    ) -> Double {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return LAPACKE_dlansy(
            _backendIndex(layout.rawValue), norm._character, triangle._lapackCharacter,
            _backendIndex(n), a, _backendIndex(lda))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateDlansy(
            layout: layout, norm: norm, triangle: triangle, n: n, a: a, lda: lda)
        #else
        return _lapackLanhe(
            layout: layout, norm: norm, triangle: triangle, n: n, a: a, lda: lda)
        #endif
    }

    public static func clanhe(
        layout: Layout, norm: Norm, triangle: Triangle, n: Int, a: UnsafePointer<Complex<Float>>,
        lda: Int
    ) -> Float {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return LAPACKE_clanhe(
            _backendIndex(layout.rawValue), norm._character, triangle._lapackCharacter,
            _backendIndex(n), _complexFloatPointer(a), _backendIndex(lda))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateClanhe(
            layout: layout, norm: norm, triangle: triangle, n: n, a: a, lda: lda)
        #else
        return _lapackLanhe(
            layout: layout, norm: norm, triangle: triangle, n: n, a: a, lda: lda)
        #endif
    }

    public static func zlanhe(
        layout: Layout, norm: Norm, triangle: Triangle, n: Int, a: UnsafePointer<Complex<Double>>,
        lda: Int
    ) -> Double {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return LAPACKE_zlanhe(
            _backendIndex(layout.rawValue), norm._character, triangle._lapackCharacter,
            _backendIndex(n), _complexDoublePointer(a), _backendIndex(lda))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateZlanhe(
            layout: layout, norm: norm, triangle: triangle, n: n, a: a, lda: lda)
        #else
        return _lapackLanhe(
            layout: layout, norm: norm, triangle: triangle, n: n, a: a, lda: lda)
        #endif
    }
}
