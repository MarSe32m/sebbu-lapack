// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: MIT

import ComplexModule
import RealModule

#if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
import COpenBLAS
#elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
import Accelerate
#endif

#if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
@inline(__always)
private func _withOpenBLASOutputPivots(
    count: Int,
    destination: UnsafeMutablePointer<Int>,
    _ body: (UnsafeMutablePointer<Int32>) -> Int32
) -> Int {
    var pivots = [Int32](repeating: 0, count: Swift.max(1, count))
    let info = pivots.withUnsafeMutableBufferPointer { pivots in
        body(pivots.baseAddress!)
    }
    for index in 0..<count { destination[index] = Int(pivots[index]) }
    return Int(info)
}

@inline(__always)
private func _withOpenBLASInputPivots(
    count: Int,
    source: UnsafePointer<Int>,
    _ body: (UnsafePointer<Int32>) -> Int32
) -> Int {
    var pivots = [Int32]()
    pivots.reserveCapacity(Swift.max(1, count))
    for index in 0..<count { pivots.append(_backendIndex(source[index])) }
    if pivots.isEmpty { pivots.append(0) }
    return Int(
        pivots.withUnsafeBufferPointer { pivots in
            body(pivots.baseAddress!)
        })
}
#endif

// MARK: - General systems and LU factorization

extension LAPACK {
    @discardableResult
    public static func sgesv(
        layout: Layout, n: Int, nrhs: Int, a: UnsafeMutablePointer<Float>, lda: Int,
        ipiv: UnsafeMutablePointer<Int>, b: UnsafeMutablePointer<Float>, ldb: Int
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return _withOpenBLASOutputPivots(count: Swift.max(0, n), destination: ipiv) { pivots in
            LAPACKE_sgesv(
                _backendIndex(layout.rawValue), _backendIndex(n), _backendIndex(nrhs), a,
                _backendIndex(lda), pivots, b, _backendIndex(ldb))
        }
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateSgesv(
            layout: layout, n: n, nrhs: nrhs, a: a, lda: lda, ipiv: ipiv, b: b, ldb: ldb)
        #else
        return _lapackGesv(
            layout: layout, n: n, nrhs: nrhs, a: a, lda: lda, ipiv: ipiv, b: b, ldb: ldb)
        #endif
    }

    @discardableResult
    public static func dgesv(
        layout: Layout, n: Int, nrhs: Int, a: UnsafeMutablePointer<Double>, lda: Int,
        ipiv: UnsafeMutablePointer<Int>, b: UnsafeMutablePointer<Double>, ldb: Int
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return _withOpenBLASOutputPivots(count: Swift.max(0, n), destination: ipiv) { pivots in
            LAPACKE_dgesv(
                _backendIndex(layout.rawValue), _backendIndex(n), _backendIndex(nrhs), a,
                _backendIndex(lda), pivots, b, _backendIndex(ldb))
        }
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateDgesv(
            layout: layout, n: n, nrhs: nrhs, a: a, lda: lda, ipiv: ipiv, b: b, ldb: ldb)
        #else
        return _lapackGesv(
            layout: layout, n: n, nrhs: nrhs, a: a, lda: lda, ipiv: ipiv, b: b, ldb: ldb)
        #endif
    }

    @discardableResult
    public static func cgesv(
        layout: Layout, n: Int, nrhs: Int, a: UnsafeMutablePointer<Complex<Float>>, lda: Int,
        ipiv: UnsafeMutablePointer<Int>, b: UnsafeMutablePointer<Complex<Float>>, ldb: Int
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return _withOpenBLASOutputPivots(count: Swift.max(0, n), destination: ipiv) { pivots in
            LAPACKE_cgesv(
                _backendIndex(layout.rawValue), _backendIndex(n), _backendIndex(nrhs),
                _complexFloatPointer(a), _backendIndex(lda), pivots, _complexFloatPointer(b),
                _backendIndex(ldb))
        }
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateCgesv(
            layout: layout, n: n, nrhs: nrhs, a: a, lda: lda, ipiv: ipiv, b: b, ldb: ldb)
        #else
        return _lapackGesv(
            layout: layout, n: n, nrhs: nrhs, a: a, lda: lda, ipiv: ipiv, b: b, ldb: ldb)
        #endif
    }

    @discardableResult
    public static func zgesv(
        layout: Layout, n: Int, nrhs: Int, a: UnsafeMutablePointer<Complex<Double>>, lda: Int,
        ipiv: UnsafeMutablePointer<Int>, b: UnsafeMutablePointer<Complex<Double>>, ldb: Int
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return _withOpenBLASOutputPivots(count: Swift.max(0, n), destination: ipiv) { pivots in
            LAPACKE_zgesv(
                _backendIndex(layout.rawValue), _backendIndex(n), _backendIndex(nrhs),
                _complexDoublePointer(a), _backendIndex(lda), pivots, _complexDoublePointer(b),
                _backendIndex(ldb))
        }
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateZgesv(
            layout: layout, n: n, nrhs: nrhs, a: a, lda: lda, ipiv: ipiv, b: b, ldb: ldb)
        #else
        return _lapackGesv(
            layout: layout, n: n, nrhs: nrhs, a: a, lda: lda, ipiv: ipiv, b: b, ldb: ldb)
        #endif
    }

    @discardableResult
    public static func sgetrf(
        layout: Layout, m: Int, n: Int, a: UnsafeMutablePointer<Float>, lda: Int,
        ipiv: UnsafeMutablePointer<Int>
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return _withOpenBLASOutputPivots(
            count: Swift.max(0, Swift.min(m, n)), destination: ipiv
        ) {
            pivots in
            LAPACKE_sgetrf(
                _backendIndex(layout.rawValue), _backendIndex(m), _backendIndex(n), a,
                _backendIndex(lda),
                pivots)
        }
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateSgetrf(layout: layout, m: m, n: n, a: a, lda: lda, ipiv: ipiv)
        #else
        return _lapackGetrf(layout: layout, m: m, n: n, a: a, lda: lda, ipiv: ipiv)
        #endif
    }

    @discardableResult
    public static func dgetrf(
        layout: Layout, m: Int, n: Int, a: UnsafeMutablePointer<Double>, lda: Int,
        ipiv: UnsafeMutablePointer<Int>
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return _withOpenBLASOutputPivots(
            count: Swift.max(0, Swift.min(m, n)), destination: ipiv
        ) {
            pivots in
            LAPACKE_dgetrf(
                _backendIndex(layout.rawValue), _backendIndex(m), _backendIndex(n), a,
                _backendIndex(lda),
                pivots)
        }
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateDgetrf(layout: layout, m: m, n: n, a: a, lda: lda, ipiv: ipiv)
        #else
        return _lapackGetrf(layout: layout, m: m, n: n, a: a, lda: lda, ipiv: ipiv)
        #endif
    }

    @discardableResult
    public static func cgetrf(
        layout: Layout, m: Int, n: Int, a: UnsafeMutablePointer<Complex<Float>>, lda: Int,
        ipiv: UnsafeMutablePointer<Int>
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return _withOpenBLASOutputPivots(
            count: Swift.max(0, Swift.min(m, n)), destination: ipiv
        ) {
            pivots in
            LAPACKE_cgetrf(
                _backendIndex(layout.rawValue), _backendIndex(m), _backendIndex(n),
                _complexFloatPointer(a), _backendIndex(lda), pivots)
        }
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateCgetrf(layout: layout, m: m, n: n, a: a, lda: lda, ipiv: ipiv)
        #else
        return _lapackGetrf(layout: layout, m: m, n: n, a: a, lda: lda, ipiv: ipiv)
        #endif
    }

    @discardableResult
    public static func zgetrf(
        layout: Layout, m: Int, n: Int, a: UnsafeMutablePointer<Complex<Double>>, lda: Int,
        ipiv: UnsafeMutablePointer<Int>
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return _withOpenBLASOutputPivots(
            count: Swift.max(0, Swift.min(m, n)), destination: ipiv
        ) {
            pivots in
            LAPACKE_zgetrf(
                _backendIndex(layout.rawValue), _backendIndex(m), _backendIndex(n),
                _complexDoublePointer(a), _backendIndex(lda), pivots)
        }
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateZgetrf(layout: layout, m: m, n: n, a: a, lda: lda, ipiv: ipiv)
        #else
        return _lapackGetrf(layout: layout, m: m, n: n, a: a, lda: lda, ipiv: ipiv)
        #endif
    }

    @discardableResult
    public static func sgetrs(
        layout: Layout, transpose: Transpose, n: Int, nrhs: Int, a: UnsafePointer<Float>, lda: Int,
        ipiv: UnsafePointer<Int>, b: UnsafeMutablePointer<Float>, ldb: Int
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return _withOpenBLASInputPivots(count: Swift.max(0, n), source: ipiv) { pivots in
            LAPACKE_sgetrs(
                _backendIndex(layout.rawValue), transpose._character, _backendIndex(n),
                _backendIndex(nrhs), a, _backendIndex(lda), pivots, b, _backendIndex(ldb))
        }
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateSgetrs(
            layout: layout, transpose: transpose, n: n, nrhs: nrhs, a: a, lda: lda, ipiv: ipiv,
            b: b,
            ldb: ldb)
        #else
        return _lapackGetrs(
            layout: layout, transpose: transpose, n: n, nrhs: nrhs, a: a, lda: lda, ipiv: ipiv,
            b: b,
            ldb: ldb)
        #endif
    }

    @discardableResult
    public static func dgetrs(
        layout: Layout, transpose: Transpose, n: Int, nrhs: Int, a: UnsafePointer<Double>, lda: Int,
        ipiv: UnsafePointer<Int>, b: UnsafeMutablePointer<Double>, ldb: Int
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return _withOpenBLASInputPivots(count: Swift.max(0, n), source: ipiv) { pivots in
            LAPACKE_dgetrs(
                _backendIndex(layout.rawValue), transpose._character, _backendIndex(n),
                _backendIndex(nrhs), a, _backendIndex(lda), pivots, b, _backendIndex(ldb))
        }
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateDgetrs(
            layout: layout, transpose: transpose, n: n, nrhs: nrhs, a: a, lda: lda, ipiv: ipiv,
            b: b,
            ldb: ldb)
        #else
        return _lapackGetrs(
            layout: layout, transpose: transpose, n: n, nrhs: nrhs, a: a, lda: lda, ipiv: ipiv,
            b: b,
            ldb: ldb)
        #endif
    }

    @discardableResult
    public static func cgetrs(
        layout: Layout, transpose: Transpose, n: Int, nrhs: Int, a: UnsafePointer<Complex<Float>>,
        lda: Int, ipiv: UnsafePointer<Int>, b: UnsafeMutablePointer<Complex<Float>>, ldb: Int
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return _withOpenBLASInputPivots(count: Swift.max(0, n), source: ipiv) { pivots in
            LAPACKE_cgetrs(
                _backendIndex(layout.rawValue), transpose._character, _backendIndex(n),
                _backendIndex(nrhs), _complexFloatPointer(a), _backendIndex(lda), pivots,
                _complexFloatPointer(b), _backendIndex(ldb))
        }
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateCgetrs(
            layout: layout, transpose: transpose, n: n, nrhs: nrhs, a: a, lda: lda, ipiv: ipiv,
            b: b,
            ldb: ldb)
        #else
        return _lapackGetrs(
            layout: layout, transpose: transpose, n: n, nrhs: nrhs, a: a, lda: lda, ipiv: ipiv,
            b: b,
            ldb: ldb)
        #endif
    }

    @discardableResult
    public static func zgetrs(
        layout: Layout, transpose: Transpose, n: Int, nrhs: Int, a: UnsafePointer<Complex<Double>>,
        lda: Int, ipiv: UnsafePointer<Int>, b: UnsafeMutablePointer<Complex<Double>>, ldb: Int
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return _withOpenBLASInputPivots(count: Swift.max(0, n), source: ipiv) { pivots in
            LAPACKE_zgetrs(
                _backendIndex(layout.rawValue), transpose._character, _backendIndex(n),
                _backendIndex(nrhs), _complexDoublePointer(a), _backendIndex(lda), pivots,
                _complexDoublePointer(b), _backendIndex(ldb))
        }
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateZgetrs(
            layout: layout, transpose: transpose, n: n, nrhs: nrhs, a: a, lda: lda, ipiv: ipiv,
            b: b,
            ldb: ldb)
        #else
        return _lapackGetrs(
            layout: layout, transpose: transpose, n: n, nrhs: nrhs, a: a, lda: lda, ipiv: ipiv,
            b: b,
            ldb: ldb)
        #endif
    }

    @discardableResult
    public static func sgetri(
        layout: Layout, n: Int, a: UnsafeMutablePointer<Float>, lda: Int, ipiv: UnsafePointer<Int>
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return _withOpenBLASInputPivots(count: Swift.max(0, n), source: ipiv) { pivots in
            LAPACKE_sgetri(
                _backendIndex(layout.rawValue), _backendIndex(n), a, _backendIndex(lda), pivots)
        }
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateSgetri(layout: layout, n: n, a: a, lda: lda, ipiv: ipiv)
        #else
        return _lapackGetri(layout: layout, n: n, a: a, lda: lda, ipiv: ipiv)
        #endif
    }

    @discardableResult
    public static func dgetri(
        layout: Layout, n: Int, a: UnsafeMutablePointer<Double>, lda: Int, ipiv: UnsafePointer<Int>
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return _withOpenBLASInputPivots(count: Swift.max(0, n), source: ipiv) { pivots in
            LAPACKE_dgetri(
                _backendIndex(layout.rawValue), _backendIndex(n), a, _backendIndex(lda), pivots)
        }
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateDgetri(layout: layout, n: n, a: a, lda: lda, ipiv: ipiv)
        #else
        return _lapackGetri(layout: layout, n: n, a: a, lda: lda, ipiv: ipiv)
        #endif
    }

    @discardableResult
    public static func cgetri(
        layout: Layout, n: Int, a: UnsafeMutablePointer<Complex<Float>>, lda: Int,
        ipiv: UnsafePointer<Int>
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return _withOpenBLASInputPivots(count: Swift.max(0, n), source: ipiv) { pivots in
            LAPACKE_cgetri(
                _backendIndex(layout.rawValue), _backendIndex(n), _complexFloatPointer(a),
                _backendIndex(lda), pivots)
        }
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateCgetri(layout: layout, n: n, a: a, lda: lda, ipiv: ipiv)
        #else
        return _lapackGetri(layout: layout, n: n, a: a, lda: lda, ipiv: ipiv)
        #endif
    }

    @discardableResult
    public static func zgetri(
        layout: Layout, n: Int, a: UnsafeMutablePointer<Complex<Double>>, lda: Int,
        ipiv: UnsafePointer<Int>
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return _withOpenBLASInputPivots(count: Swift.max(0, n), source: ipiv) { pivots in
            LAPACKE_zgetri(
                _backendIndex(layout.rawValue), _backendIndex(n), _complexDoublePointer(a),
                _backendIndex(lda), pivots)
        }
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateZgetri(layout: layout, n: n, a: a, lda: lda, ipiv: ipiv)
        #else
        return _lapackGetri(layout: layout, n: n, a: a, lda: lda, ipiv: ipiv)
        #endif
    }
}

// MARK: - Positive-definite systems and Cholesky factorization

extension LAPACK {
    @discardableResult
    public static func sposv(
        layout: Layout, triangle: Triangle, n: Int, nrhs: Int, a: UnsafeMutablePointer<Float>,
        lda: Int,
        b: UnsafeMutablePointer<Float>, ldb: Int
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_sposv(
                _backendIndex(layout.rawValue), triangle._lapackCharacter, _backendIndex(n),
                _backendIndex(nrhs), a, _backendIndex(lda), b, _backendIndex(ldb)))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateSposv(
            layout: layout, triangle: triangle, n: n, nrhs: nrhs, a: a, lda: lda, b: b, ldb: ldb
        )
        #else
        return _lapackPosv(
            layout: layout, triangle: triangle, n: n, nrhs: nrhs, a: a, lda: lda, b: b, ldb: ldb
        )
        #endif
    }

    @discardableResult
    public static func dposv(
        layout: Layout, triangle: Triangle, n: Int, nrhs: Int, a: UnsafeMutablePointer<Double>,
        lda: Int, b: UnsafeMutablePointer<Double>, ldb: Int
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_dposv(
                _backendIndex(layout.rawValue), triangle._lapackCharacter, _backendIndex(n),
                _backendIndex(nrhs), a, _backendIndex(lda), b, _backendIndex(ldb)))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateDposv(
            layout: layout, triangle: triangle, n: n, nrhs: nrhs, a: a, lda: lda, b: b, ldb: ldb
        )
        #else
        return _lapackPosv(
            layout: layout, triangle: triangle, n: n, nrhs: nrhs, a: a, lda: lda, b: b, ldb: ldb
        )
        #endif
    }

    @discardableResult
    public static func cposv(
        layout: Layout, triangle: Triangle, n: Int, nrhs: Int,
        a: UnsafeMutablePointer<Complex<Float>>,
        lda: Int, b: UnsafeMutablePointer<Complex<Float>>, ldb: Int
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_cposv(
                _backendIndex(layout.rawValue), triangle._lapackCharacter, _backendIndex(n),
                _backendIndex(nrhs), _complexFloatPointer(a), _backendIndex(lda),
                _complexFloatPointer(b),
                _backendIndex(ldb)))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateCposv(
            layout: layout, triangle: triangle, n: n, nrhs: nrhs, a: a, lda: lda, b: b, ldb: ldb
        )
        #else
        return _lapackPosv(
            layout: layout, triangle: triangle, n: n, nrhs: nrhs, a: a, lda: lda, b: b, ldb: ldb
        )
        #endif
    }

    @discardableResult
    public static func zposv(
        layout: Layout, triangle: Triangle, n: Int, nrhs: Int,
        a: UnsafeMutablePointer<Complex<Double>>,
        lda: Int, b: UnsafeMutablePointer<Complex<Double>>, ldb: Int
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_zposv(
                _backendIndex(layout.rawValue), triangle._lapackCharacter, _backendIndex(n),
                _backendIndex(nrhs), _complexDoublePointer(a), _backendIndex(lda),
                _complexDoublePointer(b), _backendIndex(ldb)))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateZposv(
            layout: layout, triangle: triangle, n: n, nrhs: nrhs, a: a, lda: lda, b: b, ldb: ldb
        )
        #else
        return _lapackPosv(
            layout: layout, triangle: triangle, n: n, nrhs: nrhs, a: a, lda: lda, b: b, ldb: ldb
        )
        #endif
    }

    @discardableResult
    public static func spotrf(
        layout: Layout, triangle: Triangle, n: Int, a: UnsafeMutablePointer<Float>, lda: Int
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_spotrf(
                _backendIndex(layout.rawValue), triangle._lapackCharacter, _backendIndex(n), a,
                _backendIndex(lda)))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateSpotrf(layout: layout, triangle: triangle, n: n, a: a, lda: lda)
        #else
        return _lapackPotrf(layout: layout, triangle: triangle, n: n, a: a, lda: lda)
        #endif
    }

    @discardableResult
    public static func dpotrf(
        layout: Layout, triangle: Triangle, n: Int, a: UnsafeMutablePointer<Double>, lda: Int
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_dpotrf(
                _backendIndex(layout.rawValue), triangle._lapackCharacter, _backendIndex(n), a,
                _backendIndex(lda)))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateDpotrf(layout: layout, triangle: triangle, n: n, a: a, lda: lda)
        #else
        return _lapackPotrf(layout: layout, triangle: triangle, n: n, a: a, lda: lda)
        #endif
    }

    @discardableResult
    public static func cpotrf(
        layout: Layout, triangle: Triangle, n: Int, a: UnsafeMutablePointer<Complex<Float>>,
        lda: Int
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_cpotrf(
                _backendIndex(layout.rawValue), triangle._lapackCharacter, _backendIndex(n),
                _complexFloatPointer(a), _backendIndex(lda)))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateCpotrf(layout: layout, triangle: triangle, n: n, a: a, lda: lda)
        #else
        return _lapackPotrf(layout: layout, triangle: triangle, n: n, a: a, lda: lda)
        #endif
    }

    @discardableResult
    public static func zpotrf(
        layout: Layout, triangle: Triangle, n: Int, a: UnsafeMutablePointer<Complex<Double>>,
        lda: Int
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_zpotrf(
                _backendIndex(layout.rawValue), triangle._lapackCharacter, _backendIndex(n),
                _complexDoublePointer(a), _backendIndex(lda)))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateZpotrf(layout: layout, triangle: triangle, n: n, a: a, lda: lda)
        #else
        return _lapackPotrf(layout: layout, triangle: triangle, n: n, a: a, lda: lda)
        #endif
    }

    @discardableResult
    public static func spotrs(
        layout: Layout, triangle: Triangle, n: Int, nrhs: Int, a: UnsafePointer<Float>, lda: Int,
        b: UnsafeMutablePointer<Float>, ldb: Int
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_spotrs(
                _backendIndex(layout.rawValue), triangle._lapackCharacter, _backendIndex(n),
                _backendIndex(nrhs), a, _backendIndex(lda), b, _backendIndex(ldb)))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateSpotrs(
            layout: layout, triangle: triangle, n: n, nrhs: nrhs, a: a, lda: lda, b: b, ldb: ldb
        )
        #else
        return _lapackPotrs(
            layout: layout, triangle: triangle, n: n, nrhs: nrhs, a: a, lda: lda, b: b, ldb: ldb
        )
        #endif
    }

    @discardableResult
    public static func dpotrs(
        layout: Layout, triangle: Triangle, n: Int, nrhs: Int, a: UnsafePointer<Double>, lda: Int,
        b: UnsafeMutablePointer<Double>, ldb: Int
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_dpotrs(
                _backendIndex(layout.rawValue), triangle._lapackCharacter, _backendIndex(n),
                _backendIndex(nrhs), a, _backendIndex(lda), b, _backendIndex(ldb)))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateDpotrs(
            layout: layout, triangle: triangle, n: n, nrhs: nrhs, a: a, lda: lda, b: b, ldb: ldb
        )
        #else
        return _lapackPotrs(
            layout: layout, triangle: triangle, n: n, nrhs: nrhs, a: a, lda: lda, b: b, ldb: ldb
        )
        #endif
    }

    @discardableResult
    public static func cpotrs(
        layout: Layout, triangle: Triangle, n: Int, nrhs: Int, a: UnsafePointer<Complex<Float>>,
        lda: Int, b: UnsafeMutablePointer<Complex<Float>>, ldb: Int
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_cpotrs(
                _backendIndex(layout.rawValue), triangle._lapackCharacter, _backendIndex(n),
                _backendIndex(nrhs), _complexFloatPointer(a), _backendIndex(lda),
                _complexFloatPointer(b),
                _backendIndex(ldb)))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateCpotrs(
            layout: layout, triangle: triangle, n: n, nrhs: nrhs, a: a, lda: lda, b: b, ldb: ldb
        )
        #else
        return _lapackPotrs(
            layout: layout, triangle: triangle, n: n, nrhs: nrhs, a: a, lda: lda, b: b, ldb: ldb
        )
        #endif
    }

    @discardableResult
    public static func zpotrs(
        layout: Layout, triangle: Triangle, n: Int, nrhs: Int, a: UnsafePointer<Complex<Double>>,
        lda: Int, b: UnsafeMutablePointer<Complex<Double>>, ldb: Int
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_zpotrs(
                _backendIndex(layout.rawValue), triangle._lapackCharacter, _backendIndex(n),
                _backendIndex(nrhs), _complexDoublePointer(a), _backendIndex(lda),
                _complexDoublePointer(b), _backendIndex(ldb)))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateZpotrs(
            layout: layout, triangle: triangle, n: n, nrhs: nrhs, a: a, lda: lda, b: b, ldb: ldb
        )
        #else
        return _lapackPotrs(
            layout: layout, triangle: triangle, n: n, nrhs: nrhs, a: a, lda: lda, b: b, ldb: ldb
        )
        #endif
    }

    @discardableResult
    public static func spotri(
        layout: Layout, triangle: Triangle, n: Int, a: UnsafeMutablePointer<Float>, lda: Int
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_spotri(
                _backendIndex(layout.rawValue), triangle._lapackCharacter, _backendIndex(n), a,
                _backendIndex(lda)))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateSpotri(layout: layout, triangle: triangle, n: n, a: a, lda: lda)
        #else
        return _lapackPotri(layout: layout, triangle: triangle, n: n, a: a, lda: lda)
        #endif
    }

    @discardableResult
    public static func dpotri(
        layout: Layout, triangle: Triangle, n: Int, a: UnsafeMutablePointer<Double>, lda: Int
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_dpotri(
                _backendIndex(layout.rawValue), triangle._lapackCharacter, _backendIndex(n), a,
                _backendIndex(lda)))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateDpotri(layout: layout, triangle: triangle, n: n, a: a, lda: lda)
        #else
        return _lapackPotri(layout: layout, triangle: triangle, n: n, a: a, lda: lda)
        #endif
    }

    @discardableResult
    public static func cpotri(
        layout: Layout, triangle: Triangle, n: Int, a: UnsafeMutablePointer<Complex<Float>>,
        lda: Int
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_cpotri(
                _backendIndex(layout.rawValue), triangle._lapackCharacter, _backendIndex(n),
                _complexFloatPointer(a), _backendIndex(lda)))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateCpotri(layout: layout, triangle: triangle, n: n, a: a, lda: lda)
        #else
        return _lapackPotri(layout: layout, triangle: triangle, n: n, a: a, lda: lda)
        #endif
    }

    @discardableResult
    public static func zpotri(
        layout: Layout, triangle: Triangle, n: Int, a: UnsafeMutablePointer<Complex<Double>>,
        lda: Int
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_zpotri(
                _backendIndex(layout.rawValue), triangle._lapackCharacter, _backendIndex(n),
                _complexDoublePointer(a), _backendIndex(lda)))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateZpotri(layout: layout, triangle: triangle, n: n, a: a, lda: lda)
        #else
        return _lapackPotri(layout: layout, triangle: triangle, n: n, a: a, lda: lda)
        #endif
    }
}

// MARK: - Triangular systems

extension LAPACK {
    @discardableResult
    public static func strtrs(
        layout: Layout, triangle: Triangle, transpose: Transpose, diagonal: Diagonal, n: Int,
        nrhs: Int,
        a: UnsafePointer<Float>, lda: Int, b: UnsafeMutablePointer<Float>, ldb: Int
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_strtrs(
                _backendIndex(layout.rawValue), triangle._lapackCharacter, transpose._character,
                diagonal._lapackCharacter, _backendIndex(n), _backendIndex(nrhs), a,
                _backendIndex(lda),
                b, _backendIndex(ldb)))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateStrtrs(
            layout: layout, triangle: triangle, transpose: transpose, diagonal: diagonal, n: n,
            nrhs: nrhs, a: a, lda: lda, b: b, ldb: ldb)
        #else
        return _lapackTrtrs(
            layout: layout, triangle: triangle, transpose: transpose, diagonal: diagonal, n: n,
            nrhs: nrhs, a: a, lda: lda, b: b, ldb: ldb)
        #endif
    }

    @discardableResult
    public static func dtrtrs(
        layout: Layout, triangle: Triangle, transpose: Transpose, diagonal: Diagonal, n: Int,
        nrhs: Int,
        a: UnsafePointer<Double>, lda: Int, b: UnsafeMutablePointer<Double>, ldb: Int
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_dtrtrs(
                _backendIndex(layout.rawValue), triangle._lapackCharacter, transpose._character,
                diagonal._lapackCharacter, _backendIndex(n), _backendIndex(nrhs), a,
                _backendIndex(lda),
                b, _backendIndex(ldb)))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateDtrtrs(
            layout: layout, triangle: triangle, transpose: transpose, diagonal: diagonal, n: n,
            nrhs: nrhs, a: a, lda: lda, b: b, ldb: ldb)
        #else
        return _lapackTrtrs(
            layout: layout, triangle: triangle, transpose: transpose, diagonal: diagonal, n: n,
            nrhs: nrhs, a: a, lda: lda, b: b, ldb: ldb)
        #endif
    }

    @discardableResult
    public static func ctrtrs(
        layout: Layout, triangle: Triangle, transpose: Transpose, diagonal: Diagonal, n: Int,
        nrhs: Int,
        a: UnsafePointer<Complex<Float>>, lda: Int, b: UnsafeMutablePointer<Complex<Float>>,
        ldb: Int
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_ctrtrs(
                _backendIndex(layout.rawValue), triangle._lapackCharacter, transpose._character,
                diagonal._lapackCharacter, _backendIndex(n), _backendIndex(nrhs),
                _complexFloatPointer(a),
                _backendIndex(lda), _complexFloatPointer(b), _backendIndex(ldb)))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateCtrtrs(
            layout: layout, triangle: triangle, transpose: transpose, diagonal: diagonal, n: n,
            nrhs: nrhs, a: a, lda: lda, b: b, ldb: ldb)
        #else
        return _lapackTrtrs(
            layout: layout, triangle: triangle, transpose: transpose, diagonal: diagonal, n: n,
            nrhs: nrhs, a: a, lda: lda, b: b, ldb: ldb)
        #endif
    }

    @discardableResult
    public static func ztrtrs(
        layout: Layout, triangle: Triangle, transpose: Transpose, diagonal: Diagonal, n: Int,
        nrhs: Int,
        a: UnsafePointer<Complex<Double>>, lda: Int, b: UnsafeMutablePointer<Complex<Double>>,
        ldb: Int
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_ztrtrs(
                _backendIndex(layout.rawValue), triangle._lapackCharacter, transpose._character,
                diagonal._lapackCharacter, _backendIndex(n), _backendIndex(nrhs),
                _complexDoublePointer(a), _backendIndex(lda), _complexDoublePointer(b),
                _backendIndex(ldb)
            ))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateZtrtrs(
            layout: layout, triangle: triangle, transpose: transpose, diagonal: diagonal, n: n,
            nrhs: nrhs, a: a, lda: lda, b: b, ldb: ldb)
        #else
        return _lapackTrtrs(
            layout: layout, triangle: triangle, transpose: transpose, diagonal: diagonal, n: n,
            nrhs: nrhs, a: a, lda: lda, b: b, ldb: ldb)
        #endif
    }

    @discardableResult
    public static func strtri(
        layout: Layout, triangle: Triangle, diagonal: Diagonal, n: Int,
        a: UnsafeMutablePointer<Float>,
        lda: Int
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_strtri(
                _backendIndex(layout.rawValue), triangle._lapackCharacter,
                diagonal._lapackCharacter,
                _backendIndex(n), a, _backendIndex(lda)))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateStrtri(
            layout: layout, triangle: triangle, diagonal: diagonal, n: n, a: a, lda: lda)
        #else
        return _lapackTrtri(
            layout: layout, triangle: triangle, diagonal: diagonal, n: n, a: a, lda: lda)
        #endif
    }

    @discardableResult
    public static func dtrtri(
        layout: Layout, triangle: Triangle, diagonal: Diagonal, n: Int,
        a: UnsafeMutablePointer<Double>,
        lda: Int
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_dtrtri(
                _backendIndex(layout.rawValue), triangle._lapackCharacter,
                diagonal._lapackCharacter,
                _backendIndex(n), a, _backendIndex(lda)))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateDtrtri(
            layout: layout, triangle: triangle, diagonal: diagonal, n: n, a: a, lda: lda)
        #else
        return _lapackTrtri(
            layout: layout, triangle: triangle, diagonal: diagonal, n: n, a: a, lda: lda)
        #endif
    }

    @discardableResult
    public static func ctrtri(
        layout: Layout, triangle: Triangle, diagonal: Diagonal, n: Int,
        a: UnsafeMutablePointer<Complex<Float>>, lda: Int
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_ctrtri(
                _backendIndex(layout.rawValue), triangle._lapackCharacter,
                diagonal._lapackCharacter,
                _backendIndex(n), _complexFloatPointer(a), _backendIndex(lda)))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateCtrtri(
            layout: layout, triangle: triangle, diagonal: diagonal, n: n, a: a, lda: lda)
        #else
        return _lapackTrtri(
            layout: layout, triangle: triangle, diagonal: diagonal, n: n, a: a, lda: lda)
        #endif
    }

    @discardableResult
    public static func ztrtri(
        layout: Layout, triangle: Triangle, diagonal: Diagonal, n: Int,
        a: UnsafeMutablePointer<Complex<Double>>, lda: Int
    ) -> Int {
        #if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
        return Int(
            LAPACKE_ztrtri(
                _backendIndex(layout.rawValue), triangle._lapackCharacter,
                diagonal._lapackCharacter,
                _backendIndex(n), _complexDoublePointer(a), _backendIndex(lda)))
        #elseif canImport(Accelerate) && !SEBBU_LAPACK_FORCE_SWIFT
        return _accelerateZtrtri(
            layout: layout, triangle: triangle, diagonal: diagonal, n: n, a: a, lda: lda)
        #else
        return _lapackTrtri(
            layout: layout, triangle: triangle, diagonal: diagonal, n: n, a: a, lda: lda)
        #endif
    }
}
