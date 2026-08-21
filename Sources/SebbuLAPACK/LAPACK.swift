// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: MIT

import SebbuBLAS

/// A namespace for LAPACK computational routines and drivers.
///
/// The API follows LAPACKE's storage rules, but uses Swift `Int` for every
/// dimension, leading dimension, pivot, and `info` value. Routines overwrite
/// their matrix arguments in the same way as their LAPACKE counterparts.
@frozen
public enum LAPACK {}

extension LAPACK {
    public typealias Index = Int
    public typealias Layout = BLAS.Layout
    public typealias Triangle = BLAS.Triangle
    public typealias Diagonal = BLAS.Diagonal

    /// The operation applied to a matrix before a solve.
    @frozen
    public enum Transpose: Sendable {
        case noTranspose
        case transpose
        case conjugateTranspose
    }

    /// Whether an eigenvalue routine computes eigenvectors.
    @frozen
    public enum Eigenvectors: Sendable {
        case none
        case vectors
    }

    /// The singular vectors returned by an SVD routine.
    @frozen
    public enum SingularVectors: Sendable {
        /// Compute every singular vector.
        case all
        /// Compute the first `min(m, n)` singular vectors.
        case some
        /// Overwrite the appropriate part of the input matrix.
        case overwrite
        /// Do not compute singular vectors.
        case none
    }

    /// A matrix norm accepted by the `*lange`, `*lansy`, and `*lanhe`
    /// families.
    @frozen
    public enum Norm: Sendable {
        case maxAbsoluteValue
        case one
        case infinity
        case frobenius
    }
}

extension LAPACK.Transpose {
    @usableFromInline
    internal var _character: CChar {
        switch self {
        case .noTranspose: 78  // N
        case .transpose: 84  // T
        case .conjugateTranspose: 67  // C
        }
    }
}

extension LAPACK.Eigenvectors {
    @usableFromInline
    internal var _character: CChar {
        switch self {
        case .none: 78  // N
        case .vectors: 86  // V
        }
    }
}

extension LAPACK.SingularVectors {
    @usableFromInline
    internal var _character: CChar {
        switch self {
        case .all: 65  // A
        case .some: 83  // S
        case .overwrite: 79  // O
        case .none: 78  // N
        }
    }
}

extension LAPACK.Norm {
    @usableFromInline
    internal var _character: CChar {
        switch self {
        case .maxAbsoluteValue: 77  // M
        case .one: 49  // 1
        case .infinity: 73  // I
        case .frobenius: 70  // F
        }
    }
}

extension BLAS.Triangle {
    @usableFromInline
    internal var _lapackCharacter: CChar {
        switch self {
        case .upper: 85  // U
        case .lower: 76  // L
        }
    }
}

extension BLAS.Diagonal {
    @usableFromInline
    internal var _lapackCharacter: CChar {
        switch self {
        case .nonUnit: 78  // N
        case .unit: 85  // U
        }
    }
}
