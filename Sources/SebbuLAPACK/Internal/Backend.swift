// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: MIT

#if canImport(COpenBLAS) && !SEBBU_LAPACK_FORCE_SWIFT
@inlinable
@inline(always)
@_transparent
internal func _backendIndex(_ value: Int) -> Int32 {
    precondition(
        value >= Int(Int32.min) && value <= Int(Int32.max),
        "The OpenBLAS LP64 interface requires every integer argument to fit in Int32"
    )
    return Int32(value)
}
#endif
