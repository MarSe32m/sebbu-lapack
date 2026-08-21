// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: MIT

import SebbuBLAS

@inlinable
internal func _matrixIndex(
    _ layout: LAPACK.Layout,
    _ row: Int,
    _ column: Int,
    _ leadingDimension: Int
) -> Int {
    switch layout {
    case .rowMajor: row * leadingDimension + column
    case .columnMajor: column * leadingDimension + row
    }
}

@inlinable
internal func _symmetricElement<T: LAPACKScalar>(
    _ pointer: UnsafePointer<T>,
    layout: LAPACK.Layout,
    triangle: LAPACK.Triangle,
    row: Int,
    column: Int,
    leadingDimension: Int,
    hermitian: Bool
) -> T {
    let stored = triangle == .upper ? row <= column : row >= column
    let sourceRow = stored ? row : column
    let sourceColumn = stored ? column : row
    let value = pointer[_matrixIndex(layout, sourceRow, sourceColumn, leadingDimension)]
    if hermitian && row == column { return value.withZeroImaginary }
    return hermitian && !stored ? value.conjugate : value
}
