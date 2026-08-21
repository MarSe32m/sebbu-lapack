// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: MIT

import SebbuBLAS

@inline(__always)
internal func _withLAPACKColumnMajorMutableMatrix<T, Result>(
    layout: LAPACK.Layout,
    rows: Int,
    columns: Int,
    matrix: UnsafeMutablePointer<T>,
    leadingDimension: Int,
    _ body: (UnsafeMutablePointer<T>, Int) -> Result
) -> Result {
    if layout == .columnMajor {
        return body(matrix, leadingDimension)
    }
    let safeRows = Swift.max(0, rows)
    let safeColumns = Swift.max(0, columns)
    if safeRows == 0 || safeColumns == 0 {
        return body(matrix, Swift.max(1, safeRows))
    }
    var storage = [T]()
    storage.reserveCapacity(safeRows * safeColumns)
    for column in 0..<safeColumns {
        for row in 0..<safeRows {
            storage.append(matrix[row * leadingDimension + column])
        }
    }
    let result = storage.withUnsafeMutableBufferPointer { storage in
        body(storage.baseAddress!, Swift.max(1, safeRows))
    }
    for column in 0..<safeColumns {
        for row in 0..<safeRows {
            matrix[row * leadingDimension + column] =
                storage[column * safeRows + row]
        }
    }
    return result
}

@inline(__always)
internal func _withLAPACKColumnMajorInputMatrix<T, Result>(
    layout: LAPACK.Layout,
    rows: Int,
    columns: Int,
    matrix: UnsafePointer<T>,
    leadingDimension: Int,
    _ body: (UnsafePointer<T>, Int) -> Result
) -> Result {
    if layout == .columnMajor {
        return body(matrix, leadingDimension)
    }
    let safeRows = Swift.max(0, rows)
    let safeColumns = Swift.max(0, columns)
    if safeRows == 0 || safeColumns == 0 {
        return body(matrix, Swift.max(1, safeRows))
    }
    var storage = [T]()
    storage.reserveCapacity(safeRows * safeColumns)
    for column in 0..<safeColumns {
        for row in 0..<safeRows {
            storage.append(matrix[row * leadingDimension + column])
        }
    }
    return storage.withUnsafeBufferPointer { storage in
        body(storage.baseAddress!, Swift.max(1, safeRows))
    }
}

@inline(__always)
internal func _withLAPACKColumnMajorOutputMatrix<T, Result>(
    layout: LAPACK.Layout,
    rows: Int,
    columns: Int,
    matrix: UnsafeMutablePointer<T>,
    leadingDimension: Int,
    initialValue: T,
    _ body: (UnsafeMutablePointer<T>, Int) -> Result
) -> Result {
    if layout == .columnMajor {
        return body(matrix, leadingDimension)
    }
    let safeRows = Swift.max(0, rows)
    let safeColumns = Swift.max(0, columns)
    var storage = [T](
        repeating: initialValue,
        count: Swift.max(1, safeRows * safeColumns)
    )
    let result = storage.withUnsafeMutableBufferPointer { storage in
        body(storage.baseAddress!, Swift.max(1, safeRows))
    }
    for column in 0..<safeColumns {
        for row in 0..<safeRows {
            matrix[row * leadingDimension + column] =
                storage[column * safeRows + row]
        }
    }
    return result
}
