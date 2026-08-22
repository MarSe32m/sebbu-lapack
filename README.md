# sebbu-lapack

`sebbu-lapack` provides Swift namespace wrappers for commonly used LAPACKE
computational routines and drivers. It complements
[`sebbu-blas`](https://github.com/MarSe32m/sebbu-blas), whose public matrix
layout, triangle and diagonal types it reuses.

The API is pointer based, follows LAPACKE's row-major and column-major storage
rules, and deliberately does not expose the lower-level `*_work` routines.
Dimensions, leading dimensions, pivot arrays and `info` values use Swift
`Int`. Pivot indices remain one-based, as in LAPACK.

```swift
import SebbuLAPACK

var a = [
    3.0, 1, -1,
    2, 4, 1,
    -1, 2, 5,
]
var b = [4.0, 1, 1]
var pivots = [Int](repeating: 0, count: 3)

let info = LAPACK.dgesv(
    layout: .rowMajor, n: 3, nrhs: 1,
    a: &a, lda: 3, ipiv: &pivots,
    b: &b, ldb: 1
)
```

## Supported routines

The initial API includes these single-, double-, complex-single-, and
complex-double-precision families:

| Area | Routines |
| --- | --- |
| General systems and LU | `*gesv`, `*getrf`, `*getrs`, `*getri` |
| Positive-definite systems and Cholesky | `*posv`, `*potrf`, `*potrs`, `*potri` |
| Triangular systems | `*trtrs`, `*trtri` |
| QR factorization | `*geqrf`, `*orgqr`, `*ungqr` |
| Least squares | `*gels` |
| General eigenproblems and Schur decompositions | `*geev`, `*gees` |
| Symmetric/Hermitian eigenproblems | `ssyev`, `dsyev`, `cheev`, `zheev` |
| Singular value decomposition | `*gesvd`, `*gesdd` |
| Matrix norms | `*lange`, `slansy`, `dlansy`, `clanhe`, `zlanhe` |

Factorization and driver routines overwrite their matrix arguments like their
LAPACKE counterparts. A zero `info` means success, a negative value identifies
an invalid argument, and a positive value reports a routine-specific numerical
failure.

Complex routines use `Complex<Float>` and `Complex<Double>` from
Swift Numerics.

## Backends

- Linux and Windows call LAPACKE through
  [`sebbu-copenblas`](https://github.com/MarSe32m/sebbu-copenblas).
- macOS, iOS, tvOS, and watchOS call Accelerate's current ILP64 LAPACK
  interface. Row-major inputs and outputs are converted around the native
  column-major routines.
- Other platforms use the pure Swift implementations.
- Pass `-Xswiftc -DSEBBU_LAPACK_FORCE_SWIFT` to force the Swift backend for
  testing.

The fallback least-squares, general-eigenproblem, Schur-decomposition, and SVD
implementations prioritize portability. For large or ill-conditioned problems,
prefer the OpenBLAS or Accelerate backend.

## Package dependency

```swift
.package(
    url: "https://github.com/MarSe32m/sebbu-lapack",
    from: "0.1.0"
)
```

Add the product to the dependencies of the target that uses it:

```swift
.product(name: "SebbuLAPACK", package: "sebbu-lapack")
```

## Validation

Run the tests against both the selected native backend and the Swift fallback:

```sh
swift test --scratch-path .build/native
swift test --scratch-path .build/fallback \
    -Xswiftc -DSEBBU_LAPACK_FORCE_SWIFT \
    -Xswiftc -DSEBBU_BLAS_FORCE_SWIFT
```

The second command also forces `sebbu-blas` onto its Swift backend because the
test suite uses BLAS to verify QR and SVD reconstructions. Separate scratch
paths keep the two compilation-condition sets isolated.

Callers are responsible for providing buffers large enough for the matrix
dimensions and leading dimensions they pass.
