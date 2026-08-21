import SebbuLAPACK
import Testing

@Test
func testSebbuLAPACKIsSufficientForThePublicAPI() {
    var a = [2.0]
    var b = [6.0]
    var pivots = [0]

    let info = LAPACK.dgesv(
        layout: .rowMajor, n: 1, nrhs: 1,
        a: &a, lda: 1, ipiv: &pivots,
        b: &b, ldb: 1
    )

    #expect(info == 0)
    #expect(b == [3.0])
}
