from rpython.jit.backend.llsupport.vector_ext import VectorExt


class Aarch64VectorExt(VectorExt):
    # NEON does not need the loop entry aligned (unaligned LD1/ST1 .2D
    # are unconstrained), unlike x86 SSE.
    should_align_unroll = False

    def setup_once(self, asm):
        # NEON Advanced-SIMD is mandatory on AArch64: 128-bit vector
        # registers, float64x2.  accum=False -> the vectorizer does not
        # introduce reduction accumulators yet (deferred stage).
        self.enable(16, accum=False)
        self._setup = True
