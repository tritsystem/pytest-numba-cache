"""Twelve cached numba functions in one module, like librosa's (the shape numba#10128 needs)."""
import numba

@numba.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def gu0(x, out):
    acc = 0.0
    for j in range(x.shape[0]):
        acc = acc * 0.1 + x[j] * 1
        out[j] = acc

@numba.jit(nopython=True, cache=True)
def jit1(x):
    s = 0.0
    for j in range(x.shape[0]):
        s += (x[j] * 1) % 7.0
    return s

@numba.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def gu2(x, out):
    acc = 0.0
    for j in range(x.shape[0]):
        acc = acc * 0.3 + x[j] * 3
        out[j] = acc

@numba.jit(nopython=True, cache=True)
def jit3(x):
    s = 0.0
    for j in range(x.shape[0]):
        s += (x[j] * 3) % 7.0
    return s

@numba.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def gu4(x, out):
    acc = 0.0
    for j in range(x.shape[0]):
        acc = acc * 0.5 + x[j] * 5
        out[j] = acc

@numba.jit(nopython=True, cache=True)
def jit5(x):
    s = 0.0
    for j in range(x.shape[0]):
        s += (x[j] * 5) % 7.0
    return s

@numba.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def gu6(x, out):
    acc = 0.0
    for j in range(x.shape[0]):
        acc = acc * 0.7 + x[j] * 7
        out[j] = acc

@numba.jit(nopython=True, cache=True)
def jit7(x):
    s = 0.0
    for j in range(x.shape[0]):
        s += (x[j] * 7) % 7.0
    return s

@numba.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def gu8(x, out):
    acc = 0.0
    for j in range(x.shape[0]):
        acc = acc * 0.9 + x[j] * 9
        out[j] = acc

@numba.jit(nopython=True, cache=True)
def jit9(x):
    s = 0.0
    for j in range(x.shape[0]):
        s += (x[j] * 9) % 7.0
    return s

@numba.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)", nopython=True, cache=True)
def gu10(x, out):
    acc = 0.0
    for j in range(x.shape[0]):
        acc = acc * 0.11 + x[j] * 11
        out[j] = acc

@numba.jit(nopython=True, cache=True)
def jit11(x):
    s = 0.0
    for j in range(x.shape[0]):
        s += (x[j] * 11) % 7.0
    return s
