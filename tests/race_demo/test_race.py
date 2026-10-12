import numpy as np
import numba_funcs as nf

X = np.linspace(-3, 3, 257)


def test_0():
    nf.jit1(X)
    nf.jit9(X)
    nf.gu8(X)
    nf.jit5(X)


def test_1():
    nf.jit7(X)
    nf.jit11(X)
    nf.gu0(X)
    nf.gu8(X)


def test_2():
    nf.jit9(X)
    nf.jit11(X)
    nf.jit3(X)
    nf.gu4(X)


def test_3():
    nf.jit1(X)
    nf.jit7(X)
    nf.gu10(X)
    nf.gu0(X)


def test_4():
    nf.jit11(X)
    nf.gu2(X)
    nf.gu8(X)
    nf.gu10(X)


def test_5():
    nf.gu10(X)
    nf.gu2(X)
    nf.jit11(X)
    nf.jit7(X)


def test_6():
    nf.gu6(X)
    nf.jit5(X)
    nf.gu2(X)
    nf.jit3(X)


def test_7():
    nf.jit7(X)
    nf.jit11(X)
    nf.jit3(X)
    nf.gu10(X)


def test_8():
    nf.jit7(X)
    nf.gu8(X)
    nf.gu4(X)
    nf.jit1(X)


def test_9():
    nf.gu8(X)
    nf.gu6(X)
    nf.jit3(X)
    nf.jit11(X)


def test_10():
    nf.jit11(X)
    nf.gu8(X)
    nf.gu4(X)
    nf.gu2(X)


def test_11():
    nf.gu2(X)
    nf.jit5(X)
    nf.gu0(X)
    nf.gu6(X)


def test_12():
    nf.jit9(X)
    nf.gu10(X)
    nf.jit1(X)
    nf.gu6(X)


def test_13():
    nf.gu8(X)
    nf.jit7(X)
    nf.gu0(X)
    nf.gu10(X)


def test_14():
    nf.gu10(X)
    nf.jit11(X)
    nf.jit7(X)
    nf.gu0(X)


def test_15():
    nf.gu4(X)
    nf.jit11(X)
    nf.jit7(X)
    nf.gu6(X)


def test_16():
    nf.gu8(X)
    nf.jit9(X)
    nf.gu2(X)
    nf.jit11(X)


def test_17():
    nf.gu10(X)
    nf.jit3(X)
    nf.gu0(X)
    nf.gu2(X)


def test_18():
    nf.gu0(X)
    nf.jit11(X)
    nf.gu4(X)
    nf.gu6(X)


def test_19():
    nf.jit9(X)
    nf.jit11(X)
    nf.gu6(X)
    nf.jit5(X)


def test_20():
    nf.jit5(X)
    nf.jit3(X)
    nf.gu0(X)
    nf.jit7(X)


def test_21():
    nf.gu8(X)
    nf.gu0(X)
    nf.jit11(X)
    nf.jit5(X)


def test_22():
    nf.gu4(X)
    nf.jit1(X)
    nf.gu10(X)
    nf.gu6(X)


def test_23():
    nf.jit7(X)
    nf.jit9(X)
    nf.jit5(X)
    nf.gu10(X)


def test_24():
    nf.gu4(X)
    nf.jit7(X)
    nf.gu8(X)
    nf.gu0(X)


def test_25():
    nf.gu8(X)
    nf.jit1(X)
    nf.jit9(X)
    nf.jit5(X)


def test_26():
    nf.gu4(X)
    nf.jit5(X)
    nf.gu2(X)
    nf.gu8(X)


def test_27():
    nf.jit5(X)
    nf.jit11(X)
    nf.jit1(X)
    nf.gu8(X)


def test_28():
    nf.gu6(X)
    nf.jit9(X)
    nf.gu0(X)
    nf.gu4(X)


def test_29():
    nf.jit7(X)
    nf.jit9(X)
    nf.gu0(X)
    nf.gu6(X)


def test_30():
    nf.jit1(X)
    nf.jit5(X)
    nf.gu6(X)
    nf.gu10(X)


def test_31():
    nf.gu4(X)
    nf.jit3(X)
    nf.gu10(X)
    nf.gu8(X)


def test_32():
    nf.gu6(X)
    nf.jit9(X)
    nf.jit11(X)
    nf.gu8(X)


def test_33():
    nf.gu0(X)
    nf.jit11(X)
    nf.gu10(X)
    nf.jit1(X)


def test_34():
    nf.gu10(X)
    nf.gu4(X)
    nf.jit1(X)
    nf.gu6(X)


def test_35():
    nf.jit11(X)
    nf.jit1(X)
    nf.gu4(X)
    nf.gu0(X)


def test_36():
    nf.gu6(X)
    nf.gu2(X)
    nf.jit3(X)
    nf.jit11(X)


def test_37():
    nf.gu4(X)
    nf.jit11(X)
    nf.gu6(X)
    nf.jit7(X)


def test_38():
    nf.jit7(X)
    nf.gu4(X)
    nf.gu0(X)
    nf.jit9(X)


def test_39():
    nf.jit5(X)
    nf.jit9(X)
    nf.gu2(X)
    nf.gu8(X)


def test_40():
    nf.gu10(X)
    nf.gu6(X)
    nf.jit11(X)
    nf.gu4(X)


def test_41():
    nf.jit9(X)
    nf.gu0(X)
    nf.jit1(X)
    nf.jit7(X)


def test_42():
    nf.jit7(X)
    nf.jit5(X)
    nf.gu2(X)
    nf.gu8(X)


def test_43():
    nf.gu10(X)
    nf.jit1(X)
    nf.jit9(X)
    nf.jit3(X)


def test_44():
    nf.jit5(X)
    nf.gu4(X)
    nf.jit11(X)
    nf.gu0(X)


def test_45():
    nf.jit5(X)
    nf.jit3(X)
    nf.jit9(X)
    nf.gu8(X)


def test_46():
    nf.jit7(X)
    nf.jit11(X)
    nf.gu8(X)
    nf.jit5(X)


def test_47():
    nf.jit9(X)
    nf.gu0(X)
    nf.gu10(X)
    nf.jit3(X)
