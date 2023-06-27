import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
import control as ct
import control.matlab as ctm

plt.style.use('seaborn-v0_8')

def c2d_feuler(H, T):
    H = ct.tf(H)

    z = sym.Symbol('z')
    s = (z - 1) / T

    num, den = ct.tfdata(H)

    num = num[0][0]
    nums = sym.Poly.from_list(num, s)
    den = den[0][0]
    dens = sym.Poly.from_list(den, s)

    Hz = sym.simplify(nums / dens)
    sym_num = sym.numer(Hz)
    sym_den = sym.denom(Hz)

    try:
        num = sym.Poly(sym_num).all_coeffs()
        num = list(map(float, num))
    except Exception:                           # fix me
        num = [1]                               # <-----
        num = list(map(float, num))
    finally:
        den = sym.Poly(sym_den).all_coeffs()
        den = list(map(float, den))

    Hz = ct.tf(num, den, dt=T)
    Hz = ct.ss(Hz)
    return(Hz)

def d2c_tustin(H, T):
    H = ct.tf(H, dt=T)

    s = sym.Symbol('s')
    z = (2 + s*T) / (2 - s*T)

    num, den = ct.tfdata(H)

    num = num[0][0]
    numz = sym.Poly.from_list(num, z)
    den = den[0][0]
    denz = sym.Poly.from_list(den, z)

    Hs = sym.simplify(numz / denz)
    sym_num = sym.numer(Hs)
    sym_den = sym.denom(Hs)

    num = sym.Poly(sym_num).all_coeffs()
    num = list(map(float, num))
    den = sym.Poly(sym_den).all_coeffs()
    den = list(map(float, den))

    Hs = ct.tf(num, den)
    Hs = ct.ss(Hs)
    return (Hs)