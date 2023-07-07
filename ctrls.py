import sympy as sym
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy.signal as sig
import control as ct
import control.matlab as ctm

plt.style.use('seaborn-v0_8')

def step(H):
    t, y = ct.step_response(H)
    plt.plot(t, y)
    plt.axhline(y=y[-1], xmin=0, color='0', linestyle='--')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')

def c2d_fe(H, T):
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

def zn1(P):
    t, y = ct.step_response(P)
    i5 = np.where(y > 0.05*y[-1])[0][0]       # 5% index
    i15 = np.where(y > 0.15*y[-1])[0][0]      # 15% index
    il = np.where(y > 0)[0][0]                # L index

    r = (y[i15] - y[i5]) / (t[i15] - t[i5])
    l = t[il]
    C_pid1 = 1.2/(r * l) + (0.6/(r * l**2)) * ct.tf([1], [1, 0]) + (0.5/r) * ct.tf([1, 0], [1/100, 1])

    L1 = P * C_pid1
    T1 = L1 / (1 + L1)
    S1 = 1 - T1

    return(C_pid1, L1, T1, S1)

def zn2_K(P, K):
    T_Ktest = P*K / (1 + P*K)
    t, y = ct.step_response(T_Ktest)
    plt.plot(t, y)

def zn2(P, Ku):
    T = P*Ku / (1 + P*Ku)
    time, y = ct.step_response(T)
    peaks = sig.find_peaks(y)

    Pu = time[peaks[0][2]] - time[peaks[0][1]]

    C_pid2 = 0.6 * Ku + (1.2 * Ku / Pu) * ct.tf(1, [1, 0]) + (0.075 * (Ku * Pu)) * ct.tf([1, 0], [1 / 100, 1])

    L2 = P * C_pid2
    T2 = L2 / (1 + L2)
    S2 = 1 - T2

    return(C_pid2, L2, T2, S2)

def pid_ct(plant : ct.TransferFunction, phasemargin : float, bandwidth = None, plot=False, maxpole=False):
    zeros = -np.roots(plant.num[0][0])                  # really the negative of the zeros & poles
    poles = -np.roots(plant.den[0][0])

    if maxpole:
        maxpole = np.max(np.abs(poles))
        if maxpole % np.floor(maxpole) != 0:            # i.e., if maxpole is not an integer
            wc = np.ceil(maxpole)
        else:
            wc = maxpole + 0.5
    else:
        wc = 2 * bandwidth / 3

    tau = 1/(wc*10)

    theta = -np.pi + np.radians(phasemargin) + np.pi/2 + np.arctan(tau * wc)
    alpha = beta = 0
    for z in zeros:
        alpha += np.arctan(wc/z)

    for p in poles:
        beta += np.arctan(wc/p)

    a = wc / np.tan((theta - alpha - beta) / 2)

    alpha = beta = 1
    for z in zeros:
        alpha *= la.norm([wc, z], 2)

    for p in poles:
        beta *= la.norm([wc, p], 2)

    num = beta * wc * la.norm([tau * wc, 1], 2)
    den = alpha * (wc**2 + a**2)
    K = num / den

    controller = K * ct.tf([1, a], [1, 0]) * ct.tf([1, a], [tau, 1])
    l = plant * controller
    t = l / (1 + l)
    s = 1 - t

    if plot:
        plt.figure()
        ct.bode([t, s])
        plt.figlegend(['$T$', '$S$'])
        plt.suptitle('Sensitivity and complementary sensitivity frequency responses')

        plt.figure()
        step(t)
        plt.title('Closed-Loop step response')

    return(controller)


    return(C_pid2, L2, T2, S2)
