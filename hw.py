from ctrls import *

plt.style.use('seaborn-v0_8')


def prob1():
    c_v = c_w = 0.1
    m = 0.2
    l = 0.2
    M = 0.4
    g = 9.81
    R = 5
    L = 0.1
    k = 0.3
    r = 0.1

    pout = 75 * ct.tf([1, 15.6], [1, 49]) * ct.tf([1, -3.1], [1, 22]) * ct.tf([1], [1, -3.3]) * ct.tf([1], [1, 0.47])
    pin = ct.tf([375, 0], [1, 49]) * ct.tf([1], [1, 22]) * ct.tf([1], [1, -3.3]) * ct.tf([1], [1, 0.47])
    p3 = ct.tf([1], [1])
    fs = 100
    Ts = 1 / fs
    pm = 50

    pout_ss = ct.tf2ss(p1)
    pin_ss = ct.tf2ss(p2)

    pin_w = ct.tf([0.003136, -0.003312, -0.002149, 0.00229], [1, -3.444, 4.391, -2.543, 0.5058])
    pout_w = ct.tf([-0.0001696, -0.3351, 69.24, 931.1, -3542], [1, 67.12, 853.2, -3087, -1633])

    zeros_in = np.roots(pin_w.num[0][0])
    poles_in = np.roots(pin_w.den[0][0])
    zeros_out = np.roots(pout_w.num[0][0])
    poles_out = np.roots(pout_w.den[0][0])

    maxpole = np.max(np.abs(poles_in), np.abs(poles_out))
    wc = np.ceil(maxpole)

    theta = -np.pi / 2 + np.radians(pm)
    a_in = b_in = 0
    a_out = b_out = 0
    for z in zeros_in:
        a_in = a_in + np.arctan(wc / z)


def prob2():
    f = 10
    Ts = 1 / f
    bandwidth = 6
    phase_margin = 45
    P = ct.tf([0.01914, -0.5815, 3.973], [1, 3.987, 3.973])  # from MATLAB (python has no d2c() command yet)

    Cct = pid_ct(P, bandwidth, phase_margin)  # script for designing PID based on in-class techniques
    Lct = P * Cct
    Tct = Lct / (1 + Lct)
    Sct = 1 - Tct

    Pdt = ct.c2d(P, Ts, 'zoh')
    Cdt = ct.c2d(Cct, Ts, 'tustin')
    print(f'C(z)={Cdt}')

    Tdt = Pdt * Cdt / (1 + Pdt * Cdt)
    Sdt = 1 - Tdt

    plt.figure()
    ct.bode(Pdt * Cdt, margins=True)
    plt.suptitle(r'$L(e^{j\omega})$ frequency response')

    plt.figure()
    ct.bode([Tdt, Sdt])
    plt.suptitle(r'$T(e^{j\omega})$ & $S(e^{j\omega})$ frequency responses')

    plt.figure()
    step(Tdt)
    plt.title('$T(s)$ step response')

    # part 2

    D = ct.tf([1], [1, 1])
    mp = ct.dcgain(P)
    md = ct.dcgain(D)
    H = 1 / (mp * md)
    print(f'H={H}')

    Tff = Sct * P * Cct + Sct * D - Sct * P * H

    plt.figure()
    ct.bode(Tff)

    plt.figure()
    step(Tff)
    plt.title('Feed Forward Closed Loop step response')

    Tff = ct.c2d(Tff, Ts, 'tustin')
    print(f'T={Tff}')

    # plt.figure()
    # ct.bode([Tff, Tdt])
    #
    # plt.figure()
    # step(Tff)

def obs_ex():
    p = ct.tf([1, 2],[1, 3]) * ct.tf([1], [1, 4]) # want closed-loop poles of observer at -2+2i, -2-2i
    pss = ct.tf2ss(p)
    A = pss.A[0][0]
    B = pss.B[0][0]
    C = pss.C[0][0]

    K = np.array([0, 1]).reshape(1, 2) @ la.inv(np.hstack(B, A @ B)) @ (la.matrix_power(A, 2) + 4 * A + 8)
    L = np.array([0, 1]).reshape(1, 2) @ la.inv(np.hstack(C.T, A.T @ C.T)) @ (la.matrix_power(A.T, 2) + 4 * A.T + 8)

    print(L.T)
    return(L.T)

