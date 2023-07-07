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

    # pvel = ctm.zpk([-15.6, 3.1], [-49, -22, 3.3, -0.47], 75)
    # pang = ctm.zpk([0], [-49, -22, 3.3, -0.47], 375)
    # pvol = ct.tf([1], [1])
    #
    # pvel_ss = ct.tf2ss(pvel)
    # pang_ss = ct.tf2ss(pang)
    # pvol_ss = ct.tf2ss(pvol)
    #
    # pvel_dt = ct.c2d(pvel, Ts, 'zoh')
    # pang_dt = ct.c2d(pang, Ts, 'zoh')
    # pvol_dt = ct.c2d(pvol, Ts, 'zoh')

    E = np.array([[L, 0, 0, 0], [0, 1, 0, 0], [0, 0, m*l**2, -m*l], [0, 0, -m*l, m+M]]) # from hand calculations
    A = np.array([[-R, 0, 0, -k], [0, 0, 1, 0], [0, m*g*l, -c_w, 0], [k/r, 0, 0, -c_v]])
    B = np.array([1, 0, 0, 0]).reshape(4, 1)
    C = np.array([[0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 0, 0]])
    D = np.array([0, 0, 1]).reshape(3, 1)

    A = la.inv(E) @ A
    B = la.inv(E) @ B

    P = ct.ss(A, B, C, D)

    vars = vars()
    plist = []                                      # 0: velocity, 1: angle, 2: voltage
    dlist = []
    for i in range(3):
        dlist.append(f'P{i + 1}')
        plist.append(ct.ss(A, B, C[i], D[i]))
        vars[dlist[i]] = plist[i]

    fs = 100
    Ts = 1 / fs
    pm = 50

    # w-plane method from MATLAB
    pvel_w = ct.tf([-0.00016966, -0.3351, 69.24, 931.1, -3542], [1, 67.12, 853.2, -3087, -1633])
    pang_w = ct.tf([1.505e-05, -0.004047, 1.623, 366.2, -5.691e-11], [1, 67.12, 853.2, -3087, -1633])
    pvol_w = pvol

    c_i = pid_ct(pang_w, 40, 31.4942, plot=True) # literally if bw is 0.0001 higher it becomes unstable
    c_i = ct.c2d(c_i, Ts, 'tustin')




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
    p = ct.tf([1, 2],np.convolve([1, 3], [1, 4])) # want closed-loop poles of observer at -2+2i, -2-2i
    pss = ct.tf2ss(p)
    A = pss.A
    B = pss.B
    C = pss.C
    #Ackerman's formula
    L = np.array([0, 1]).reshape(1, 2) @ la.inv(np.hstack((C.T, A.T @ C.T))) @ (la.matrix_power(A.T, 2) + 4 * A.T + 8 * np.eye(2))
    L = L.T

    return(L, la.eigvals(A - L@C))

