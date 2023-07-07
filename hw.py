from ctrls import *

plt.style.use('seaborn-v0_8')


def prob1():
    c_v = c_w = L = r = 0.1
    m = l = 0.2
    k = 0.3
    M = 0.4
    R = 5
    g = 9.81

    E = np.array([[L, 0, 0, 0], [0, 1, 0, 0], [0, 0, m*l**2, -m*l], [0, 0, -m*l, m+M]])        # from hand calculations
    A = np.array([[-R, 0, 0, -k], [0, 0, 1, 0], [0, m*g*l, -c_w, 0], [k/r, 0, 0, -c_v]])
    B = np.array([1, 0, 0, 0]).reshape(4, 1)
    C = np.array([[0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 0, 0]])
    D = np.array([0, 0, 1]).reshape(3, 1)

    A = la.inv(E) @ A
    B = la.inv(E) @ B

    P = ct.ss(A, B, C, D)

    vars = globals()
    plist = []
    deflist_ct = []
    deflist_dt = []
    deflist_w = []                                # 0: velocity, 1: angle, 2: voltage
    fs = 100
    Ts = 1 / fs

    for i in range(3):
        deflist_ct.append(f'P{i + 1}')
        # deflist_dt.append(f'P{i + 1}_dt')
        # deflist_w.append(f'P{i + 1}_w')
        sys = ct.ss(A, B, C[i], D[i])
        plist.append(ct.tf(sys))
        vars[deflist_ct[i]] = plist[i]

    P1_dt = ct.c2d(P1, Ts, 'zoh')
    P1_w = d2c_tustin(P1_dt, Ts)
    P2_dt = ct.c2d(P2, Ts, 'zoh')
    P2_w = d2c_tustin(P2_dt, Ts)

    pm = 50

    return(P1, P2, P3, P1_dt, P2_dt, P1_w, P2_w)

def prob1_2(plant: ct.TransferFunction):
    a = np.linspace(1, 49, 49)

    index = []
    for i in range(len(a)):
        c = pid_ct(plant, 45, a[i], 50*3/2)
        t, y = ct.step_response(plant*c / (1 + plant * c))
        index.append(np.max(y))

    a = np.argmin(index)
    return(a)




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
