import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
import control as ct
import control.matlab as ctm

plt.style.use('seaborn-v0_8')

def pid(BW):

    PM = 60                                                                          # design parameters
    tau = 0.00
    wc = (2/3) * BW

    alpha = np.arctan(0.01*wc) + np.arctan(0.1*wc) - np.arctan(10*wc)           # solving for z
    z = wc / np.tan(np.radians(PM) + np.pi/2 + alpha)

    numK = wc * np.sqrt(((tau*wc)**2 + 1)*(wc**2 + (-0.1)**2)*((0.01*wc)**2 + 1))   # solving for controller gain k
    denK = np.sqrt((wc**2 + z**2)*((-0.01*wc)**2 + 1))
    k = numK/denK

    numL = [0, -0.01*k, k*(1-0.02*z), 2*k*z-0.01*z**2, z**2]                        # specific to this problems plant
    denL = [0.01*tau, 0.999*tau + 0.01, 0.999 - 0.1*tau, -0.1, 0]

    L = ct.TransferFunction(numL, denL)                                            # algebra to get transfer functions
    T = L / (1 + L)
    S = 1 - T

    gm, pm, wcg, wcp = ct.margin(T)                                                # get margins

    plt.figure()
    ct.nyquist(T)
    plt.title(r"Nyquist plot of $T(s)$; $BW=$"+str(BW))

    plt.figure()
    ct.bode(T)
    plt.suptitle(r"Bode Plot of $T(s)$; $BW=$"+str(BW))

    plt.figure()
    ct.bode(S)
    plt.suptitle(r"Bode Plot of $S(s)$; $BW=$"+str(BW))

    plt.figure()
    ct.step_response(T)
    plt.title(r"Step response of $T(s)$; $BW=$"+str(BW))

    plt.figure()
    ct.step_response(S)
    plt.title(r"Step response of $S(s)$; $BW=$"+str(BW))


    print(
    "PM + 90 deg + alpha = ", (PM + 90 + alpha), '\n'
    '\n'
    "Design specs: \n"
        "Bandwidth:", BW,"\n"       
        "Phase margin: 60 deg\n"
        'Tau: 0.01\n'
        'Omega c = 0.667 * BW\n'
    '\n'
    "results: \n"
        "k = ", k,'\n'
        'z = ', z,'\n'
    '\n'
    'Closed-loop params: \n'
        'Gain margin: ', gm, '\n'
        'Gain Crossover freq:', wcg,'\n'
        'Phase margin:', pm,'\n'
        'Phase crossover freq:', wcp
    )

def prob1():
    A = np.array([[0.2, 0.2],[0.2, 0.2]])
    B = np.array([[0.], [1.]]).reshape(2, 1)
    C = np.array([0., 1.]).reshape(1, 2)
    D = np.array([0.])

    x0 = np.array([1, 0]).reshape(2, 1)

    x = []
    y = []

    for k in range(1, 6):
        S = np.array([0.0, 0.0]).reshape(2, 1)
        for j in range(1, k):
            S = S + np.linalg.matrix_power(A, k - j) @ B
        xk = np.linalg.matrix_power(A, k) @ x0 + B + S
        x.append(xk)

    for k in range(1, 6):
        S = np.array([0.0, 0.0]).reshape(2, 1)
        for j in range(1, k):
            S = S + C @ np.linalg.matrix_power(A, k - j) @ B
        yk = C @ (np.linalg.matrix_power(A, k) @ x0 + B + S) + D
        y.append(yk)

    print(x, '\n', y)

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
    except Exception:
        num = [1]
        num = list(map(float, num))
    finally:
        den = sym.Poly(sym_den).all_coeffs()
        den = list(map(float, den))

    Hz = ct.tf(num, den, dt=T)
    Hz = ct.ss(Hz)
    return(Hz)

def prob5(H, T):
    Htust = ctm.c2d(H, T, 'tustin')
    Hzoh = ctm.c2d(H, T, 'zoh')
    Hfe = ct.tf(c2d_feuler(H, T))

    plt.figure()
    ct.bode([H, Hfe, Htust, Hzoh])
    plt.figlegend(['CT', 'Forward Euler', 'Tustin', 'ZOH'])
    plt.legend(['CT', 'For Eu', 'Tust', 'ZOH'])
    plt.suptitle('T=' + str(T))

def lab3(BW, tau, PM):
    P = ct.tf([-2.776e-17, 0.165555],[1, -0.1995, 0])
    wc = 2 * BW / 3
    beta = np.radians(PM) + np.arctan(tau*wc) + np.arctan(wc / 0.1995) - np.arctan(2.77e-17 / 0.1655)

    a = wc / (np.tan(beta / 2))

    num = (wc**2) * np.sqrt(((tau*wc)**2 + 1) * (wc**2 + 0.1995**2))
    den = (wc**2 + a**2) * np.sqrt(-2.77e-17**2 + 0.1655**2)

    K = 2 * num / den
    print('K=', K, '\n a=', a, '\n')

    C = K * ct.tf([1, a],[1, 0]) * ct.tf([1, a],[tau, 1])
    print(C)

    Ki = K*(a**2)
    Kp = K * (2*a - tau*a**2)
    Kd = K - 2*a*K*tau - K*(a*tau)**2

    return(Ki, Kp, Kd)

def hwprob2():
    P1s = ct.tf([-1, 5], [1, 1]) * ct.tf([1], [1, 1]) * ct.tf([1], [1, 1])
    P2s = ct.tf([5], [1, 1, 1])

    ### PM TUNING METHODS ###

    BW = 10
    wc = 2 * BW / 3
    tau = 0.01

    beta = np.radians(-220) + np.arctan(tau * wc) + np.arctan(wc / 5) + 3 * np.arctan(wc)
    a1 = wc / np.tan(beta/2)

    alpha = np.radians(500) + np.arctan(tau * wc) - np.arctan(wc / (1+1.7320508075688772j)) - np.arctan(wc / (1-1.7320508075688772j))
    a2

