import numpy as np

# Set precision (PyTorch uses 32-bit or 64-bit floats by default)
np.set_printoptions(precision=20)


def get_polynomial(Ext):
    n = len(Ext) - 1
    M = np.zeros((n + 1, n + 1), dtype=np.float64)
    for i in range(n + 1):
        for j in range(n):
            M[i, j] = Ext[i] ** (2 * j + 1)
        M[i, n] = (-1) ** (i + 1)
    c = np.linalg.solve(M, np.ones(n + 1, dtype=np.float64))
    return c


def get_ext(c):
    n = len(c)
    coeffs = [(2 * j + 1) * c[j] for j in range(n)]
    rts = np.roots(coeffs[::-1])
    return np.sqrt(rts)


def remez_step(Ext):
    n = len(Ext) - 1
    c = get_polynomial(Ext)
    coeffs = [c[j // 2 - 1] if j % 2 == 0 else 0 for j in range(1, 2 * n + 1)]
    p = np.poly1d(coeffs[::-1])
    e = c[n].item()
    NewExt = np.concatenate(([Ext[0]], get_ext(c[:n]), [Ext[-1]]))
    f = lambda x: np.abs(p(x) - 1)
    newe = np.max(f(NewExt))
    return p, NewExt, e, newe


def remez(A, B, degree):
    # Remez algorithm for finding optimal polynomial approximating the unity function f==1
    # on the segment [A, B]
    n = (degree + 1) // 2
    Ext = np.linspace(B, A, n + 1, dtype=np.float64)
    p = np.poly1d([0])
    newe = 0
    for i in range(100):
        try:
            p, NewExt, e, newe = remez_step(Ext)
        except np.linalg.LinAlgError:
            return kovarik_formula(degree), 0  # if the segment converged to [A, B]=[1, 1]
        if newe < abs(e) + 1e-20:
            return p, newe
        Ext = np.array(NewExt, dtype=np.float64)
    return p, newe


def c_n_k(n, k):
    s = 1
    for i in range(n - k + 1, n + 1):
        s *= i
    for i in range(1, k + 1):
        s /= i
    return s


def kovarik_formula(degree):
    # generates polynomial of specified degree from the paper
    # Zdislav Kovarik, "SOME ITERATIVE METHODS FOR IMPROVING ORTHONORMALITY", 1970
    p = np.zeros(degree + 1)
    p[1] += 1
    a = 1
    for i in range(1, (degree + 1) // 2):
        for j in range(2 * (i - 1) + 1, 2 * i + 1):
            a *= j / 2
        a /= i ** 2
        sign = 1
        for k in range(0, i + 1):
            p[2 * k + 1] += a * sign * c_n_k(i, k)
            sign *= -1
    return np.poly1d(p[::-1])


def explicit3(A, B):
    # explicit formula for optimal 3-rd order polynomial on the segment [A, B]
    e = np.sqrt((A ** 2 + A * B + B ** 2) / 3)
    a = 2 / (2 * e ** 3 + A ** 2 * B + B ** 2 * A)
    p = np.poly1d([-a, 0, a * (A ** 2 + A * B + B ** 2), 0])
    err = (2 * e ** 3 - A ** 2 * B - B ** 2 * A) / (2 * e ** 3 + A ** 2 * B + B ** 2 * A)
    return p, err


def find_left_bd(delta, B, degree):
    # find one optimal polynomial with high derivative at zero on the interval [0, B], which falls into [1-delta, 1+delta]
    # B is the right boundary of the interval
    # delta is the desired accuracy of approximation
    Al = 0.0
    Ar = B
    A = (Al + Ar) / 2
    p, f = remez(A, B, degree)
    while abs(delta - f) > 1e-15:
        if f < delta:
            Ar = (Ar + Al) / 2
        else:
            Al = (Al + Ar) / 2
        p, f = remez((Al + Ar) / 2, B, degree)
    return p, f, (Al + Ar) / 2


def delta_orthogonalization(n=1, degree=3, delta=0.3, B=1):
    # find composition of n polynomials of specified degree on the interval [0, B], which falls into [1-delta, 1+delta]
    # the derivative of composition at zero is maximized
    Al = 0.0
    Ar = B
    e = 100
    while abs(e - delta) > 1e-7:
        a, b = (Al + Ar) / 2, B
        lst = []
        for i in range(n):
            if degree == 3:
                Q, e = explicit3(a, b)
            else:
                Q, e = remez(a, b, degree)
            lst.append(Q)
            a, b = 1 - e, 1 + e
        if e < delta:
            Ar = (Ar + Al) / 2
        else:
            Al = (Al + Ar) / 2
    return lst, (Al + Ar) / 2

def curve_iteration(a, n, a_index=0.01, degree=5, preprocess=True, preprocess_iters=1, delta=0.99):
    b_index = 1
    e = 10
    if preprocess:
        lst, _ = delta_orthogonalization(preprocess_iters, degree, delta)
        print(lst)
        for i in range(preprocess_iters):
            c = a
            a_2 = c * c
            b = lst[i][1] * a
            for j in range(3, degree + 1, 2):
                c = c * a_2
                b += lst[i][j] * c
            a = b
        a_index, b_index = 1 - delta, 1 + delta
    cnt = 0
    while cnt < n:
        if degree == 3:
            p, e = explicit3(a_index, b_index)
            a_index, b_index = 1 - e, 1 + e
            a = p[1] * a + p[3] * a * a * a
        else:
            p, e = remez(a_index, b_index, degree)
            c = a
            a_2 = a * a
            b = p[1] * a
            for i in range(3, degree + 1, 2):
                c = c * a_2
                b += p[i] * c
            a_index, b_index = 1 - e, 1 + e
            a = b
        cnt += 1
    return a

if __name__ == "__main__":
    x = np.sort(np.random.rand(100000))
    model = curve_iteration
    y = model(x,n=4,preprocess_iters=1,degree=5,preprocess=True)
    import matplotlib.pyplot as plt
    plt.plot(x, y)
    plt.show()