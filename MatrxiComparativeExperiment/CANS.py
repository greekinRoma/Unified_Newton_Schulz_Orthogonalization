from turtle import forward
import numpy as np
import torch

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


def cans_iteration(A, n=4, a_index=0.01, degree=3, preprocess=False, preprocess_iters=1, delta=0.99):
    # CANS iteration for orthogonalization
    # a is the left boundary of the segment
    # n is the maximum number of iterations
    b_index = 1  # assume that matrix is normalized
    err = []
    matmuls = [0]
    I = np.eye(A.shape[0])
    err.append(np.linalg.norm(A @ A.T - I))
    e = 10
    if preprocess:
        lst, _ = delta_orthogonalization(preprocess_iters, degree, delta)
        for i in range(preprocess_iters):
            C = A
            AtA = A.T @ A
            B = lst[i][1] * A
            for j in range(3, degree + 1, 2):
                C = C @ AtA
                B += lst[i][j] * C
            A = B
            matmuls.append(matmuls[-1] + (degree + 1) // 2)
            err.append(np.linalg.norm(A.T @ A - I))
        a_index, b_index = 1 - delta, 1 + delta
    cnt = 0
    while cnt < n and (len(err) == 0 or err[-1] > 1e-13):
        if degree == 3:
            p, e = explicit3(a_index, b_index)
            a_index, b_index = 1 - e, 1 + e
            A = p[1] * A + p[3] * A @ A.T @ A
        else:
            p, e = remez(a_index, b_index, degree)
            C = A
            AtA = A.T @ A
            B = p[1] * A
            for i in range(3, degree + 1, 2):
                C = C @ AtA
                B += p[i] * C
            a_index, b_index = 1 - e, 1 + e
            A = B
        matmuls.append(matmuls[-1] + (degree + 1) // 2)
        err.append(np.linalg.norm(A @ A.T - I))
        cnt += 1
    return A, err, matmuls
class CaNS:
    def __init__(self, iter_num=5):
        self.iter_num = iter_num
        self.weights = [
            (8.462385123610193, -25.076703591296525,18.604318451139527 ),
            (4.169671637309271, -3.0995520439584285, 0.57956324300095),
            (3.913676673069997, -2.920037812686962, 0.5590987905684336),
            (3.1710465180967007, -2.3777373620051665, 0.49752193655456894),
            (2.187030637313693, -1.5649500065194997, 0.4076449910177318),
        ]
    def newtonschulz5(self, X, iteration):
        """
        X: (N, N) torch.Tensor
        """
        a, b, c = self.weights[iteration]
        X_2 = X @ X.T
        A = X
        B = X_2 @ X
        C = X_2 @ B
        return a * A + b *B + c * C
    def forward(self, X):
        for i in range(self.iter_num):
            X = self.newtonschulz5(X, i)
        return X