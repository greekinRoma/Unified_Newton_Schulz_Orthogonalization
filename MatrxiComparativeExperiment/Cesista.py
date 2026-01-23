"""
Tool for optimizing the coefficients of the Newton-Schulz iterators in Muon.

Usage notes:
- Use a high `epsilon` value to prevent the singular values from either blowing up or switching signs.
- Set --enable_flatness_aux_loss to get flatter composite curves.
"""

import argparse
from functools import partial

import sympy as sp

DEFAULT_EPS = 1. / 16
DEFAULT_PRECISION = 4

class CesistaNS:
    def __init__(self, iter_num=5):
        self.iter_num = iter_num
        self.weights = [
            (4.0249, -6.4216, 2.6016),
            (3.9924, -6.2874, 2.5399),
            (3.3319, -4.8337, 1.9453),
            (2.8797, -3.6304, 1.6253),
            (2.9954, -3.6247, 1.6126),
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
