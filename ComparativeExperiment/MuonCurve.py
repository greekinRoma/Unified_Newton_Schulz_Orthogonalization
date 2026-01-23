from turtle import forward
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
class MuonCurve:
    def __init__(self):
        self.iter_num = 5
    # Pytorch code
    def newtonschulz5(self, g):
        a, b, c = (3.4445, -4.7750, 2.0315)
        return a*g + b*g**3 + c*g**5
    def forward(self,x):
        for _ in range(self.iter_num):
            x = self.newtonschulz5(x)
        return x
if __name__ == "__main__":
    model = MuonCurve()
    import torch
    x = torch.linspace(0, 1, 1000)
    y = model.forward(x)
    print(y.shape)
    plt.plot(x, y)
    plt.show()