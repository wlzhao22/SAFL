"""
Cubic Spline library on python
author Atsushi Sakai
"""

import torch

class CubicSpline(torch.nn.Module):

    def __init__(self, x, y):
        super(CubicSpline, self).__init__()

        self.x = x
        self.y = y
        self.a = y
        self.nx = x.shape[0] # dimension of x
        h = x[1:] - x[:-1]

        A = self.__calc_A(h)
        B = self.__calc_B(h)

        self.c = torch.matmul(torch.inverse(A), B)
        self.d = (self.c[1:self.nx] - self.c[:self.nx-1])/(3 * h[:self.nx-1])
        self.b = (self.a[1:self.nx] - self.a[:self.nx-1])/h[:self.nx-1] - h[:self.nx-1]*(self.c[1:self.nx] + 2*self.c[:self.nx-1])/3

    def __calc_A(self, h):
        A = torch.zeros((self.nx * self.nx)).cuda()
        index_list = torch.arange(0, self.nx-1, 1)
        A[index_list[1:self.nx-1] + (index_list[1:self.nx-1])*self.nx] = 2 * (h[:self.nx-2] + h[1:self.nx-1])
        A[index_list + (index_list+1) * self.nx] = h[:self.nx-1]
        A[index_list + 1 + index_list* self.nx] = h[:self.nx-1]

        A[0], A[1], A[(self.nx-1)*self.nx + self.nx-2], A[(self.nx-1)*self.nx + self.nx-1] = 1, 0, 0, 1
        A = A.view(self.nx, self.nx)
        return A

    def __calc_B(self, h):
        B = torch.zeros(self.nx).cuda()
        B[1:self.nx-1] = 3 * (self.a[2:self.nx] - self.a[1:self.nx-1])/h[1:self.nx-1] - 3 * (self.a[1:self.nx-1] - self.a[:self.nx-2])/h[:self.nx-2]
        return B

    def forward(self, t):
        # i = ((t - self.x[0])/(self.x[1] - self.x[0])).long()
        data = self.x[1] - self.x[0]
        print(f".....{data}......")
        i = ((t - self.x[0])/12).long()
        dx = t - self.x[i]
        result = self.a[i] + self.b[i] * dx + self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0
        return result
