import matplotlib.pyplot as plt
import math
import numpy as np
from scipy import optimize as op
import sys
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False



def get_K(file,name,jpg):
    X, Y = [], []
    for line in open(file,'r'):
        values = [float(s) for s in line.split()]
        X.append(values[0])
        Y.append(values[1])
    plt.scatter(X, Y, marker='o',label='真实值')



    '''拟合'''
    # 需要拟合的函数
    def f_1(x, A,b):
        return A * x+b
    A1,b1= op.curve_fit(f_1, X, Y)[0]

    print(A1)
    print(b1)


    H=[]
    Z=[]
    for x in range(500, 800):
        Z.append(x)
        y = A1 * x+b1
        H.append(y)
    plt.plot(Z, H,color='red',label='拟合曲线')
    plt.legend() # 显示label
    # plt.xlabel("Angles (deg)")
    # plt.ylabel("Potntial")

    plt.title(name)
    # plt.xlim(50,185,20)
    # plt.ylim(0,30,5)
    plt.savefig(jpg)
    plt.show()


file='1base.txt'
name='1base'
jpg='1base.jpg'
get_K(file,name,jpg )
'''
0.0008847568569774664
0.02847844759547282
'''