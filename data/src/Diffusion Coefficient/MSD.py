import MDAnalysis
import MDAnalysis.coordinates.DCD as DCD
import sys
import math
import os

from math import sqrt
from math import fabs
from math import floor
from copy import deepcopy
import matplotlib.pyplot as plt


def msd(timestep, dotnum, mult, tau_t, dcd_filename):
    # compute the path for particle
    num_dim = 3
    dcd = DCD.DCDReader(dcd_filename)  # get the DCD data

    # box dimensions (set once and assume same for all time steps)
    domainL = [None] * num_dim
    for d in range(0, num_dim):
        domainL[d] = dcd[0].dimensions[d]
    X = domainL[0] / 2
    Y = domainL[1] / 2
    Z = domainL[2] / 2
    print(X, Y, Z)  # 40  40  40

    numframes = dcd.n_frames
    numatoms = dcd.n_atoms
    startframe = int(numframes / 5)
    freq = dcd.skip_timestep  # freq
    framenum = numframes - startframe
    print('原子数量是:', numatoms)
    print('timestep:', timestep)
    print('freq', freq)
    print('从dcd中第几帧开始取:', startframe)

    # read in dcd data to list Xdcd
    Xdcd = num_dim * [[[]]]
    for d in range(num_dim):
        Xdcd[d] = framenum * [[]]
    i = 0
    for timeIndex in range(startframe, numframes):
        ts0 = dcd[timeIndex]
        Xdcd[0][i] = deepcopy(ts0._x)
        Xdcd[1][i] = deepcopy(ts0._y)
        Xdcd[2][i] = deepcopy(ts0._z)
        i = i + 1

    msd = [[0 for ii in range(30000)] for jj in range(3)]  # [[0, 0.... 0], [0, 0.... 0], [0, 0.... 0], [0, 0.... 0]]
    MSD = [0] * 30000
    time1 = [0]
    for dot in range(1, dotnum):  # 画几个点

        delt_t = dot * mult  # 每个点对应dcd中的20帧
        time1.append(delt_t * timestep * freq / tau_t)
        for index in range(numatoms):
            Fd = framenum - delt_t
            for n in range(0, Fd):
                for d in range(3):  # 3维
                    if (Xdcd[d][n + delt_t][index] - Xdcd[d][n][index]) >= X:
                        msd[d][delt_t] += (Xdcd[d][n + delt_t][index] - Xdcd[d][n][index] - 2 * X) ** 2
                    if (Xdcd[d][n + delt_t][index] - Xdcd[d][n][index]) <= (-1 * X):
                        msd[d][delt_t] += (Xdcd[d][n + delt_t][index] - Xdcd[d][n][index] + 2 * X) ** 2
                    if (Xdcd[d][n + delt_t][index] - Xdcd[d][n][index]) > (-1 * X) and (Xdcd[d][n + delt_t][index] - Xdcd[d][n][index]) < X:
                        msd[d][delt_t] += (Xdcd[d][n + delt_t][index] - Xdcd[d][n][index]) ** 2

            # MSD[dot] = (msd[0][delt_t] + msd[1][delt_t] + msd[2][delt_t]) / (Fd*numatoms)
            MSD[dot] = (msd[0][delt_t] + msd[1][delt_t] + msd[2][delt_t]) / (Fd * numatoms)

    print('time1:', time1)
    # print(MSD[:dot])
    print(MSD[:dotnum])
    plt.plot(time1, MSD[:dotnum], 'o', color='r', label=' 1 molecules of length 1')
    plt.xlabel("t/ps", fontsize=12)
    plt.ylabel("MSD/nm^2")
    plt.title("SELM All Beads Mean Square Displacement")
    plt.legend()  # 显示上面的label

    # 从文件名中提取基本名称（不包含路径和扩展名）
    base_filename = os.path.splitext(os.path.basename(dcd_filename))[0]
    plt.savefig(f'{base_filename}.jpg')
    plt.show()

    # 输出数据到文本文件，文件名与dcd文件名一致
    with open(f'{base_filename}.txt', "w") as file:
        com = zip(time1, MSD[:dotnum])
        for data in com:
            file.write(f"{data[0]} {data[1]}\n")

    print('done!')


# 调用函数时传入DCD文件的路径
msd(0.005, 41, 1, 1, '1base.dcd')
