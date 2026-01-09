import matplotlib.pyplot as plt
import math
import numpy as np
from scipy import optimize

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def base_MSD(list1, list2, list3, list4, list5, list6, list7, list8, xl, yl, figname):

    list11, lista = [], []


    # A
    for i in list1:
        list11.append(i / 130.032)
    for i in list2:
        lista.append(i / 0.08847568569774664)
    plt.scatter(list11, lista, color='r', marker='o', label='A')

    # 绘制散点 C
    list31, listb = [], []

    for i in list3:
        list31.append(i / 130.032)
    for i in list4:
        listb.append(i / 0.08686479867795335)
    plt.scatter(list31, listb, color='b', marker='s', label='C')

    # 绘制散点 T
    list51, listf = [], []

    for i in list5:
        list51.append(i / 130.032)
    for i in list6:
        listf.append(i / 0.08652728084774365)
    plt.scatter(list51, listf, color='g', marker='^', label='T')

    # 绘制散点 G
    list71, listg = [], []

    for i in list7:
        list71.append(i / 130.032)
    for i in list8:
        listg.append(i / 0.0861359824583352)
    plt.scatter(list71, listg, color='m', marker='D', label='G')

    # 直线方程函数
    def f_1(x, A, B):
        return A * x + B

    # 直线拟合与绘制
    A1, B1 = optimize.curve_fit(f_1, list11, lista)[0]
    x1 = np.arange(0, 2, 0.2)
    y1 = A1 * x1 + B1
    plt.plot(x1, y1, "red")

    A2, B2 = optimize.curve_fit(f_1, list31, listb)[0]
    x2 = np.arange(0, 2, 0.2)
    y2 = A2 * x2 + B2
    plt.plot(x2, y2, "b")

    A3, B3 = optimize.curve_fit(f_1, list51, listf)[0]
    x3 = np.arange(0, 2, 0.2)
    y3 = A3 * x3 + B3
    plt.plot(x3, y3, "g")

    A4, B4 = optimize.curve_fit(f_1, list71, listg)[0]
    x4 = np.arange(0, 2, 0.2)
    y4 = A4 * x4 + B4
    plt.plot(x4, y4, "m")

    # 图像细节设置
    plt.ylabel(yl, fontdict={'family': 'Times New Roman', 'size': 12})
    plt.xlabel(xl, fontdict={'family': 'Times New Roman', 'size': 12})
    plt.xticks(fontproperties='Times New Roman', size=12)
    plt.yticks(fontproperties='Times New Roman', size=12)
    plt.legend(prop={'family': 'Times New Roman', 'size': 12})
    # plt.savefig(figname)

    # ------------ save as eps-------------
    plt.savefig(f"{figname}.eps", dpi=600, bbox_inches='tight')

    # ------------ save as pdf -------------
    # plt.savefig(f"{figname}.eps", dpi=300, bbox_inches='tight')

    plt.show()


# A
list1 = [1, 40, 60, 80, 96, 112, 140, 168, 224]
list2=[0.08847568569774664,0.05494720063026705, 0.0511226900897256, 0.04562096837092343,
       0.04253707051919581,0.040031220225552754,0.03590837783744626,0.03249833596249332, 0.018874211802640374]


# T
list5 = [1, 40, 60, 80, 96, 112, 140, 168, 224]
list6 = [0.08652728084774365, 0.05914547447073333, 0.05477737189205385, 0.04939901971381085,
          0.045461776003048824, 0.043863591803704237, 0.04062965329550983, 0.030507908790396084, 0.019200721424543076]

# C
list3 = [1, 40, 60, 80, 96, 112, 140, 168, 224]
list4 = [0.08686479867795335, 0.0817843289359967, 0.0762831457640899, 0.0713697102827398, 0.06665345552236345,
         0.06350744584072675, 0.05469950341169441, 0.046873103232915716, 0.030620130070280985]


# G
list7 = [1, 40, 60, 80, 96, 112, 140, 168, 224]
list8 = [0.0861359824583352, 0.07827867207986578, 0.07392786987565137, 0.06778790920590744, 0.06311090597510305,
         0.05889580843120432, 0.04956355770840535, 0.04002859929111666, 0.02589751020535041]

base_MSD(list1, list2, list3, list4, list5, list6, list7, list8, 'C (mol/L)', 'D$_s$(C)/D$_0$', 'base_MSD.png')
