import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def linear_fit(x, y, xmin, xmax):
    x = np.array(x)
    y = np.array(y)

    mask = (x >= xmin) & (x <= xmax)
    x_fit = x[mask]
    y_fit = y[mask]

    k, b = np.polyfit(x_fit, y_fit, 1)

    return x_fit, y_fit, k, b

def get_property(file1, file2, title, yl, xl, figname,
                 fit_range1=None, fit_range2=None):

    plt.figure()

    T1, Z1 = [], []
    for line in open(file1, 'r'):
        values = [float(s) for s in line.split()]
        T1.append(values[0])
        Z1.append(values[1] * 0.01)

    plt.scatter(T1, Z1, color='b', marker='s',
                label='C=0.008 mol/L', linewidth=1.0)

    if fit_range1 is not None:
        xmin, xmax = fit_range1
        x_fit, y_fit, k1, b1 = linear_fit(T1, Z1, xmin, xmax)

        plt.plot(x_fit, k1 * x_fit + b1, 'b--',
                 linewidth=3,
                 label=f'Fit1 slope={k1:.4e}')

        print(f"Data1 slope = {k1:.6e}")

    X1, Y1 = [], []
    for line in open(file2, 'r'):
        values = [float(s) for s in line.split()]
        X1.append(values[0])
        Y1.append(values[1] * 0.01)

    plt.scatter(X1, Y1, color='r', marker='.',
                label='C=1.293 mol/L', linewidth=1.0)

    if fit_range2 is not None:
        xmin, xmax = fit_range2
        x_fit, y_fit, k2, b2 = linear_fit(X1, Y1, xmin, xmax)

        plt.plot(x_fit, k2 * x_fit + b2, 'r--',
                 linewidth=3,
                 label=f'Fit2 slope={k2:.4e}')

        print(f"Data2 slope = {k2:.6e}")

    plt.title(title, fontdict={'family': 'Times New Roman', 'size': 12})
    plt.ylabel(yl, fontdict={'family': 'Times New Roman', 'size': 12})
    plt.xlabel(xl, fontdict={'family': 'Times New Roman', 'size': 12})

    plt.yticks(fontproperties='Times New Roman', size=12)
    plt.xticks(fontproperties='Times New Roman', size=12)
    plt.legend(prop={'family': 'Times New Roman', 'size': 12})

    plt.savefig('compare_MSD.pdf', dpi=300)
    plt.show()

get_property(
    '1base.txt',
    '168base.txt',
    'Mean Square Displacement',
    'MSD (nm^2)',
    'Time(ps)',
    'compare_SELM.eps',
    fit_range1=(2000, 4000),
    fit_range2=(2000, 4000)
)
