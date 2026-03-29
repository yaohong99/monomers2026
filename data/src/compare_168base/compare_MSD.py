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

def read_data(file):
    X, Y = [], []
    for line in open(file, 'r'):
        values = [float(s) for s in line.split()]
        X.append(values[0])
        Y.append(values[1] * 0.01)
    return X, Y

def get_property(file1, file2, file3, file4,
                 title, yl, xl, figname,
                 fit_ranges=None):

    plt.figure()

    T1, Z1 = read_data(file1)
    T2, Z2 = read_data(file2)
    X1, Y1 = read_data(file3)
    X2, Y2 = read_data(file4)

    plt.scatter(T1, Z1, color='b', marker='o', label='period-x-direction')
    plt.scatter(T2, Z2, color='magenta', marker='s', label='period-z-direction')
    plt.scatter(X1, Y1, color='r', marker='^', label='channel-x-direction')
    plt.scatter(X2, Y2, color='orange', marker='v', label='channel-z-direction')

    if fit_ranges is not None:
        datasets = [
            (T1, Z1, 'b', 'period-x'),
            (T2, Z2, 'magenta', 'period-z'),
            (X1, Y1, 'r', 'channel-x'),
            (X2, Y2, 'orange', 'channel-z')
        ]

        for i, (x, y, color, name) in enumerate(datasets):
            if fit_ranges[i] is not None:
                xmin, xmax = fit_ranges[i]

                x_fit, y_fit, k, b = linear_fit(x, y, xmin, xmax)

                plt.plot(x_fit, k * x_fit + b,
                         linestyle='--',
                         color=color,
                         linewidth=2.5,
                         label=f'{name} fit slope={k:.2e}')

                print(f"{name} slope = {k:.6e}")

    plt.title(title, fontdict={'family': 'Times New Roman', 'size': 12})
    plt.ylabel(yl, fontdict={'family': 'Times New Roman', 'size': 12})
    plt.xlabel(xl, fontdict={'family': 'Times New Roman', 'size': 12})

    plt.yticks(fontproperties='Times New Roman', size=12)
    plt.xticks(fontproperties='Times New Roman', size=12)
    plt.legend(prop={'family': 'Times New Roman', 'size': 10})

    plt.savefig(figname, dpi=300, bbox_inches='tight')
    plt.show()

get_property(
    'period_168base_X.txt',
    'period_168base_Z.txt',
    'dirichlet_168base_X.txt',
    'dirichlet_168base_Z.txt',
    'Mean Square Displacement',
    'MSD (nm^2)',
    'Time(ps)',
    'compare_SELM.pdf',
    fit_ranges=[
        (2000, 3500),
        (2000, 3500),
        (2000, 3500),
        (2000, 3500)
    ]
)
