import numpy as np
import matplotlib.pyplot as plt


def main():
    labels = ['FirstNet', 'SecondNet', 'ThirdNet', 'LargeNet']
    color = ['b'] * 3 + ['r']
    mean = np.array([87.9820, 88.9140, 89.8480, 90.2480])
    std = np.array([0.1662, 0.1951, 0.2470, 0.5296])
    pvalue = np.array([0.0001, 0.0002, 0.261])

    plt.figure(figsize=(8, 6))

    bar = plt.bar(labels, 100 - mean, yerr=std, capsize=8, zorder=3,
                  color=color, alpha=0.5)

    text = np.where(pvalue < 0.001, '(*)', 'n.s.')
    for r, s, t in zip(bar[1:], std[1:], text):
        plt.text(r.get_x() + r.get_width() / 2.0,
                 r.get_height() + s + 0.5 * max(std),
                 t, ha='center', va='bottom')

    plt.ylabel('Tasa de error de evaluación [%]')
    plt.savefig('../img/error_rates.svg')
    plt.show()


if __name__ == '__main__':
    main()
