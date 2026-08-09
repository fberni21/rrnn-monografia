import numpy as np
import matplotlib.pyplot as plt


def main():
    labels = ['FirstNet', 'SecondNet', 'ThirdNet', 'LargeNet']
    means = np.array([87.9820, 88.9140, 89.8480, 90.7080])
    stds = np.array([0.1662, 0.1951, 0.2470, 0.2276])
    pvalues = np.array([-1, 0.0001, 0.0002, 0.0004])
    error_rates = 100 - means

    _, ax = plt.subplots(figsize=(6, 4))

    bars = ax.bar(labels, error_rates, yerr=stds, capsize=8, zorder=3)

    symbols = np.where(pvalues < 0.001, '(*)', 'n.s.')
    labels = list(f"{e:.1f} %" + (f" {s}" if i > 0 else "")
                  for i, (e, s) in enumerate(zip(error_rates, symbols)))
    ax.bar_label(bars, padding=2, labels=labels)

    plt.ylabel('Tasa de error de evaluación [%]')
    plt.grid(zorder=0)
    plt.tight_layout()
    plt.savefig('../img/error_rates.svg')
    plt.show()


if __name__ == '__main__':
    main()
