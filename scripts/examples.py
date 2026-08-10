import matplotlib.pyplot as plt

from train import load_data


class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def main():
    data = load_data(batch_size=10000, train=False)
    x, y = next(iter(data))

    fig, axs = plt.subplots(10, 10, figsize=(6, 6))
    for i, class_name in enumerate(class_names):
        imgs = x[y == i][:10]
        for j in range(10):
            ax = axs[i][j]
            ax.imshow(imgs[j].view(28, 28), cmap='Grays')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if j == 0:
                ax.text(-0.3, 0.5, class_name,
                        fontsize=11, rotation=0,
                        transform=ax.transAxes,
                        ha='right', va='center')

    plt.tight_layout()
    plt.subplots_adjust(left=0.16)
    plt.savefig('../img/examples.svg')
    plt.show()


if __name__ == '__main__':
    main()
