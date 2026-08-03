import matplotlib.pyplot as plt
import numpy as np
import torch

import models
from train import load_data


def plot_improvements(x, y, first_pred, second_pred, save_path=None):
    improved_imgs_mask = (first_pred != y) & (second_pred == y)
    improved_imgs = x[improved_imgs_mask, :]

    fig = plt.figure(figsize=(12, 8))
    axs = np.array(fig.subplots(4, 6)).flatten()
    for i, ax in enumerate(axs):
        ax.imshow(improved_imgs[i].view(28, 28), cmap='Greys_r')
        ax.grid(False)
        ax.axis(False)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def main():
    with torch.no_grad():
        loader = load_data(batch_size=10000, train=False)

        x, y = next(iter(loader))

        first = models.FirstNet(pth_file='weights/firstnet_fashion.pth')
        first.eval()
        first_output = first(x)
        _, first_pred = torch.max(first_output.data, 1)

        second = models.SecondNet(pth_file='weights/secondnet_fashion.pth')
        second.eval()
        second_output = second(x)
        _, second_pred = torch.max(second_output.data, 1)

        third = models.ThirdNet(pth_file='weights/thirdnet_fashion.pth')
        third.eval()
        third_output = third(x)
        _, third_pred = torch.max(third_output.data, 1)

        plt.ion()
        plot_improvements(x, y, first_pred, second_pred,
                          save_path='img/first_to_second.svg')
        plot_improvements(x, y, second_pred, third_pred,
                          save_path='img/second_to_third.svg')
        input()


if __name__ == '__main__':
    main()
