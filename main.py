import matplotlib.pyplot as plt
import torch
from torch.nn.functional import relu

import models
from train import load_data


class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def main():
    with torch.no_grad():
        loader = load_data(train=False)

        x, y = next(iter(loader))
        idx = 0

        model = models.FberNet(pth_file='fbernet_fashion.pth')
        model.eval()
        output = model(x)
        _, pred = torch.max(output.data, 1)

        plt.figure()
        plt.subplot(121)
        plt.title(f'{class_names[y[idx]]} (pred: {class_names[pred[idx]]})')
        plt.imshow(x[idx, :, :, :].view(28, 28), cmap='Greys_r')
        plt.axis(False)

        deconv = models.DeconvFberNet(pth_file='fbernet_fashion.pth')
        deconv.eval()

        layer_idx = 6
        out_channels = model.activations[layer_idx].shape[1]
        #_, map_idx = torch.max(
        #        torch.sum(model.activations[layer_idx]
        #                  .view(x.shape[0], out_channels, -1), 2), 1)
        map_idx = [7] * 64

        activation = model.activations[layer_idx][:, map_idx[idx], :, :]
        reconstruction = relu(deconv.forward(activation[:, None, :, :],
                                             layer_idx,
                                             map_idx[idx],
                                             model.pool_indices))

        plt.subplot(122)
        plt.title(f'Layer/Map: {layer_idx}/{map_idx[idx]}')
        plt.imshow(reconstruction[idx, :, :, :].view(28, 28), cmap='Greys_r')
        plt.axis(False)

        plt.show()


if __name__ == '__main__':
    main()
