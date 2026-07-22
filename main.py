import numpy as np
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
        loader = load_data(batch_size=10000, train=False)

        x, y = next(iter(loader))

        model = models.FberNet2(pth_file='fbernet2_fashion.pth')
        model.eval()
        output = model(x)
        _, pred = torch.max(output.data, 1)

        deconv = models.DeconvFberNet2(pth_file='fbernet2_fashion.pth')
        deconv.eval()

        # 0 2 5 7 10
        layer_idx = 5
        imgs = []
        indices = []
        true_max = -1000

        for map_idx in range(model.activations[layer_idx].shape[1]):
            map_idx = [map_idx]
            acts = relu(model.activations[layer_idx][:, map_idx, :, :])

            isolated = torch.zeros((1, 1, acts.shape[2], acts.shape[3]))
            max_values, max_indices = torch.max(acts.view(acts.shape[0], 1, -1), 2, keepdim=True)
            max_val, index = torch.max(max_values, 0, keepdim=True)
            index = [index.item()]
            max_idx = max_indices[index]

            max_h = max_idx // acts.shape[3]
            max_w = max_idx % acts.shape[3]
            isolated[:, :, max_h, max_w] = max_val

            reconstruction = deconv.forward(isolated,
                                            layer_idx,
                                            map_idx,
                                            {k: v[index] for k, v
                                             in model.pool_indices.items()})
            img = reconstruction.view(28, 28).numpy()
            imgs.append(img)
            indices.append(index)
            val_max = np.max(np.abs(img))
            true_max = max([true_max, val_max.item()])

        plt.ion()
        plt.figure(figsize=(8, 4))
        for i, (index, img) in enumerate(zip(indices, imgs)):
            plt.subplot(121)
            plt.title(f'{class_names[y[index]]} (pred: {class_names[pred[index]]})')
            plt.imshow(x[index].view(28, 28), cmap='Greys_r')
            plt.axis(False)

            plt.subplot(122)
            plt.title(f'Layer/Map: {layer_idx}/{i}')
            plt.imshow(img, cmap='Greys_r', vmin=-true_max, vmax=true_max)
            # plt.imshow(img, cmap='RdBu_r')
            plt.axis(False)
            input()


if __name__ == '__main__':
    main()
