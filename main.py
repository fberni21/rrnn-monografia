import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn.functional import relu
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score

import models
from train import load_data


class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def visualize_all_maps(model, deconv, layer_idx, x, y, pred):
    imgs = []
    indices = []
    true_max = 0.0

    num_maps = model.activations[layer_idx].shape[1]

    for map_idx in range(num_maps):
        map_list = [map_idx]
        acts = relu(model.activations[layer_idx][:, map_list, :, :])

        max_values, max_indices = torch.max(
            acts.view(acts.shape[0], 1, -1), dim=2, keepdim=True
        )
        max_val, winning_img_tensor = torch.max(max_values, dim=0, keepdim=True)

        win_idx = winning_img_tensor.item()
        indices.append(win_idx)

        max_spatial_idx = max_indices[win_idx].item()
        max_h = max_spatial_idx // acts.shape[3]
        max_w = max_spatial_idx % acts.shape[3]

        isolated = torch.zeros((1, 1, acts.shape[2], acts.shape[3]))
        isolated[0, 0, max_h, max_w] = max_val

        winning_pool_indices = {
            k: v[win_idx : win_idx + 1] for k, v in model.pool_indices.items()
        }

        reconstruction = deconv.forward(
            isolated, layer_idx, map_list, winning_pool_indices
        )

        img = reconstruction.view(28, 28).cpu().numpy()
        imgs.append(img)

        val_max = np.max(np.abs(img))
        true_max = max(true_max, val_max.item())

    ncols = int(math.ceil(math.sqrt(num_maps)))
    nrows = int(math.ceil(num_maps / ncols))

    fig = plt.figure(figsize=(ncols * 4, nrows * 2))
    subfigs = fig.subfigures(1, 2)

    subfigs[0].suptitle(
        'Corresponding Input Images (Peak Activation)', fontsize=14
    )
    axes_inputs = np.array(subfigs[0].subplots(nrows, ncols)).flatten()

    subfigs[1].suptitle(f'Reconstructions (Layer {layer_idx})', fontsize=14)
    axes_deconv = np.array(subfigs[1].subplots(nrows, ncols)).flatten()

    for i in range(num_maps):
        win_idx = indices[i]
        img = imgs[i]

        ax_in = axes_inputs[i]
        ax_in.imshow(x[win_idx].view(28, 28).cpu(), cmap='Greys_r')
        ax_in.axis('off')

        ax_rec = axes_deconv[i]
        ax_rec.imshow(img, cmap='Greys_r', vmin=-true_max, vmax=true_max)
        ax_rec.axis('off')

    for i in range(num_maps, len(axes_inputs)):
        axes_inputs[i].axis('off')
        axes_deconv[i].axis('off')

    plt.show()


def main():
    with torch.no_grad():
        loader = load_data(batch_size=10000, train=False)

        x, y = next(iter(loader))

        model = models.FirstNet(pth_file='firstnet_fashion.pth')
        # model = models.SecondNet(pth_file='secondnet_fashion.pth')
        num_parameters = sum([p.numel()
                              for p in model.parameters() if p.requires_grad])
        print(f'Model has {num_parameters} parameters.')
        model.eval()
        output = model(x)
        _, pred = torch.max(output.data, 1)

        accuracy = 100 * accuracy_score(y, pred)

        print(f'Model accuracy: {accuracy:.2f} %',
              f'- Model error rate {100 - accuracy:.2f} %')

        ConfusionMatrixDisplay.from_predictions(y, pred, normalize='true', display_labels=class_names)

        deconv = models.DeconvFirstNet(pth_file='firstnet_fashion.pth')
        # deconv = models.DeconvSecondNet(pth_file='secondnet_fashion.pth')
        deconv.eval()

        plt.ion()
        # for i in [0, 3, 6, 8]:
        for i in [0, 3]:
            visualize_all_maps(model, deconv, i, x, y, pred)

        input()

if __name__ == '__main__':
    main()
