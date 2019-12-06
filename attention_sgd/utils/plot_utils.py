from typing import Optional, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import time
import seaborn as sn

import torch
from IPython import display
from PIL import Image
from matplotlib.figure import Figure
from torch import Tensor
import torchvision.transforms.functional as F


def norm_red_green():
    return matplotlib.colors.Normalize(-1, 1, clip=True)


def cmap_red_green():
    return matplotlib.colors.LinearSegmentedColormap.from_list("", [[0.0, "#ff0000"],
                                                                    [0.5, "#000000"],
                                                                    [1.0, "#00ff00"]])


def plot_forwarding_task_targets(output, target, count=10, dim0=0, dim1=1, plot=plt):
    target = target.cpu().detach().numpy()
    output = output.cpu().detach().numpy()
    plot.scatter(target[:count, dim0], target[:count, dim1])
    plot.scatter(output[:count, dim0], output[:count, dim1])
    plot.plot([target[:count, dim0], output[:count, dim0]], [target[:count, dim1], output[:count, dim1]], 'k--')


def plot_painter_output(output):
    out = np.clip((output.cpu().detach().numpy() + 1) / 2, 0, 1)
    out = out[0].transpose(1, 2, 0)
    plt.imshow(out)


def tensor_to_image(tensor: Tensor) -> Image:
    """
    Converts Tensor to Image
    :param tensor: Tensor float[colors, height, width], values in range <-1.0, 1.0>
    :return: PIL Image
    """
    tensor = (tensor + 1) / 2
    return F.to_pil_image(tensor.cpu().detach())
    # out = np.clip(tensor.cpu().detach().numpy(), 0, 256).astype(np.uint8)
    # out = out[0].transpose(1, 2, 0)
    # return Image.fromarray(out)


def image_to_tensor(image: Image, device: Optional[str] = None) -> Tensor:
    """
    Converts image to Tensor of shape: [colors, height, width] with values in range <-1.0, 1.0>
    i.e. [3, 128, 128]
    :param image: PIL image
    :param device: 'cpu' or 'cuda'
    :return: Tensor float[batch_size, colors, height, width], values in range <-1.0, 1.0>
    """
    result = F.to_tensor(image) * 2 - 1
    return result if device is None else result.to(device)
    # result = (torch.tensor(np.array(image), dtype=torch.float, device=device) - 128.0) / 128.0
    # return result.permute(2, 0, 1).unsqueeze(0)


def notebook_show_plot(size=(16, 8)):
    plt.gcf().set_size_inches(size)
    display.clear_output(wait=True)
    display.display(plt.gcf())
    time.sleep(0.01)


def plot_matrix(data: Tensor, clip=True, include_values=True) -> Figure:
    fig = matplotlib.figure.Figure()
    ax = fig.add_subplot(1, 1, 1)
    plot_matrix_to_axis(ax, data, clip, include_values)
    size = get_plot_size(data)
    fig.set_size_inches(size[0], size[1])
    return fig


def plot_matrix_to_axis(axis, data: Tensor, clip=True, include_values=True):
    data = data.detach().cpu().numpy()
    if include_values:
        sn.heatmap(data, annot=True, cmap=cmap_red_green(), vmin=-1, vmax=1, ax=axis, square=True, fmt='.2f',
                   cbar=False)
    else:
        axis.matshow(data, cmap=cmap_red_green(), norm=norm_red_green() if clip else None)


def get_plot_size(data: Tensor):
    size_coef = 0.5
    max_size = 50
    size = np.array(data.shape) * size_coef
    scale = max(size / max_size)
    if scale > 1.0:
        size = size / scale
    # height, width
    return size[1], size[0]


def plot_matrices(data: List[Tensor], clip=True, include_values=True) -> Figure:
    plots = len(data)
    fig = matplotlib.figure.Figure()
    total_size = [0, 0]

    # sanitize data
    def sanitize(t: Tensor):
        if t.dim() < 2:
            return t.unsqueeze(0)
        elif t.dim() > 2:
            return t.view(-1, t.size(-1))
        return t

    data = [sanitize(t) for t in data]

    ratios = [get_plot_size(t)[1] for t in data]
    axes = fig.subplots(plots, 1, gridspec_kw={'height_ratios': ratios})
    # Note: figures created by plt has to be closed manually (no garbage collection happening) - not this case.
    # following commands may be useful according to https://stackoverflow.com/a/21884375/2203164,
    # but not necessary for now:
    # plt.cla()
    # plt.clf()
    # plt.close()

    for p, tensor in enumerate(data):
        plot_matrix_to_axis(axes[p], tensor, clip, include_values)
        size = get_plot_size(tensor)
        total_size[1] += size[1]
        total_size[0] = max(total_size[0], size[0])

    fig.set_size_inches(total_size[0], total_size[1])
    return fig


def plot_line(data: List[float]) -> Figure:
    fig = matplotlib.figure.Figure()
    ax = fig.add_subplot(1, 1, 1, yscale="log")
    ax.plot(data)
    return fig
