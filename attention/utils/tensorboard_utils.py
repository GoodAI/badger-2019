import io
from datetime import datetime

from pathlib import Path
from typing import Optional

from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from attention.utils import project_root
import tensorflow as tf


class TensorboardUtils:
    def __init__(self, experiment_name: str, log_root: Optional[Path] = None, append_timestamp: bool = False):
        # Disable GPU usage
        tf.config.experimental.set_visible_devices([], 'GPU')

        if log_root is None:
            log_root = project_root() / 'data' / 'tensorboard'
        timestamp = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
        if append_timestamp:
            experiment_name = f'{experiment_name}_{timestamp}'
        log_dir = log_root / experiment_name

        file_writer = tf.summary.create_file_writer(str(log_dir / "metrics"))
        file_writer.set_as_default()

    def add_matplot(self, figure: Figure, name: str, step: int):
        tf.summary.image(name, self._matplot_to_tfplot(figure), step=step)
        plt.close(figure)

    def add_scalar(self, value: float, name: str, step: int):
        tf.summary.scalar(name, value, step)

    def _matplot_to_tfplot(self, fig: Figure):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches="tight")
        buf.seek(0)

        # Prepare the plot
        plot_buf = buf

        # Convert PNG buffer to TF image
        image = tf.image.decode_png(plot_buf.getvalue(), channels=4)

        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image


