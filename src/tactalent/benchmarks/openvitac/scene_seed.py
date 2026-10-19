"""Seed native scene construction without changing the reset random stream."""

import numpy as np


def seeded_task_class(native_task, seed):
    """Native BaseTask creates an entropy RNG before constructing actors.

    A later reset(seed) cannot reproduce those one-time choices. Seed only the
    instance scene RNG at its construction boundary; native reset still starts
    its own unchanged random stream. No global RNG or upstream class is patched.
    """

    class SeededTask(native_task):
        def _setup_scene(self):
            self.rng = np.random.default_rng(seed)
            return super()._setup_scene()

    SeededTask.__name__ = native_task.__name__
    SeededTask.__module__ = native_task.__module__
    return SeededTask
