"""Loss registry. To add a loss, write `fn(pred, target) -> scalar` and decorate it:

    @register("my_loss")
    def my_loss(pred, target):
        ...

then train with `--loss my_loss`. Every registered loss is also reported on the
validation set, so runs trained with different losses stay comparable.
Both tensors are already divided by the input mean (see train.py).
"""
import torch.nn.functional as F

LOSSES = {}


def register(name):
    def wrap(fn):
        LOSSES[name] = fn
        return fn
    return wrap


@register("l1")
def l1(pred, target):
    return F.l1_loss(pred, target)


@register("l2")
def l2(pred, target):
    return F.mse_loss(pred, target)
