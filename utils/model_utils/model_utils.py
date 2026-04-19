# model_utils.py - Model utilities

import torch


def strip_optimizer(filename):
    """
    Strip optimizer from checkpoint and convert model to FP16.
    """
    x = torch.load(filename, map_location=torch.device('cpu'), weights_only=False)
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, filename)


def clip_gradients(model, max_norm=10.0):
    """
    Clip gradients to prevent exploding gradients.
    """
    parameters = model.parameters()
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)

