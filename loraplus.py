from functools import reduce

# MIT License
#
# Copyright (c) 2024 nikhil-ghosh-berkeley
# https://github.com/nikhil-ghosh-berkeley/loraplus

import torch.nn as nn

from peft.tuners import lora
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from transformers.utils import  logging


logger = logging.get_logger(__name__)

def get_module(name, opt_model):
    """
    Retrieve a module from a model using its parameter name.
    Args:
        name (str): Full name of the parameter, typically including module path.
        opt_model (torch.nn.Module): The model from which to retrieve the module.

    Returns:
        Module corresponding to the given name.
    """
    parent_idx = 2 if "lora" in name else 1
    module_names = name.split(sep=".")[:-parent_idx]
    module = reduce(getattr, module_names, opt_model)
    return module

def create_loraplus_params(
    opt_model,
#     optimizer_cls,
    optimizer_kwargs,
    lr_ratio=1.314,
    lr_embedding=None,
):
    """
    Creates an params_group for the given model, applying LoRA-specific learning rate adjustments to different parameter groups.
    
    Args:
        opt_model (torch.nn.Module): The model for which the optimizer is being created.
        optimizer_kwargs (dict): A dictionary of keyword arguments for the optimizer's initialization.
        lr_ratio (float): The learning rate ratio to be applied to LoRA parameters.
        lr_embedding (float, optional): A specific learning rate for embedding parameters, with a default value if not provided.
    
    Returns:
        List of Params that needs to be updated
        
    example use:-
    update_params=create_loraplus_params(
    model,
    optimizer_kwargs={'lr':1e-5},
    lr_ratio=3,
    lr_embedding=2e-6,
    )

    
    optimizer=AdamW(update_params,**kwargs)
    """

    assert lr_ratio is not None

    if lr_embedding is None:
        lr_embedding = 1e-6

    decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    param_groups = {
        "groupA": {},
        "groupB": {},
        "groupB_no_decay": {},
        "embedding": {},
    }

    for name, param in opt_model.named_parameters():
        if not param.requires_grad:
            continue

        module = get_module(name, opt_model)
        if isinstance(module, lora.Embedding):
            param_groups["embedding"][name] = param
        elif "lora_B" in name or param.ndim == 1:
            if name in decay_parameters:
                param_groups["groupB"][name] = param
            else:
                param_groups["groupB_no_decay"][name] = param
        else:
            param_groups["groupA"][name] = param

    assigned_param_groups = ""
    for group in param_groups:
        assigned_param_groups += f"{group}\n {list(param_groups[group].keys())}\n\n"
    logger.debug(assigned_param_groups)

    lr = optimizer_kwargs["lr"]
    weight_decay = optimizer_kwargs.get("weight_decay", 0.0)

    optimizer_grouped_parameters = [
        {
            "params": list(param_groups["groupA"].values()),
            "weight_decay": weight_decay,
            "lr": lr,
        },
        {
            "params": list(param_groups["embedding"].values()),
            "weight_decay": weight_decay,
            "lr": lr_embedding,
        },
        {
            "params": list(param_groups["groupB"].values()),
            "weight_decay": weight_decay,
            "lr": lr * lr_ratio,
        },
        {
            "params": list(param_groups["groupB_no_decay"].values()),
            "weight_decay": 0.0,
            "lr": lr * lr_ratio,
        },
    ]

#     optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    return optimizer_grouped_parameters ## return updated params with specific weight decay
