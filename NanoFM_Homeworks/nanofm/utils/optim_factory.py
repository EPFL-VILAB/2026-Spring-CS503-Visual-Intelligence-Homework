# Copyright 2025 EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------
# Based on 4M, BEiT, timm, DINO, DeiT code bases
# https://github.com/apple/ml-4m/
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------

import json

import torch
from torch import optim as optim


def get_parameter_groups(model, weight_decay=1e-5, skip_list=()):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        # Remove wrapped module to be compatible with FSDP
        name = name.replace("_fsdp_wrapped_module.", "")

        if not param.requires_grad:
            continue  # frozen weights

        # Assign weight decay values
        # Only norm and bias terms should have no decay
        # Previously, this checked if (param.shape) == 1 which is incompatible with FSDP which flattens all params
        if "norm." in name or ".norm" in name or name.endswith(".bias") or name.endswith(".lookup_table_weight") or name.endswith(".gamma") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay

        if group_name not in parameter_group_names:
            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def create_adamw_optimizer(args, model, filter_bias_and_bn=True, skip_list=None):
    weight_decay = args.weight_decay

    def get_parameters(m):
        if weight_decay and filter_bias_and_bn:
            skip = {}
            if skip_list is not None:
                skip = skip_list
            elif hasattr(m, 'no_weight_decay'):
                skip = m.no_weight_decay()
            parameters = get_parameter_groups(m, weight_decay, skip)
            wd = 0.
        else:
            parameters = m.parameters()
            wd = weight_decay
        return parameters, wd
    
    parameters, weight_decay = get_parameters(model)

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas

    print("optimizer settings:", opt_args)

    optimizer = optim.AdamW(parameters, **opt_args)

    return optimizer