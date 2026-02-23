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
# Based on timm and 4M code bases
# https://github.com/huggingface/pytorch-image-models
# https://github.com/apple/ml-4m/
# --------------------------------------------------------

from typing import Optional, Union
import io
import os
import ast
import json
from pathlib import Path
from hydra.utils import instantiate
from safetensors.torch import load as load_st
from safetensors.torch import save_file

import torch

from .dist import save_on_main, is_main_process


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model

def get_state_dict(model, unwrap_fn=unwrap_model):
    return unwrap_fn(model).state_dict()


def load_state_dict(model, state_dict, prefix='', ignore_missing=''):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


def save_model(
        args, iteration, model, model_without_ddp, optimizer, loss_scaler, loss_balancer=None, 
        ckpt_name=None, all_nodes=False, save_as_safetensors=False, model_args=None
    ):
    output_dir = Path(args.output_dir)
    iteration_name = str(iteration)
    ckpt_name = ckpt_name or iteration_name

    # Only create the save_dict on the main process, unless all_nodes is set to True
    if is_main_process() or (all_nodes and args.gpu == 0): 
        checkpoint_path = os.path.join(output_dir, f'checkpoint-{ckpt_name}.pth')

        to_save = {
            'model': model_without_ddp.state_dict(),
            'iteration': iteration,
            'args': args,
            'scaler': loss_scaler.state_dict(),
        }

        if optimizer is not None:
            to_save['optimizer'] = optimizer.state_dict()

        if loss_balancer is not None:
            to_save['loss_balancer'] = loss_balancer.state_dict()

        save_on_main(to_save, checkpoint_path)

        # Save only weights as .safetensors, including model args as metadata
        if save_as_safetensors:
            checkpoint_path_st = os.path.join(output_dir, f"checkpoint-{ckpt_name}.safetensors")
            save_safetensors(to_save["model"], checkpoint_path_st, metadata_dict=model_args)


def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler):
    output_dir = Path(args.output_dir)
    # torch.amp
    if args.auto_resume and len(args.resume) == 0:
        import glob
        all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
        latest_ckpt = -1
        for ckpt in all_checkpoints:
            t = ckpt.split('-')[-1].split('.')[0]
            if t.isdigit():
                latest_ckpt = max(int(t), latest_ckpt)
        if latest_ckpt >= 0:
            args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
            
    if args.resume:
        print("Auto resume checkpoint: %s" % args.resume)

        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu')
        else:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)

        if 'optimizer' in checkpoint and 'iteration' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_iteration = checkpoint['iteration'] + 1

            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


def save_safetensors(state_dict, ckpt_path, metadata_dict=None):
    for k, v in state_dict.items():
        state_dict[k] = v.contiguous()
    if metadata_dict is not None:
        metadata = {k: str(v) for k, v in metadata_dict.items()}
    else:
        metadata = None
    save_file(state_dict, ckpt_path, metadata=metadata)


def parse_metadata(metadata_str):
    metadata = {}
    for k, v in metadata_str.items():
        try:
            v_parsed = ast.literal_eval(v)
        except:
            v_parsed = v
        metadata[k] = v_parsed
    return metadata


def load_safetensors(safetensors_path, return_metadata=True):
    with open(safetensors_path, "rb") as f:
        data = f.read()

    tensors = load_st(data)

    if not return_metadata:
        return tensors

    n_header = data[:8]
    n = int.from_bytes(n_header, "little")
    metadata_bytes = data[8 : 8 + n]
    header = json.loads(metadata_bytes)
    metadata = header.get("__metadata__", {})
    metadata = parse_metadata(metadata)

    return tensors, metadata


def load_model_from_safetensors(
    ckpt_path: str,
    device: Optional[Union[str, torch.device]] = None,
    to_eval: bool = True,
) -> torch.nn.Module:
    ckpt, config = load_safetensors(ckpt_path)
    model = instantiate(config)
    model.load_state_dict(ckpt)
    if device is not None:
        model = model.to(device)
    if to_eval:
        model = model.eval()
    return model