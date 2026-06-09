#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
 
import os, sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.ttnn as ttnn
from workloads.ttnn.tt_transformers.model import Transformer
from workloads.ttnn.tt_transformers.model_config import ModelArgs
from ttsim.front.ttnn.device import Device as TTNNDevice
from loguru import logger
from ttsim.utils.common import setup_logger

def run_llama3(wlname: str, ttnn_device: TTNNDevice, cfg: dict):
    assert isinstance(ttnn_device, TTNNDevice), "ttnn_device must be a TTNNDevice"
    assert isinstance(cfg, dict), "cfg must be a dictionary"
    assert isinstance(wlname, str), "wlname must be a string"

    model_name = cfg.get('model_name', 'llama3-8B')
    paged_attention = False
    page_params = [{"page_block_size": 32, "page_max_num_blocks": 1024}]
    batch_size = cfg.get('bs', 1)
    if model_name == "llama3-8B":
        max_seq_len = 4096 #256
        layers = 32
    elif model_name == "llama3-3B":
        max_seq_len = 2048 #256
        layers = 28
    elif model_name == "llama3-1B":
        max_seq_len = 2048 #256
        layers = 16
    elif model_name == "llama3-70B":
        max_seq_len = 8192
        layers = 80
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    weights = "random"

    dtype = ttnn.bfloat8_b
    instruct = False  # True if weights == "instruct" else False
    dummy_weights = True if weights == "random" else False
    cache_pcc = False # layers == 1 and not dummy_weights

    model_args = ModelArgs(
        ttnn_device,
        model_name=model_name,
        instruct=instruct,
        dummy_weights=dummy_weights,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
    )
    iterations = cfg.get('iterations', 1)

    if layers is not None:
        model_args.n_layers = layers
    state_dict = None

    prompts = ["This is a test"] * model_args.max_batch_size
    encoded_prompts = [128000]
    generation_start_pos = cfg.get('start_pos', 512)
    generation_length = iterations
    kv_slice = cfg.get('kv_slice', True)
    dram_model = cfg.get('dram_model', True)
    page_table_tt = None
    paged_attention_config = None

    # Load TTNN model
    tt_model = Transformer(
        args=model_args,
        mesh_device=ttnn_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=None,
        paged_attention_config=paged_attention_config,
    )
    logger.info("Model and caches loaded.")

    # ✅ FIX: Build kv_cache from each layer's pre-allocated layer_past
    kv_cache = [
        [
            tt_model.layers[i].attention.layer_past[0],
            tt_model.layers[i].attention.layer_past[1],
        ]
        for i in range(model_args.n_layers)
    ]
    logger.info(f"KV cache built for {model_args.n_layers} layers.")

    seqlen = 1  # Generating one token per user at a time
    batch = model_args.max_batch_size

    # Select the first token from the prompts for initial decoding
    encoded_prompts_tensor = ttnn._rand(shape=(len(encoded_prompts), batch), device=ttnn_device, dtype=ttnn.int32)
    tt_decode_input = tt_model.embd(encoded_prompts_tensor).view(seqlen, batch, -1)

    for i in range(generation_length):
        pos = generation_start_pos + i
        pos_np = np.array([pos] * batch_size, dtype=np.int32)
        current_pos_tensor = ttnn.Tensor(
            shape=(batch_size,),
            device=ttnn_device,
            dtype=ttnn.int32,
            data=pos_np,
        )
        rot_idxs_np = pos_np.reshape(1, batch_size)
        rot_idxs = ttnn.Tensor(
            shape=(1, batch_size),
            device=ttnn_device,
            dtype=ttnn.int32,
            data=rot_idxs_np,
        )

        logger.info(f"[Model] Generating token {i} at position {pos} (kv_slice={kv_slice}, dram_model={dram_model})")
        decode_input = model_args.prepare_residual_tensor_decode(
            tt_decode_input,
            None, #model_args.model_config["DECODE_RESIDUAL_MEMCFG"],
        )
        rot_mats = tt_model.rope_setup.get_rot_mats(rot_idxs)
        # Run TT model
        tt_out = tt_model(
            decode_input,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table_tt,
            kv_cache=kv_cache,
            kv_slice=kv_slice,
            dram_model=dram_model,
        )
        tt_output_torch = ttnn.permute(ttnn.to_torch(tt_out), (1, 2, 0, 3)).squeeze(2)#[: model_args.max_batch_size, 0:1, : model_args.vocab_size]
        
        if (tt_output_torch.shape == [batch_size, seqlen, 128256]): # 128256 is the vocab_size for llama3 8B and llama3 3B, 1B
            logger.info(f'tt_output_torch is correctly shaped: {tt_output_torch.shape}')
        else:
            logger.info(f'tt_output_torch is incorrectly shaped: {tt_output_torch.shape}')

        logger.info(f"Finished running TT model {model_name}.")
        
        ttnn.deallocate(tt_out)
    # print("Generating Model Graph...")
    # g = mesh_device.get_graph()
    # g.graph2onnx('ttnn_llama32_model.onnx', do_model_check=False,
    #                 filter_op_attrs=filter_ttnn_attrs)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = "llama3-3B"
    ttnn_device = ttnn.open_device(device_id=0)
    run_llama3(wlname='llama3', ttnn_device=ttnn_device, cfg={'model_name': model_name, 'bs': 1})
    ttnn.close_device(ttnn_device)
