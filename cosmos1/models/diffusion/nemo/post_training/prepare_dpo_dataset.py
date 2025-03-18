# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import argparse
import glob
import json
import os

import torch
import torchvision
from einops import rearrange
from huggingface_hub import snapshot_download
from nemo.collections.diffusion.models.model import DiT7BConfig
from tqdm import tqdm
from transformers import T5EncoderModel, T5TokenizerFast

from cosmos1.utils import log


def get_parser():
    parser = argparse.ArgumentParser(description="Process some configurations.")
    parser.add_argument("--tokenizer_dir", type=str, default="", help="Path to the VAE model")
    parser.add_argument("--dataset_path", type=str, default="video_dataset", help="Path to the dataset. Should contain 'videos' and 'instructions' folders.",)
    parser.add_argument("--output_path", type=str, default="video_dataset_cached", help="Path to the output directory (latents, etc.)")
    parser.add_argument("--num_chunks", type=int, default=1, help="Number of random 130-frame samples to generate per video")
    parser.add_argument("--chunk", type=int, default=0, help="Chunk index to process")
    parser.add_argument("--height", type=int, default=704, help="Height to resize video frames")
    parser.add_argument("--width", type=int, default=1280, help="Width to resize video frames")
    return parser


def init_t5():
    """Initialize and return the T5 tokenizer and text encoder."""
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-11b")
    text_encoder = T5EncoderModel.from_pretrained("google-t5/t5-11b")
    text_encoder.to("cuda")
    text_encoder.eval()
    return tokenizer, text_encoder


def init_video_tokenizer(tokenizer_dir: str):
    """Initialize and return the Cosmos Video tokenizer."""
    dit_config = DiT7BConfig(vae_path=tokenizer_dir)
    vae = dit_config.configure_vae()
    return vae


@torch.no_grad()
def encode_for_batch(tokenizer, encoder, prompts, max_length=512):
    """
    Encode a batch of text prompts to a batch of T5 embeddings.
    Parameters:
        tokenizer: T5 embedding tokenizer.
        encoder: T5 embedding text encoder.
        prompts: A batch of text prompts.
        max_length: Sequence length of text embedding (defaults to 512).
    Returns:
        torch.FloatTensor: Encoded text of shape (batch_size, seq_len, hidden_dim).
    """

    batch_encoding = tokenizer.batch_encode_plus(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_length=True,
        return_offsets_mapping=False,
    )

    # We expect all the processing is done on GPU.
    input_ids = batch_encoding.input_ids.cuda()
    attn_mask = batch_encoding.attention_mask.cuda()

    outputs = encoder(input_ids=input_ids, attention_mask=attn_mask)
    encoded_text = outputs.last_hidden_state  # shape: (batch_size, seq_len, hidden_dim)

    lengths = attn_mask.sum(dim=1).cpu()
    # Zero out positions beyond each sequence's actual length
    for batch_id in range(encoded_text.shape[0]):
        encoded_text[batch_id][lengths[batch_id] :] = 0

    return encoded_text


def create_condition_latent_from_input_frames(vae, input_frames, num_frames_condition=9):
    """
    Encode the first `num_frames_condition` frames as conditioning.

    NOTE: This is a simplified helper. 
    We'll assume the final chunk we pass in has at least `num_frames_condition` frames.

    Args:
        vae: The video VAE tokenizer.
        input_frames (torch.Tensor): shape (B, C, T, H, W), already on GPU in [-1,1].
        num_frames_condition (int): Number of frames from the start to use as conditioning.
    """
    B, C, T, H, W = input_frames.shape
    num_frames_encode = vae.pixel_chunk_duration
    assert T >= num_frames_condition, (
        f"Not enough frames for conditioning: need {num_frames_condition}, got {T}."
    )
    assert (
        num_frames_encode >= num_frames_condition
    ), f"num_frames_encode should be larger than num_frames_condition, get {num_frames_encode}, {num_frames_condition}"

    # We'll simply encode the first N frames directly
    condition_clip = input_frames[:, :, :num_frames_condition]  # shape: (B, C, 9, H, W)

    padding_frames = condition_clip.new_zeros(B, C, num_frames_encode - num_frames_condition, H, W)
    condition_clip = torch.cat([condition_clip, padding_frames], dim=2).to("cuda")
    vae = vae.to(condition_clip.device)
    latent = vae.encode(condition_clip) # shape: (B, latent_dim, 9/factor, H/factor, W/factor)
    return latent


def resample_video_to_24fps(video_tensor, orig_fps):
    """
    Resample (down/up) a video tensor to exactly 24 fps by sampling frames.
    
    Args:
        video_tensor (torch.Tensor): shape (T, H, W, C), in [0..255] range (uint8 or float).
        orig_fps (float): original fps from meta data.
    
    Returns:
        video_24 (torch.Tensor): shape (T_new, H, W, C) with T_new ~ round( (T / orig_fps) * 24 ).
    """
    # Original number of frames
    T = video_tensor.shape[0]
    if orig_fps <= 0 or int(orig_fps)==24:
        # Fallback if fps is not valid
        return video_tensor

    duration_seconds = T / orig_fps
    target_num_frames = int(round(24.0 * duration_seconds))

    if target_num_frames <= 0:
        # If something goes wrong with metadata, return as-is
        return video_tensor

    # Create floating indices from 0..T-1
    idxs = torch.linspace(0, T - 1, steps=target_num_frames)
    idxs = torch.round(idxs).long().clamp(0, T - 1)

    video_24 = video_tensor[idxs]
    return video_24


def main(args):
    # 1. Create output directories for latents, sample videos, sample conditions.
    os.makedirs(args.output_path, exist_ok=True)

    # Initialize the VAE tokenizer
    if args.tokenizer_dir == "":
        args.tokenizer_dir = snapshot_download("nvidia/Cosmos-1.0-Tokenizer-CV8x8x8")
    vae = init_video_tokenizer(args.tokenizer_dir)

    # Constants:
    #  - 9 frames for conditioning - 121 frames for the "main" chunk  - total = 130 frames
    conditioning_frames = 13
    main_video_frames = vae.video_vae.pixel_chunk_duration or 121  
    chunk_total_frames =  main_video_frames # 9 + 121 = 130

    video_folder = args.dataset_path
    video_paths = glob.glob(os.path.join(video_folder, "*.mp4"))
    
    video_paths.sort()
    # Split sorted video paths based on args.num_chunks
    video_paths_split = [video_paths[i::args.num_chunks] for i in range(args.num_chunks)]
    
    # Select the chunk for this run based on args.chunk
    if args.chunk < 0 or args.chunk >= args.num_chunks:
        raise ValueError(f"Invalid chunk index: {args.chunk}. Must be between 0 and {args.num_chunks - 1}.")
    
    video_paths = video_paths_split[args.chunk]

    if not video_paths:
        raise ValueError(f"No .mp4 files found in {video_folder}. Check dataset_path?")

    #ci = 0
    with torch.no_grad():
        for video_path in tqdm(video_paths):
            # NO text embedding, i.e resue same instruction for all videos
            cnt = int(os.path.splitext(os.path.basename(video_path))[0])

            if os.path.exists(os.path.join(args.output_path, f"{cnt}.info.json")):
                log.info(f"Video {video_path} already processed, skipping.")
                continue

            # Read the entire video
            video, _, meta = torchvision.io.read_video(video_path) #  shape: (T, H, W, C), dtype uint8, range [0..255]
            orig_fps = meta.get("video_fps", 24.0)  # default fallback

            # --- (1) Resample video to 24 fps ---
            video = resample_video_to_24fps(video, orig_fps)
            T, H, W, C = video.shape
            #print(f"Video shape at 24fps: {video.shape}, T: {T}")
            new_fps = 24.0  # after resampling

            if T < 1:
                log.info(f"Video {video_path} is empty after resampling, skipping.")
                continue

            assert chunk_total_frames == T, f"Expected {chunk_total_frames} frames, got {T}."

            # --- (2) If T < 130 but > 121, pad up to 130 by repeating the FIRST frame ---
            if T < chunk_total_frames:
                if T > main_video_frames:  # i.e. T in (121, 130)
                    frames_to_add = chunk_total_frames - T
                    #first_frame = video[0:1, ...]  # shape (1, H, W, C)
                    last_frame = video[-1:, ...]                 # <--- changed line
                    repeat_block = last_frame.repeat(frames_to_add, 1, 1, 1)
                    video = torch.cat([video, repeat_block], dim=0)
                    T = video.shape[0]
                else:
                    # If T <= 121, we skip
                    log.info(f"Video {video_path} has {T} frames (<=121) after resampling. Skipped.")
                    continue

            # Extract chunk of shape (130, H, W, C)
            chunk = video[:]  # shape: (120, H, W, C)

            # (4) Convert chunk to shape (B=1, C, T=130, H, W) in [-1,1] float for the VAE
            # Rearrange dimensions: (T, H, W, C) -> (T, C, H, W)
            chunk = rearrange(chunk, "t h w c -> t c h w")
            chunk = torchvision.transforms.functional.resize(chunk, [args.height, args.width])
            t, c, h, w = chunk.shape
            chunk = rearrange(chunk, "(b t) c h w -> b c t h w", b=1)

            # Convert to bf16 and normalize from [0, 255] to [-1, 1]
            chunk = chunk.to(device="cuda", dtype=torch.bfloat16, non_blocking=True) / 127.5 - 1.0

            # (5) Create latents for the entire 130 frames
            latent = vae.encode(chunk).cpu()  # shape: (1, latent_dim, T//factor, H//factor, W//factor)

            # (8) Save the latents embeddings
            torch.save(latent[0], os.path.join(args.output_path, f"{cnt}.video_latent.pth"))

            # (9) Save metadata (info.json)
            info = {
                "height": h, # changed to new H after resizing
                "width": w, # changed to new H after resizing
                "fps": new_fps,
                "prior_fps": orig_fps,
                "num_frames": main_video_frames,
                "video_path": os.path.basename(video_path),
                "start_frame": 0,
                "conditioning_frames": conditioning_frames,
            }
            with open(os.path.join(args.output_path, f"{cnt}.info.json"), "w") as json_file:
                json.dump(info, json_file)

    
    print(f"Done. Created {cnt} chunks.")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # Create output_path if not exist
    os.makedirs(args.output_path, exist_ok=True)

    main(args)
    # aim is to encode the generated videos in _yl
        # i.e get the /processed folder with the latent embeddings
        # no need for/to extract the conditioning frames from the genrated video or text embeddings
        # no need to extract various clips from videos longer than 5 secs as the videos are genereted to be exact
    # use the metadata.jsonl file to get 

