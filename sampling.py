import math
import os
import sys
from glob import glob
from pathlib import Path
from typing import List, Optional

import cv2
import imageio
import numpy as np
import torch
from einops import rearrange, repeat
from fire import Fire
from omegaconf import OmegaConf
from PIL import Image
from rembg import remove
from util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.inference.helpers import embed_watermark
from sgm.util import default, instantiate_from_config
from torchvision.transforms import ToTensor
from tqdm import tqdm
from camera import Warper

def sample(
    input_path: str = "assets/images/cat.jpg",  # Can either be image file or folder with image files
    prompt: str="a cat wandering in garden",
    neg_prompt: str=" ",
    pcd_mode: str = 'complex default 14 mode_4',
    add_index: int = 10,
    num_frames: int = 14,
    num_steps: Optional[int] = 25,
    fps_id: int = 6,
    motion_bucket_id: int = 127,
    version: str = 'svd',
    cond_aug: float = 0.02,
    seed: int = 1,
    decoding_t: int = 4,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    output_folder: Optional[str] = None,
    verbose: Optional[bool] = False,
    save_warps: Optional[bool] = False,
    load_warps: Optional[str] = None,
):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """
    pcd_mode = pcd_mode.split(' ')
    num_frames = default(num_frames, 14)
    num_steps = default(num_steps, 25)
    output_folder = default(output_folder, "outputs")
    model_config = "sgm/svd.yaml"
    pcd_dir = os.path.join(output_folder,'renderings')
    os.makedirs(output_folder, exist_ok=True)
    if save_warps == True:
        os.makedirs(pcd_dir, exist_ok=True)

    model, filter = load_model(
        model_config,
        device,
        num_frames,
        num_steps,
        verbose,
    )
    torch.manual_seed(seed)

    path = Path(input_path)
    all_img_paths = []
    if path.is_file():
        if any([input_path.endswith(x) for x in ["jpg", "jpeg", "png"]]):
            all_img_paths = [input_path]
        else:
            raise ValueError("Path is not valid image file.")
    elif path.is_dir():
        all_img_paths = sorted(
            [
                f
                for f in path.iterdir()
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )
        if len(all_img_paths) == 0:
            raise ValueError("Folder does not contain any images.")
    else:
        raise ValueError

    for input_img_path in all_img_paths:
        with Image.open(input_img_path) as image:
            input_image = image.convert("RGB") 
            w, h = image.size
            if h % 64 != 0 or w % 64 != 0:
                width, height = map(lambda x: x - x % 64, (w, h))
                input_image = input_image.resize((width, height))
                print(
                    f"WARNING: Your image is of size {h}x{w} which is not divisible by 64. We are resizing to {height}x{width}!"
                )

        image = ToTensor()(input_image)
        image = image * 2.0 - 1.0

        image = image.unsqueeze(0).to(device)
        H, W = image.shape[2:]
        assert image.shape[1] == 3
        F = 8
        C = 4
        shape = (num_frames, C, H // F, W // F)
        if (H, W) != (576, 1024) and "sv3d" not in version:
            print(
                "WARNING: The conditioning frame you provided is not 576x1024. This leads to suboptimal performance as model was only trained on 576x1024. Consider increasing `cond_aug`."
            )
        if motion_bucket_id > 255:
            print(
                "WARNING: High motion bucket! This may lead to suboptimal performance."
            )
        if fps_id < 5:
            print("WARNING: Small fps value! This may lead to suboptimal performance.")

        if fps_id > 30:
            print("WARNING: Large fps value! This may lead to suboptimal performance.")

        value_dict = {}
        value_dict["cond_frames_without_noise"] = image
        value_dict["motion_bucket_id"] = motion_bucket_id
        value_dict["fps_id"] = fps_id
        value_dict["cond_aug"] = cond_aug
        value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)

        with torch.no_grad():
            with torch.autocast(device):
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    value_dict,
                    [1, num_frames],
                    T=num_frames,
                    device=device,
                )
                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=[
                        "cond_frames",
                        "cond_frames_without_noise",
                    ],
                )

                for k in ["crossattn", "concat"]:
                    uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                    uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                    c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                    c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

                additional_model_inputs = {}
                additional_model_inputs["image_only_indicator"] = torch.zeros(
                    2, num_frames
                ).to(device)
                additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

                def denoiser(input, sigma, c):
                    return model.denoiser(
                        model.model, input, sigma, c, **additional_model_inputs
                    )
                
                if load_warps != None:
                    print('warp path provided, reading from folder')
                    images = concat_warp_start(image, num_frames, load_warps)
                else:
                    warper = Warper(H, W)
                    images = warper.generate_pcd(input_image, prompt, neg_prompt, pcd_mode, seed, num_steps, pcd_dir, save_warps)
                latent_images = model.encode_first_stage(images)

                # samples_z = model.sampler(denoiser, randn, cond=c, uc=uc)
                randn = torch.randn(shape, device=device)
                _, s_in, sigmas, num_sigmas, cond, uc = model.sampler.prepare_sampling_loop(randn, cond=c, uc=uc, num_steps=num_steps)            
                
                noise = torch.randn(shape, device=device)
                x  = latent_images + noise * sigmas[add_index]

                for i in tqdm(model.sampler.get_sigma_gen(num_sigmas)[add_index:]):
                    gamma = (
                        min(model.sampler.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                        if model.sampler.s_tmin <= sigmas[i] <= model.sampler.s_tmax
                        else 0.0
                    )

                    x = model.sampler.sampler_step(
                        s_in * sigmas[i],
                        s_in * sigmas[i + 1],
                        denoiser,
                        x,
                        cond,
                        uc,
                        gamma,
                    )

                model.en_and_decode_n_samples_a_time = decoding_t
                samples_x = model.decode_first_stage(x)
                if "sv3d" in version:
                    samples_x[-1:] = value_dict["cond_frames_without_noise"]
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

                base_count = len(glob(os.path.join(output_folder, "*.gif")))

                samples = embed_watermark(samples)
                samples = filter(samples)
                vid = (
                    (rearrange(samples, "t c h w -> t h w c") * 255)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )
                video_path = os.path.join(output_folder, f"{base_count:06d}_{'_'.join(pcd_mode)}_i_{add_index}_seed_{seed}.gif")
                imageio.mimwrite(video_path, vid)

def concat_warp_start(image, num_frames, concat_path, device = 'cuda'):
    images = torch.Tensor([]).to(device)
    h, w = image.shape[2:]
    for i in range(num_frames):
        if i == 0:
            new_image = image
        else:
            new_image = Image.open(f'{concat_path}/{i}_concat.png').resize((w, h))
            new_image = ToTensor()(new_image)
            new_image = new_image * 2.0 - 1.0
            new_image = new_image.unsqueeze(0).to(device)
        images = torch.cat([images, new_image])
    
    return images

def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames" or key == "cond_frames_without_noise":
            batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=N[0])
        elif key == "polars_rad" or key == "azimuths_rad":
            batch[key] = torch.tensor(value_dict[key]).to(device).repeat(N[0])
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def load_model(
    config: str,
    device: str,
    num_frames: int,
    num_steps: int,
    verbose: bool = False,
):
    config = OmegaConf.load(config)
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.verbose = verbose
    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )
    if device == "cuda":
        with torch.device(device):
            model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()

    filter = DeepFloydDataFiltering(verbose=False, device=device)
    return model, filter


if __name__ == "__main__":
    Fire(sample)