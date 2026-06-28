# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
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

import os
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

from torch.utils.data import DataLoader, Dataset 
from datasets import load_dataset
from torchvision import transforms

from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model

# 导入你的核心组件
from dreamlite import DreamLitePipelineLoRA

def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA for DreamLite")
    parser.add_argument("--model_id", type=str, default="models/DreamLite-base")
    parser.add_argument("--dataset_id", type=str, default="showlab/OmniConsistency")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="./output/output_lora/edit_Snoopy")
    parser.add_argument("--rank", type=int, default=16, help="LoRA Rank")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size only can be 1 here.")
    parser.add_argument("--max_train_steps", type=int, default=3500)
    parser.add_argument("--default_prompt", type=str, default="transfer the image into Snoopy style")
    parser.add_argument("--low-vram", action="store_true", help="Keep the text encoder on CPU and move it only around prompt encoding to reduce VRAM usage.")
    return parser.parse_args()



def main():
    args = parse_args()

    if torch.cuda.is_available():
        if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            precision = "bf16"
            print("Using bfloat16 (BF16)")
        else:
            dtype = torch.float16
            precision = "fp16"
            print("Using float16 (FP16)")
    else:
        # Fallback to float32 for CPU or if neither bf16 nor fp16 are suitable
        dtype = torch.float32
        precision=None
        print("Using float32 (FP32)")

    # 1. Initialize Accelerator
    accelerator = Accelerator(
        mixed_precision=precision,
        gradient_accumulation_steps=4,
    )
    
    # 2. Load DreamLite Pipeline
    pipe = DreamLitePipelineLoRA.from_pretrained(args.model_id, torch_dtype=dtype)
    
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet
    noise_scheduler = pipe.scheduler

    # Frozen other modules
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # 3. LoRA (Based on PEFT)
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        target_modules=[
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
        ],
    )
    unet = get_peft_model(unet, lora_config)
    
    # print
    unet.print_trainable_parameters()

    # 4. configure optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    
    # 5. prepare dataloader
    # =======================================================
    print("Loading dataset...")
    train_dataset = load_dataset(args.dataset_id, split=args.dataset_split)

    image_transforms = transforms.Compose([
        transforms.Resize(1024, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(1024),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]), # Normalize
    ])

    def preprocess_train(examples):
        target_imgs = []
        source_imgs = []
        source_imgs_pil = []
        
        # 1. 处理图片 (tar 列)
        for tar_item in examples["tar"]:
            # 情况 A: HF Dataset 已经自动把它解析成了 PIL Image 对象
            if hasattr(tar_item, "convert"):
                img = tar_item.convert("RGB")
            else:
                raise ValueError(f"无法识别的图像格式: {type(tar_item)}")
                
            target_imgs.append(image_transforms(img))

        for tar_item in examples["src"]:
            # 情况 A: HF Dataset 已经自动把它解析成了 PIL Image 对象
            if hasattr(tar_item, "convert"):
                img = tar_item.convert("RGB")
            else:
                raise ValueError(f"无法识别的图像格式: {type(tar_item)}")
                
            source_imgs_pil.append(img)
            source_imgs.append(image_transforms(img))
        
        # 2. 处理文本 (prompt 列)
        prompts = examples["prompt"]
        # prompts = [args.default_prompt] * len(examples["prompt"])
                
        return {
            "target_imgs": target_imgs,
            "source_imgs": source_imgs,
            "source_imgs_pil": source_imgs_pil,
            "prompt": prompts 
        }

    train_dataset.set_transform(preprocess_train)

    def collate_fn(examples):
        target_imgs = torch.stack([example["target_imgs"] for example in examples])
        source_imgs = torch.stack([example["source_imgs"] for example in examples])
        prompts = [example["prompt"] for example in examples]
        source_imgs_pil = [example["source_imgs_pil"] for example in examples]
        return {"target_imgs": target_imgs, "source_imgs": source_imgs, "source_imgs_pil": source_imgs_pil, "prompts": prompts}

    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=1,
    )

    # =======================================================

    # 6. Accelerator
    # unet, optimizer = accelerator.prepare(unet, optimizer)
    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

    vae.to(accelerator.device, dtype=dtype)
    text_encoder.eval()
    if args.low_vram:
        text_encoder.to("cpu")
    else:
        text_encoder.to(accelerator.device, dtype=dtype)

    # 7. Train
    global_step = 0
    progress_bar = tqdm(total=args.max_train_steps, disable=not accelerator.is_local_main_process)
    
    unet.train()
    
    while global_step < args.max_train_steps:
        # =======================================================
        # TODO: get data from DataLoader
        # for batch in dataloader:
        #     images = batch["pixel_values"]
        #     prompts = batch["text"]
        for batch in dataloader:
            if global_step >= args.max_train_steps:
                break
            images = batch['target_imgs'].to(accelerator.device, dtype=dtype)
            conds = batch['source_imgs'].to(accelerator.device, dtype=dtype)
            conds_pil = batch['source_imgs_pil'][0]
            prompts = batch['prompts']
        # =======================================================

            with accelerator.accumulate(unet):
                # 1. encode Latents (Ground Truth x_0)
                latents = vae.encode(images).latents
                latents = latents * vae.config.scaling_factor
                src_latents = vae.encode(conds).latents
                src_latents = src_latents * vae.config.scaling_factor

                # 2. noise and timestep
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                sigmas = torch.rand((bsz,), dtype=latents.dtype, device=latents.device)
                sigmas_expanded = sigmas.view(bsz, 1, 1, 1)

                timesteps = (sigmas * 1000.0).long() 

                # 3. Add noise to Latents
                noisy_latents = (1.0 - sigmas_expanded) * latents + sigmas_expanded * noise

                # 4. Encode Prompt
                if args.low_vram:
                    text_encoder.to(accelerator.device, dtype=dtype)
                    with torch.no_grad():
                        prompt_embeds, text_attention_mask = pipe.encode_prompt(
                            mode="edit",
                            image=conds_pil,
                            prompts=prompts,
                            device=accelerator.device,
                            dtype=dtype,
                        )
                    text_encoder.to("cpu")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    prompt_embeds, text_attention_mask = pipe.encode_prompt(
                        mode="edit",
                        image=conds_pil,
                        prompts=prompts,
                        device=accelerator.device,
                        dtype=dtype,
                    )

                # 5. Time IDs, Image Latents
                # Generate mode, condition image = 0
                model_input = torch.cat([noisy_latents, src_latents], dim=3) # In-context Concat
                
                add_time_ids = torch.tensor([[1024, 1024]], dtype=dtype, device=accelerator.device).repeat(bsz, 1)

                # 6. UNet Predict Noise
                noise_pred = unet(
                    model_input,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=text_attention_mask,
                    added_cond_kwargs={"time_ids": add_time_ids},
                    return_dict=False,
                )[0]
                
                noise_pred = noise_pred[..., :latents.shape[-1]]

                # 7. Loss (Flow Matching, MSE)
                target = noise - latents
                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                # 8. backward and update params
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(filter(lambda p: p.requires_grad, unet.parameters()), 1.0)
                
                optimizer.step()
                optimizer.zero_grad()

            # update
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                progress_bar.set_postfix({"loss": loss.item()})

    accelerator.wait_for_everyone()
    
    # 8. Save LoRA weights
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        os.makedirs(args.output_dir, exist_ok=True)
        unet.save_pretrained(args.output_dir)
        print(f"LoRA weights saved to {args.output_dir}")

if __name__ == "__main__":
    main()