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
import torch
from safetensors.torch import load_file
from dreamlite import DreamLitePipelineLoRA
# from diffusers.utils import load_image
from PIL import Image
from peft import PeftModel

pipe = DreamLitePipelineLoRA.from_pretrained("models/DreamLite-base", torch_dtype=torch.bfloat16).to("cuda")
os.makedirs("lora-outputs", exist_ok=True)

image_path = ""
prompt = ""
lora_path = ""
steps=28

# 1. Load LoRA Weights
input_image = Image.open(image_path)  # for Edit lora
w,h = input_image.size

print(f"Injecting LoRA weights from {lora_path}...")
pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)

# 3. Inference
image = pipe(
    prompt=prompt, 
    image=input_image,
    num_inference_steps=steps,
    image_guidance_scale=1.5
).images[0]

# (1024, 1024) -> image ar
image.save("lora-outputs/image.jpg")
resized_image = image.resize((w,h))
resized_image.save("lora-outputs/image-resized.jpg")