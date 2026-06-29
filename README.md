# DreamLite: Edit LoRA Trainer

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saurabhv749/dreamlite-lora/blob/main/demo.ipynb)

Minimal training and inference scripts for [DreamLite](https://github.com/ByteVisionLab/DreamLite) Edit LoRA on consumer GPU.

## Requirements

This repo is meant to work with the original DreamLite repository.

- Install the original DreamLite repo first.
- The training script expects the huggingface dataset to include `tar`, `src`, and `prompt` fields.
- VRAM usage ~7.5GB


## Train

see [demo.ipynb](./demo.ipynb) for training example.

```bash
python train_edit_lora.py \
  --model_id models/DreamLite-base \
  --dataset_id showlab/OmniConsistency \
  --dataset_split Snoopy \
  --output_dir ./output/output_lora/Snoopy
```

## Inference

Edit [infer.py](infer.py) with your `image_path`, `prompt`, and `lora_path`, then run:

```bash
python infer.py
```
