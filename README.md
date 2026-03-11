# Multimodal Distribution Matching for Vision-Language Dataset Distillation (CVPR 2026)

This repository contains the implementation for **Multimodal Distribution Matching for Vision-Language Dataset Distillation**, a method for condensing large vision-language datasets into smaller, synthetic datasets while preserving training efficacy.

## 📊 Dataset Structure

The project expects datasets to be organized as follows:

```
data/
├── datasets/
│   ├── Flickr30k/          # Flickr30k images
│   ├── Flickr8k/           # Flickr8k images
│   └── COCO/               # MS-COCO images
└── annotations/             # Annotation files (captions, etc.)
    ├── flickr30k/
    ├── flickr8k/
    └── coco/
```

## 🚀 Usage

### Training (Distillation)

Train a distilled dataset using the main script:

```bash
python distill_mdm.py \
    --dataset flickr8k \
    --buffer_path ./buffer/flickr8k/nfnet_bert/InfoNCE \
    --num_queries 100 \
    --batch_size_train 64 \
    --batch_size_test 64 \
    --lr_txt 100 \
    --lr_img 100 \
    --Iteration 3000 \
    --image_encoder nfnet \
    --text_encoder bert \
    --w_nce 1.0 \
    --w_sph_u_mmd 0.8 \
    --w_sph_g_mmd 0.8 \
    --wandb \
    --wandb_project MDD_CVPR2026_Submission
```

**Key parameters:**

- `--dataset`: Dataset to use (`flickr8k`, `flickr30k`, `coco`, or `flickr`)
- `--num_queries`: Number of synthetic image-text pairs to generate
- `--image_encoder`: Image encoder architecture (`nfnet`, `nf_resnet50`, `nf_regnet`, `vit`)
- `--text_encoder`: Text encoder (`bert`, `distilbert`, `clip`)
- `--buffer_path`: Path to teacher model buffers (required)
- `--w_nce`, `--w_sph_u_mmd`, `--w_sph_g_mmd`: Loss weights for InfoNCE and spherical MMD losses

**Example shell scripts:**

```bash
# Run Flickr8k distillation with 100 queries
bash sh/run_distill_final_flickr8k_100.sh <gpu_id> <experiment_name>

```

### Evaluation

Evaluate distilled data on downstream tasks:

```bash
python eval.py \
    --dataset flickr8k \
    --num_eval 1 \
    --ckpt_path ./logs/flickr8k_100_distilled.pt \
    --loss_type WBCE \
    --image_encoder nf_resnet50 \
    --text_encoder bert \
    --batch_train 64
```

```bash

# Run cross-architecture experiments
bash sh/run_crossarch_8k100.sh

```

**Evaluation metrics:**

- Image Retrieval: R@1, R@5, R@10
- Text Retrieval: R@1, R@5, R@10
- Mean retrieval score



## Star History
<picture>
  <source
    media="(prefers-color-scheme: dark)"
    srcset="
      https://api.star-history.com/svg?repos=andyj1/mdm&type=Date&theme=dark
    "
  />
  <source
    media="(prefers-color-scheme: light)"
    srcset="
      https://api.star-history.com/svg?repos=andyj1/mdm&type=Date
    "
  />
  <img
    alt="Star History Chart"
    src="https://api.star-history.com/svg?repos=andyj1/mdm&type=Date"
  />
</picture>

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@article{jeong2026mdm,
  title={Multimodal Distribution Matching for Vision-Language Dataset Distillation},
  author={},
  journal={},
  year={},
  note={}
}
```

_Note: Please update the citation with the actual publication details._

## 🙏 Acknowledgements

We thank the authors and contributors of:
VL-Distill, LoRS-Distill for disclosing their codes for open research.

---

For questions or issues, please open an issue on the repository.
