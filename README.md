# Multimodal Distribution Matching for Vision-Language Dataset Distillation (CVPR 2026)

Official implementation of Multimodal Distribution Matching for Vision-Language Dataset Distillation,a method for condensing large vision-language datasets into smaller synthetic sets while preserving training efficacy.

[Paper](docs/paper.pdf)

## About

Experts: [buffers](https://drive.google.com/drive/folders/1Q9etol246RjeB_XZ3on5aEGOirW4o1aB?usp=sharing).

```text
.
├── distill_mdm.py       # Main dataset distillation entry point
├── eval.py              # Retrieval evaluation on distilled checkpoints
├── src/                 
│   ├── clustering_utils.py
│   ├── epoch.py
│   ├── geo_utils.py
│   ├── model.py
│   ├── model_utils.py
│   ├── networks.py
│   ├── reparam_module.py
│   ├── similarity_mining.py
│   ├── utils.py
│   └── vl_distill_utils.py
├── utils/               
├── sh/
│   ├── distill.sh       
│   └── eval.sh           
```

## Dataset

```text
data/
├── datasets/
│   ├── Flickr30k/
│   ├── Flickr8k/
│   └── COCO/
└── annotations/
    ├── flickr30k/
    ├── flickr8k/
    └── coco/
```

Defaults in `distill_mdm.py` include:

image roots such as `./data/datasets/Flickr30k/` and annotation root `./data/annotations/` when using the `flickr` / `flickr8k` / `coco` options.

## Dependencies and Usage

```bash
conda env create -f mdm.yml
conda activate mdm
```

### Distillation
```bash
export CKPT_PATH=/path/to/distilled.pt
./sh/distill.sh <gpu_id> [run_name]
```

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{jeong2026mdm,
  title={Multimodal Distribution Matching for Vision-Language Dataset Distillation},
  author={Jeong, Jongoh and Kwon, Hoyong and Kim, Minseok and Yoon, Kuk-Jin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
