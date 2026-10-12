
<div align=center>

# DocRes: A Generalist Model Toward Unifying Document Image Restoration Tasks

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/qubvel-hf/documents-restoration)

</div>

<p align="center">
<img src="images/motivation.jpg" width="400">
</p>

This is the official implementation of our paper [DocRes: A Generalist Model Toward Unifying Document Image Restoration Tasks](https://arxiv.org/abs/2405.04408).

## News 
🔥 [2025.7] Our paper ["Aesthetics is Cheap, Show me the Text: An Empirical Evaluation of State-of-the-Art Generative Models for OCR"](https://arxiv.org/abs/2507.15085), which conducts a comprehensive evaluation of SOTA generative models has been online at arXiv. 🔥

🔥 [2025.6] Beyond GPT-4o, we evaluate more SOTA generative models' image generation abilities in various document processing tasks. Check [here](https://github.com/NiceRingNode/Awesome-Image-Generators-for-OCR-Image-Generation-and-Editing)! 🔥

🎉 [2025.5] We evaluate the image generation ability of GPT-4o, including various document processing tasks. Check [here](https://github.com/NiceRingNode/Awesome-Image-Generators-for-OCR-Image-Generation-and-Editing)! 🔥

🎉 [2025.2] Our new work [LGGPT](https://github.com/NiceRingNode/LGGPT) has been accepted to IJCV 2025, an LLM that unifies versatile layout generation tasks! Welcome to follow!

🔥 A comprehensive [Recommendation for Document Image Processing](https://github.com/ZZZHANG-jx/Recommendations-Document-Image-Processing) is available.


## Inference 
1. Put MBD model weights [mbd.pkl](https://1drv.ms/f/s!Ak15mSdV3Wy4iahoKckhDPVP5e2Czw?e=iClwdK) to `./data/MBD/checkpoint/`
2. Put DocRes model weights [docres.pkl](https://1drv.ms/f/s!Ak15mSdV3Wy4iahoKckhDPVP5e2Czw?e=iClwdK) to `./checkpoints/`
3. Run the following script and the results will be saved in `./restorted/`. We have provided some distorted examples in `./input/`.
```bash
python inference.py --im_path ./input/for_dewarping.png --task dewarping --save_dtsprompt 1
```

- `--im_path`: the path of input document image
- `--task`: task that need to be executed, it must be one of _dewarping_, _deshadowing_, _appearance_, _deblurring_, _binarization_, or _end2end_
- `--save_dtsprompt`: whether to save the DTSPrompt

### Apple Silicon / MPS

Use a native arm64 Python environment (Python 3.10 or newer) and install
`requirements.txt`. The standard PyTorch macOS wheels include MPS support.
The existing checkpoint files work without conversion.

```bash
python inference.py --device mps --im_path ./input/for_dewarping.png --task dewarping
python inference.py --device mps --im_path ./distorted/ --task end2end --out_folder ./restored/
```

- `--device auto` (default) selects CUDA, MPS, then CPU according to availability.
- `--precision auto` (default) uses FP16 for restoration on GPU and FP32 on CPU.
  Dewarping and MBD segmentation always use FP32. Use `--precision fp32` to
  compare precision or troubleshoot numerical issues.
- `--max_size` controls the square inference size for appearance enhancement
  and deshadowing. The default is `1024` on MPS to fit Apple Silicon laptops,
  and `1600` on CPU/CUDA. Set `--max_size 1600` to reproduce the original size,
  or use a smaller multiple of 8 to reduce memory use further. Deblurring and
  binarization retain the original image resolution.
- `--mbd_model_path` overrides the segmentation checkpoint path. MBD is loaded
  only for dewarping/end2end and reused across images.
- Default resource paths are resolved relative to the project. Explicit relative
  paths are resolved relative to your current working directory.
- The original output directory spelling `restorted/` is retained for compatibility;
  `--out_folder` can use any directory. End2end intermediate images are temporary
  and are removed automatically.

The same runtime options are available in `eval.py`. If an unsupported MPS operator
is encountered, `PYTORCH_ENABLE_MPS_FALLBACK=1` can be set before starting Python
to enable PyTorch's CPU fallback.

## Evaluation

1. Dataset preparation, see [dataset instruction](./data/README.md)
2. Put MBD model weights [mbd.pkl](https://1drv.ms/f/s!Ak15mSdV3Wy4iahoKckhDPVP5e2Czw?e=iClwdK) to `data/MBD/checkpoint/`
3. Put DocRes model weights [docres.pkl](https://1drv.ms/f/s!Ak15mSdV3Wy4iahoKckhDPVP5e2Czw?e=iClwdK) to `./checkpoints/`
2. Run the following script
```bash
python eval.py --dataset realdae
```
- `--dataset`: dataset that need to be evaluated, it can be set as _dir300_, _kligler_, _jung_, _osr_, _docunet\_docaligner_, _realdae_, _tdd_, and _dibco18_.

## Training 
1. Dataset preparation, see [dataset instruction](./data/README.md)
2. Specify the datasets_setting within `train.py` based on your dataset path and experimental setting.
3. Run the following script
```bash
bash start_train.sh
```


## Citation
```
@inproceedings{zhangdocres2024, 
Author = {Jiaxin Zhang, Dezhi Peng, Chongyu Liu , Peirong Zhang and Lianwen Jin}, 
Booktitle = {In Proceedings of the IEEE/CV Conference on Computer Vision and Pattern Recognition}, 
Title = {{DocRes: A Generalist Model Toward Unifying Document Image Restoration Tasks}}, 
Year = {2024}}   
```
## ⭐ Star Rising
[![Star Rising](https://api.star-history.com/svg?repos=ZZZHANG-jx/DocRes&type=Timeline)](https://star-history.com/#ZZZHANG-jx/DocRes&Timeline)
