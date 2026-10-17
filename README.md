
<div align=center>

# DocRes: A Generalist Model Toward Unifying Document Image Restoration Tasks

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/qubvel-hf/documents-restoration)

</div>

<p align="center">
<img src="images/motivation.jpg" width="400">
</p>

This repository is a personal reproduction of the official implementation of [DocRes: A Generalist Model Toward Unifying Document Image Restoration Tasks](https://arxiv.org/abs/2405.04408).

The main modifications include support for mps backend.

See below for more details.

## Cross-backend Support

Use a native arm64 Python environment (Python 3.10 or newer) and install
`requirements.txt`. The standard PyTorch macOS wheels include MPS support.
The existing checkpoint files work without conversion.

```bash
python inference.py --device mps --im_path ./input/for_dewarping.png --task dewarping
python inference.py --device mps --im_path ./input/local/ --task end2end --out_folder ./restored/inference/
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
- The default output directory is `restored/inference/`; `--out_folder` can use
  any directory. End2end intermediate images are temporary
  and are removed automatically.

The same runtime options are available in `eval.py`. If an unsupported MPS operator
is encountered, `PYTORCH_ENABLE_MPS_FALLBACK=1` can be set before starting Python
to enable PyTorch's CPU fallback.

## Inference
1. Put MBD model weights [mbd.pkl](https://1drv.ms/f/s!Ak15mSdV3Wy4iahoKckhDPVP5e2Czw?e=iClwdK) to `./data/MBD/checkpoint/`
2. Put DocRes model weights [docres.pkl](https://1drv.ms/f/s!Ak15mSdV3Wy4iahoKckhDPVP5e2Czw?e=iClwdK) to `./checkpoints/`
3. Run the following script and the results will be saved in `./restored/inference/`. We have provided some distorted examples in `./input/`.
```bash
python inference.py --im_path ./input/for_dewarping.png --task dewarping --save_dtsprompt 1
```

- `--im_path`: required; the path of an input document image or directory
- `--task`: task that need to be executed, it must be one of _dewarping_, _deshadowing_, _appearance_, _deblurring_, _binarization_, or _end2end_
- `--save_dtsprompt`: whether to save the DTSPrompt

Keep your own inputs in `input/local/`, which is ignored by Git. Optional subdirectories
such as `dewarping/`, `deshadowing/` and `deblurring/` can help organize files by task.
Directory names do not select the task; use `--task` explicitly when choosing a task:

```bash
python inference.py --im_path input/local/deshadowing --task deshadowing
```

Both baseline and experimental inference require `--im_path`; neither selects a
default input folder. Example images shipped with the repository remain in `input/`.

## Project Structure

The root scripts provide baseline inference, evaluation and training. Each experiment
has its own directory under `experiments/`; inference and experiment outputs share
the local `restored/` directory.

```text
inference.py                     # Baseline inference entry point
inference_runtime.py             # Shared device, checkpoint and image I/O helpers
eval.py                          # Baseline dataset evaluation
train.py                         # Baseline training entry point
models/                          # Baseline model implementation
loaders/                         # Training data loaders
data/                            # MBD implementation, preprocessing and dataset instructions
  MBD/checkpoint/                # Downloaded MBD weights (local)
checkpoints/                     # Downloaded DocRes weights (local)
input/                           # Versioned example inputs
  local/                         # User-provided inputs for any task (Git-ignored)
experiments/
  RESEARCH_NOTES.md               # Research hypotheses and experiment plans
  dewarp_smoothing/
    inference.py                 # Independent smoothing experiment entry point
    README.md                    # Experiment details and commands
restored/                        # Generated outputs (local)
  inference/                     # Default inference outputs
    legacy/                      # Archived example outputs
  debug/legacy/                  # Archived debugging outputs
  dewarp_smoothing/
    legacy/                      # Archived exploratory outputs
    <timestamp>/                 # New runs, including run.json
  migration.json                 # Local file-migration paths and SHA-256 hashes
```

### Independent experiments

Keep experiment-specific changes in `experiments/<experiment_name>/`. Copy the
functions that need to change and reuse unchanged helpers from the baseline.
The root inference script processes each input once and retains the baseline
dewarping postprocessing: a 3×3 blur applied 15 times.

Run experimental entry points as modules from the repository root:

```bash
python -m experiments.dewarp_smoothing.inference \
  --im_path input/189_origin.png --device mps \
  --kernel_sizes 3 5 20 --smooth_times 0 1 5 15 30
```

See the [smoothing experiment instructions](experiments/dewarp_smoothing/README.md)
and [research notes](experiments/RESEARCH_NOTES.md) for details. The smoothing
experiment predicts one coordinate map per image and applies each setting to a
separate copy. Each run gets a new output directory and records its parameters,
input paths, weight paths, device and Git state in `run.json`.

For future training experiments, add a local `train.py` to the corresponding
experiment directory. If the model changes, add a local `model.py` shared by that
experiment's training and inference entry points. Calls to baseline functions still
use baseline implementations internally; explicitly connect experimental functions
in the experiment's own pipeline.

### Output consolidation

The previous mixed-purpose directories have been consolidated as follows:

| Previous location | Current location |
| --- | --- |
| `tweaks/01_explicit_task_conditioning_and_output_heads.md` | `experiments/RESEARCH_NOTES.md` |
| `tweaks/dewarp_smooth_times/` | `restored/dewarp_smoothing/legacy/` |
| `restored_debug/` | `restored/debug/legacy/` |
| `restorted/` (previous misspelling) | `restored/inference/legacy/` |

New baseline outputs default to `restored/inference/`. New smoothing runs default
to `restored/dewarp_smoothing/<UTC timestamp>/`; `--out_folder` can override either
location. The experiment requires a new directory to avoid overwriting earlier runs.

Code and research notes remain versioned. Generated images, prompts and run records
under `restored/` are ignored by Git. Previously tracked output images have been
removed from version control and retained locally in the archive directories;
fresh clones do not include these images. Local inputs and downloaded weights are
also ignored. Dataset evaluation continues to write predictions next to its inputs
under `--data_root`, as described in the evaluation workflow.

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
