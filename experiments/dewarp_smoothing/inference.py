"""Run with python -m experiments.dewarp_smoothing.inference from the repo root."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
import torch

import inference as baseline
from inference_runtime import PROJECT_ROOT, image_paths, read_image

# Copied from inference.py::dewarping at this revision, then split locally so
# every smoothing variant can reuse the same prediction. Keep experiment edits here.
BASELINE_COMMIT = '134b614760893d33ea368bf7ac1fe87cde56592f'


@torch.inference_mode()
def predict_dewarping(model, im_path, mbd_model_path=baseline.DEFAULT_MBD_PATH):
    """Return the original image, uint8 prompts and unsmoothed normalized map."""
    input_size = 256
    device = next(model.parameters()).device
    im_org = read_image(im_path)
    im_masked, prompt_org = baseline.dewarp_prompt(im_org.copy(), device, mbd_model_path)

    im_masked = cv2.resize(im_masked.copy(), (input_size, input_size)) / 255.0
    im_masked = torch.from_numpy(im_masked.transpose(2, 0, 1)).unsqueeze(0)
    im_masked = im_masked.to(device=device, dtype=torch.float32)
    prompt = torch.from_numpy(prompt_org.transpose(2, 0, 1)).unsqueeze(0)
    prompt = prompt.to(device=device, dtype=torch.float32)
    in_im = torch.cat((im_masked, prompt), dim=1)

    base_coord = baseline.utils.getBasecoord(input_size, input_size) / input_size
    pred = model.float()(in_im)
    if not torch.isfinite(pred).all().item():
        raise RuntimeError('Dewarping produced non-finite values.')
    pred = pred[0][:2].permute(1, 2, 0).cpu().numpy() + base_coord

    prompt_org = (prompt_org * 255).astype(np.uint8)
    prompt_org = cv2.resize(prompt_org, im_org.shape[:2][::-1])
    return im_org, prompt_org, pred


def render_dewarping(image, backward_map, *, smooth_times=15, kernel_size=20):
    """Apply one experiment setting without changing the cached prediction."""
    if smooth_times < 0 or kernel_size < 1:
        raise ValueError('smooth_times must be >= 0 and kernel_size must be >= 1.')
    pred = backward_map.copy()
    for _ in range(smooth_times):
        pred = cv2.blur(pred, (kernel_size, kernel_size), borderType=cv2.BORDER_REPLICATE)
    h, w = image.shape[:2]
    pred = (cv2.resize(pred, (w, h)) * (w, h)).astype(np.float32)
    return cv2.remap(image, pred[:, :, 0], pred[:, :, 1], cv2.INTER_LINEAR)


def dewarping(model, im_path, mbd_model_path=baseline.DEFAULT_MBD_PATH, *,
              smooth_times=15, kernel_size=20):
    """Single-setting entry point with the same result shape as the baseline."""
    image, prompts, backward_map = predict_dewarping(model, im_path, mbd_model_path)
    restored = render_dewarping(image, backward_map,
                               smooth_times=smooth_times, kernel_size=kernel_size)
    return prompts[:, :, 0], prompts[:, :, 1], prompts[:, :, 2], restored


def get_args(argv=None):
    parser = argparse.ArgumentParser(description='Independent dewarping smoothing sweep')
    parser.add_argument('--im_path', default=str(PROJECT_ROOT / 'input/distorted'))
    parser.add_argument('--model_path', default=str(PROJECT_ROOT / 'checkpoints/docres.pkl'))
    parser.add_argument('--mbd_model_path', default=str(baseline.DEFAULT_MBD_PATH))
    parser.add_argument('--device', choices=['auto', 'mps', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--out_folder', help='New run directory; defaults to restored/dewarp_smoothing/<timestamp>')
    parser.add_argument('--smooth_times', type=int, nargs='+', default=[0, 1, 3, 5, 10, 15, 30, 60, 100])
    parser.add_argument('--kernel_sizes', type=int, nargs='+', default=[20])
    parser.add_argument('--save_dtsprompt', action='store_true')
    args = parser.parse_args(argv)
    if any(value < 0 for value in args.smooth_times):
        parser.error('--smooth_times values must be >= 0.')
    if any(value < 1 for value in args.kernel_sizes):
        parser.error('--kernel_sizes values must be >= 1.')
    args.smooth_times = list(dict.fromkeys(args.smooth_times))
    args.kernel_sizes = list(dict.fromkeys(args.kernel_sizes))
    return args


def code_version():
    """Read provenance without changing the repository."""
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, text=True).strip()
        status = subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=PROJECT_ROOT, text=True).strip()
        return {'commit': commit, 'dirty': bool(status), 'status': status}
    except (OSError, subprocess.CalledProcessError):
        return {'commit': None, 'dirty': None}


def main(argv=None):
    args = get_args(argv)
    paths = image_paths(args.im_path)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out_folder = (Path(args.out_folder) if args.out_folder else
                  PROJECT_ROOT / 'restored/dewarp_smoothing' / run_id)
    # Each invocation owns a fresh directory, so previous results cannot be overwritten.
    out_folder.mkdir(parents=True, exist_ok=False)
    model = baseline.model_init(args)
    device = str(next(model.parameters()).device)
    manifest = {
        'experiment': 'dewarp_smoothing', 'baseline_commit': BASELINE_COMMIT,
        'code': code_version(),
        'arguments': {**vars(args), 'out_folder': str(out_folder.resolve())},
        'inputs': [str(path.resolve()) for path in paths],
        'model_path': str(Path(args.model_path).resolve()),
        'mbd_model_path': str(Path(args.mbd_model_path).resolve()),
        'device': device, 'precision': 'fp32', 'torch_version': str(torch.__version__),
    }
    (out_folder / 'run.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Device: {device}; precision: fp32; output: {out_folder}')
    for path in paths:
        image, prompts, backward_map = predict_dewarping(model, path, args.mbd_model_path)
        # Keep original filenames as directories to distinguish e.g. page.jpg/page.png.
        image_folder = out_folder / path.name
        for kernel_size in args.kernel_sizes:
            for smooth_times in args.smooth_times:
                restored = render_dewarping(image, backward_map,
                                           smooth_times=smooth_times, kernel_size=kernel_size)
                label = f'dewarpingx{kernel_size}_{smooth_times:02d}'
                saved = baseline.save_results(
                    path.with_suffix('.png'), image_folder, label, args.save_dtsprompt,
                    restored, (prompts[:, :, 0], prompts[:, :, 1], prompts[:, :, 2]),
                )
                print(f'Saved: {saved}')
        baseline.release_mps_cache(model)


if __name__ == '__main__':
    main()
