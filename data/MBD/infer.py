"""Document mask inference, shared by DocRes and the standalone MBD CLI."""

import argparse
from functools import lru_cache
from pathlib import Path
import sys
import warnings

# Support both `python data/MBD/infer.py` and `python -m data.MBD.infer`.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'data.MBD'

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .model.deep_lab_model.deeplab import DeepLab
from .MBD import mask_base_dewarper
from .utils import cvimg2torch, torch2cvimg
from inference_runtime import PROJECT_ROOT, load_model_state, read_image, select_device, write_image


@lru_cache(maxsize=1)
def _load_segmentation_model(model_path, device_name):
    model = DeepLab(num_classes=1, backbone='resnet', output_stride=16,
                    sync_bn=False, freeze_bn=False)
    model.load_state_dict(load_model_state(model_path))
    return model.eval().to(device=device_name, dtype=torch.float32)


def get_segmentation_model(model_path, device='auto'):
    device = select_device(device)
    # Cache by absolute path and actual device, so auto and mps share one model.
    return _load_segmentation_model(str(Path(model_path).resolve()), str(device))


@torch.inference_mode()
def predict_mask(model, image):
    h, w = image.shape[:2]
    image = cv2.resize(image, (448, 448))
    image = cv2.GaussianBlur(image, (15, 15), 0, 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    parameter = next(model.parameters())
    tensor = cvimg2torch(image).to(device=parameter.device, dtype=parameter.dtype)
    prediction = model(tensor)[:, :1]
    if not torch.isfinite(prediction).all().item():
        raise RuntimeError('MBD produced non-finite values.')
    prediction = F.interpolate(prediction, (h, w))
    mask = prediction[0, 0].float().cpu().numpy()
    mask = (mask * 255).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    mask = cv2.erode(mask, kernel, iterations=3)
    mask[mask > 100] = 255
    mask[mask < 100] = 0
    return mask


def net1_net2_infer_single_im(img, model_path, device='auto'):
    return predict_mask(get_segmentation_model(model_path, device), img)


def net1_net2_infer(model, img_paths, args=None):
    model.eval()
    device = next(model.parameters()).device
    for img_path in tqdm(img_paths):
        img_path = Path(img_path)
        capture_path = img_path.with_name(img_path.name.replace('_origin', '_capture'))
        if capture_path.exists():
            continue
        image = read_image(img_path)
        h, w = image.shape[:2]
        mask = predict_mask(model, image)
        try:
            dewarped, grid = mask_base_dewarper(image, mask, device=device)
        except (ValueError, IndexError, cv2.error) as exc:
            warnings.warn(f'TPS failed for {img_path}: {exc}; using an identity grid.')
            grid = np.meshgrid(np.arange(w), np.arange(h)) / np.array([w, h]).reshape(2, 1, 1)
            grid = torch.from_numpy((grid - 0.5) * 2).float().unsqueeze(0).permute(0, 2, 3, 1)
            dewarped = torch2cvimg(F.grid_sample(cvimg2torch(image), grid, align_corners=False))[0]
            grid = grid[0].numpy()
        write_image(capture_path, dewarped)
        write_image(img_path.with_name(img_path.name.replace('_origin', '_mask_new')), mask)
        grid = cv2.resize(grid, (128, 128))
        np.save(img_path.with_name(img_path.name.replace('_origin', '_grid1')), grid)


def main():
    parser = argparse.ArgumentParser(description='MBD segmentation and TPS dewarping')
    parser.add_argument('--img_folder', type=str, default=str(PROJECT_ROOT / 'data/MBD/all_data'))
    parser.add_argument('--seg_model_path', type=str,
                        default=str(PROJECT_ROOT / 'data/MBD/checkpoint/mbd.pkl'))
    parser.add_argument('--device', choices=['auto', 'mps', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    paths = sorted(Path(args.img_folder).glob('*_origin.*'))
    if not paths:
        parser.error(f'No *_origin images in: {args.img_folder}')
    model = get_segmentation_model(args.seg_model_path, args.device)
    print(f'Device: {next(model.parameters()).device}')
    net1_net2_infer(model, paths, args)


if __name__ == '__main__':
    main()
