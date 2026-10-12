"""Dataset evaluation using the same device-aware inference as inference.py."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from tqdm import tqdm

import utils
from inference import inference_one_im, model_init
from inference_runtime import (
    PROJECT_ROOT, IMAGE_SUFFIXES, add_runtime_arguments, read_image,
    validate_runtime_arguments, write_image,
)

ALL_DATASETS = {
    'dir300': 'dewarping', 'kligler': 'deshadowing', 'jung': 'deshadowing',
    'osr': 'deshadowing', 'docunet_docaligner': 'appearance',
    'realdae': 'appearance', 'tdd': 'deblurring', 'dibco18': 'binarization',
}


def get_args():
    parser = argparse.ArgumentParser(description='DocRes dataset evaluation')
    parser.add_argument('--model_path', type=str,
                        default=str(PROJECT_ROOT / 'checkpoints/docres.pkl'))
    parser.add_argument('--dataset', required=True, choices=ALL_DATASETS)
    parser.add_argument('--data_root', type=str, default=str(PROJECT_ROOT / 'data/eval'))
    add_runtime_arguments(parser)
    args = parser.parse_args()
    validate_runtime_arguments(parser, args)
    return args


def prediction_path(path):
    path = Path(path)
    return path.with_name(path.name.replace('_in.', '_docres.'))


def ground_truth_path(path):
    path = Path(path)
    return path.with_name(path.name.replace('_in.', '_gt.'))


def main():
    args = get_args()
    folder = Path(args.data_root) / args.dataset
    paths = sorted(path for path in folder.glob('*_in.*')
                   if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise ValueError(f'No *_in images in: {folder}')
    task = ALL_DATASETS[args.dataset]
    if task != 'dewarping':
        for path in paths:
            if not ground_truth_path(path).is_file():
                raise FileNotFoundError(f'Missing ground truth: {ground_truth_path(path)}')
    model = model_init(args)
    print(f'Device: {next(model.parameters()).device}; task: {task}')
    print('Predicting')
    for path in tqdm(paths):
        _, _, _, restored = inference_one_im(
            model, path, task, max_size=args.max_size, precision=args.precision,
            mbd_model_path=args.mbd_model_path,
        )
        write_image(prediction_path(path), restored)

    if task == 'dewarping':
        return
    print('Metric calculating')
    if task in ('deshadowing', 'appearance', 'deblurring'):
        psnr, ssim = [], []
        for path in tqdm(paths):
            pred = read_image(prediction_path(path))
            gt = read_image(ground_truth_path(path))
            ssim.append(structural_similarity(pred, gt, channel_axis=-1, data_range=255))
            psnr.append(peak_signal_noise_ratio(pred, gt, data_range=255))
        print(args.dataset)
        print('ssim:', np.mean(ssim))
        print('psnr:', np.mean(psnr))
    else:
        fmeasures, pfmeasures, psnrs = [], [], []
        for path in tqdm(paths):
            pred = cv2.cvtColor(read_image(prediction_path(path)), cv2.COLOR_BGR2GRAY)
            gt = cv2.cvtColor(read_image(ground_truth_path(path)), cv2.COLOR_BGR2GRAY)
            pred = np.where(pred > 155, 255, 0).astype(np.uint8)
            gt = np.where(gt > 155, 255, 0).astype(np.uint8)
            fmeasure, pfmeasure, psnr, _, _, _ = utils.bin_metric(pred, gt)
            fmeasures.append(fmeasure)
            pfmeasures.append(pfmeasure)
            psnrs.append(psnr)
        print(args.dataset)
        print('fmeasure:', np.mean(fmeasures))
        print('pfmeasure:', np.mean(pfmeasures))
        print('psnr:', np.mean(psnrs))


if __name__ == '__main__':
    main()
