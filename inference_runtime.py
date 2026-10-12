"""Device, checkpoint and image I/O helpers shared by inference entry points."""

from pathlib import Path

import cv2
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def select_device(name='auto'):
    if name == 'auto':
        name = ('cuda' if torch.cuda.is_available() else
                'mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device(name)
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable in this Python environment; use --device cpu.')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this Python environment.')
    # Tensor.device reports mps:0, while CLI/auto commonly supplies mps.
    # Canonicalize both so the segmentation cache does not reload the same GPU.
    if device.type == 'mps':
        device = torch.device('mps')
    elif device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    return device


def inference_dtype(device, precision='auto'):
    if precision == 'auto':
        return torch.float32 if device.type == 'cpu' else torch.float16
    return {'fp32': torch.float32, 'fp16': torch.float16}[precision]


def default_max_size(device):
    # 1600 can exhaust the working set of a 24 GB Apple Silicon laptop.
    return 1024 if device.type == 'mps' else 1600


def load_model_state(path):
    # Keep optimizer tensors in CPU memory and discard them after extracting weights.
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    state = checkpoint['model_state']
    return {key.removeprefix('module.'): value for key, value in state.items()}


def read_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f'Cannot read image: {path}')
    return image


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f'Cannot write image: {path}')


def image_paths(source):
    source = Path(source)
    if source.is_dir():
        paths = sorted(path for path in source.iterdir()
                       if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        if not paths:
            raise ValueError(f'No supported images in: {source}')
        return paths
    if not source.is_file():
        raise FileNotFoundError(f'Input image or directory does not exist: {source}')
    return [source]


def add_runtime_arguments(parser):
    parser.add_argument('--device', choices=['auto', 'mps', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--precision', choices=['auto', 'fp32', 'fp16'], default='auto',
                        help='Restoration precision; auto uses FP16 on GPU and FP32 on CPU. Dewarping uses FP32.')
    parser.add_argument('--max_size', type=int, default=None,
                        help='Appearance/deshadowing size, a multiple of 8; default: 1024 on MPS, 1600 otherwise.')
    parser.add_argument('--mbd_model_path', type=str,
                        default=str(PROJECT_ROOT / 'data/MBD/checkpoint/mbd.pkl'))


def validate_runtime_arguments(parser, args):
    if args.max_size is None:
        args.max_size = default_max_size(select_device(args.device))
    if args.max_size < 8 or args.max_size % 8:
        parser.error('--max_size must be a positive multiple of 8 (at least 8).')
