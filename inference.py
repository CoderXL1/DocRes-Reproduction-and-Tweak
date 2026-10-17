import cv2
from pathlib import Path
from tempfile import TemporaryDirectory
import utils
import argparse
import numpy as np

import torch

from models import restormer_arch
from data.preprocess.crop_merge_image import stride_integral
from data.MBD.infer import net1_net2_infer_single_im
from inference_runtime import (
    PROJECT_ROOT, add_runtime_arguments, default_max_size, image_paths, inference_dtype,
    load_model_state, read_image, select_device, validate_runtime_arguments, write_image,
)

TASKS = ['dewarping', 'deshadowing', 'appearance', 'deblurring', 'binarization', 'end2end']
DEFAULT_MBD_PATH = PROJECT_ROOT / 'data/MBD/checkpoint/mbd.pkl'


def release_mps_cache(model):
    if next(model.parameters()).device.type == 'mps':
        torch.mps.empty_cache()


@torch.inference_mode()
def forward_image(model, image, precision='auto'):
    device = next(model.parameters()).device
    dtype = inference_dtype(device, precision)
    # NumPy division otherwise produces float64, which cannot be copied to MPS.
    image = image.astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0)
    tensor = tensor.to(device=device, dtype=dtype)
    prediction = model.to(dtype=dtype)(tensor)
    if not torch.isfinite(prediction).all().item():
        raise RuntimeError('Model produced non-finite values; try --precision fp32 or a smaller --max_size.')
    return prediction


def dewarp_prompt(img, device='auto', mbd_model_path=DEFAULT_MBD_PATH):
    mask = net1_net2_infer_single_im(img, mbd_model_path, device=device)
    base_coord = utils.getBasecoord(256,256)/256
    img[mask==0]=0
    mask = cv2.resize(mask,(256,256))/255
    return img,np.concatenate((base_coord,np.expand_dims(mask,-1)),-1)

def deshadow_prompt(img):
    h,w = img.shape[:2]
    # img = cv2.resize(img,(128,128))
    img = cv2.resize(img,(1024,1024))
    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []
    bg_imgs = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        bg_imgs.append(bg_img)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
    bg_imgs = cv2.merge(bg_imgs)
    bg_imgs = cv2.resize(bg_imgs,(w,h))
    # result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    result_norm[result_norm==0]=1
    shadow_map = np.clip(img.astype(float)/result_norm.astype(float)*255,0,255).astype(np.uint8)
    shadow_map = cv2.resize(shadow_map,(w,h))
    shadow_map = cv2.cvtColor(shadow_map,cv2.COLOR_BGR2GRAY)
    shadow_map = cv2.cvtColor(shadow_map,cv2.COLOR_GRAY2BGR)
    # return shadow_map
    return bg_imgs

def deblur_prompt(img):
    x = cv2.Sobel(img,cv2.CV_16S,1,0)  
    y = cv2.Sobel(img,cv2.CV_16S,0,1)  
    absX = cv2.convertScaleAbs(x)   # 转回uint8  
    absY = cv2.convertScaleAbs(y)  
    high_frequency = cv2.addWeighted(absX,0.5,absY,0.5,0)
    high_frequency = cv2.cvtColor(high_frequency,cv2.COLOR_BGR2GRAY)
    high_frequency = cv2.cvtColor(high_frequency,cv2.COLOR_GRAY2BGR)
    return high_frequency

def appearance_prompt(img):
    h,w = img.shape[:2]
    # img = cv2.resize(img,(128,128))
    img = cv2.resize(img,(1024,1024))
    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
    result_norm = cv2.merge(result_norm_planes)
    result_norm = cv2.resize(result_norm,(w,h))
    return result_norm

def binarization_promptv2(img):
    result,thresh = utils.SauvolaModBinarization(img)
    thresh = thresh.astype(np.uint8)
    result[result>155]=255
    result[result<=155]=0

    x = cv2.Sobel(img,cv2.CV_16S,1,0)  
    y = cv2.Sobel(img,cv2.CV_16S,0,1)  
    absX = cv2.convertScaleAbs(x)   # 转回uint8  
    absY = cv2.convertScaleAbs(y)  
    high_frequency = cv2.addWeighted(absX,0.5,absY,0.5,0)
    high_frequency = cv2.cvtColor(high_frequency,cv2.COLOR_BGR2GRAY)
    return np.concatenate((np.expand_dims(thresh,-1),np.expand_dims(high_frequency,-1),np.expand_dims(result,-1)),-1)

def dewarping(model,im_path,mbd_model_path=DEFAULT_MBD_PATH):
    INPUT_SIZE=256
    device = next(model.parameters()).device
    im_org = read_image(im_path)
    im_masked, prompt_org = dewarp_prompt(im_org.copy(), device, mbd_model_path)

    h,w = im_masked.shape[:2]
    im_masked = im_masked.copy()
    im_masked = cv2.resize(im_masked,(INPUT_SIZE,INPUT_SIZE))
    im_masked = im_masked / 255.0
    im_masked = torch.from_numpy(im_masked.transpose(2,0,1)).unsqueeze(0)
    im_masked = im_masked.to(device=device, dtype=torch.float32)

    prompt = torch.from_numpy(prompt_org.transpose(2,0,1)).unsqueeze(0)
    prompt = prompt.to(device=device, dtype=torch.float32)

    in_im = torch.cat((im_masked,prompt),dim=1)

    # inference
    base_coord = utils.getBasecoord(INPUT_SIZE,INPUT_SIZE)/INPUT_SIZE
    model = model.float()
    with torch.no_grad():
        pred = model(in_im)
        if not torch.isfinite(pred).all().item():
            raise RuntimeError('Dewarping produced non-finite values.')
        pred = pred[0][:2].permute(1,2,0).cpu().numpy()
        pred = pred+base_coord
    ## smooth
    for i in range(15):
        pred = cv2.blur(pred,(3,3),borderType=cv2.BORDER_REPLICATE)
    pred = cv2.resize(pred,(w,h))*(w,h)
    pred = pred.astype(np.float32)
    out_im = cv2.remap(im_org,pred[:,:,0],pred[:,:,1],cv2.INTER_LINEAR)

    prompt_org = (prompt_org*255).astype(np.uint8)
    prompt_org = cv2.resize(prompt_org,im_org.shape[:2][::-1])

    return prompt_org[:,:,0],prompt_org[:,:,1],prompt_org[:,:,2],out_im

def appearance(model,im_path,max_size=None,precision='auto'):
    MAX_SIZE = max_size if max_size is not None else default_max_size(next(model.parameters()).device)
    # obtain im and prompt
    im_org = read_image(im_path)
    h,w = im_org.shape[:2]
    prompt = appearance_prompt(im_org)
    in_im = np.concatenate((im_org,prompt),-1)

    # constrain the max resolution 
    if max(w,h) < MAX_SIZE:
        in_im,padding_h,padding_w = stride_integral(in_im,8)
    else:
        in_im = cv2.resize(in_im,(MAX_SIZE,MAX_SIZE))
    
    with torch.no_grad():
        pred = forward_image(model, in_im, precision)
        pred = torch.clamp(pred,0,1)
        pred = pred[0].permute(1,2,0).float().cpu().numpy()
        pred = (pred*255).astype(np.uint8)

        if max(w,h) < MAX_SIZE:
            out_im = pred[padding_h:,padding_w:]
        else:
            pred[pred==0] = 1
            shadow_map = cv2.resize(im_org,(MAX_SIZE,MAX_SIZE)).astype(float)/pred.astype(float)
            shadow_map = cv2.resize(shadow_map,(w,h))
            shadow_map[shadow_map==0]=0.00001
            out_im = np.clip(im_org.astype(float)/shadow_map,0,255).astype(np.uint8)

    return prompt[:,:,0],prompt[:,:,1],prompt[:,:,2],out_im
        

def deshadowing(model,im_path,max_size=None,precision='auto'):
    MAX_SIZE = max_size if max_size is not None else default_max_size(next(model.parameters()).device)
    # obtain im and prompt
    im_org = read_image(im_path)
    h,w = im_org.shape[:2]
    prompt = deshadow_prompt(im_org)
    in_im = np.concatenate((im_org,prompt),-1)

    # constrain the max resolution 
    if max(w,h) < MAX_SIZE:
        in_im,padding_h,padding_w = stride_integral(in_im,8)
    else:
        in_im = cv2.resize(in_im,(MAX_SIZE,MAX_SIZE))
    
    with torch.no_grad():
        pred = forward_image(model, in_im, precision)
        pred = torch.clamp(pred,0,1)
        pred = pred[0].permute(1,2,0).float().cpu().numpy()
        pred = (pred*255).astype(np.uint8)

        if max(w,h) < MAX_SIZE:
            out_im = pred[padding_h:,padding_w:]
        else:
            pred[pred==0]=1
            shadow_map = cv2.resize(im_org,(MAX_SIZE,MAX_SIZE)).astype(float)/pred.astype(float)
            shadow_map = cv2.resize(shadow_map,(w,h))
            shadow_map[shadow_map==0]=0.00001
            out_im = np.clip(im_org.astype(float)/shadow_map,0,255).astype(np.uint8)

    return prompt[:,:,0],prompt[:,:,1],prompt[:,:,2],out_im


def deblurring(model,im_path,precision='auto'):
    # setup image
    im_org = read_image(im_path)
    in_im,padding_h,padding_w = stride_integral(im_org,8)
    prompt = deblur_prompt(in_im)
    in_im = np.concatenate((in_im,prompt),-1)
    with torch.no_grad():
        pred = forward_image(model, in_im, precision)
        pred = torch.clamp(pred,0,1)
        pred = pred[0].permute(1,2,0).float().cpu().numpy()
        pred = (pred*255).astype(np.uint8)
        out_im = pred[padding_h:,padding_w:]
    
    return prompt[:,:,0],prompt[:,:,1],prompt[:,:,2],out_im



def binarization(model,im_path,precision='auto'):
    im_org = read_image(im_path)
    im,padding_h,padding_w = stride_integral(im_org,8)
    prompt = binarization_promptv2(im)
    h,w = im.shape[:2]
    in_im = np.concatenate((im,prompt),-1)

    with torch.no_grad():
        pred = forward_image(model, in_im, precision)
        pred = pred[:,:2,:,:]
        pred = torch.max(torch.softmax(pred,1),1)[1]  # torch.max()[1] stands for argmax
        pred = pred[0].cpu().numpy()
        pred = (pred*255).astype(np.uint8)
        pred = cv2.resize(pred,(w,h))
        out_im = pred[padding_h:,padding_w:]
    
    return prompt[:,:,0],prompt[:,:,1],prompt[:,:,2],out_im

def get_args():
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--model_path', type=str, default=str(PROJECT_ROOT / 'checkpoints/docres.pkl'),help='Path of the saved checkpoint')
    parser.add_argument('--im_path', type=str, required=True,
                        help='Path of an input document image or directory (required)')
    parser.add_argument('--out_folder', type=str, default=str(PROJECT_ROOT / 'restored/inference'),
                        help='Folder of the output images')
    parser.add_argument('--task', choices=TASKS, default='dewarping',
                        help='task that need to be executed')
    parser.add_argument('--save_dtsprompt', nargs='?', type=int, default=0, 
                        help='Whether to save the three prompt images')
    add_runtime_arguments(parser)
    args = parser.parse_args()
    validate_runtime_arguments(parser, args)
    return args

def model_init(args):
    device = select_device(getattr(args, 'device', 'auto'))
   # prepare model
    model = restormer_arch.Restormer( 
        inp_channels=6, 
        out_channels=3, 
        dim = 48,
        num_blocks = [2,3,3,4], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',
        dual_pixel_task = True        
    )

    state = load_model_state(args.model_path)
    model.load_state_dict(state)

    model.eval()
    model = model.to(device)
    return model

def inference_one_im(model, im_path, task, *, max_size=None, precision='auto',
                     mbd_model_path=DEFAULT_MBD_PATH, out_folder=None):
    if task=='dewarping':
        return dewarping(model, im_path, mbd_model_path)
    elif task=='deshadowing':
        return deshadowing(model, im_path, max_size, precision)
    elif task=='appearance':
        return appearance(model, im_path, max_size, precision)
    elif task=='deblurring':
        return deblurring(model, im_path, precision)
    elif task=='binarization':
        return binarization(model, im_path, precision)
    elif task=='end2end':
        if out_folder is not None:
            Path(out_folder).mkdir(parents=True, exist_ok=True)
        # Keep intermediate files isolated per image and clean them up on failure too.
        with TemporaryDirectory(prefix='docres-', dir=out_folder) as temp_dir:
            step1 = Path(temp_dir) / 'step1.jpg'
            step2 = Path(temp_dir) / 'step2.jpg'
            _, _, _, restored = dewarping(model, im_path, mbd_model_path)
            write_image(step1, restored)
            release_mps_cache(model)
            _, _, _, restored = deshadowing(model, step1, max_size, precision)
            write_image(step2, restored)
            release_mps_cache(model)
            return appearance(model, step2, max_size, precision)
    raise ValueError(f'Unsupported task: {task}')


def save_results(
    img_path: str,
    out_folder: str,
    task: str,
    save_dtsprompt: bool,
    restored,
    prompts,
):
    source = Path(img_path)
    save_path = Path(out_folder) / f'{source.stem}_{task}{source.suffix}'
    write_image(save_path, restored)
    if save_dtsprompt:
        for index, prompt in enumerate(prompts, 1):
            write_image(save_path.with_name(f'{save_path.stem}_prompt{index}{save_path.suffix}'), prompt)
    return save_path


def main():
    args = get_args()
    paths = image_paths(args.im_path)
    Path(args.out_folder).mkdir(parents=True, exist_ok=True)
    model = model_init(args)
    print(f'Device: {next(model.parameters()).device}; precision: {args.precision}; task: {args.task}')
    for path in paths:
        prompt1, prompt2, prompt3, restored = inference_one_im(
            model, path, args.task, max_size=args.max_size, precision=args.precision,
            mbd_model_path=args.mbd_model_path, out_folder=args.out_folder,
        )
        saved = save_results(path, args.out_folder, args.task, args.save_dtsprompt,
                             restored, (prompt1, prompt2, prompt3))
        print(f'Saved: {saved}')
        release_mps_cache(model)


if __name__ == '__main__':
    main()
