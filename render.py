#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import imageio
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
import threading
import concurrent.futures
from scipy import signal
from plenoptic.simulate import SteerablePyramidFreq as Steerable_Pyramid_Freq
def bandpass_filter(data_3d, mask):
    d_data_3d = data_3d

    data_f = torch.fft.fft(d_data_3d)
    data_f = data_f * mask.to(data_f.device)

    out = torch.fft.ifft(data_f)

    return out.real


def bandpass_filter_np(data_3d, mask):
    d_data_3d = data_3d - data_3d[..., 0:1]
    data_f = np.fft.fft(d_data_3d)
    data_f[~mask] = 0
    filtered = np.fft.ifft(data_f)

    return filtered.real


def create_bandpass_mask(lowcut, highcut, n, fs):
    fl, fh = lowcut / fs, highcut / fs
    fl, fh = fl * 2, fh * 2
    B = signal.firwin(n, cutoff=[fl, fh], window="hamming", pass_zero="bandpass")
    B = torch.FloatTensor(B[:n])

    mask = torch.fft.fft(torch.fft.ifftshift(B))
    mask = mask.unsqueeze(0).unsqueeze(0)

    return mask


def create_bandpass_mask_ideal(lowcut, highcut, n, fs):
    freq = np.arange(n) / n
    freq = freq * fs

    mask = (freq > lowcut) * (freq <= highcut)
    mask = torch.from_numpy(mask)

    return mask


def amplify_one_channel_vid(
    f_vid, alpha, low, high, fs, ideal=False, suppress_others=False, flt_thrs=0.0
):
    if ideal:
        mask = create_bandpass_mask_ideal(
            lowcut=low, highcut=high, n=f_vid.shape[-1], fs=fs
        )
    else:
        mask = create_bandpass_mask(lowcut=low, highcut=high, n=f_vid.shape[-1], fs=fs)

    mask = mask.repeat(f_vid.shape[0], f_vid.shape[1], 1)

    ref_phs = f_vid.angle()[..., 0:1]

    delta = f_vid.angle() - ref_phs
    delta = ((np.pi + delta) % (2 * np.pi)) - np.pi

    flt_phs = bandpass_filter(delta, mask)

    thrs = torch.quantile(flt_phs.abs(), q=flt_thrs)
    flt_phs[flt_phs.abs() < thrs] = 0.0

    amp_phs = alpha * flt_phs
    amp_phs = ((np.pi + amp_phs) % (2 * np.pi)) - np.pi

    if suppress_others:
        amp_vid = f_vid[..., 0:1] * torch.exp(1j * amp_phs)
    else:
        amp_vid = f_vid * torch.exp(1j * amp_phs)

    return amp_vid


def process_feature_images(f_im, alpha, low, high, fs, device, args):
    f_im = torch.cat(f_im, dim=0)

    pyr = Steerable_Pyramid_Freq(
        height="auto",
        image_shape=[f_im.shape[-2], f_im.shape[-1]],
        order=8,
        is_complex=True,
        downsample=True,
        twidth=0.75,
    )
    pyr.to(device)
    mbsize = 30

    for channel in tqdm(range(f_im.shape[1]), disable=True):
        all_coeffs = {}

        for mb in range(0, len(f_im), mbsize):
            coeff = pyr.forward(
                f_im[mb : mb + mbsize, channel : channel + 1].to(device)
            )
            for key in coeff.keys():
                all_coeffs.setdefault(key, []).append(coeff[key])
            del coeff
        for key in all_coeffs.keys():
            all_coeffs[key] = torch.cat(all_coeffs[key])

        for key in all_coeffs.keys():
            _vid = all_coeffs[key]
            if "residual_highpass" in key or "residual_lowpass" in key:
                continue
            else:
                amp_vid = amplify_one_channel_vid(
                    _vid[:, 0].permute(1, 2, 0),
                    alpha,
                    low,
                    high,
                    fs,
                    ideal=False,
                    suppress_others=False,
                    flt_thrs=0.0,
                )
            all_coeffs[key] = amp_vid.permute(2, 0, 1)[:, None]
            del amp_vid

        for mb in range(0, len(f_im), mbsize):
            mb_coeffs = {}
            for key in all_coeffs.keys():
                mb_coeffs[key] = all_coeffs[key][mb : mb + mbsize]

            out = pyr.recon_pyr(mb_coeffs)
            out = out.cpu()

            f_im[mb : mb + mbsize, channel : channel + 1] = out

            del out, mb_coeffs
        del all_coeffs

    return torch.unbind(f_im.unsqueeze(1), 0)
def vis_grid_features(new_field_feature_imgs, base_logdir):
    planes = torch.cat(new_field_feature_imgs["grid.grids.0.0"]).cpu()
    out = planes.permute(0, 2, 3, 1)[..., :3]
    out = (out - out.min()) / (out.max() - out.min())
    out = np.uint8(out * 255)
    imageio.mimwrite(f"{base_logdir}/xy.mp4", out, fps=10, quality=8)

    planes = torch.cat(new_field_feature_imgs["grid.grids.0.1"]).cpu()
    out = planes.permute(0, 2, 3, 1)[..., :3]
    out = (out - out.min()) / (out.max() - out.min())
    out = np.uint8(out * 255)
    imageio.mimwrite(f"{base_logdir}/xz.mp4", out, fps=10, quality=8)

    planes = torch.cat(new_field_feature_imgs["grid.grids.0.2"]).cpu()
    out = planes.permute(0, 2, 3, 1)[..., :3]
    out = (out - out.min()) / (out.max() - out.min())
    out = np.uint8(out * 255)
    imageio.mimwrite(f"{base_logdir}/yz.mp4", out, fps=10, quality=8)
def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)
    
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)
def render_set(model_path, name, iteration, views, gaussians, pipeline, background, cam_type, gaussians_lst):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    render_images = []
    gt_list = []
    render_list = []
    print("point nums:",gaussians._xyz.shape[0])
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 0:time1 = time()
        print('current idx:', idx)
        gaussians._deformation.deformation_net.grid.grids[0][0] = gaussians_lst[idx%30]._deformation.deformation_net.grid.grids[0][0]
        gaussians._deformation.deformation_net.grid.grids[0][1] = gaussians_lst[idx % 30]._deformation.deformation_net.grid.grids[0][1]
        gaussians._deformation.deformation_net.grid.grids[0][2] = gaussians_lst[idx % 30]._deformation.deformation_net.grid.grids[0][2]
        rendering = render(view, gaussians, pipeline, background,cam_type=cam_type)["render"]
        render_images.append(to8b(rendering).transpose(1,2,0))
        render_list.append(rendering)
        if name in ["train", "test"]:
            if cam_type != "PanopticSports":
                gt = view.original_image[0:3, :, :]
            else:
                gt  = view['image'].cuda()
            gt_list.append(gt)

    time2=time()
    print("FPS:",(len(views)-1)/(time2-time1))

    multithread_write(gt_list, gts_path)

    multithread_write(render_list, render_path)

    
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rgb.mp4'), render_images, fps=30)
def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        cam_type=scene.dataset_type
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # 基础路径
        base_dir = "/root/autodl-tmp/4DGS_MotionMag/output/synthetic/mic_5Hz"
        file_name = "point_cloud/iteration_27000"
        gaussians_lst = []
        # 遍历编号目录
        for i in range(1,30):  # 从 000 到 029
            sub_dir = f"{base_dir}/{i:03d}"  # 格式化为三位数字

            file_path = os.path.join(sub_dir, file_name)
            gaussians_tmp = GaussianModel(dataset.sh_degree, hyperparam)
            # 检查文件是否存在
            if os.path.exists(file_path+"/deformation.pth"):
                # print(f"Found: {file_path}")
                # 如果需要进一步处理文件，可以在这里操作
                gaussians_tmp.load_model(file_path)
                gaussians_lst.append(gaussians_tmp)
                # gaussians_tmp
            else:
                print(f"Not found: {file_path}")
        field_feature_imgs = {}
        field_feature_imgs2 = {}

        for i in range(30):
            for n,p in gaussians_lst[i]._deformation.deformation_net.named_parameters():
                if 'grid.grids.' in n:
                    field_feature_imgs2.setdefault(n,[]).append(p.clone().cpu())

            for n, p in gaussians_lst[i]._deformation.deformation_net.named_parameters():

                if "delta_grids" in n:
                    n2 = n.replace("delta_", "")
                    x = field_feature_imgs2[n2][0]
                    field_feature_imgs2.setdefault(n, []).append(p.data.clone().cpu() + x)
        #             print(n)
        #         print(f'{n} device is {p.device} !')
        # assert 1==2
        new_field_feature_imgs = {}
        for n, f_im in field_feature_imgs.items():
            print(f"Processing: {n}")
            alpha = 50
            low = 1.5
            high = 4.5
            fs = 30
            args = ''
            processed = process_feature_images(f_im, alpha, low, high, fs, 'cpu', args)
            # processed = process_feature_images(processed, alpha, low, high, fs, 'cpu', args)
            new_field_feature_imgs[n] = processed
        vis_grid_features(new_field_feature_imgs,'/root/autodl-tmp/4DGS_MotionMag/output/synthetic/mic_5Hz')

        for i in range(30):
            for n,p in gaussians_lst[i]._deformation.deformation_net.named_parameters():
                if 'grid.grids.' in n:
                    new_data = new_field_feature_imgs[n][i].to('cuda')
                    # print(f"Tensor {n} is on device: {new_data.device}")
                    p.data = new_data.to('cuda')
                    # print(i)
                    # print(n)

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,cam_type)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background,cam_type,gaussians_lst)
        if not skip_video:
            render_set(dataset.model_path,"video",scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,background,cam_type,gaussians_lst)
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video)