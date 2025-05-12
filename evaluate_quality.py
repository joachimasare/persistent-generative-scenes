#!/usr/bin/env python3
import os, numpy as np
import skimage.metrics as metrics
from gen_scenes import render_scene
from run_phase3_diffusion import mem, intr

# define held-out camera angles
heldout = [30,60,120,150]
results = []

for ang in heldout:
    # render GT
    rad = np.deg2rad(ang)
    center = np.array([8,8,8],dtype=np.float32)
    eye = center + np.array([20*np.cos(rad),0,20*np.sin(rad)],dtype=np.float32)
    R = look_at(eye, center)
    P = np.eye(4,dtype=np.float32); P[:3,:3],P[:3,3]=R,eye
    rgb_gt, _ = render_scene(create_block_world(16,30), P, 256,256)

    # generate
    hint = mem.query(P, intr, view_size=(256,256))
    rgb_gen = hallucinate_texture(hint)

    # compute metrics
    psnr = metrics.peak_signal_noise_ratio(rgb_gt, rgb_gen, data_range=255)
    ssim = metrics.structural_similarity(rgb_gt, rgb_gen, multichannel=True, data_range=255)
    results.append((ang, psnr, ssim))
    print(f"Angle {ang}: PSNR={psnr:.2f}, SSIM={ssim:.3f}")

# summary
arr = np.array(results)
print("Average:", np.mean(arr[:,1]), "PSNR,", np.mean(arr[:,2]), "SSIM")
