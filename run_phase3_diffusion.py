#!/usr/bin/env python3
import os, sys
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
import open3d as o3d
from skimage.measure import marching_cubes
from tqdm import tqdm

# allow imports
repo = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(repo, "src"))
from memory.voxel_memory import VoxelMemory
from scripts.gen_scenes import create_block_world, render_scene  # headless CPU render

# -----------------------------------------------------------------------------
# 1. Load pre-trained ControlNet + Stable Diffusion model
# -----------------------------------------------------------------------------
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth",
    torch_dtype=torch.float16
)
#pipe = StableDiffusionControlNetPipeline.from_pretrained(
#    "runwayml/stable-diffusion-v1-5",
 #   controlnet=controlnet,
 #   torch_dtype=torch.float16
#)
#pipe.enable_xformers_memory_efficient_attention()
#pipe.to("cuda")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
).to("cuda")
# pipe.enable_xformers_memory_efficient_attention()
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)



def hallucinate_texture(depth_hint: np.ndarray) -> np.ndarray:
    """
    Runs depth->RGB via ControlNet. Expects depth_hint in meters.
    Returns H×W×3 uint8 image.
    """
    # normalize depth map to [0,255] and stack as single-channel PIL Image
    dh = depth_hint.copy()
    valid = dh > 0
    if valid.any():
        dh[valid] = (dh[valid] / dh[valid].max()) * 255
    dh_img = Image.fromarray(dh.astype(np.uint8))

    prompt = "minecraft style block world, voxel art"
    neg_prompt = "lowres, bad anatomy"

    # run inference
    out = pipe(
        prompt=prompt,
        negative_prompt=neg_prompt,
        image=dh_img,
        num_inference_steps=20,
        guidance_scale=7.5,
        controlnet_conditioning_scale=1.0,
    )
    rgb = np.asarray(out.images[0])
    return rgb

# -----------------------------------------------------------------------------
# 2. Main: fuse real + generated, extract mesh
# -----------------------------------------------------------------------------
def main():
    # --- Phase 1 memory setup ---
    mem = VoxelMemory(dims=(128,128,128), voxel_size=0.25)

    # regenerate scene
    mesh = create_block_world(size=16, num_blocks=30)
    W = H = 256; fx = fy = W; cx = cy = W/2

    class Intrinsic:
        def __init__(self, fx, fy, cx, cy):
            self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
    intr = Intrinsic(fx, fy, cx, cy)

    # four canonical views
    center = np.array([8,8,8], dtype=np.float32)
    radius = 20.0
    poses = []
    for ang in [0,90,180,270]:
        rad = np.deg2rad(ang)
        eye = center + np.array([radius*np.cos(rad),0,radius*np.sin(rad)],dtype=np.float32)
        R = look_at(eye, center)
        P = np.eye(4, dtype=np.float32); P[:3,:3], P[:3,3] = R, eye
        poses.append(P)

    # 1) Fuse real captures
    print("Fusing real captures…")
    for i, P in enumerate(poses,1):
        rgb, depth = render_scene(mesh, P, W, H)
        mem.update(P, rgb, depth, intr)
        cov = mem.counts.sum()/(W*H)*100
        print(f"  View {i}: coverage = {cov:.1f}%")

    # 2) Hallucinate with diffusion + fuse
    print("Hallucinating & fusing textures…")
    for i, P in enumerate(poses,1):
        hint = mem.query(P, intr, view_size=(H,W))
        rgb_gen = hallucinate_texture(hint)
        mem.update(P, rgb_gen, hint, intr)
        cov = mem.counts.sum()/(W*H)*100
        print(f"  Gen {i}: coverage = {cov:.1f}%")

    # --- Extract colored mesh via Marching Cubes ---
    print("Extracting final mesh…")
    occ = (mem.counts>0).astype(np.float32)
    vs = mem.voxel_size
    verts, faces, normals, _ = marching_cubes(occ, level=0.5, spacing=(vs,vs,vs))

    # color vertices
    ijk = np.floor(verts/vs).astype(int)
    np.clip(ijk, 0, np.array(mem.dims)-1, out=ijk)
    cols_acc = mem.color_accum[ijk[:,0],ijk[:,1],ijk[:,2]]
    counts   = mem.counts[ijk[:,0],ijk[:,1],ijk[:,2]].reshape(-1,1)
    vert_colors = (cols_acc/np.maximum(counts,1)).clip(0,1)

    mesh_o3d = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts),
        triangles=o3d.utility.Vector3iVector(faces)
    )
    mesh_o3d.vertex_normals = o3d.utility.Vector3dVector(normals)
    mesh_o3d.vertex_colors  = o3d.utility.Vector3dVector(vert_colors)

    out = "data/phase3_diffusion"
    os.makedirs(out, exist_ok=True)
    pth = os.path.join(out, "mesh_phase3.ply")
    o3d.io.write_triangle_mesh(pth, mesh_o3d)
    print("Saved:", pth)
    o3d.visualization.draw_geometries([mesh_o3d], window_name="Phase 3 Diffusion")

if __name__=="__main__":
    main()
