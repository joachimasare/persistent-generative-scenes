#!/usr/bin/env python3
import numpy as np
from run_phase3_diffusion import mem, intr, poses, render_scene, hallucinate_texture
from tqdm import tqdm

def multi_view_consistency(novel_angles=[45,135,225,315]):
    print("Enforcing multi-view consistency…")
    center = np.array([8,8,8],dtype=np.float32); radius = 20.0

    for ang in tqdm(novel_angles, desc="Novel-view loop"):
        rad = np.deg2rad(ang)
        eye = center + np.array([radius*np.cos(rad),0,radius*np.sin(rad)],dtype=np.float32)
        R = look_at(eye, center)
        P = np.eye(4,dtype=np.float32); P[:3,:3],P[:3,3]=R,eye

        # render hint → hallucinate → fuse
        hint = mem.query(P, intr)
        rgb_gen = hallucinate_texture(hint)
        mem.update(P, rgb_gen, hint, intr)

    print("Consistency pass complete.")

if __name__=="__main__":
    # Assumes you have already run run_phase3_diffusion to populate mem
    multi_view_consistency()
