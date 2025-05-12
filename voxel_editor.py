#!/usr/bin/env python3
import numpy as np
import open3d as o3d
from memory.voxel_memory import VoxelMemory
from skimage.measure import marching_cubes

mem = VoxelMemory(dims=(128,128,128), voxel_size=0.25)

def add_block(x,y,z,r,g,b):
    mem.counts[x,y,z] = 1
    mem.color_accum[x,y,z] = np.array([r,g,b],dtype=float)
    print(f"Added block at ({x},{y},{z})")

def remove_block(x,y,z):
    mem.counts[x,y,z] = 0
    mem.color_accum[x,y,z].fill(0)
    print(f"Removed block at ({x},{y},{z})")

def render_mesh():
    occ = (mem.counts>0).astype(np.float32)
    vs = mem.voxel_size
    verts, faces, normals, _ = marching_cubes(occ, level=0.5, spacing=(vs,vs,vs))
    ijk = np.floor(verts/vs).astype(int)
    np.clip(ijk,0,np.array(mem.dims)-1,out=ijk)
    cols = mem.color_accum[ijk[:,0],ijk[:,1],ijk[:,2]]
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts),
        triangles=o3d.utility.Vector3iVector(faces)
    )
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    mesh.vertex_colors  = o3d.utility.Vector3dVector(cols)
    o3d.visualization.draw_geometries([mesh], window_name="Editor Mesh")

def repl():
    print("Voxel Editor REPL. Commands: add x y z r g b | remove x y z | show | exit")
    while True:
        cmd = input(">> ").split()
        if not cmd: continue
        if cmd[0]=="add" and len(cmd)==7:
            add_block(*map(float,cmd[1:]))
        elif cmd[0]=="remove" and len(cmd)==4:
            remove_block(*map(int,cmd[1:4]))
        elif cmd[0]=="show":
            render_mesh()
        elif cmd[0]=="exit":
            break
        else:
            print("Unknown command")

if __name__=="__main__":
    repl()
