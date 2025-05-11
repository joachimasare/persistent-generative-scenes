import numpy as np

class VoxelMemory:
    def __init__(self, dims=(32, 32, 32), voxel_size=1.0):
        """
        dims: tuple of ints (Nx, Ny, Nz), number of voxels along each axis
        voxel_size: float, world-unit length per voxel
        """
        self.dims = np.array(dims, dtype=int)
        self.voxel_size = float(voxel_size)
        # Accumulate color sums and sample counts per voxel
        self.color_accum = np.zeros((*self.dims, 3), dtype=np.float32)
        self.counts = np.zeros(self.dims, dtype=np.int32)

    def world_to_grid(self, pts_world):
        """
        Convert world coordinates to voxel indices.
        pts_world: (N,3) array of points in world space.
        Returns: (N,3) integer indices clipped to [0, dims-1].
        """
        idx = np.floor(pts_world / self.voxel_size).astype(int)
        idx = np.clip(idx, [0,0,0], self.dims - 1)
        return idx

    def update(self, pose, rgb, depth, intrinsic):
        """
        Integrate a new RGB-D view into memory.
        pose: (4,4) camera-to-world transform matrix
        rgb: (H,W,3) uint8 image array
        depth: (H,W) float or int, depth in the same scale as intrinsic (meters)
        intrinsic: object with fx, fy, cx, cy attributes
        """
        H, W = depth.shape
        # Flatten pixel grid
        u, v = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')  # u: x, v: y
        z = depth.astype(np.float32)
        # Backproject to camera coords
        x = (u - intrinsic.cx) * z / intrinsic.fx
        y = (v - intrinsic.cy) * z / intrinsic.fy
        pts_cam = np.stack([x, y, z], axis=-1).reshape(-1, 3)  # (H*W, 3)
        # Transform to world
        R = pose[:3, :3]
        t = pose[:3, 3]
        pts_world = (R @ pts_cam.T).T + t  # (H*W, 3)
        # Map to voxel grid
        vox_idx = self.world_to_grid(pts_world)  # (H*W, 3)
        flat_idx = (vox_idx[:,0] * self.dims[1] * self.dims[2]
                    + vox_idx[:,1] * self.dims[2]
                    + vox_idx[:,2])
        # Normalize color to [0,1]
        colors = rgb.reshape(-1, 3).astype(np.float32) / 255.0
        # Accumulate color sums and counts
        flat_accum = self.color_accum.reshape(-1, 3)
        flat_counts = self.counts.reshape(-1)
        np.add.at(flat_accum, flat_idx, colors)
        np.add.at(flat_counts, flat_idx, 1)
        # Reshape back
        self.color_accum = flat_accum.reshape(*self.dims, 3)
        self.counts = flat_counts.reshape(*self.dims)

    def query(self, pose, intrinsic, view_size=(256, 256)):
        """
        Project stored voxel centers into the camera to produce a depth hint map.
        pose: (4,4) camera-to-world transform
        intrinsic: object with fx, fy, cx, cy attributes
        view_size: (H, W) output hint map size
        Returns: depth_hint (H,W) float32, zeros where unknown
        """
        H, W = view_size
        # Compute voxel center coordinates in world
        grid_idxs = np.indices(self.dims).reshape(3, -1).T  # (Nx*Ny*Nz, 3)
        centers = (grid_idxs + 0.5) * self.voxel_size  # center of each voxel
        # Filter only voxels with data
        counts_flat = self.counts.reshape(-1)
        valid = counts_flat > 0
        if not np.any(valid):
            return np.zeros((H, W), dtype=np.float32)
        valid_centers = centers[valid]
        # Transform to camera coords
        R = pose[:3, :3]
        t = pose[:3, 3]
        pts_cam = (R.T @ (valid_centers - t).T).T  # world->camera: invert transform
        # Project to pixel coords
        x = pts_cam[:, 0]
        y = pts_cam[:, 1]
        z = pts_cam[:, 2]
        u = (x * intrinsic.fx / z) + intrinsic.cx
        v = (y * intrinsic.fy / z) + intrinsic.cy
        u = np.round(u).astype(int)
        v = np.round(v).astype(int)
        # Initialize depth hint map
        depth_hint = np.zeros((H, W), dtype=np.float32)
        # Keep smallest z per pixel
        for ui, vi, zi in zip(u, v, z):
            if 0 <= ui < W and 0 <= vi < H:
                if depth_hint[vi, ui] == 0 or zi < depth_hint[vi, ui]:
                    depth_hint[vi, ui] = zi
        return depth_hint