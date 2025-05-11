import open3d as o3d
import numpy as np

def create_block_world(size=16, num_blocks=50):
    """
    Returns a combined TriangleMesh of random 1×1×1 cubes
    within a size^3 grid.
    """
    mesh = o3d.geometry.TriangleMesh()
    for _ in range(num_blocks):
        x, y, z = np.random.randint(0, size, 3)
        cube = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)
        cube.translate((x, y, z))
        mesh += cube
    mesh.compute_vertex_normals()
    return mesh


def render_scene(mesh, pose, width=256, height=256):
    """
    Offscreen render with Open3D.
    pose: 4×4 camera-to-world transform.
    Returns: rgb (H,W,3) uint8, depth (H,W) float32 in meters.
    """
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    renderer.scene.add_geometry("world", mesh, material)
    # Camera intrinsics
    fx = fy = width
    cx, cy = width / 2, height / 2
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    extrinsic = np.linalg.inv(pose)  # world->camera
    renderer.setup_camera(intrinsic, extrinsic)
    # Render
    img = renderer.render_to_image()
    depth = renderer.render_to_depth_image(True)
    rgb = np.asarray(img)
    depth_map = np.asarray(depth) * 1.0  # depth in meters
    return rgb, depth_map