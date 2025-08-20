import open3d as o3d
from copy import deepcopy

if __name__ == "__main__":
    # load cloud
    ref_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[0, 0, 0]
    )
    # Load the point cloud and the SMPL mesh
    pcd = o3d.io.read_point_cloud("build/target.ply")
    smpl = o3d.io.read_triangle_mesh("build/predicted.obj")
    pcd_original = deepcopy(pcd)
    pcd = pcd.translate((-2.0, 0, 0))
    o3d.visualization.draw_geometries([pcd_original, pcd, smpl, ref_frame])
