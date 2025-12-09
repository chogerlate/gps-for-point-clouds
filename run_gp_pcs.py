import argparse
import time
import torch
import numpy as np
import os
from gp_point_clouds.algorithm import SubsetAlgorithm
from pytorch3d.io import IO
from pytorch3d.structures.pointclouds import Pointclouds
from jakteristics import compute_features

def load_data(file_path, neigh_size, device="cpu"):
    print(f"Loading data from {file_path}...")
    
    # Try to load with open3d first to preserve colors
    colors = None
    try:
        import open3d as o3d
        pcd_o3d = o3d.io.read_point_cloud(file_path)
        if pcd_o3d.has_colors():
            colors = np.asarray(pcd_o3d.colors)
            print(f"Loaded colors from PLY file: {colors.shape}")
    except (ImportError, Exception) as e:
        print(f"Could not load colors with open3d: {e}")
    
    # Load mesh for coordinates and faces
    mesh = IO().load_mesh(file_path, device=device)
    coords = mesh.verts_list()[0].double()
    if len(mesh.faces_list()) > 0:
        faces = mesh.faces_list()[0].double()
    else:
        faces = None # or empty tensor?
    
    pcd = Pointclouds(points=mesh.verts_list())
    bounding_box = pcd.get_bounding_boxes()
    diag = bounding_box[0, :, 1] - bounding_box[0, :, 0]
    volume = diag[0] * diag[1] * diag[2]
    surface = volume ** (2 / 3)
    surface_per_point = surface / coords.size(0)
    radius = torch.sqrt(surface_per_point * neigh_size)
    
    print(f"Computing features with radius: {radius.item()}...")
    # Use torch.as_tensor to tolerate any numpy subclass returned by jakteristics
    curv_np = np.asarray(
        compute_features(
            coords.cpu().numpy(),
            search_radius=radius.item(),
            feature_names=["surface_variation"],
        ),
        dtype=np.float64,
    )
    curv = torch.as_tensor(curv_np, device=device, dtype=torch.double).squeeze(1)
    
    return coords, curv, faces, colors

def main():
    parser = argparse.ArgumentParser(description="Run GP-PCS point cloud simplification")
    parser.add_argument("input_file", type=str, help="Path to input .ply file")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save results")
    parser.add_argument("--neigh_size", type=int, default=30, help="Neighbourhood size for curvature computation")
    parser.add_argument("--target_points", type=int, default=None, help="Target number of points in simplified cloud")
    parser.add_argument("--ratio", type=float, default=None, help="Simplification ratio (0.0 to 1.0). e.g., 0.1 for 10% of original.")
    parser.add_argument("--random_cloud_size", type=int, default=25000, help="Size of random subset for processing")
    parser.add_argument("--opt_subset_size", type=int, default=300, help="Size of subset for hyperparameter optimization")
    parser.add_argument("--n_iter", type=int, default=100, help="Number of optimization iterations")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    
    args = parser.parse_args()

    # Device setup
    if args.gpu and torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.enabled = True
        print("Using CUDA")
    else:
        device = "cpu"
        print("Using CPU")

    # Load Data
    if not os.path.exists(args.input_file):
        print(f"Error: File {args.input_file} not found.")
        return

    start_time = time.time()
    coords, curv, faces, colors = load_data(args.input_file, args.neigh_size, device=device)
    original_data_size = coords.shape[0]
    print(f"Original point cloud size: {original_data_size}")

    # Parameters
    random_cloud_size = args.random_cloud_size
    if original_data_size > random_cloud_size:
        random_cloud_size = random_cloud_size
    else:
        random_cloud_size = original_data_size

    if args.ratio is not None:
        if args.ratio <= 0 or args.ratio > 1.0:
            print("Error: Ratio must be between 0 and 1.")
            return
        target_num_points = int(original_data_size * args.ratio)
        print(f"Calculated target points from ratio {args.ratio}: {target_num_points}")
    elif args.target_points is not None:
        target_num_points = args.target_points
    else:
        # Default to 5000 if neither is specified
        target_num_points = 5000
        print("No target specified, defaulting to 5000 points.")

    if target_num_points > original_data_size:
        print(f"Warning: target points ({target_num_points}) > original size. Clamping.")
        target_num_points = original_data_size

    # Ensure target is at least minimal size
    if target_num_points < 10:
        target_num_points = 10
        print("Warning: target points too small, clamped to 10.")

    initial_set_size = int(target_num_points / 3)
    
    print(f"Settings:")
    print(f"  Target points: {target_num_points}")
    print(f"  Random cloud size: {random_cloud_size}")
    print(f"  Optimization subset size: {args.opt_subset_size}")
    print(f"  Iterations: {args.n_iter}")
    print(f"  Initial set size: {initial_set_size}")

    # Algorithm
    alg = SubsetAlgorithm(
        coords,
        curv,
        target_num_points,
        random_cloud_size,
        args.opt_subset_size,
        args.n_iter,
        initial_set_size,
        device,
    )
    
    print("Starting simplification...")
    simp_loop_start = time.time()
    simp_coords, simp_time, original_indices = alg.run()
    total_time = time.time() - start_time
    
    print(f"Simplification complete in {simp_time:.2f}s (Total: {total_time:.2f}s)")

    # Save results
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    base_name = os.path.basename(args.input_file).replace(".ply", "")
    output_base = os.path.join(args.output_dir, f"{base_name}_{target_num_points}")
    
    # Save XYZ
    xyz_path = output_base + ".xyz"
    np.savetxt(xyz_path, simp_coords, delimiter=" ")
    print(f"Saved XYZ to {xyz_path}")
    
    # Save NPZ
    npz_path = output_base + ".npz"
    np.savez(
        npz_path,
        org_coords=np.asarray(coords.cpu().numpy()),
        org_faces=np.asarray(faces.cpu().numpy()) if faces is not None else None,
        simp_coords=np.asarray(simp_coords),
        org_curv=np.asarray(curv.cpu().numpy()),
        original_indices=np.asarray(original_indices),
    )
    print(f"Saved NPZ to {npz_path}")

    # Save PLY with colors if available
    try:
        import open3d as o3d
        ply_path = output_base + ".ply"
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(simp_coords)
        
        # Map colors from original point cloud to simplified points
        if colors is not None and len(colors) == original_data_size:
            try:
                simp_colors = colors[original_indices]
                # Ensure colors are in [0, 1] range (open3d expects this)
                if simp_colors.max() > 1.0:
                    simp_colors = simp_colors / 255.0
                # Clamp to valid range
                simp_colors = np.clip(simp_colors, 0.0, 1.0)
                pcd.colors = o3d.utility.Vector3dVector(simp_colors)
                print(f"Saved PLY with colors to {ply_path} ({len(simp_colors)} points)")
            except (IndexError, ValueError) as e:
                print(f"Warning: Could not map colors: {e}. Saving PLY without colors.")
        elif colors is not None:
            print(f"Warning: Color array size ({len(colors)}) doesn't match point cloud size ({original_data_size}). Saving PLY without colors.")
        else:
            print(f"Saved PLY (no colors in input file) to {ply_path}")
        
        o3d.io.write_point_cloud(ply_path, pcd)
    except ImportError:
        print("Open3D not installed, skipping PLY save.")

if __name__ == "__main__":
    main()

