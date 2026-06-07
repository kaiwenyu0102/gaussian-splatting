"""
RGB-to-Thermal 位姿迁移脚本（简化版）
====================================

核心思路：
  RGB 和 Thermal 相机刚性固定在同一 DJI 无人机上，物理偏移极小（<10cm），
  方向基本一致。因此可以直接用 RGB 的 COLMAP 位姿作为 Thermal 的位姿。

  关键区分：
  - 外参（位姿）：来自 RGB 重建，因为 RGB 的 305 张全部注册成功
  - 内参（焦距、FOV、畸变）：来自 Thermal 相机，因为两个相机 FOV 不同
  
  这样所有 305 张 Thermal 图像都在 RGB 的世界坐标系下，保证一致性。

  3DGS 训练时，由于高斯点会自适应优化位置，小偏移（<10cm）不会
  显著影响重建质量，远好于只有 135 张或坐标系不一致的情况。

使用方法:
python rgb_to_thermal_pose_transfer.py \
    --rgb_sparse_dir data/rgb/sparse/0 \
    --thermal_sparse_dir data/thermal/sparse/0 \
    --thermal_input_dir data/thermal/input \
    --output_dir data/thermal/sparse_transfer/0
"""

import os
import sys
import numpy as np
import struct
from pathlib import Path

# ---- COLMAP binary read/write utilities ----

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    fid.write(struct.pack(endian_character + format_char_sequence, *data))

CameraModel = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}

def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name, num_params = CameraModel.get(model_id, ("PINHOLE", 4))
            width = camera_properties[2]
            height = camera_properties[3]
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[camera_id] = {
                "id": camera_id, "model": model_name,
                "width": width, "height": height,
                "params": list(params)
            }
    return cameras

def read_images_binary(path):
    images = {}
    with open(path, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            _ = read_next_bytes(fid, 24 * num_points2D, "ddq" * num_points2D)
            images[image_id] = {
                "id": image_id, "qvec": qvec, "tvec": tvec,
                "camera_id": camera_id, "name": image_name
            }
    return images

def write_cameras_binary(cameras, path):
    with open(path, "wb") as fid:
        write_next_bytes(fid, [len(cameras)], "Q")
        for cam_id in sorted(cameras.keys()):
            cam = cameras[cam_id]
            model_name_to_id = {name: mid for mid, (name, _) in CameraModel.items()}
            model_id = model_name_to_id[cam["model"]]
            write_next_bytes(fid, [cam_id, model_id, cam["width"], cam["height"]], "iiQQ")
            write_next_bytes(fid, cam["params"], "d" * len(cam["params"]))

def write_images_binary(images, path):
    with open(path, "wb") as fid:
        write_next_bytes(fid, [len(images)], "Q")
        for img_id in sorted(images.keys()):
            img = images[img_id]
            write_next_bytes(fid, [img_id] + list(img["qvec"]) + list(img["tvec"]) + [img["camera_id"]], "idddddddi")
            for c in img["name"]:
                write_next_bytes(fid, [c.encode("utf-8")], "c")
            write_next_bytes(fid, [b"\x00"], "c")
            write_next_bytes(fid, [0], "Q")  # no 2D points

def write_points3D_binary(points, path):
    with open(path, "wb") as fid:
        write_next_bytes(fid, [len(points)], "Q")
        for pt_id in sorted(points.keys()):
            pt = points[pt_id]
            write_next_bytes(fid, [pt_id] + list(pt["xyz"]) + list(pt["rgb"]) + [pt["error"]], "QdddBBBd")
            write_next_bytes(fid, [0], "Q")

def write_rigs_binary(rigs, path):
    """Write COLMAP 4.x rigs.bin.
    rigs: dict {rig_id: [(camera_id, ref_tvec_3), ...]}
    Each rig has one or more cameras with reference translations.
    """
    with open(path, "wb") as fid:
        write_next_bytes(fid, [len(rigs)], "Q")
        for rig_id in sorted(rigs.keys()):
            cameras = rigs[rig_id]
            write_next_bytes(fid, [rig_id, len(cameras)], "II")
            for camera_id, ref_tvec in cameras:
                write_next_bytes(fid, [camera_id], "I")
                write_next_bytes(fid, list(ref_tvec), "ddd")

def write_frames_binary(frames, path):
    """Write COLMAP 4.x frames.bin.
    frames: dict {frame_id: (rig_id, [(data_id, camera_id), ...])}
    Each frame belongs to a rig and has one or more data_ids.
    data_id is the image_id, camera_id is which camera in the rig.
    """
    with open(path, "wb") as fid:
        write_next_bytes(fid, [len(frames)], "Q")
        for frame_id in sorted(frames.keys()):
            rig_id, data_ids = frames[frame_id]
            write_next_bytes(fid, [frame_id, rig_id, len(data_ids)], "III")
            for data_id, camera_id in data_ids:
                write_next_bytes(fid, [data_id], "Q")
                write_next_bytes(fid, [camera_id], "I")

def get_base_stem(filename):
    """
    Extract the base stem from DJI filename (legacy, used for sorting).
    DJI_20230103153418_0002_W.JPG -> DJI_20230103153418_0002
    """
    stem = Path(filename).stem
    if stem.endswith("_W"):
        return stem[:-2]
    elif stem.endswith("_T"):
        return stem[:-2]
    return stem

def get_seq_number(filename):
    """Extract sequence number from DJI filename for matching and sorting.
    DJI_20230103153428_0006_T.JPG -> 6
    DJI_20230103153427_0006_W.JPG -> 6
    RGB and Thermal share the same sequence number but differ in timestamp.
    """
    stem = Path(filename).stem
    # Remove _W or _T suffix
    if stem.endswith("_W") or stem.endswith("_T"):
        stem = stem[:-2]
    # Split by underscore: DJI, YYYYMMDDHHMMSS, NNNN
    parts = stem.split("_")
    try:
        return int(parts[-1])
    except (ValueError, IndexError):
        return 0


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2*qvec[2]**2 - 2*qvec[3]**2, 2*qvec[1]*qvec[2] - 2*qvec[0]*qvec[3], 2*qvec[3]*qvec[1] + 2*qvec[0]*qvec[2]],
        [2*qvec[1]*qvec[2] + 2*qvec[0]*qvec[3], 1 - 2*qvec[1]**2 - 2*qvec[3]**2, 2*qvec[2]*qvec[3] - 2*qvec[0]*qvec[1]],
        [2*qvec[3]*qvec[1] - 2*qvec[0]*qvec[2], 2*qvec[2]*qvec[3] + 2*qvec[0]*qvec[1], 1 - 2*qvec[1]**2 - 2*qvec[2]**2]
    ])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def slerp(q0, q1, t):
    """Spherical linear interpolation between two quaternions."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    result = s0 * q0 + s1 * q1
    return result / np.linalg.norm(result)

def interpolate_pose(img_before, img_after, t):
    """
    Interpolate a camera pose between two images.
    t in [0, 1]: 0 = img_before pose, 1 = img_after pose.
    Uses SLERP for rotation and linear interpolation for translation.
    """
    qvec_before = img_before["qvec"]
    qvec_after = img_after["qvec"]
    tvec_before = img_before["tvec"]
    tvec_after = img_after["tvec"]
    
    # SLERP for quaternion (rotation)
    qvec_interp = slerp(qvec_before, qvec_after, t)
    
    # Linear interpolation for translation
    tvec_interp = (1 - t) * tvec_before + t * tvec_after
    
    return qvec_interp, tvec_interp


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Transfer RGB poses to Thermal images")
    parser.add_argument("--rgb_sparse_dir", required=True, help="RGB COLMAP sparse/0 directory")
    parser.add_argument("--thermal_sparse_dir", default=None, help="Thermal COLMAP sparse/0 directory (for intrinsics, optional if --thermal_camera_params provided)")
    parser.add_argument("--thermal_input_dir", required=True, help="Thermal input images directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for new Thermal sparse/0")
    parser.add_argument("--thermal_camera_params", default=None,
        help="Thermal camera params as 'MODEL,W,H,fx,fy,cx,cy,...' "
             "e.g. 'PINHOLE,1280,1024,1515.32,1515.32,640,512' "
             "Overrides --thermal_sparse_dir for intrinsics")
    parser.add_argument("--per_image_camera", action="store_true",
        help="Create one camera per image (all with same intrinsics). "
             "Required for COLMAP 4.x point_triangulator compatibility (avoids frame/rig conflicts).")
    parser.add_argument("--skip_rigs_frames", action="store_true",
        help="Skip writing rigs.bin and frames.bin. Use this when running "
             "point_triangulator after cleaning database frame/rig data.")
    args = parser.parse_args()
    
    # ============================================================
    # STEP 1: Read RGB reconstruction (all 305 images registered)
    # ============================================================
    print("=" * 60)
    print("[STEP 1] Reading RGB COLMAP reconstruction...")
    print("=" * 60)
    rgb_cameras = read_cameras_binary(os.path.join(args.rgb_sparse_dir, "cameras.bin"))
    rgb_images = read_images_binary(os.path.join(args.rgb_sparse_dir, "images.bin"))
    print(f"[INFO] RGB: {len(rgb_cameras)} cameras, {len(rgb_images)} images registered")
    
    # Build lookup: seq_number -> rgb_image
    # RGB and Thermal share the same sequence number (e.g., 0006)
    # but have different timestamps (RGB may be 1 sec earlier)
    rgb_by_seq = {}
    for rid, rimg in rgb_images.items():
        seq = get_seq_number(rimg["name"])
        rgb_by_seq[seq] = rimg
    
    # ============================================================
    # STEP 2: Get Thermal camera intrinsics
    # ============================================================
    print("=" * 60)
    print("[STEP 2] Getting Thermal camera intrinsics...")
    print("=" * 60)
    
    if args.thermal_camera_params:
        # Parse manually specified camera params
        parts = args.thermal_camera_params.split(",")
        model_name = parts[0]
        width = int(parts[1])
        height = int(parts[2])
        params = [float(p) for p in parts[3:]]
        th_cam = {"id": 1, "model": model_name, "width": width, "height": height, "params": params}
        print(f"[INFO] Using manually specified Thermal camera intrinsics")
    elif args.thermal_sparse_dir:
        th_cam_path = os.path.join(args.thermal_sparse_dir, "cameras.bin")
        if os.path.exists(th_cam_path):
            thermal_cameras = read_cameras_binary(th_cam_path)
            th_cam = thermal_cameras[1]
            print(f"[INFO] Using Thermal camera intrinsics from COLMAP reconstruction")
        else:
            print("[ERROR] Thermal cameras.bin not found and no --thermal_camera_params provided.")
            print("[HINT] Re-run COLMAP on thermal, or use --thermal_camera_params like:")
            print("       --thermal_camera_params PINHOLE,1280,1024,1515.32,1515.32,640,512")
            sys.exit(1)
    else:
        print("[ERROR] Either --thermal_sparse_dir or --thermal_camera_params must be provided.")
        sys.exit(1)
    
    print(f"[INFO] Thermal camera: model={th_cam['model']}, "
          f"width={th_cam['width']}, height={th_cam['height']}")
    print(f"[INFO] Thermal params: {th_cam['params']}")
    
    # Compare with RGB camera for reference
    rgb_cam = rgb_cameras[1]
    print(f"[INFO] RGB camera for comparison: model={rgb_cam['model']}, "
          f"width={rgb_cam['width']}, height={rgb_cam['height']}")
    print(f"[INFO] RGB params: {rgb_cam['params']}")
    
    # ============================================================
    # STEP 3: Assign poses to ALL Thermal images
    #   - Matched with RGB -> use RGB pose directly
    #   - No RGB match -> interpolate from nearest neighbors
    # ============================================================
    print("=" * 60)
    print("[STEP 3] Assigning poses to all 305 Thermal images...")
    print("=" * 60)
    
    # Sort thermal files by sequence number (flight order)
    thermal_files = sorted(os.listdir(args.thermal_input_dir), key=get_seq_number)
    
    # First pass: assign RGB poses where available (match by sequence number)
    matched_indices = []  # (index, rgb_img) for images with RGB match
    unmatched_indices = []  # indices without RGB match
    
    for i, th_file in enumerate(thermal_files):
        th_seq = get_seq_number(th_file)
        if th_seq in rgb_by_seq:
            matched_indices.append((i, rgb_by_seq[th_seq]))
        else:
            unmatched_indices.append(i)
    
    print(f"[INFO] Directly matched with RGB: {len(matched_indices)}")
    print(f"[INFO] Need pose interpolation: {len(unmatched_indices)}")
    
    # Build pose array for all thermal files
    # Each entry: {"qvec": ..., "tvec": ..., "camera_id": 1, "name": ...}
    thermal_poses = [None] * len(thermal_files)
    
    # Assign matched poses
    for i, rgb_img in matched_indices:
        thermal_poses[i] = {
            "qvec": rgb_img["qvec"],
            "tvec": rgb_img["tvec"],
            "camera_id": 1,
            "name": thermal_files[i]
        }
    
    
    # Interpolate poses for unmatched images
    # For each unmatched image, find nearest matched neighbors before and after
    matched_idx_set = {i for i, _ in matched_indices}
    matched_idx_sorted = sorted(matched_idx_set)
    
    interpolated = 0
    for i in unmatched_indices:
        # Find nearest matched neighbor before
        before_idx = None
        for j in reversed(matched_idx_sorted):
            if j < i:
                before_idx = j
                break
        
        # Find nearest matched neighbor after
        after_idx = None
        for j in matched_idx_sorted:
            if j > i:
                after_idx = j
                break
        
        if before_idx is not None and after_idx is not None:
            # Interpolate between before and after
            t = (i - before_idx) / (after_idx - before_idx)
            qvec_interp, tvec_interp = interpolate_pose(
                thermal_poses[before_idx], thermal_poses[after_idx], t)
            thermal_poses[i] = {
                "qvec": qvec_interp,
                "tvec": tvec_interp,
                "camera_id": 1,
                "name": thermal_files[i]
            }
            interpolated += 1
        elif before_idx is not None:
            # Only have before -> use before's pose
            thermal_poses[i] = {
                "qvec": thermal_poses[before_idx]["qvec"].copy(),
                "tvec": thermal_poses[before_idx]["tvec"].copy(),
                "camera_id": 1,
                "name": thermal_files[i]
            }
            interpolated += 1
        elif after_idx is not None:
            # Only have after -> use after's pose
            thermal_poses[i] = {
                "qvec": thermal_poses[after_idx]["qvec"].copy(),
                "tvec": thermal_poses[after_idx]["tvec"].copy(),
                "camera_id": 1,
                "name": thermal_files[i]
            }
            interpolated += 1
        else:
            print(f"[WARNING] Cannot assign pose to {thermal_files[i]}, no neighbors available")
    
    # Build final dict
    new_thermal_images = {}
    new_id = 1
    skipped = 0
    for i, pose in enumerate(thermal_poses):
        if pose is not None:
            pose["id"] = new_id
            new_thermal_images[new_id] = pose
            new_id += 1
        else:
            skipped += 1
    
    print(f"[INFO] Matched: {len(matched_indices)}, Interpolated: {interpolated}, Skipped: {skipped}")
    print(f"[INFO] Total thermal images with poses: {len(new_thermal_images)}")
    
    if len(new_thermal_images) < 100:
        print("[ERROR] Too few images with poses. Check filename matching.")
        sys.exit(1)
    
    # ============================================================
    # STEP 4: Read RGB 3D points (same coordinate system as RGB poses)
    # ============================================================
    print("=" * 60)
    print("[STEP 4] Reading RGB 3D points (same coordinate system)...")
    print("=" * 60)
    
    rgb_points_path = os.path.join(args.rgb_sparse_dir, "points3D.bin")
    sparse_points = {}
    
    if os.path.exists(rgb_points_path):
        with open(rgb_points_path, "rb") as fid:
            num_points = read_next_bytes(fid, 8, "Q")[0]
            for pt_id in range(num_points):
                binary_point_line = read_next_bytes(fid, 43, "QdddBBBd")
                xyz = np.array(binary_point_line[1:4])
                rgb_color = np.array(binary_point_line[4:7])
                error = binary_point_line[7]
                track_length = read_next_bytes(fid, 8, "Q")[0]
                if track_length > 0:
                    _ = read_next_bytes(fid, 8 * track_length, "ii" * track_length)
                sparse_points[pt_id + 1] = {
                    "xyz": list(xyz), "rgb": list(rgb_color.astype(int)),
                    "error": error
                }
        print(f"[INFO] Using {len(sparse_points)} RGB 3D points as initial sparse points")
        print("[INFO] Points are in RGB world coordinate system (same as all image poses)")
    else:
        print("[WARNING] No RGB points3D.bin found, creating empty point cloud")
    
    # ============================================================
    # STEP 5: Write new COLMAP sparse files
    # ============================================================
    print("=" * 60)
    print("[STEP 5] Writing new Thermal COLMAP sparse files...")
    print("=" * 60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # cameras.bin: Thermal intrinsics (1 shared camera)
    new_cameras = {1: th_cam}
    write_cameras_binary(new_cameras, os.path.join(args.output_dir, "cameras.bin"))
    print(f"[INFO] Created 1 shared camera")

    # Ensure all images reference camera_id=1
    for img_id in new_thermal_images:
        new_thermal_images[img_id]["camera_id"] = 1

    if not args.skip_rigs_frames:
        # rigs.bin: 1 rig with 1 camera
        new_rigs = {1: [(1, [0.0, 0.0, 0.0])]}  # rig_id=1, camera_id=1, ref_tvec=[0,0,0]
        write_rigs_binary(new_rigs, os.path.join(args.output_dir, "rigs.bin"))
        print(f"[INFO] Created 1 rig (single-camera)")

        # frames.bin: 305 frames
        new_frames = {}
        for img_id in sorted(new_thermal_images.keys()):
            new_frames[img_id] = (1, [(img_id, 1)])
        write_frames_binary(new_frames, os.path.join(args.output_dir, "frames.bin"))
        print(f"[INFO] Created {len(new_frames)} frames")
    else:
        new_rigs = {}
        new_frames = {}
        print(f"[INFO] Skipped rigs.bin and frames.bin (--skip_rigs_frames)")
    
    # images.bin: RGB poses + Thermal filenames + Thermal camera_id
    write_images_binary(new_thermal_images, os.path.join(args.output_dir, "images.bin"))
    
    # points3D.bin: RGB points (same coordinate system)
    write_points3D_binary(sparse_points, os.path.join(args.output_dir, "points3D.bin"))
    
    print(f"\n{'=' * 60}")
    print(f"DONE! Output at {args.output_dir}")
    print(f"  Cameras: {len(new_cameras)} (shared Thermal intrinsics)")
    print(f"  Images:  {len(new_thermal_images)} (305 with RGB poses)")
    if not args.skip_rigs_frames:
        print(f"  Rigs:    {len(new_rigs)} (single-camera rig)")
        print(f"  Frames:  {len(new_frames)} (one per image)")
    else:
        print(f"  Rigs:    skipped")
        print(f"  Frames:  skipped")
    print(f"  Points:  {len(sparse_points)} (RGB world frame)")
    print(f"{'=' * 60}")
    
    print("\nNext steps:")
    print("1. Run image_undistorter:")
    print(f"   colmap image_undistorter --image_path data/thermal/input "
          f"--input_path {args.output_dir} --output_path data/thermal_thermal "
          f"--output_type COLMAP")
    print("2. Check images count:")
    print("   ls data/thermal_thermal/images | wc -l")
    print("3. Move sparse to correct location:")
    print("   mkdir -p data/thermal_thermal/sparse/0")
    print("   mv data/thermal_thermal/sparse/* data/thermal_thermal/sparse/0/ 2>/dev/null")
    print("4. Train:")
    print("   CUDA_VISIBLE_DEVICES=1 python train.py -s data/thermal_thermal")


if __name__ == "__main__":
    main()