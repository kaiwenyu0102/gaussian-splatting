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

def get_base_stem(filename):
    """
    Extract the base stem from DJI filename.
    DJI_20230103153418_0002_W.JPG -> DJI_20230103153418_0002
    DJI_20230103153418_0002_T.JPG -> DJI_20230103153418_0002
    """
    stem = Path(filename).stem
    if stem.endswith("_W"):
        return stem[:-2]
    elif stem.endswith("_T"):
        return stem[:-2]
    return stem


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
    
    # Build lookup: base_stem -> rgb_image
    rgb_by_base = {}
    for rid, rimg in rgb_images.items():
        base = get_base_stem(rimg["name"])
        rgb_by_base[base] = rimg
    
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
    # STEP 3: For each Thermal image, use its corresponding RGB
    #          image's pose directly (same drone, small offset)
    # ============================================================
    print("=" * 60)
    print("[STEP 3] Assigning RGB poses to Thermal images...")
    print("=" * 60)
    
    new_thermal_images = {}
    thermal_files = sorted(os.listdir(args.thermal_input_dir))
    
    matched = 0
    skipped = 0
    
    new_id = 1
    for th_file in thermal_files:
        th_base = get_base_stem(th_file)
        
        if th_base in rgb_by_base:
            rgb_img = rgb_by_base[th_base]
            
            # Directly use RGB W2C pose for Thermal
            # The physical offset between RGB and Thermal cameras on DJI
            # is very small (<10cm, <5deg), so this approximation works.
            # 3DGS training optimizes Gaussian positions to compensate.
            new_thermal_images[new_id] = {
                "id": new_id,
                "qvec": rgb_img["qvec"],     # RGB rotation
                "tvec": rgb_img["tvec"],     # RGB translation
                "camera_id": 1,              # Use Thermal camera intrinsics!
                "name": th_file              # Thermal filename
            }
            matched += 1
            new_id += 1
        else:
            print(f"[WARNING] No RGB match for {th_file} (base={th_base}), skipping")
            skipped += 1
    
    print(f"[INFO] Matched: {matched}, Skipped: {skipped}")
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
    
    # cameras.bin: Thermal intrinsics (different FOV from RGB!)
    new_cameras = {1: th_cam}
    write_cameras_binary(new_cameras, os.path.join(args.output_dir, "cameras.bin"))
    
    # images.bin: RGB poses + Thermal filenames + Thermal camera_id
    write_images_binary(new_thermal_images, os.path.join(args.output_dir, "images.bin"))
    
    # points3D.bin: RGB points (same coordinate system)
    write_points3D_binary(sparse_points, os.path.join(args.output_dir, "points3D.bin"))
    
    print(f"\n{'=' * 60}")
    print(f"DONE! Output at {args.output_dir}")
    print(f"  Cameras: {len(new_cameras)} (Thermal intrinsics)")
    print(f"  Images:  {len(new_thermal_images)} (RGB poses)")
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