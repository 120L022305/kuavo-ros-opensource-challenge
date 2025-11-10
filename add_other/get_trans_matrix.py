#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一次性从 TF 树导出 base_link->camera_* 外参，并派生光学坐标(RDF)；支持零参数运行。
覆盖方式（可选，用其一即可）：
  1) CLI：--parent --child --out --npy --wait
  2) ROS参数：~parent ~children ~out ~npy ~wait
"""
import time, yaml, numpy as np
import argparse, rospy, tf2_ros
from tf.transformations import quaternion_matrix, quaternion_from_matrix

# 你提供的：RDF(optical) -> FLU(physical)
R_PHYSICAL_FROM_OPTICAL = np.array([
    [ 0.,  0.,  1.],
    [-1.,  0.,  0.],
    [ 0., -1.,  0.]
], dtype=float)

def tf_to_hmat(tf_msg):
    t = tf_msg.transform.translation
    q = tf_msg.transform.rotation
    H = quaternion_matrix([q.x, q.y, q.z, q.w])
    H[:3, 3] = [t.x, t.y, t.z]
    return H

def hmat_to_yaml_dict(H):
    qx, qy, qz, qw = quaternion_from_matrix(H)
    tx, ty, tz = H[:3, 3]
    return dict(
        translation=[float(tx), float(ty), float(tz)],
        quaternion_xyzw=[float(qx), float(qy), float(qz), float(qw)],
    )

def lookup_once(buf, parent, child, timeout=5.0):
    return buf.lookup_transform(parent, child, rospy.Time(0), rospy.Duration(timeout))

def print_block(title, base, frame, H):
    d = hmat_to_yaml_dict(H)
    print(f"# === {title}: {base} -> {frame} ===")
    print(f"camera_frame: {frame}")
    print(f"base_frame:   {base}")
    print("translation: [{:.9f}, {:.9f}, {:.9f}]".format(*d["translation"]))
    qx, qy, qz, qw = d["quaternion_xyzw"]
    print("quaternion_xyzw: [{:.9f}, {:.9f}, {:.9f}, {:.9f}]".format(qx, qy, qz, qw))
    print("H (4x4):")
    np.set_printoptions(precision=9, suppress=True)
    print(H, "\n")

def main():
    # 1) CLI 默认值（零参数即可）
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--parent", default="base_link")
    parser.add_argument("--child", action="append", default=["camera_base", "camera"])
    parser.add_argument("--wait", type=float, default=2.0)
    parser.add_argument("--out", default="extrinsics.yaml")
    parser.add_argument("--npy", default="extrinsics.npy")
    args, _ = parser.parse_known_args()

    rospy.init_node("dump_tf_extrinsics_once", anonymous=True)

    # 2) ROS 参数（若提供则覆盖 CLI/默认）
    parent  = rospy.get_param("~parent",  args.parent)
    children= rospy.get_param("~children",args.child)  # 支持 list，例如 ['camera_base','camera']
    out_yml = rospy.get_param("~out",     args.out)
    out_npy = rospy.get_param("~npy",     args.npy)
    wait_s  = float(rospy.get_param("~wait", args.wait))

    buf = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
    _ = tf2_ros.TransformListener(buf)
    time.sleep(wait_s)

    results_yaml, results_npy = {}, {}
    found_cam_base, H_base_cam_base = False, None

    # 3) 逐个子帧查询（不用你手动给起点/终点：target=parent, source=child）
    for child in children:
        try:
            tfmsg = lookup_once(buf, parent, child, timeout=5.0)
        except Exception as e:
            rospy.logwarn("未找到变换 %s -> %s: %s", parent, child, str(e))
            continue

        H = tf_to_hmat(tfmsg)
        print_block("TF", parent, child, H)
        results_yaml[child] = dict(base_frame=parent, **hmat_to_yaml_dict(H))
        results_npy[child]  = H

        if child == "camera_base":
            found_cam_base = True
            H_base_cam_base = H.copy()

    # 4) 若拿到 camera_base(FLU)，派生出 camera_optical_derived(RDF)
    if found_cam_base:
        H_opt = H_base_cam_base.copy()
        # R^base_cam_opt = R^base_cam_base * R^physical_optical  （同点，平移不变）
        H_opt[:3, :3] = H_base_cam_base[:3, :3].dot(R_PHYSICAL_FROM_OPTICAL)
        frame_name = "camera_optical_derived"
        print_block("Derived (FLU -> RDF)", parent, frame_name, H_opt)
        results_yaml[frame_name] = dict(base_frame=parent, **hmat_to_yaml_dict(H_opt))
        results_npy[frame_name]  = H_opt

    # 5) 保存
    with open(out_yml, "w") as f:
        yaml.safe_dump(results_yaml, f, sort_keys=False, allow_unicode=True)
    np.save(out_npy, results_npy, allow_pickle=True)
    print(f"Saved YAML -> {out_yml}")
    print(f"Saved NPY  -> {out_npy}")

if __name__ == "__main__":
    main()
