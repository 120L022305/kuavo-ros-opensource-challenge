#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线批处理：只对 /foundationpose-hztest 做坐标变换（相机系 -> base_link），
再把结果以两个新增话题写入新 bag；其余所有原话题原样保留。

- 输入目录：/home/lab/data/Datasets/kuavo_data_challenge/origin/test
- 输出目录：/home/lab/data/Datasets/kuavo_data_challenge/origin/add_pose
- 新增话题：
    /foundationpose_base         (std_msgs/String，4x4 行优先)
    /foundationpose_base_pose    (geometry_msgs/PoseStamped，frame_id=base_link)
- 外参：内置为你给的 camera_optical_derived（base_link <- camera_optical）
- 输入矩阵默认认为是“相机光学坐标系(RDF)”；如你后续发现输入是“相机物理(FLU)”，
  只需将 INPUT_FRAME = "physical"。

依赖：numpy、rosbag、std_msgs、geometry_msgs（随 ROS 自带）
"""

import os, glob, re, math
import numpy as np
import rosbag
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped

# ------------------- 固定路径（无需参数） -------------------
IN_DIR  = "/home/lab/data/Datasets/kuavo_data_challenge/origin/add_foundation"
OUT_DIR = "/home/lab/data/Datasets/kuavo_data_challenge/origin/add_pose_trans"

# 只转换这个输入话题（其余话题全保留）
FOUNDATIONPOSE_TOPIC_IN = "/foundationpose-hztest"

# 新增输出话题名
FOUNDATIONPOSE_TOPIC_OUT_STR  = "/foundationpose_base"
FOUNDATIONPOSE_TOPIC_OUT_POSE = "/foundationpose_base_pose"

# 输入4x4矩阵所属相机坐标系： "optical" (RDF) 或 "physical" (FLU)
INPUT_FRAME = "optical"   # 如需切换，改成 "physical"

# ------------------- 外参（内置自你提供的 YAML） -------------------
# 这里用 camera_optical_derived：base_link <- camera_optical
EXTRINSICS = {
    "translation": [0.06822573754593178, -1.0755729148748657e-05, 0.7178047690557512],
    "quaternion_xyzw": [-0.6119166374409138, 0.6119925321992996, -0.3542981098668795, 0.3542541724525645],
    "base_frame": "base_link",
    "camera_frame": "cam_h_color_optical_frame"
}

# 你之前给的 “RDF(光学) -> FLU(物理)” 固定旋转
R_PHYSICAL_FROM_OPTICAL = np.array([
    [ 0.,  0.,  1.],
    [-1.,  0.,  0.],
    [ 0., -1.,  0.]
], dtype=float)
R_OPTICAL_FROM_PHYSICAL = R_PHYSICAL_FROM_OPTICAL.T  # 物理->光学

# ------------------- 工具函数 -------------------
def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)

def list_bags(in_dir):
    return sorted(glob.glob(os.path.join(in_dir, "*.bag")))

def parse_matrix_string(s):
    """从 std_msgs/String 中稳健解析 4x4 浮点（支持科学计数法、逗号/空格分隔）"""
    floats = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    nums = [float(x) for x in floats]
    if len(nums) != 16:
        raise ValueError(f"解析到 {len(nums)} 个数，期望 16；原串前60字节: {s[:60]!r}")
    return np.array(nums, dtype=float).reshape(4, 4)

def h_from_qt(q_xyzw, t_xyz):
    x,y,z,w = q_xyzw
    tx,ty,tz = t_xyz
    n = math.sqrt(x*x+y*y+z*z+w*w)
    x,y,z,w = x/n, y/n, z/n, w/n
    R = np.array([
        [1-2*(y*y+z*z),   2*(x*y - z*w), 2*(x*z + y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z - x*w)],
        [  2*(x*z-y*w),   2*(y*z + x*w), 1-2*(x*x+y*y)]
    ], dtype=float)
    H = np.eye(4); H[:3,:3]=R; H[:3,3]=[tx,ty,tz]
    return H

def quat_from_R(R):
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2,1] - R[1,2]) / s
        qy = (R[0,2] - R[2,0]) / s
        qz = (R[1,0] - R[0,1]) / s
    else:
        i = np.argmax([R[0,0], R[1,1], R[2,2]])
        if i == 0:
            s = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            qx = 0.25 * s
            qy = (R[0,1] + R[1,0]) / s
            qz = (R[0,2] + R[2,0]) / s
            qw = (R[2,1] - R[1,2]) / s
        elif i == 1:
            s = math.sqrt(1.0 - R[0,0] + R[1,1] - R[2,2]) * 2
            qx = (R[0,1] + R[1,0]) / s
            qy = 0.25 * s
            qz = (R[1,2] + R[2,1]) / s
            qw = (R[0,2] - R[2,0]) / s
        else:
            s = math.sqrt(1.0 - R[0,0] - R[1,1] + R[2,2]) * 2
            qx = (R[0,2] + R[2,0]) / s
            qy = (R[1,2] + R[2,1]) / s
            qz = 0.25 * s
            qw = (R[1,0] - R[0,1]) / s
    q = np.array([qx,qy,qz,qw], dtype=float)
    q = q / np.linalg.norm(q)
    return q.tolist()

# ------------------- 主处理 -------------------
def process_one_bag(bag_in):
    base_frame = EXTRINSICS.get("base_frame", "base_link")
    H_base_camopt = h_from_qt(EXTRINSICS["quaternion_xyzw"], EXTRINSICS["translation"])

    # 物理->光学 4x4（纯旋转）
    H_opt_from_phys = np.eye(4)
    H_opt_from_phys[:3,:3] = R_OPTICAL_FROM_PHYSICAL

    bag_name = os.path.basename(bag_in)
    out_path = os.path.join(OUT_DIR, os.path.splitext(bag_name)[0] + "__with_base_obj.bag")

    print(f"\n[INFO] 处理: {bag_in}")
    cnt_total = cnt_fp = 0

    with rosbag.Bag(bag_in, "r") as ib, rosbag.Bag(out_path, "w") as ob:
        for topic, msg, t in ib.read_messages():
            # 1) 先把原消息原样写回（保留所有话题，时间戳不变）
            ob.write(topic, msg, t)
            cnt_total += 1

            # 2) 针对 foundationpose 做坐标变换并写入两个新增话题
            if topic == FOUNDATIONPOSE_TOPIC_IN:
                cnt_fp += 1

                # 2.1 解析输入 4x4（相机系）
                H_cam_obj = parse_matrix_string(msg.data)

                # 2.2 如输入是“相机物理(FLU)”，先转成光学(RDF)
                if INPUT_FRAME == "physical":
                    H_cam_obj = H_opt_from_phys @ H_cam_obj

                # 2.3 相机光学 -> base_link
                H_base_obj = H_base_camopt @ H_cam_obj

                # 2.4 以 String(4x4) 写新话题
                s = String()
                s.data = " ".join(f"{x:.9f}" for x in H_base_obj.reshape(-1))
                ob.write(FOUNDATIONPOSE_TOPIC_OUT_STR, s, t)

                # 2.5 以 PoseStamped 写新话题
                ps = PoseStamped()
                ps.header.stamp = t
                ps.header.frame_id = base_frame
                ps.pose.position.x = float(H_base_obj[0,3])
                ps.pose.position.y = float(H_base_obj[1,3])
                ps.pose.position.z = float(H_base_obj[2,3])
                qx,qy,qz,qw = quat_from_R(H_base_obj[:3,:3])
                ps.pose.orientation.x = qx
                ps.pose.orientation.y = qy
                ps.pose.orientation.z = qz
                ps.pose.orientation.w = qw
                ob.write(FOUNDATIONPOSE_TOPIC_OUT_POSE, ps, t)

    print(f"[OK] 完成: 原消息 {cnt_total} 条；"
          f"从 {FOUNDATIONPOSE_TOPIC_IN} 生成 {cnt_fp*2} 条新增消息；"
          f"输出 -> {out_path}")

def main():
    ensure_out_dir()
    bags = list_bags(IN_DIR)
    if not bags:
        print(f"[WARN] 输入目录没有 .bag：{IN_DIR}")
        return
    for b in bags:
        process_one_bag(b)

if __name__ == "__main__":
    main()
