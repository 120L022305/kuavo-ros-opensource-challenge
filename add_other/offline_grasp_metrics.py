#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批处理：在不改动原话题的基础上，计算并追加以下新话题到新 bag：
  - /foundationpose_base            (std_msgs/String 4x4)
  - /foundationpose_base_pose       (geometry_msgs/PoseStamped, frame_id=base_link)
  - /ee_base                        (std_msgs/String 4x4)
  - /ee_base_pose                   (geometry_msgs/PoseStamped, frame_id=base_link)
  - /ee_to_obj                      (std_msgs/String 4x4)               # H_ee_obj
  - /ee_to_obj_pose                 (geometry_msgs/PoseStamped, frame_id=EE_SITE)

输入目录：/root/kuavo_ws/add_other/add_pose
输出目录：/root/kuavo_ws/add_other/add_grasp_metrics
输出 bag 文件名与原包“同名”。

依赖：numpy、rosbag、geometry_msgs、std_msgs、mujoco(>=2.3)
MJCF：用你的 biped_s45.xml 做 FK（默认右手 right_pinch 作为末端）。
"""

import os, glob, re, math
import numpy as np
import rosbag
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import mujoco  # pip install mujoco

# ------------------- 固定路径（零参数运行） -------------------
IN_DIR  = "/root/kuavo_ws/add_other/add_pose"
OUT_DIR = "/root/kuavo_ws/add_other/add_grasp_metrics"

# MJCF（MuJoCo XML）
MJCF_PATH = "/root/kuavo_ws/src/data_challenge_simulator/models/biped_s45/xml/biped_s45.xml"

# 话题名
TOPIC_FOUND_IN          = "/foundationpose-hztest"
TOPIC_SENSOR_DATA       = "/sensors_data_raw"  # 已改为传感器话题
TOPIC_FOUND_OUT_STR     = "/foundationpose_base"
TOPIC_FOUND_OUT_POSE    = "/foundationpose_base_pose"
TOPIC_EE_OUT_STR        = "/ee_base"
TOPIC_EE_OUT_POSE       = "/ee_base_pose"
TOPIC_REL_OUT_STR       = "/ee_to_obj"
TOPIC_REL_OUT_POSE      = "/ee_to_obj_pose"

# 输入4x4矩阵所属相机坐标系： "optical" (RDF) 或 "physical" (FLU)
INPUT_FRAME = "optical"

# ------------------- 外参（base_link <- camera_optical） -------------------
# 支持两种模式：固定外参 or 自动计算外参

USE_AUTO_EXTRINSICS =  True  # True: 自动计算，False: 固定
print(f"[EXTRINSICS] USE_AUTO_EXTRINSICS = {USE_AUTO_EXTRINSICS}")

EXTRINSICS = {
    "translation": [0.06822573754593178, -1.0755729148748657e-05, 0.7178047690557512],
    "quaternion_xyzw": [-0.6119166374409138, 0.6119925321992996, -0.3542981098668795, 0.3542541724525645],
    "base_frame": "base_link",
    "camera_frame": "cam_h_color_optical_frame"
}

# --- 自动计算外参的独立子模块 ---
def compute_extrinsics_from_bag(mjcf_path, bag_path, base_body="base_link", camera_body="head_camera", joint_names=None):
    print(f"[EXTRINSICS] 自动计算外参: mjcf_path={mjcf_path}, bag_path={bag_path}, base_body={base_body}, camera_body={camera_body}")
    import mujoco
    import rosbag
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    # 1. 读取一帧 joint_q
    with rosbag.Bag(bag_path, "r") as bag:
        found = False
        for topic, msg, t in bag.read_messages():
            if hasattr(msg, "joint_data") and hasattr(msg.joint_data, "joint_q"):
                q = list(msg.joint_data.joint_q)
                print(f"[EXTRINSICS] 读取到 joint_q: {q}")
                found = True
                break
        if not found:
            raise RuntimeError("未找到 joint_q 数据")

    # 2. 关节名与 joint_q 对应（这里只取头部关节，默认采用 joint_q 的末尾两个值，
    #    因为有些数据流把头部放在最后；如果需要其它索引，可传入 joint_names
    if joint_names is None:
        print("[EXTRINSICS] 未传入 joint_names，默认使用头部关节 zhead_1_joint 和 zhead_2_joint")
        joint_names = ["zhead_1_joint", "zhead_2_joint"]
    qpos_dict = {}
    qlen = len(q)
    n_head = len(joint_names)
    # 默认从 joint_q 末尾取对应数量的关节值
    for i, name in enumerate(joint_names):
        idx = qlen - n_head + i
        if 0 <= idx < qlen:
            qpos_dict[name] = float(q[idx])
        else:
            # 退回到从头开始的映射（向后兼容）
            qpos_dict[name] = float(q[i])
    print(f"[EXTRINSICS] joint_q length={qlen}, using indices {[qlen-n_head+i for i in range(n_head)]}")
    print(f"[EXTRINSICS] qpos_dict: {qpos_dict}")

    # 3. MuJoCo FK
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)
    joint_name_to_adr = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j_id): int(model.jnt_qposadr[j_id]) for j_id in range(model.njnt)}
    for name, pos in qpos_dict.items():
        adr = joint_name_to_adr.get(name)
        if adr is not None:
            data.qpos[adr] = pos
    mujoco.mj_forward(model, data)
    print(f"[EXTRINSICS] MuJoCo qpos: {data.qpos}")

    # 4. 获取 base_link（body）和 head_camera（camera）的世界位姿
    bid_base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body)
    print(f"[EXTRINSICS] base_body id: {bid_base}")
    H_world_base = np.eye(4)
    H_world_base[:3, :3] = data.xmat[bid_base].reshape(3, 3)
    H_world_base[:3, 3] = data.xpos[bid_base]

    #   注意：head_camera 是 MuJoCo 的 camera 对象，不是 body，这里按 camera 读取
    cid_cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_body)
    if cid_cam < 0:
        raise KeyError(f"找不到 MuJoCo camera '{camera_body}'")
    print(f"[EXTRINSICS] camera id: {cid_cam}")
    H_world_cam_mj = np.eye(4)
    H_world_cam_mj[:3, :3] = data.cam_xmat[cid_cam].reshape(3, 3)
    H_world_cam_mj[:3, 3] = data.cam_xpos[cid_cam]
    print(f"[EXTRINSICS] H_world_base: {H_world_base}")
    print(f"[EXTRINSICS] H_world_cam(mj): {H_world_cam_mj}")

    # 5. MuJoCo camera 坐标系 -> 物理相机(FLU)坐标系
    #    MuJoCo camera: +X右 +Y上 +Z朝后(看向 -Z)
    #    物理相机(FLU): +X前 +Y左 +Z上
    R_PHYS_FROM_MJ = np.array([
        [ 0.,  0., -1.],
        [-1.,  0.,  0.],
        [ 0.,  1.,  0.],
    ], dtype=float)
    #   右乘时应使用 C_mj_T_C_phys 的旋转，即 R_mj_from_phys = (R_phys_from_mj)^T
    R_MJ_FROM_PHYS = R_PHYS_FROM_MJ.T
    H_cam_phys_in_cam_mj = np.eye(4); H_cam_phys_in_cam_mj[:3, :3] = R_MJ_FROM_PHYS

    #   世界到相机(物理)位姿：在相机局部右乘坐标变换
    H_world_cam_phys = H_world_cam_mj @ H_cam_phys_in_cam_mj

    # 6. 先算 base_link 到 camera（物理）
    H_base_cam_phys = np.linalg.inv(H_world_base) @ H_world_cam_phys
    print(f"[EXTRINSICS] H_base_cam_phys: {H_base_cam_phys}")

    # 7. 再右乘物理到光学的旋转，得到 base_link 到 camera_optical（沿用你原有定义）
    H_physical_optical = np.eye(4)
    H_physical_optical[:3, :3] = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]], dtype=float)
    H_base_camopt = H_base_cam_phys @ H_physical_optical
    print(f"[EXTRINSICS] H_base_camopt: {H_base_camopt}")

    t = H_base_camopt[:3, 3].tolist()
    q = R.from_matrix(H_base_camopt[:3, :3]).as_quat()  # xyzw
    print(f"[EXTRINSICS] translation: {t}")
    print(f"[EXTRINSICS] quaternion_xyzw: {q}")
    return {
        "translation": t,
        "quaternion_xyzw": q.tolist(),
        "base_frame": base_body,
        "camera_frame": "cam_h_color_optical_frame"
    }

# --- 选择模式 ---
if USE_AUTO_EXTRINSICS:
    # 这里 bag_path 需指定一包含头部关节的 bag 文件，默认用 IN_DIR 下第一个 bag
    bags = sorted(glob.glob(os.path.join(IN_DIR, "*.bag")))
    if not bags:
        raise RuntimeError(f"未找到 bag 文件于 {IN_DIR}")
    print(f"[EXTRINSICS] 自动外参计算将使用 bag: {bags[0]}")
    EXTRINSICS = compute_extrinsics_from_bag(
        mjcf_path=MJCF_PATH,
        bag_path=bags[0],
        base_body="base_link",
        camera_body="head_camera",
        joint_names=["zhead_1_joint", "zhead_2_joint"]
    )

# 光学(RDF) <-> 物理(FLU) 的固定旋转
R_PHYSICAL_FROM_OPTICAL = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]], dtype=float)
R_OPTICAL_FROM_PHYSICAL = R_PHYSICAL_FROM_OPTICAL.T

# 末端 site 和 躯干 body
EE_SITE = "right_pinch"
BASE_BODY_NAME = "base_link"  # 按照您确认的设置为 base_link

# --- 关节映射 (基于 /sensors_data_raw/joint_data/joint_q 的固定顺序) ---
L_ARM_START_IDX = 12
MJCF_JOINT_NAMES_LEFT_ARM = ["zarm_l1_joint", "zarm_l2_joint", "zarm_l3_joint", "zarm_l4_joint", "zarm_l5_joint", "zarm_l6_joint", "zarm_l7_joint"]
R_ARM_START_IDX = 19
MJCF_JOINT_NAMES_RIGHT_ARM = ["zarm_r1_joint", "zarm_r2_joint", "zarm_r3_joint", "zarm_r4_joint", "zarm_r5_joint", "zarm_r6_joint", "zarm_r7_joint"]

# ------------------- 常用矩阵/四元数工具 -------------------
def parse_matrix_string(s: str) -> np.ndarray:
    floats = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    nums = [float(x) for x in floats]
    if len(nums) != 16:
        raise ValueError(f"解析到 {len(nums)} 个数，期望 16；原串: {s[:120]!r}")
    return np.array(nums, dtype=float).reshape(4, 4)

def h_from_qt(q_xyzw, t_xyz):
    x,y,z,w = q_xyzw; tx,ty,tz = t_xyz
    n = math.sqrt(x*x+y*y+z*z+w*w)
    if n == 0: n = 1.0
    x,y,z,w = x/n, y/n, z/n, w/n
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]], dtype=float)
    H = np.eye(4); H[:3,:3]=R; H[:3,3]=[tx,ty,tz]
    return H

def quat_from_R(R):
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2; qw = 0.25 * s; qx = (R[2,1] - R[1,2]) / s; qy = (R[0,2] - R[2,0]) / s; qz = (R[1,0] - R[0,1]) / s
    else:
        i = np.argmax([R[0,0], R[1,1], R[2,2]])
        if i == 0:
            s = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2; qx = 0.25 * s; qy = (R[0,1] + R[1,0]) / s; qz = (R[0,2] + R[2,0]) / s; qw = (R[2,1] - R[1,2]) / s
        elif i == 1:
            s = math.sqrt(1.0 - R[0,0] + R[1,1] - R[2,2]) * 2; qx = (R[0,1] + R[1,0]) / s; qy = 0.25 * s; qz = (R[1,2] + R[2,1]) / s; qw = (R[0,2] - R[2,0]) / s
        else:
            s = math.sqrt(1.0 - R[0,0] - R[1,1] + R[2,2]) * 2; qx = (R[0,2] + R[2,0]) / s; qy = (R[1,2] + R[2,1]) / s; qz = 0.25 * s; qw = (R[1,0] - R[0,1]) / s
    q = np.array([qx,qy,qz,qw], dtype=float); q = q / np.linalg.norm(q)
    return q.tolist()

# ------------------- MuJoCo 正解（FK） -------------------
class MujocoKDL:
    def __init__(self, mjcf_path, ee_site, base_body_name):
        if not os.path.exists(mjcf_path): raise FileNotFoundError(mjcf_path)
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data  = mujoco.MjData(self.model)
        self.sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, ee_site)
        if self.sid < 0: raise KeyError(f"找不到 site '{ee_site}'")
        self.bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
        if self.bid < 0: raise KeyError(f"找不到 body '{base_body_name}'")
        self.joint_name_to_adr = {mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j_id): int(self.model.jnt_qposadr[j_id]) for j_id in range(self.model.njnt)}

    def set_qpos(self, mjcf_qpos_dict):
        for mj_name, pos in mjcf_qpos_dict.items():
            adr = self.joint_name_to_adr.get(mj_name)
            if adr is not None:
                self.data.qpos[adr] = float(pos)
        mujoco.mj_forward(self.model, self.data)

    def ee_pose_in_world(self):
        pos = self.data.site_xpos[self.sid].copy(); mat = self.data.site_xmat[self.sid].reshape(3, 3).copy()
        H = np.eye(4); H[:3,:3] = mat; H[:3,3] = pos
        return H
        
    def base_pose_in_world(self):
        pos = self.data.xpos[self.bid].copy(); mat = self.data.xmat[self.bid].reshape(3, 3).copy()
        H = np.eye(4); H[:3,:3] = mat; H[:3,3] = pos
        return H

# ------------------- 目录/枚举 -------------------
def ensure_out_dir(): os.makedirs(OUT_DIR, exist_ok=True)
def list_bags(in_dir): return sorted(glob.glob(os.path.join(in_dir, "*.bag")))

# ------------------- 主处理 -------------------
def process_one_bag(bag_in):
    base_frame = EXTRINSICS.get("base_frame", "base_link")
    H_base_camopt = h_from_qt(EXTRINSICS["quaternion_xyzw"], EXTRINSICS["translation"])
    H_opt_from_phys = np.eye(4); H_opt_from_phys[:3,:3] = R_OPTICAL_FROM_PHYSICAL

    bag_name = os.path.basename(bag_in); out_path = os.path.join(OUT_DIR, bag_name)
    print(f"\n[INFO] 处理: {bag_in}")
    ensure_out_dir()

    obj_t_ros, obj_t_sec, obj_H_cam = [], [], []
    arm_t_sec, arm_q_values = [], []
    total = 0

    with rosbag.Bag(bag_in, "r") as ib, rosbag.Bag(out_path, "w") as ob:
        for topic, msg, t in ib.read_messages():
            ob.write(topic, msg, t)
            total += 1
            if topic == TOPIC_FOUND_IN:
                try: H = parse_matrix_string(msg.data)
                except Exception as e: print(f"[WARN] 解析 {TOPIC_FOUND_IN} 失败：{e}"); continue
                obj_t_ros.append(t); obj_t_sec.append(t.to_sec()); obj_H_cam.append(H)
            
            # 按照您的要求，改用 sensors_data_raw 话题
            elif topic == TOPIC_SENSOR_DATA:
                joint_data = getattr(msg, "joint_data", None)
                if joint_data:
                    q_values = list(getattr(joint_data, "joint_q", []))
                    # 确保数据长度足够
                    if len(q_values) >= R_ARM_START_IDX + len(MJCF_JOINT_NAMES_RIGHT_ARM):
                        arm_t_sec.append(t.to_sec())
                        arm_q_values.append(q_values)

    if not obj_t_sec: print(f"[WARN] 没有 {TOPIC_FOUND_IN}，仅完成拷贝：{out_path}"); return
    have_arm = len(arm_t_sec) > 0
    if not have_arm: print(f"[WARN] 没有 {TOPIC_SENSOR_DATA}，仅生成 base->obj 两个话题。")

    mj = None
    if have_arm:
        try: mj = MujocoKDL(MJCF_PATH, ee_site=EE_SITE, base_body_name=BASE_BODY_NAME)
        except Exception as e: print(f"[WARN] MuJoCo 初始化失败：{e}"); have_arm = False

    new_msgs = []
    obj_t_sec = np.array(obj_t_sec, dtype=float)
    if have_arm: arm_t_sec = np.array(arm_t_sec, dtype=float)

    for i in range(len(obj_t_sec)):
        t_ros = obj_t_ros[i]; t_sec = float(obj_t_sec[i])
        H_cam_obj = obj_H_cam[i]
        if INPUT_FRAME == "physical": H_cam_obj = H_opt_from_phys @ H_cam_obj
        H_base_obj = H_base_camopt @ H_cam_obj

        s = String(); s.data = " ".join(f"{x:.9f}" for x in H_base_obj.reshape(-1))
        ps = PoseStamped(); ps.header.frame_id = base_frame; ps.header.stamp = t_ros
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = H_base_obj[0,3], H_base_obj[1,3], H_base_obj[2,3]
        qx,qy,qz,qw = quat_from_R(H_base_obj[:3,:3])
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = qx,qy,qz,qw
        new_msgs.append((TOPIC_FOUND_OUT_STR, s, t_ros)); new_msgs.append((TOPIC_FOUND_OUT_POSE, ps, t_ros))

        if have_arm:
            j = int(np.argmin(np.abs(arm_t_sec - t_sec)))
            q_values = arm_q_values[j]
            if q_values:
                qpos_dict = {}
                left_arm_q = q_values[L_ARM_START_IDX : R_ARM_START_IDX]
                for name, pos in zip(MJCF_JOINT_NAMES_LEFT_ARM, left_arm_q): qpos_dict[name] = float(pos)
                right_arm_q = q_values[R_ARM_START_IDX : R_ARM_START_IDX + len(MJCF_JOINT_NAMES_RIGHT_ARM)]
                for name, pos in zip(MJCF_JOINT_NAMES_RIGHT_ARM, right_arm_q): qpos_dict[name] = float(pos)
                
                mj.set_qpos(qpos_dict)
                H_world_ee = mj.ee_pose_in_world(); H_world_base = mj.base_pose_in_world()
                H_base_ee = np.linalg.inv(H_world_base) @ H_world_ee

                s2 = String(); s2.data = " ".join(f"{x:.9f}" for x in H_base_ee.reshape(-1))
                ps2 = PoseStamped(); ps2.header.frame_id = base_frame; ps2.header.stamp = t_ros
                ps2.pose.position.x, ps2.pose.position.y, ps2.pose.position.z = H_base_ee[0,3], H_base_ee[1,3], H_base_ee[2,3]
                qx2,qy2,qz2,qw2 = quat_from_R(H_base_ee[:3,:3])
                ps2.pose.orientation.x, ps2.pose.orientation.y, ps2.pose.orientation.z, ps2.pose.orientation.w = qx2,qy2,qz2,qw2
                new_msgs.append((TOPIC_EE_OUT_STR, s2, t_ros)); new_msgs.append((TOPIC_EE_OUT_POSE, ps2, t_ros))

                H_ee_obj = np.linalg.inv(H_base_ee) @ H_base_obj
                s3 = String(); s3.data = " ".join(f"{x:.9f}" for x in H_ee_obj.reshape(-1))
                ps3 = PoseStamped(); ps3.header.frame_id = EE_SITE; ps3.header.stamp = t_ros
                ps3.pose.position.x, ps3.pose.position.y, ps3.pose.position.z = H_ee_obj[0,3], H_ee_obj[1,3], H_ee_obj[2,3]
                qx3,qy3,qz3,qw3 = quat_from_R(H_ee_obj[:3,:3])
                ps3.pose.orientation.x, ps3.pose.orientation.y, ps3.pose.orientation.z, ps3.pose.orientation.w = qx3,qy3,qz3,qw3
                new_msgs.append((TOPIC_REL_OUT_STR, s3, t_ros)); new_msgs.append((TOPIC_REL_OUT_POSE, ps3, t_ros))

    with rosbag.Bag(out_path, "a") as ob:
        for topic, msg, t in new_msgs: ob.write(topic, msg, t)

    print(f"[OK] 完成: 原始消息 {total} 条；新增 {len(new_msgs)} 条。输出 -> {out_path}")

def main():
    ensure_out_dir()
    bags = list_bags(IN_DIR)
    if not bags: print(f"[WARN] 输入目录没有 .bag：{IN_DIR}"); return
    for b in bags: process_one_bag(b)

if __name__ == "__main__":
    main()
