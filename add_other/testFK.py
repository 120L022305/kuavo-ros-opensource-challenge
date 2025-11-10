#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批处理：动态解算相机外参和末端位姿，并追加多个新话题到新 bag。
（修正版：修复 MuJoCo model.jnt_qposadr 索引问题，增加健壮性和调试信息）
"""
import os, glob, re, math
import numpy as np
import rosbag
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import mujoco

# ------------------- 固定路径 -------------------
IN_DIR  = "/root/kuavo_ws/add_other/add_pose"
OUT_DIR = "/root/kuavo_ws/add_other/add_grasp_metrics"
MJCF_PATH = "/root/kuavo_ws/src/data_challenge_simulator/models/biped_s45/xml/biped_s45.xml"

# ------------------- 话题与配置 -------------------
TOPIC_FOUND_IN          = "/foundationpose-hztest"
TOPIC_SENSOR_DATA       = "/sensors_data_raw"
TOPIC_FOUND_OUT_STR     = "/foundationpose_base"
TOPIC_FOUND_OUT_POSE    = "/foundationpose_base_pose"
TOPIC_EE_OUT_STR        = "/ee_base"
TOPIC_EE_OUT_POSE       = "/ee_base_pose"
TOPIC_REL_OUT_STR       = "/ee_to_obj"
TOPIC_REL_OUT_POSE      = "/ee_to_obj_pose"

INPUT_FRAME = "optical"
BASE_BODY_NAME = "base_link"
EE_SITE = "right_pinch"
HEAD_CAMERA_NAME = "head_camera"  # MJCF中定义的头部相机名

# 光学(RDF) <-> 物理(FLU) 的固定旋转
R_OPTICAL_FROM_PHYSICAL = np.array([[0,-1,0], [0,0,-1], [1,0,0]], dtype=float).T

# --- 关节映射 (基于 /sensors_data_raw/joint_data/joint_q 的固定顺序) ---
# 顺序: 12个腿部, 7个左臂, 7个右臂, 2个头部
JOINT_MAPPING = {
    # 左臂
    "zarm_l1_joint": 12, "zarm_l2_joint": 13, "zarm_l3_joint": 14,
    "zarm_l4_joint": 15, "zarm_l5_joint": 16, "zarm_l6_joint": 17, "zarm_l7_joint": 18,
    # 右臂
    "zarm_r1_joint": 19, "zarm_r2_joint": 20, "zarm_r3_joint": 21,
    "zarm_r4_joint": 22, "zarm_r5_joint": 23, "zarm_r6_joint": 24, "zarm_r7_joint": 25,
    # 头部
    "zhead_1_joint": 26, "zhead_2_joint": 27,
}

# ------------------- 矩阵/四元数工具 -------------------
def parse_matrix_string(s: str) -> np.ndarray:
    floats = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if len(floats) != 16: raise ValueError(f"期望16个数字，实际解析到{len(floats)}")
    return np.array([float(x) for x in floats], dtype=float).reshape(4, 4)

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
    q = np.array([qx,qy,qz,qw], dtype=float); q /= np.linalg.norm(q)
    return q.tolist()

# ------------------- MuJoCo 正解（FK） -------------------
class MujocoFKSolver:
    def __init__(self, mjcf_path, base_body, ee_site, head_camera):
        if not os.path.exists(mjcf_path): raise FileNotFoundError(mjcf_path)
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)

        # 获取ID
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, base_body)
        self.ee_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, ee_site)
        self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, head_camera)

        if self.base_id < 0: raise KeyError(f"找不到 body '{base_body}'")
        if self.ee_sid < 0: raise KeyError(f"找不到 site '{ee_site}'")
        if self.cam_id < 0: raise KeyError(f"找不到 camera '{head_camera}'")
        
        # 预计算 MuJoCo 关节名到 qpos 地址的映射（兼容多种 binding）
        self.joint_name_to_qpos_adr = {}
        for j_id in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j_id)
            # 尝试把 model.jnt_qposadr[j_id] 转成整数地址
            adr = -1
            try:
                adr = int(self.model.jnt_qposadr[j_id])
            except Exception:
                try:
                    arr = np.asarray(self.model.jnt_qposadr[j_id])
                    if arr.size > 0:
                        adr = int(arr.ravel()[0])
                except Exception:
                    adr = -1
            if adr >= 0:
                self.joint_name_to_qpos_adr[name] = adr
            else:
                # 固定关节或不可用 qpos 地址，跳过
                pass

        # 调试信息，帮助检查哪些 JOINT_MAPPING 没有被匹配到
        missing = set(JOINT_MAPPING.keys()) - set(self.joint_name_to_qpos_adr.keys())
        if missing:
            print(f"[WARN] 以下 JOINT_MAPPING 中的关节在 MuJoCo 模型中没有对应的 qpos 地址（可能是名字不匹配或是固定关节）：{sorted(missing)}")
        # 打印统计信息（可根据需要注释掉）
        print(f"[INFO] MuJoCo model: njnt={self.model.njnt}, nq={self.model.nq}, mapped_joint_count={len(self.joint_name_to_qpos_adr)}")

    def debug_model(self):
        try:
            print("[MODEL DEBUG] njnt=", self.model.njnt, " nq=", self.model.nq, " nv=", self.model.nv)
            print("[MODEL DEBUG] len(data.qpos)=", len(self.data.qpos))
            # 列出前 N 个 joint 的 qpos adr
            for j_id in range(min(20, self.model.njnt)):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j_id)
                try:
                    adr_raw = self.model.jnt_qposadr[j_id]
                    try:
                        adr = int(adr_raw)
                    except Exception:
                        arr = np.asarray(adr_raw); adr = int(arr.ravel()[0]) if arr.size>0 else -1
                except Exception:
                    adr = -1
                print(f"  j_id={j_id:2d} name={name:30s} qpos_adr={adr}")
        except Exception as e:
            print("[MODEL DEBUG] failed:", e)

    def set_joint_states(self, sensor_q_values):
        """根据/sensors_data_raw的q值列表设置模型关节状态（带边界检查）"""
        n_qpos = len(self.data.qpos)
        for mj_name, sensor_idx in JOINT_MAPPING.items():
            adr = self.joint_name_to_qpos_adr.get(mj_name)
            if adr is None:
                # 没有找到对应 qpos 地址（名字或模型问题）
                # print(f"[DEBUG] joint '{mj_name}' not mapped to qpos (skipped).")
                continue
            if not (0 <= adr < n_qpos):
                print(f"[WARN] joint '{mj_name}' mapped adr {adr} 超出 qpos 长度 {n_qpos}，跳过")
                continue
            if sensor_idx >= len(sensor_q_values):
                print(f"[WARN] sensor_q_values 长度 {len(sensor_q_values)} 不足以索引 sensor_idx={sensor_idx}，跳过 joint {mj_name}")
                continue
            # 设置 qpos
            try:
                self.data.qpos[adr] = float(sensor_q_values[sensor_idx])
            except Exception as e:
                print(f"[WARN] 无法写入 qpos adr {adr} for joint {mj_name}: {e}")
        # 前向传播以更新 xpos/xmat 等
        mujoco.mj_forward(self.model, self.data)

    def get_pose_in_world(self, obj_type, obj_id):
        """通用函数，获取body, site或camera在世界坐标系下的位姿"""
        if obj_type == 'body':
            pos = self.data.xpos[obj_id]
            mat = self.data.xmat[obj_id].reshape(3, 3)
        elif obj_type == 'site':
            pos = self.data.site_xpos[obj_id]
            mat = self.data.site_xmat[obj_id].reshape(3, 3)
        elif obj_type == 'camera':
            pos = self.data.cam_xpos[obj_id]
            mat = self.data.cam_xmat[obj_id].reshape(3, 3)
        else:
            raise ValueError("Unsupported object type")
            
        H = np.eye(4); H[:3, :3] = mat.copy(); H[:3, 3] = pos.copy()
        return H

# ------------------- 目录与主处理 -------------------
def ensure_out_dir(): os.makedirs(OUT_DIR, exist_ok=True)

def process_one_bag(bag_in, fk_solver):
    bag_name = os.path.basename(bag_in)
    out_path = os.path.join(OUT_DIR, bag_name)
    print(f"\n[INFO] 处理: {bag_in}")
    
    # 预加载数据
    obj_data = [] # (t_ros, H_cam_obj)
    arm_data = [] # (t_sec, q_values)
    
    with rosbag.Bag(bag_in, "r") as ib:
        for topic, msg, t in ib.read_messages():
            if topic == TOPIC_FOUND_IN:
                try:
                    H = parse_matrix_string(msg.data)
                    obj_data.append((t, H))
                except Exception as e:
                    print(f"[WARN] 解析 {TOPIC_FOUND_IN} 失败: {e}")
            elif topic == TOPIC_SENSOR_DATA:
                q = getattr(getattr(msg, "joint_data", None), "joint_q", [])
                if len(q) >= max(JOINT_MAPPING.values()) + 1:
                    arm_data.append((t.to_sec(), list(q)))

    if not obj_data:
        print(f"[WARN] Bag中没有找到 {TOPIC_FOUND_IN} 消息, 将只复制原包。")
        os.system(f'cp "{bag_in}" "{out_path}"')
        return
    if not arm_data:
        print(f"[ERROR] Bag中没有找到 {TOPIC_SENSOR_DATA} 消息, 无法进行FK计算，跳过此包。")
        return

    arm_times = np.array([d[0] for d in arm_data])
    arm_q_values = [d[1] for d in arm_data]
    
    new_msgs = []
    for t_ros, H_cam_obj_raw in obj_data:
        # 1. 找到时间最接近的关节数据
        t_sec = t_ros.to_sec()
        best_idx = np.argmin(np.abs(arm_times - t_sec))
        q_values = arm_q_values[best_idx]
        
        # 2. 设置关节状态并执行FK
        fk_solver.set_joint_states(q_values)
        
        # 3. 获取所有需要的世界坐标系位姿
        H_world_base = fk_solver.get_pose_in_world('body', fk_solver.base_id)
        H_world_ee = fk_solver.get_pose_in_world('site', fk_solver.ee_sid)
        H_world_cam = fk_solver.get_pose_in_world('camera', fk_solver.cam_id)
        print(f"[DEBUG] 时间 {t_ros.to_sec():.3f}: base_pos={H_world_base[:3,3]}, ee_pos={H_world_ee[:3,3]}, cam_pos={H_world_cam[:3,3]}")
        
        # 4. 计算相对位姿
        H_base_cam = np.linalg.inv(H_world_base) @ H_world_cam
        H_base_ee = np.linalg.inv(H_world_base) @ H_world_ee
        
        # 5. 处理输入姿态的坐标系 (optical vs physical)
        H_cam_obj = H_cam_obj_raw
        if INPUT_FRAME == "physical":
            H_opt_from_phys = np.eye(4); H_opt_from_phys[:3,:3] = R_OPTICAL_FROM_PHYSICAL
            H_cam_obj = H_opt_from_phys @ H_cam_obj
            
        # 6. 计算最终目标位姿
        H_base_obj = H_base_cam @ H_cam_obj
        H_ee_obj = np.linalg.inv(H_base_ee) @ H_base_obj

        # 7. 生成所有新话题的消息
        # /foundationpose_base
        s1 = String(data=" ".join(f"{x:.9f}" for x in H_base_obj.flatten()))
        qx,qy,qz,qw = quat_from_R(H_base_obj[:3,:3])
        ps1 = PoseStamped()
        ps1.header.stamp = t_ros; ps1.header.frame_id = BASE_BODY_NAME
        ps1.pose.position.x, ps1.pose.position.y, ps1.pose.position.z = H_base_obj[:3,3]
        ps1.pose.orientation.x, ps1.pose.orientation.y, ps1.pose.orientation.z, ps1.pose.orientation.w = qx,qy,qz,qw
        new_msgs.extend([(TOPIC_FOUND_OUT_STR, s1, t_ros), (TOPIC_FOUND_OUT_POSE, ps1, t_ros)])

        # /ee_base
        s2 = String(data=" ".join(f"{x:.9f}" for x in H_base_ee.flatten()))
        qx,qy,qz,qw = quat_from_R(H_base_ee[:3,:3])
        ps2 = PoseStamped()
        ps2.header.stamp = t_ros; ps2.header.frame_id = BASE_BODY_NAME
        ps2.pose.position.x, ps2.pose.position.y, ps2.pose.position.z = H_base_ee[:3,3]
        ps2.pose.orientation.x, ps2.pose.orientation.y, ps2.pose.orientation.z, ps2.pose.orientation.w = qx,qy,qz,qw
        new_msgs.extend([(TOPIC_EE_OUT_STR, s2, t_ros), (TOPIC_EE_OUT_POSE, ps2, t_ros)])

        # /ee_to_obj
        s3 = String(data=" ".join(f"{x:.9f}" for x in H_ee_obj.flatten()))
        qx,qy,qz,qw = quat_from_R(H_ee_obj[:3,:3])
        ps3 = PoseStamped()
        ps3.header.stamp = t_ros; ps3.header.frame_id = EE_SITE
        ps3.pose.position.x, ps3.pose.position.y, ps3.pose.position.z = H_ee_obj[:3,3]
        ps3.pose.orientation.x, ps3.pose.orientation.y, ps3.pose.orientation.z, ps3.pose.orientation.w = qx,qy,qz,qw
        new_msgs.extend([(TOPIC_REL_OUT_STR, s3, t_ros), (TOPIC_REL_OUT_POSE, ps3, t_ros)])
        
    # 8. 写入新bag
    with rosbag.Bag(bag_in, "r") as ib, rosbag.Bag(out_path, "w") as ob:
        total_original = 0
        for topic, msg, t in ib.read_messages():
            ob.write(topic, msg, t)
            total_original += 1
        for topic, msg, t in new_msgs:
            ob.write(topic, msg, t)

    print(f"[OK] 完成: 原始消息 {total_original} 条；新增 {len(new_msgs)} 条。输出 -> {out_path}")

def main():
    ensure_out_dir()
    
    try:
        print(f"[INFO] 正在从 {MJCF_PATH} 初始化FK求解器...")
        fk_solver = MujocoFKSolver(MJCF_PATH, BASE_BODY_NAME, EE_SITE, HEAD_CAMERA_NAME)
        # 可选：打印 model debug 信息，便于检查 qpos 布局
        fk_solver.debug_model()
    except (FileNotFoundError, KeyError) as e:
        print(f"[FATAL] 初始化FK求解器失败: {e}")
        return
        
    bags = glob.glob(os.path.join(IN_DIR, "*.bag"))
    if not bags:
        print(f"[WARN] 输入目录没有 .bag 文件：{IN_DIR}")
        return
        
    for b in sorted(bags):
        process_one_bag(b, fk_solver)

if __name__ == "__main__":
    main()
