## 对应kuavo-ros-opensource下的add_other

包含批处理rosbag包，在已经加入位姿话题的基础上，计算fk，计算obj的基座坐标系下的坐标，计算末端与obj的相对位置和姿态


## sensor_data_raw
### `/sensors_data_raw` 传感器数据

1. 功能描述

`/sensors_data_raw` 话题用于发布实物机器人或仿真器的传感器原始数据，包括关节数据、IMU数据和末端执行器数据。

2. 消息类型

- **类型**: `kuavo_msgs/sensorsData`

3. 消息字段

| 字段               | 类型                        | 描述                              |
| ----------------- | -------------------------- | ------------------------------- |
| sensor_time       | time                       | 时间戳                           |
| joint_data        | kuavo_msgs/jointData       | 关节数据: 位置、速度、加速度、电流 |
| imu_data          | kuavo_msgs/imuData         | 包含陀螺仪、加速度计、自由加速度、四元数 |
| end_effector_data | kuavo_msgs/endEffectorData | 末端数据，暂未使用                |

4. 关节数据说明

- **数组长度**: `NUM_JOINT`
- **数据顺序**:
  - 前 12 个数据为下肢电机数据:
    - 0~5 为左下肢数据 (l_leg_roll, l_leg_yaw, l_leg_pitch, l_knee, l_foot_pitch, l_foot_roll)
    - 6~11 为右下肢数据 (r_leg_roll, r_leg_yaw, r_leg_pitch, r_knee, r_foot_pitch, r_foot_roll)
  - 接着 14 个数据为手臂电机数据:
    - 12~18 左臂电机数据 ("l_arm_pitch", "l_arm_roll", "l_arm_yaw", "l_forearm_pitch", "l_hand_yaw", "l_hand_pitch", "l_hand_roll")
    - 19~25 为右臂电机数据 ("r_arm_pitch", "r_arm_roll", "r_arm_yaw", "r_forearm_pitch", "r_hand_yaw", "r_hand_pitch", "r_hand_roll")
  - 最后 2 个为头部电机数据: head_yaw 和 head_pitch

- **单位**:
  - 位置: 弧度 (radian)
  - 速度: 弧度每秒 (radian/s)
  - 加速度: 弧度每平方秒 (radian/s²)
  - 电流: 安培 (A)

5. IMU 数据说明

- **gyro**: 陀螺仪的角速度，单位弧度每秒（rad/s）
- **acc**: 加速度计的加速度，单位米每平方秒（m/s²）
- **quat**: IMU的姿态（orientation）

---
