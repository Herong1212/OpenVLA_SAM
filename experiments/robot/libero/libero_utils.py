"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os

import imageio
import numpy as np
import tensorflow as tf
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
)


# 创建 LIBERO 虚拟环境（机器人 + 任务场景）
def get_libero_env(task, model_family, resolution=256):
    """
    Initializes and returns the LIBERO environment, along with the task description.

    Args:
        task: Task object containing language description and BDDL file information
        model_family: Model family identifier (unused in current implementation)
        resolution (int): Camera resolution for both height and width, defaults to 256

    Returns:
        tuple: A tuple containing:
            - env: Initialized OffScreenRenderEnv environment instance
            - task_description: Language description of the task from task.language
    """
    # 1、任务的自然语言描述
    task_description = task.language

    # 2、任务的虚拟环境
    task_bddl_file = os.path.join(get_libero_path(
        "bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file,
                "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)

    # NOTE IMPORTANT: seed seems to affect object positions even when using fixed initial state
    env.seed(0)

    return env, task_description


# 生成 “无效动作”（比如让机器人不动）
def get_libero_dummy_action(model_family: str):
    """
    Get dummy/no-op action, used to roll out the simulation while the robot does nothing.

    Args:
        model_family (str): The family of the robot model to determine appropriate dummy action

    Returns:
        list: A list of 7 elements representing the dummy action [0, 0, 0, 0, 0, 0, -1]
              where the first 6 elements are zero motion commands and the last element
              is a gripper close command (-1)
    """
    return [0, 0, 0, 0, 0, 0, -1]


# 调整单张图像的大小，将输入的numpy数组格式图像转换为模型训练时使用的标准尺寸
def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)

    # Resize to image size expected by model
    # Encode as JPEG, as done in RLDS dataset builder, 即: 首先将图像编码为JPEG格式（模仿RLDS数据集构建器的做法）
    img = tf.image.encode_jpeg(img)

    # Immediately decode back, 即: 立即解码回图像格式，确保数据一致性
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
    # 使用"Lanczos3"算法进行高质量缩放，并启用抗锯齿功能
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    # 将像素值限制在 0-255 范围内并转换为8位整数格式
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)

    # 最终返回 numpy 数组格式的调整后图像
    img = img.numpy()
    return img


# 从环境中获取 “模型能识别的图像”（预处理）
def get_libero_image(obs, resize_size):
    """
    从观测数据中提取图像并进行预处理。

    参数:
        obs (dict): 包含观测数据的字典，必须包含键 "agentview_image"
        resize_size (int or tuple): 图像调整大小的尺寸。如果为整数，则认为是正方形尺寸；
                                   如果为元组，则格式为(height, width)

    返回:
        处理后的图像数据

    异常:
        AssertionError: 当 resize_size 既不是整数也不是元组时抛出
    """
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)

    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)

    img = obs["agentview_image"]

    # 重要：旋转 180度 以匹配训练时的预处理方式
    img = img[::-1, ::-1]
    img = resize_image(img, resize_size)

    return img


# 保存试验过程的视频
def save_rollout_video(rollout_images, idx, success, task_description, log_file=None):
    """Saves an MP4 replay of an episode.

    Args:
        rollout_images: 包含回合中所有帧图像的列表
        idx: 回合的索引编号
        success: 任务是否成功的布尔值
        task_description: 任务描述字符串
        log_file: 可选的日志文件对象，用于记录保存路径

    Returns:
        str: 保存的MP4文件的完整路径
    """
    # 创建回放视频保存目录
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)

    # 处理任务描述字符串，用于文件命名
    processed_task_description = task_description.lower().replace(
        " ", "_").replace("\n", "_").replace(".", "_")[:50]

    # 构造MP4文件保存路径
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"

    # 写入视频帧数据
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()

    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


# 四元数转轴角（机器人姿态格式转换）
def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion to valid range [-1, 1] for arccosine calculation
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
