"""Utils for evaluating robot policies in various environments."""

import os
import random
import time

import numpy as np
import torch

from experiments.robot.openvla_utils import (
    get_vla,
    get_vla_action,
)

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


# 设置随机种子（让试验结果可复现）
def set_seed_everywhere(seed: int):
    """
    设置 Python、NumPy 和 PyTorch 的随机种子，确保实验结果可复现
    
    参数:
        seed (int): 随机种子值
        
    返回值:
        无
    """
    # 设置 PyTorch 的 CPU 和 GPU 随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 设置 NumPy 和 Python 标准库的随机种子
    np.random.seed(seed)
    random.seed(seed)
    
    # 设置 CUDA 的确定性选项以确保结果可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置 Python 哈希种子
    os.environ["PYTHONHASHSEED"] = str(seed)


# 加载预训练的 OpenVLA 模型（“唤醒 AI 大脑”）
def get_model(cfg, wrap_diffusion_policy_for_droid=False):
    """
    加载用于评估的模型。
    
    参数:
        cfg: 配置对象，包含模型配置信息
        wrap_diffusion_policy_for_droid: 布尔值，是否为 droid 包装扩散策略，默认为 False
    
    返回:
        加载的模型对象
    
    异常:
        ValueError: 当配置中的 model_family 不是预期值时抛出
    """
    
    # 根据模型家族类型加载对应的模型
    if cfg.model_family == "openvla":
        model = get_vla(cfg)
    else:
        raise ValueError("Unexpected [model_family](file:///root/private/openvla/prismatic/models/backbones/llm/prompting/base_prompter.py#L0-L0) found in config.")
    
    print(f"Loaded model: {type(model)}")
    
    return model


# 获取模型要求的图像尺寸（统一输入）
def get_image_resize_size(cfg):
    """
    Gets image resize size for a model class.
    If `resize_size` is an int, then the resized image will be a square. Else, the image will be a rectangle.

    Args:
        cfg: 配置对象，包含模型配置信息

    Returns:
        int: 图像调整大小的尺寸，用于模型输入

    Raises:
        ValueError: 当配置中的 model_family 不被支持时抛出异常
    """
    # 根据模型系列确定图像 resize 尺寸
    if cfg.model_family == "openvla":
        resize_size = 224
    else:
        raise ValueError(
            "Unexpected [model_family](file:///root/private/openvla/prismatic/models/backbones/llm/prompting/base_prompter.py#L0-L0) found in config.")
    
    return resize_size


# 调用模型，生成机器人的动作数组（比如 [dx, dy, dz, dθ, gripper_action]，表示夹爪的位置变化、姿态变化、张开 / 闭合动作）
def get_action(cfg, model, obs, task_label, processor=None):
    """
    Queries the model to get an action.

    Args:
        cfg: Configuration object containing model settings and parameters
        model: The trained model used for action prediction
        obs: Observation input data for the model
        task_label: Label indicating the specific task to perform
        processor: Optional processor for handling input data preprocessing

    Returns:
        action: Predicted action array with shape (ACTION_DIM,)

    Raises:
        ValueError: When an unexpected model family is found in the configuration
    """
    # Handle different model families for action prediction
    if cfg.model_family == "openvla":
        action = get_vla_action(
            model, processor, cfg.pretrained_checkpoint, obs, task_label, cfg.unnorm_key, center_crop=cfg.center_crop
        )
        assert action.shape == (ACTION_DIM,)
    else:
        raise ValueError(
            "Unexpected [model_family](file:///root/private/openvla/prismatic/models/backbones/llm/prompting/base_prompter.py#L0-L0) found in config.")

    return action


# Note 归一化夹爪动作 —— 模型输出的夹爪动作是 [0,1] 范围，环境需要 [-1,1] 范围
def normalize_gripper_action(action, binarize=True):
    """
    将夹爪动作（动作向量的最后一维）从[0,1]范围转换到[-1,+1]范围。
    对于某些环境（非Bridge环境）是必要的，因为数据集包装器将夹爪动作标准化为[0,1]范围。
    需要注意的是，与其他动作维度不同，夹爪动作默认不会被数据集包装器标准化到[-1,+1]范围。
    
    标准化公式: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1

    参数:
        action: 动作数组，最后一个维度表示夹爪动作，数值范围应在[0,1]之间
        binarize: 布尔值，是否将结果二值化为-1或+1，默认为 True

    返回:
        归一化后的动作数组，最后一个维度的数值范围为[-1,1]或二值化的 -1/+1
    """
    # 将最后一个动作维度从[0,1]标准化到[-1,+1]
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / \
        (orig_high - orig_low) - 1

    if binarize:
        # 将夹爪动作二值化为-1（关闭）或+1（打开）
        action[..., -1] = np.sign(action[..., -1])

    return action


# Note 反转夹爪动作符号 —— 模型训练时，夹爪动作的定义是 “0 = 闭合，1 = 张开”，但环境的定义是 “-1 = 张开，+1 = 闭合”，所以需要反转符号，否则模型让夹爪 “张开”，环境会执行 “闭合”。
def invert_gripper_action(action):
    """
    反转夹爪动作符号以适配不同的动作约定。
    
    在模型训练中，夹爪动作定义为 0=闭合, 1=张开，而在环境中则定义为 -1=张开, +1=闭合，因此需要反转最后一维的符号来保持一致性。

    Args:
        action: 动作数组，最后一个维度表示夹爪动作。
               夹爪动作值应该在范围 [-1, 1] 内。

    Returns:
        action: 修改后的动作数组，夹爪动作（最后一个维度）的符号已被反转以匹配环境的动作约定。
    """
    # Flip the sign of the gripper action (last dimension) to adapt between 
    # different action conventions used by the model and environment
    # 反转夹爪动作（最后一维）的符号，以适配模型和环境之间不同的动作约定
    action[..., -1] = action[..., -1] * -1.0
    return action
