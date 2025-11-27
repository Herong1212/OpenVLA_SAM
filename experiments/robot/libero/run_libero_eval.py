"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

# ps1：Basic dependencies
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys

sys.path.append("/root/private/LIBERO")
# print("当前Python解释器路径：", sys.executable)
# print("当前环境的搜索路径：", sys.path)

# 快速创建 “配置类”（用来管理脚本的所有参数，不用写一堆变量定义）
from dataclasses import dataclass

# 更方便地处理文件路径（比 os.path 更简洁）
from pathlib import Path

# 指定变量的 “类型注解”（让代码更易读，比如 Optional[str] 表示变量可以是字符串或 None）
from typing import Optional, Union

# ps2：Third-party dependency
# 解析命令行参数的工具（把命令行输入的--model_family openvla这类参数，自动传给配置类）
import draccus
import numpy as np
import tqdm

from libero.libero import benchmark

# Weights & Biases 工具，wandb 是专门的「实验跟踪 + 可视化工具」，用于记录、可视化评估过程和结果，可选启用（默认不启用）；
# 不启用的话，评估结果会默认保存在本地 results/ 目录（文本格式），也能查看成功率，只是没有可视化图表
import wandb

# ps3：Custom tools
# Append current directory so that interpreter can find experiments.robot
# 脚本在 experiments/robot/libero/ 目录，要导入上级目录的 experiments.robot 模块，必须把 “上级的上级目录” 加入搜索路径，否则 Python 找不到
sys.path.append("../..")
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)


# 用@dataclass装饰器，把所有脚本参数整理成一个 “参数容器”，方便管理和修改
@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path，即：必填项，不能为空！
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization，即：与上面的 load_in_8bit 不能同时设为 True（会冲突）

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug) ，即：必须设置，因模型训练时使用了随机裁剪增强（90% 区域），测试时需对应使用中心 90% 裁剪

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim，即：仿真环境初始化后，等待物体稳定的步数
    num_trials_per_task: int = 50                    # Number of rollouts per task，即：默认每个任务做 50次

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging，即：日志文件名的额外备注（方便区分不同试验）
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases，即：若需可视化日志，添加该参数并指定 --wandb_project 和 --wandb_entity
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


# Note draccus 的装饰器，作用是 “自动解析命令行参数，生成配置类对象”
@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    # step1: 参数校验（避免配置错误）
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    # 如果模型路径里包含 “image_aug”（说明模型训练时用了图像增强），就强制要求 center_crop=True，否则报错
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    # 检查不能同时启用 8 位和 4 位量化，否则报错
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # step2: Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    # 给配置类加一个新属性 unnorm_key（动作反归一化的键），值等于任务套件名称,
    # 因为模型输出的动作是 “归一化后的值”（比如 [-1,1]），需要根据任务套件的规则还原成 “环境能识别的动作”（比如机器人关节角度）
    cfg.unnorm_key = cfg.task_suite_name

    # Notice Load model, 把预训练好的模型加载到 GPU/CPU 中，准备后续决策动作
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    # 校验模型是否有 “动作反归一化的规则”：
    #   ps1: OpenVLA 模型训练时会保存 norm_stats（归一化统计信息，比如动作的最大值、最小值），unnorm_key 是获取这些信息的 “钥匙”。
    #   ps2: 如果 unnorm_key 不在模型的 norm_stats 里，但带 _no_noops 后缀的钥匙在，就自动修改 unnorm_key（处理数据集修改的情况）。
    #   ps3: 最后断言钥匙必须存在，否则报错（确保动作能正确还原）。
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    # 加载 OpenVLA 的 “输入处理器”：
    #   ps1: 处理器的作用是 “把环境的原始图像、任务描述，转成模型能读懂的格式”（比如图像缩放到 224×224，文本转成数字编码）。
    #   ps2: 只有 OpenVLA 需要这个处理器，所以加了模型家族的判断。
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    # 如果额外指定了 run_id_note
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # step3: Initialize LIBERO task suite, 这部分不是很懂 ?
    # 获取 LIBERO 所有可用的任务套件字典（键是任务套件名称，值是任务套件类）
    benchmark_dict = benchmark.get_benchmark_dict()
    # 根据配置的 task_suite_name，创建对应的任务套件对象（比如 libero_spatial 任务套件）
    task_suite = benchmark_dict[cfg.task_suite_name]()
    # 获取该任务套件包含的 “具体任务数量”（比如 libero_spatial 有 10 个具体任务）, int 类型
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    # 打印并写入日志，告诉用户当前评估的任务套件
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    # 根据配置获取图像预处理后的尺寸（比如 224×224），后续所有图像都会缩放到这个尺寸
    resize_size = get_image_resize_size(cfg)

    # step4: Start evaluation!@!
    total_episodes = 0  # 总试验次数（所有任务的试验次数之和）
    total_successes = 0  # 总成功次数（所有任务的成功次数之和）

    # 循环每个具体任务：tqdm.tqdm(...)显示任务进度条（比如 10 个任务，完成 1 个就显示 10%）
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        # 获取当前任务 ID 对应的具体任务对象（比如 “把红色方块移到左边”）
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        # 获取该任务的 “默认初始状态”（比如物体的初始位置、机器人的初始姿态，每个试验的初始状态可能不同）
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        # 创建该任务的虚拟环境和任务描述（比如环境包含机器人、红色方块、目标位置；任务描述是 “Move the red block to the left platform”）
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        # Start episodes
        task_episodes = 0  # 当前任务的试验次数
        task_successes = 0  # 当前任务的成功次数

        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")

            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()

            # Set initial states
            # 设置当前试验的初始状态（每个试验的初始状态可能不同，保证评估的泛化性）
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0  # 当前试验的步数计数器（记录机器人已经执行了多少步）
            replay_images = []  # 保存试验过程的图像（后续用来生成视频）

            # 每个试验的最大步数（根据任务套件设置，参考训练时的最长演示步数，避免机器人无限循环）
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")

            # 开始试验的动作循环：while 循环条件是 “当前步数 < 最大步数 + 等待步数”（等待步数是前面说的让物体稳定的步数）
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # Notice IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    # Get preprocessed image --- 从环境观测中提取图像，缩放到模型要求的尺寸
                    # ? obs 的结构是怎样的?
                    img = get_libero_image(obs, resize_size)

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Note1: Prepare observations dict
                    # Notice: OpenVLA does not take proprio state as input
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (
                                # 机器人末端执行器（夹爪）的位置（x,y,z）
                                obs["robot0_eef_pos"],
                                # 机器人夹爪的姿态（四元数转轴角）
                                quat2axisangle(obs["robot0_eef_quat"]),
                                # 夹爪的位置（比如张开程度）
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                    }

                    # Note2: Query model to get action
                    action = get_action(
                        cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                    )

                    # ps1 Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    action = normalize_gripper_action(action, binarize=True)

                    # ps2 [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    # Execute action in environment
                    # 把动作数组转成列表，让环境执行（机器人真的动起来）
                    obs, reward, done, info = env.step(action.tolist())
                    # 如果任务完成（done=True），更新当前任务和全局的成功次数，跳出动作循环（该试验结束）
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                # 异常处理：如果试验过程中报错（比如模型崩溃、环境异常），打印并记录错误信息，跳出循环（不影响其他试验）
                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            # 更新当前任务和全局的试验次数
            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
            )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")

            # 强制把日志写入文件（避免缓存导致日志丢失）
            log_file.flush()

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )

    # Save local log file
    log_file.close()  # 关闭日志文件（确保所有内容都写入）

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()
