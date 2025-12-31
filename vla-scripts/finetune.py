"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    参数详解：
        --standalone: 告诉 PyTorch 我是单机运行，不需要去连接其他服务器的主节点；
        --nnodes 1: 只有 1 台机器（就是你这一台）；
        --nproc-per-node $K: $K 代表你要用几张卡，将 nproc-per-node 设置为可用 GPU 数量
            如果你想 8 卡全开：就把 $K 换成 8。
            如果你只想用前 2 张卡：就把 $K 换成 2。
        
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...
"""

import os
from datetime import datetime
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# 一个基于 dataclass 的参数解析库，比 argparse 更高级，支持层级配置
import draccus

import torch
import torch.distributed as dist  # 用于多卡训练时的进程间通信
import tqdm

from accelerate import PartialState  # HuggingFace Accelerate 库，简化多 GPU 设备管理

# PEFT (Parameter-Efficient Fine-Tuning) 库核心组件：
#   - LoraConfig: 配置 LoRA 的参数（如秩 r, alpha）
#   - PeftModel: PEFT 模型的包装类
#   - get_peft_model: 将基础模型包装成 PEFT 模型（冻结原参数，插入 LoRA 层）
#   - prepare_model_for_kbit_training: 专为量化训练（QLoRA）做的预处理，比如稳定 LayerNorm
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader

# Transformers 库核心：
#   - AutoModelForVision2Seq: 自动加载视觉-语言模型（VLM）
#   - AutoProcessor: 自动加载处理器（包含 Tokenizer 和 ImageProcessor）
#   - BitsAndBytesConfig: 用于配置 4-bit/8-bit 量化的参数
#   - Autoconfig：加载模型的 “配置” (config.json)，而不加载那几十 GB 的权重文件
#   - AutoImageProcessor：加载图像的 “预处理器”，决定图片在喂给模型前怎么处理
#   - CausalLMOutputWithPast: 一个数据类 (Data Class)，专门用来定义模型 “输出结果”的格式
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

# 用于实验记录和可视化的工具
import wandb

from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Sane Defaults
# 强制关闭 Hugging Face tokenizers 库内部的多线程并行功能
# 原因：Hugging Face 的分词器（Tokenizer）默认是用 Rust 写的，为了快，它自己会在后台开多线程；
# 冲突：但是，你的数据加载器 DataLoader（基于 PyTorch 或 RLDS）通常也会开多进程 (num_workers) 来读取数据；
# 后果：当“多进程”里面嵌套“多线程”时，在 Linux 系统下极易发生 死锁(Deadlock)，导致程序卡死不动，或者 CPU 占用率 100 % 却不干活；
# 解决：因此，这里将 Hugging Face tokenizers 库内部多线程并行功能关闭，防止训练过程中程序莫名其妙卡死。
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# # === Utilities ===
# # fmt: off
# def create_vision_transform(vla: nn.Module, input_size: int) -> Callable[[Image.Image], torch.Tensor]:
#     """Gets image transform for the vision encoder."""
#     data_cfg = timm.data.resolve_model_data_config(vla.vision_backbone)
#     data_cfg["input_size"] = (3, input_size, input_size)
#     return timm.data.create_transform(
#         input_size=data_cfg["input_size"],
#         interpolation=data_cfg["interpolation"],
#         mean=data_cfg["mean"],
#         std=data_cfg["std"],
#         crop_pct=1.0,           # Set to 1.0 to disable cropping
#         crop_mode="center",     # Default crop mode --> no-op when `crop_pct == 1.0`
#         is_training=False,      # Disable image_aug when loading transform; handled by RLDS dataloader
#     )
#
# # fmt: on


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("modified_libero_rlds")              # Path to Open-X dataset directory，即：数据集路径。这里必须填 RLDS 格式数据的根目录。
    dataset_name: str = "libero_spatial_no_noops"                   # Name of fine-tuning dataset (e.g., `droid_wipe`)，即：数据集名称。对应 RLDS 目录下的子文件夹名。
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints，即：实验结果存放处。日志、训练过程中的 Checkpoint、最终模型都会存在这。
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing，即：临时目录。OpenVLA 训练时会先保存 LoRA 的小文件到这里，然后再把它们合并到底座模型里存到 run_root_dir。

    # Fine-tuning Parameters
    batch_size: int = 4                                             # Fine-tuning batch size，即：单张显卡上的 Batch Size
    max_steps: int = 200_000                                        # Max number of fine-tuning steps，即：最大训练步数（Step），不是 Epoch
    save_steps: int = 5000                                          # Interval for checkpoint saving，即：每隔多少步保存一次检查点（Checkpoint）
    learning_rate: float = 5e-4                                     # Fine-tuning learning rate，即：学习率 ———— 5e-4 是 LoRA 微调的标准值

    # 梯度累积步数。如果显存只能跑 batch_size=2，但你想达到 batch_size=16 的效果，就把这个设为 8 (2*8=16)。它会累积 8 次前向传播的梯度后，才执行一次参数更新
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps，即：梯度累积步数 ———— 显存不够时，累积几次梯度再更新一次参数，变相增大 Batch Size

    image_aug: bool = True                                          # Whether to train with image augmentations，即：图像增强 ———— 是否在训练时随机改变亮度、对比度等，增加模型泛化性
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)，即：RLDS 数据流的随机缓冲区大小。越大随机性越好，但越占内存。

    # 如果为 True，每次保存时会覆盖上一次的检查点，只留最新的。节省硬盘空间。如果是 False，则保存所有模型
    save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run and continually overwrite the latest checkpoint (If False, saves all checkpoints)

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning，即：是否使用 LoRA。OpenVLA 强烈建议 True。
    lora_rank: int = 32                                             # Rank of LoRA weight matrix，即：秩（Rank）———— LoRA 矩阵的维度，越大参数越多，拟合能力越强但显存占用越高。决定了微调参数量。32 是平衡点，太小拟合不够，太大显存不够。
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights，即：Dropout 率 ———— 防止过拟合的随机失活比例。0.0 表示不使用。

    use_quantization: bool = True                                   # Whether to 4-bit quantize VLA for LoRA fine-tuning，即：量化开关 ———— 是否使用 4-bit 量化加载底座模型（即 QLoRA），极大节省显存
                                                                    #  ps：=> CAUTION: Reduces memory but hurts performance ——> 会降低一些性能

    # Tracking Parameters
    wandb_project: str = "openvla-debug"                            # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"                          # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases

    # fmt: on


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(
        f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    # [校验] 必须有 GPU
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"

    # 初始化分布式状态。PartialState 是 accelerate 库提供的工具，处理多卡分布式。
    # 它能自动识别你是单卡运行还是多卡 DDP 运行，并获取当前进程的 ID (local_process_index)。
    distributed_state = PartialState()

    # 设置当前进程使用的 GPU ID
    # 强制设定当前进程只使用分配给它的那块 GPU。比如在 8 卡机器上，进程 3 就只能看到 GPU 3。
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    # 清空显存碎片
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    # Notice [构建实验 ID] 生成一个类似 "openvla-7b+libero_spatial+b16+lr-5e-4..." 这样包含所有关键参数的字符串
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}" f"+b{cfg.batch_size * cfg.grad_accumulation_steps}" f"+lr-{cfg.learning_rate}")
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    exp_id += f"--{timestamp}"

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    # 如果开启量化，模型将以 4-bit 精度加载
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,  # 开启 4-bit 量化
            # 重要！计算时将上面 4-bit 反量化回 bfloat16 进行计算，保证精度
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"  # 使用 NormalFloat4 数据类型，针对正态分布的神经网络权重设计的量化数据类型，比标准的线性量化精度更高
        )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    # ps 注册 OpenVLA 专属配置类。由于 OpenVLA 是斯坦福团队自定义的模型结构，故需要显式注册到系统的自动加载器中
    AutoConfig.register("openvla", OpenVLAConfig)
    # 加载一系列处理器 (Processor)，包含 Tokenizer (处理文本) 和 ImageProcessor (处理图像)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    # trust_remote_code=True: 必须为 True。允许从模型文件夹里加载 `modeling_openvla.py` 这样的 python 代码并执行。
    processor = AutoProcessor.from_pretrained(
        cfg.vla_path, trust_remote_code=True)

    # Notice 加载原始大模型 OpenVLA-7B
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,  # 模型路径
        torch_dtype=torch.bfloat16,  # 显式指定权重精度使用 BF16 精度加载，防止默认 FP32 撑爆显存
        quantization_config=quantization_config,  # 传入上面的量化配置
        low_cpu_mem_usage=True,  # 优化 CPU 内存加载策略（分层加载），避免一次性占用过多 CPU 内存
        trust_remote_code=True,  # 允许运行模型仓库里的自定义 Python 代码
    )

    # Notice 【微调的核心】LoRA 适配器挂载 (PEFT Setup)
    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        # prepare_model_for_kbit_training: 对量化模型进行一系列处理（如开启梯度检查点、转换 LayerNorm 精度），使其可以被训练
        # 这个函数的作用包括：
        #   1. 冻结所有参数。
        #   2. 将 LayerNorm 层强制转回 float32（保证稳定性）。
        #   3. 开启 gradient checkpointing（以计算换显存）。
        vla = prepare_model_for_kbit_training(vla)
        # 【新增这行】强制使用 use_reentrant=False 来解决 DDP 报错
        vla.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    else:
        # 若未开启量化，则直接移入 GPU
        vla = vla.to(device_id)

    # Notice [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,  # 秩 (Rank)，决定了可训练参数量的大小。32 是个适中的值 (两个小矩阵的宽度)
            lora_alpha=min(cfg.lora_rank, 16),  # 缩放系数。通常设置为 r 的 1倍或 2倍
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",  # 将 LoRA 适配器挂载到模型中所有的线性层（包括 Q, K, V, Projection, FFN）
            init_lora_weights="gaussian",  # 使用高斯分布初始化 LoRA 权重
        )

        # Notice 【核心动作】把 LoRA 挂载到模型上去，即：冻结原模型的所有参数，只将 LoRA 参数设为可训练（requires_grad=True）
        # 它会在原模型外部套一层 PeftModelWrapper。此时，vla.parameters() 里只有 LoRA 的 A, B 矩阵是 requires_grad=True 的，原有的 7B 参数全部变成了 requires_grad=False。
        vla = get_peft_model(vla, lora_config)

        # 打印可训练参数量，确认只有极少比例（如0.2%）的参数参与训练
        vla.print_trainable_parameters()

    # ps 多卡并行训练：Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    # find_unused_parameters=True: 允许模型中有部分参数在前向传播中未被使用（OpenVLA 中常见）
    vla = DDP(
        vla,
        device_ids=[device_id],
        find_unused_parameters=True,
        gradient_as_bucket_view=True  # 减少内存拷贝，优化显存
    )

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    # 创建优化器。过滤出 requires_grad=True 的参数，只传入 trainable_params（也就是只有 LoRA 参数）
    trainable_params = [param for param in vla.parameters()
                        if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Create Action Tokenizer：将连续的机械臂动作数值（如 x=0.25）离散化为 Token ID（如 512）
    # ps 这是 OpenVLA/RT-2 架构的核心，把动作预测变成了“完形填空”的分类问题
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # vla_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    # )
    # ---

    # RLDSBatchTransform: 这是一个回调函数类
    # 定义数据预处理逻辑：图片缩放、指令添加 Prompt 模板等
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        # 构建提示词模板（比如 "USER: What to do? ASSISTANT: <action>"）
        image_transform=processor.image_processor.apply_transform,
        # 构建提示词模板（比如 "USER: What to do? ASSISTANT: <action>"）
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )

    # RLDSDataset: 实际上是调用的 dlimp 库去专门读取 TFRecord 格式的 RLDS 数据
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),  # 再次确认图片尺寸
        shuffle_buffer_size=cfg.shuffle_buffer_size,  # 决定数据打乱的程度
        image_aug=cfg.image_aug,
    )

    # Notice [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoader：整理器
    # 因为每个 batch 里的指令长度可能不一样（有的指令长有的短），Collator 负责用 pad_token 将它们填充对齐到当前 batch 的最大长度，这样才能堆叠成 Tensor。
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right")

    # 通常情况 PyTorch 训练我们会设为 4 或 8。
    # 但此处必须设为 0！因为 RLDS 底层依赖 TensorFlow 的 data loader，它自己内部已经有多线程并行了。
    # 如果 PyTorch 再开多进程去 fork TensorFlow 的进程，会导致死锁或显存爆炸！
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,  # 负责将不同长度的数据 Padding 到同一长度
        num_workers=0,  # ps Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity,
                   project=cfg.wandb_project, name=f"ft+{exp_id}")

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    # 用于平滑日志曲线的队列
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

    # Note Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:

        # 进入训练模式
        vla.train()

        optimizer.zero_grad()

        # 开始循环 Batch
        for batch_idx, batch in enumerate(dataloader):
            # 1. 开启混合精度上下文 (BF16)，自动将计算转为 bfloat16，加快速度并减少显存
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # 详解：OpenVLA 本质上是一个“因果语言模型”（Causal LM，就像 GPT 一样）。当运行模型时，它返回的不仅仅是一个数字，而是一个包含多个字段的包裹：
                #   loss: 损失值（训练时用来反向传播）。
                #   logits: 模型预测下一个 Token 的概率分布。
                #   past_key_values: KV Cache（用于加速推理的缓存，虽然微调时通常用不到，但结构里必须有）。
                # 作用：为了 类型提示 (Type Hinting)。告诉阅读代码的人和 IDE 工具：“此 vla() 函数跑完后，会输出一个标准的 LM 输出对象”，方便开发人员知道怎么去取里面的 .loss 或 .logits
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),  # 文本指令
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(
                        torch.bfloat16).to(device_id),  # 图像数据 tensor
                    labels=batch["labels"],  # 动作的真值（Ground Truth），用于计算 Loss
                )

                # 模型内部自动计算 CrossEntropyLoss
                loss = output.loss

            # 梯度累积归一化：因为 loss 是累积多次才 step，所以这里要除以累积步数，保持梯度尺度一致
            normalized_loss = loss / cfg.grad_accumulation_steps

            # 反向传播 (Backward)：计算梯度
            normalized_loss.backward()

            ################################ 日志记录相关，可忽略👇 ################################
            # Compute Accuracy and L1 Loss for Logging
            action_logits = output.logits[:,
                                          vla.module.vision_backbone.featurizer.patch_embed.num_patches: -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx

            # Compute Accuracy
            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()

            # Compute L1 Loss on Predicted (Continuous) Actions
            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy()))
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy()))
            action_l1_loss = torch.nn.functional.l1_loss(
                continuous_actions_pred, continuous_actions_gt)

            # Store recent train metrics
            recent_losses.append(loss.item())
            recent_action_accuracies.append(action_accuracy.item())
            recent_l1_losses.append(action_l1_loss.item())

            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            #   =>> Equal to current step metrics when not using gradient accumulation
            #   =>> Otherwise, equal to the average of metrics observed over micro-batches used for gradient accumulation
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_accuracy = sum(
                recent_action_accuracies) / len(recent_action_accuracies)
            smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)

            # Push Metrics to W&B (every 10 gradient steps)
            if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
                wandb.log(
                    {
                        "train_loss": smoothened_loss,
                        "action_accuracy": smoothened_action_accuracy,
                        "l1_loss": smoothened_l1_loss,
                    },
                    step=gradient_step_idx,
                )
            ####################################################################################

            # Notice Optimizer Step：优化器更新
            # 只有当累积了足够的步数后，才真正更新一次参数。
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()  # 清空梯度，防止累积到下一轮
                progress.update()  # 进度条 +1

            # Notice 【Save Model Checkpoint】=>> by default, only keeps the latest checkpoint, continually overwriting it!
            # ps 这是 OpenVLA 复现中最独特的一步！
            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                if distributed_state.is_main_process:
                    print(
                        f"Saving Model Checkpoint for Step {gradient_step_idx}")

                    # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
                    save_dir = adapter_dir if cfg.use_lora else run_dir

                    # Save Processor & Weights
                    # step1. 保存 Processor (包含 Tokenizer 配置等)，此时只保存了几百 MB 的 adapter 文件夹
                    processor.save_pretrained(run_dir)
                    # step2. 保存 Adapter (LoRA 权重) 到临时目录
                    vla.module.save_pretrained(save_dir)

                # 分布式同步屏障：确保主进程保存完，其他进程再继续，防止文件读写冲突。
                dist.barrier()

                # Merge LoRA weights into model backbone for faster inference
                #   =>> Note that merging is slow and can be done post-hoc to speed up training
                # step3. [合并权重] Merge Logic！！！
                # 为了推理方便，我们不希望每次推理都分别加载 Base 模型和 Adapter。所以这里做了一个“融合”操作。
                if cfg.use_lora:
                    # 重新加载一个干净的、未冻结的底座模型 (base_vla)
                    base_vla = AutoModelForVision2Seq.from_pretrained(
                        cfg.vla_path,
                        torch_dtype=torch.bfloat16,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True
                    )

                    # 加载刚才保存的 LoRA 权重
                    merged_vla = PeftModel.from_pretrained(
                        base_vla, adapter_dir)

                    # 核心函数 merge_and_unload(): 将 LoRA 的矩阵乘积加回到原模型的权重矩阵中
                    # 数学原理：W_new = W_base + (A * B * scale)
                    # 执行完后，merged_vla 就变成了一个普通的模型，没有 LoRA 层了，但权重已经包含了微调的信息。
                    # 结果是一个结构与原模型完全一致，但参数已更新的标准模型。
                    merged_vla = merged_vla.merge_and_unload()

                    if distributed_state.is_main_process:
                        # step4. 保存最终的融合模型
                        if cfg.save_latest_checkpoint_only:
                            # Overwrite latest checkpoint：保存最终的全量模型，这样推理时就不需要加载两个文件，直接加载这一个大模型即可
                            merged_vla.save_pretrained(run_dir)

                            print(
                                f"Saved Model Checkpoint for Step {gradient_step_idx} at: {run_dir}")
                        else:
                            # Prepare to save checkpoint in new directory
                            checkpoint_dir = Path(
                                str(run_dir) + f"--{gradient_step_idx}_chkpt")
                            os.makedirs(checkpoint_dir, exist_ok=True)

                            # Save dataset statistics to new directory
                            save_dataset_statistics(
                                vla_dataset.dataset_statistics, checkpoint_dir)

                            # Save processor and model weights to new directory
                            processor.save_pretrained(checkpoint_dir)
                            merged_vla.save_pretrained(checkpoint_dir)

                            print(
                                f"Saved Model Checkpoint for Step {gradient_step_idx} at: {checkpoint_dir}")

                # Block on Main Process Checkpointing
                dist.barrier()

            # Stop training when max_steps is reached
            if gradient_step_idx == cfg.max_steps:
                print(
                    f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()
