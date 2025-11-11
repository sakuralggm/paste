# Copyright 2025 The HuggingFace Team. All rights reserved.
# 版权所有 2025 HuggingFace 团队。保留所有权利。
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache 许可证 2.0 版本（"许可证"）获得许可；
# you may not use this file except in compliance with the License.
# 除非遵守许可证，否则您不得使用此文件。
# You may obtain a copy of the License at
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# 除非适用法律要求或书面同意，
# distributed under the License is distributed on an "AS IS" BASIS,
# 否则根据许可证分发的软件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 是以"按原样"基础分发的，不附带任何明示或暗示的保证或条件。
# See the License for the specific language governing permissions and
# 参见许可证，了解有关管辖许可的特定语言
# limitations under the License.
# 和限制。
import time  # 导入 time 模块，用于时间相关操作
import os  # 导入 os 模块，用于与操作系统交互（例如文件路径）
import textwrap  # 导入 textwrap 模块，用于文本格式化（如自动换行）
from collections import defaultdict  # 从 collections 导入 defaultdict，一种带默认值的字典
from typing import Any, Callable, Optional, Union  # 导入类型提示，用于代码注解

from trl import create_reference_model  # 从 trl 库导入 (这行在下面又导入了一次)
import numpy as np  # 导入 numpy，用于数值计算
import torch.distributed as dist  # 导入 torch.distributed，用于分布式训练


import torch  # 导入 PyTorch 核心库
import torch.utils.data  # 导入 PyTorch 数据处理工具
import transformers  # 导入 Hugging Face transformers 库
from datasets import Dataset, IterableDataset  # 从 datasets 库导入数据集类
from packaging import version  # 导入 packaging.version 用于比较版本号
from transformers import (  # 从 transformers 库导入各种组件
    TrainingArguments,  # 训练参数配置类
    AriaForConditionalGeneration,  # Aria 条件生成模型 (可能是特定模型或占位符)
    AriaProcessor,  # Aria 处理器 (可能是特定模型或占位符)
    AutoModelForCausalLM,  # 自动加载因果语言模型
    AutoModelForSequenceClassification,  # 自动加载序列分类模型
    AutoProcessor,  # 自动加载处理器 (通常包含 tokenizer 和 feature extractor)
    AutoTokenizer,  # 自动加载分词器
    GenerationConfig,  # 生成配置
    PreTrainedModel,  # 预训练模型基类
    PreTrainedTokenizerBase,  # 预训练分词器基类
    Qwen2VLForConditionalGeneration,  # Qwen2-VL 条件生成模型
    Qwen2_5_VLForConditionalGeneration,  # Qwen2.5-VL 条件生成模型
    Qwen2VLProcessor,  # Qwen2-VL 处理器
    Qwen2_5_VLProcessor,  # Qwen2.5-VL 处理器
    Trainer,  # 训练器基类
    TrainerCallback,  # 训练器回调
    is_wandb_available,  # 检查 wandb (Weights & Biases) 是否可用
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled  # 检查 DeepSpeed ZeRO 3 是否启用
from transformers.utils import is_peft_available  # 检查 PEFT (Parameter-Efficient Fine-Tuning) 是否可用

from trl.data_utils import (  # 从 trl 的数据工具中导入
    apply_chat_template,  # 应用聊天模板
    is_conversational,  # 检查是否为对话格式
    maybe_apply_chat_template,  # 可能应用聊天模板
)
from trl.models import (  # 从 trl 的模型工具中导入
    create_reference_model,  # 创建参考模型 (用于 DPO, PPO 等)
    prepare_deepspeed,  # 准备 DeepSpeed
    unwrap_model_for_generation,  # 为生成任务解包模型 (例如去除 PEFT 包装)
)

from trl.trainer.utils import generate_model_card, get_comet_experiment_url  # 从 trl 训练器工具中导入
import PIL.Image  # 导入 PIL (Pillow) 库的 Image 模块，用于图像处理
from typing import List  # 导入 List 类型提示
import copy  # 导入 copy 模块
from PIL import Image  # 再次导入 Image (与 PIL.Image 相同)
import json  # 导入 json 模块，用于处理 JSON 数据


from vlm_modules.vlm_module import VLMBaseModule  # 从自定义的 vlm_modules 导入 VLM 基模块

if is_peft_available():  # 如果 PEFT 可用
    from peft import PeftConfig, get_peft_model  # 导入 PEFT 配置和模型函数

if is_wandb_available():  # 如果 wandb 可用
    import wandb  # 导入 wandb


class Qwen2VLSFTToolTrainer(Trainer):  # 定义一个继承自 Hf Trainer 的自定义训练器
    def __init__(  # 构造函数
        self,
        vlm_module: VLMBaseModule,  # VLM 基础模块，用于处理 VLM 特定的逻辑
        model: Union[str, PreTrainedModel],  # 模型，可以是模型名称字符串或是已加载的 PreTrainedModel 对象
        args: TrainingArguments = None,  # 训练参数
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,  # 训练数据集
        eval_dataset: Optional[  # 评估数据集
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,  # 处理器类 (类型提示可能是 Tokenizer，但变量名是 processing_class)
        reward_processing_classes: Optional[  # 奖励模型的处理器 (在此 SFT trainer 中似乎未使用)
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,  # 回调函数列表
        optimizers: tuple[  # 优化器和学习率调度器
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
        peft_config: Optional["PeftConfig"] = None,  # PEFT 配置
        max_pixels: Optional[int] = 12845056,  # 最大像素数 (似乎未使用)
        min_pixels: Optional[int] = 3136,  # 最小像素数 (似乎未使用)
        attn_implementation: str = "flash_attention_2",  # 注意力实现，默认为 flash attention 2
        torch_dtype: str = "bfloat16",  # torch 数据类型，默认为 bfloat16
        is_base_model: Optional[bool] = False,  # 是否是基础模型 (似乎未使用)
        tools: Optional[list[str]] = None,  # 工具列表
        weight: Optional[torch.Tensor] = None,  # 损失权重
        freeze_vision_modules: Optional[bool] = False,  # 是否冻结视觉模块
        use_final_answer: Optional[bool] = False,  # 是否使用 final_answer (似乎未使用)
    ):
        # Args (参数)
        if args is None:  # 如果没有提供训练参数
            model_name = model if isinstance(model, str) else model.config._name_or_path  # 获取模型名称
            model_name = model_name.split("/")[-1]  # 取名称的最后一部分
            args = GRPOConfig(f"{model_name}-GRPO")  # !! 注意：这里硬编码了 GRPOConfig，可能是一个遗留错误

        # Models (模型)
        # Trained model (训练的模型)
        self.vlm_module = vlm_module  # 保存 VLM 模块实例

        model_init_kwargs = {}  # 初始化模型参数字典
        model_init_kwargs["attn_implementation"] = attn_implementation  # 设置注意力实现
        if model_init_kwargs.get("torch_dtype") is None:  # 如果 torch_dtype 未设置
            model_init_kwargs["torch_dtype"] = torch_dtype  # 使用传入的 torch_dtype
        if isinstance(model, str):  # 如果模型是一个字符串 (路径或名称)
            model_id = model  # 保存模型 ID
            torch_dtype = model_init_kwargs.get("torch_dtype")  # 获取 torch_dtype
            if (
                isinstance(torch_dtype, torch.dtype)  # 如果已经是 torch.dtype
                or torch_dtype == "auto"  # 或者是 "auto"
                or torch_dtype is None  # 或者是 None
            ):
                pass  # 无需转换
            elif isinstance(torch_dtype, str):  # 如果是字符串 (例如 "bfloat16")
                torch_dtype = getattr(torch, torch_dtype)  # 转换为 torch.dtype 对象
                model_init_kwargs["torch_dtype"] = torch_dtype  # 更新字典
            else:
                raise ValueError(  # 否则，抛出错误
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            # 如果启用了梯度检查点，则禁用缓存（不支持）
            model_init_kwargs["use_cache"] = (
                False
                if args.gradient_checkpointing  # 检查是否启用梯度检查点
                else model_init_kwargs.get("use_cache")  # 否则使用字典中已有的值
            )
            # model_init_kwargs["use_cache"] = True # (这行被注释掉了)
            model = vlm_module.get_model(model_id, model_init_kwargs=model_init_kwargs)  # 使用 VLM 模块加载模型

        else:  # 如果模型已经是 PreTrainedModel 对象
            model_id = model.config._name_or_path  # 获取模型 ID
            if args.model_init_kwargs is not None:  # 检查是否传入了 model_init_kwargs
                raise ValueError(  # 如果传入了，则抛出错误，因为模型已实例化
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )
        # TODO: freeze vision modules (待办：冻结视觉模块)
        if freeze_vision_modules:  # 如果设置了冻结视觉模块
            print("Freezing vision modules...")  # 打印提示信息
            for n, p in model.named_parameters():  # 遍历模型所有命名参数
                if any(  # 如果参数名 n 包含
                    keyword in n
                    for keyword in self.vlm_module.get_vision_modules_keywords()  # VLM 模块定义的任何视觉关键词
                ):
                    p.requires_grad = False  # 将该参数的 requires_grad 设置为 False

        if use_final_answer:  # 如果使用 final_answer (似乎未使用)
            self.final_answer = FinalAnswer()  # (FinalAnswer 类未在此定义)
            self.funcs = [self.final_answer.function]
        else:
            self.funcs = []  # 默认功能列表为空
        # tools (工具)
        self.tools = [tools] if isinstance(tools, str) else tools  # 将工具转换为列表

        self.funcs = self.vlm_module.tool_des_postprocess(self.funcs)  # 对工具描述进行后处理

        # Processing class (处理器类)
        if processing_class is None:  # 如果未提供处理器
            processing_class = vlm_module.get_processor(model_id)  # 使用 VLM 模块获取处理器

        # Data collator (数据整理器)
        def data_collator(features):  # 定义一个数据整理器
            return features  # 直接返回特征 (SFT 在 compute_loss 中处理)

        # Training arguments (训练参数)
        # = G in the GRPO paper (GRPO 论文中的 G - 可能是 GRPO 遗留代码)

        self.weight = weight  # 保存权重 (可能用于损失计算)

        # The trainer estimates the number of FLOPs... 
        # (训练器会估计 FLOPs... 下面这行是为了抑制相关警告)
        model.warnings_issued["estimate_tokens"] = True  # 标记为 True 以抑制 FLOPs 计算警告

        # Initialize the metrics (初始化指标)
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}  # 训练和评估指标字典

        super().__init__(  # 调用父类 (Trainer) 的构造函数
            model=model,  # 模型
            args=args,  # 训练参数
            data_collator=data_collator,  # 数据整理器
            train_dataset=train_dataset,  # 训练集
            eval_dataset=eval_dataset,  # 评估集
            processing_class=processing_class,  # 处理器 (!! Hf Trainer 原本没有此参数，这是 trl DPO/GRPO 的特性)
            callbacks=callbacks,  # 回调
            optimizers=optimizers,  # 优化器
        )
        # Gradient accumulation requires scaled loss... 
        # (梯度累积需要缩放损失... 下面这行用于启用损失缩放)
        self.model_accepts_loss_kwargs = False  # 强制启用损失缩放 (如果需要梯度累积)
        self.processing_class.tokenizer.padding_side = "left"  # 设置分词器的填充侧为 "left" (用于生成)

    def _set_signature_columns_if_needed(self):  # 重写方法：设置签名列
        # If `self.args.remove_unused_columns` is True...
        # (如果 remove_unused_columns=True... 我们需要指定保留哪些列)
        if self._signature_columns is None:  # 如果签名列未设置
            self._signature_columns = ["message_list"]  # 将其设置为 "message_list"，对应 training_step 的输入

    # Get the per-token log probabilities for the completions for the model and the reference model
    # (获取模型 [此处无参考模型] 的 token 级 log-probabilities)
    def _get_per_token_logps(
        self, model, inputs, logits_to_keep, weight: Optional[torch.Tensor] = None
    ):
        inputs.to(model.device)  # 将输入移动到模型所在的设备
        logits = model(**inputs).logits  # 模型前向传播，获取 logits
        input_ids = inputs.input_ids  # 获取 input_ids
        # (B, L, V) (形状注释)
        if weight is not None:  # 如果提供了权重 (用于对特定 token 加权)
            answer = self.processing_class.tokenizer.encode(  # 编码 "answer" 字符串
                "answer", add_special_tokens=False
            )[0]
            numbers = self.processing_class.tokenizer.encode(  # 编码数字 "1234567890"
                "1234567890", add_special_tokens=False
            )

        # print(f'logits shape: {logits.shape}') # (调试打印)
        logits = logits[
            :, :-1, :
        ]  # (B, L-1, V), 移除最后一个 logit (它对应下一个 token 的预测)
        input_ids = input_ids[
            :, 1:
        ]  # (B, L-1), 移除第一个 input ID (没有 logit 对应它)
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        # (计算输入 token 的 log probabilities。使用循环以减少内存峰值。)
        logits_to_keep = logits_to_keep[:, 1:]  # 对应 input_ids，也移除第一个位置

        per_token_logps = []  # 存储每个 token 的 logp
        for seq_logits, seq_input_ids, seq_mask in zip(  # 遍历 batch 中的每个样本
            logits, input_ids, logits_to_keep
        ):
            # Select only the tokens where mask is True (只选择 mask 为 True 的 token)
            masked_logits = seq_logits[
                seq_mask
            ]  # (N, V)，N 是 True 的数量
            masked_input_ids = seq_input_ids[seq_mask]  # (N,)

            # Calculate log probabilities for the selected tokens (计算所选 token 的 log probabilities)
            log_probs = masked_logits.log_softmax(dim=-1)  # (N, V)
            if weight is not None:  # 如果使用权重
                answer_positions = (masked_input_ids == answer).nonzero().flatten()  # 找到 "answer" token 的位置

                if len(answer_positions) == 1:  # 如果找到了一个 "answer"
                    answer_zone = masked_input_ids[  # 取 "answer" 后的 10 个 token
                        answer_positions[0] : answer_positions[0] + 10
                    ]
                    # Find positions of number tokens in the answer zone (在 answer 区域找数字 token)
                    number_positions = []
                    for i, token_id in enumerate(answer_zone):
                        if token_id in numbers:  # 如果 token 是数字
                            number_positions.append(answer_positions[0] + i)  # 记录位置

                    # If number positions found, apply weight to those positions (如果找到数字，应用权重)
                    if number_positions:
                        weight_mask = torch.ones_like(log_probs)  # 创建一个全 1 的权重掩码
                        for pos in number_positions:
                            weight_mask[pos] = weight  # 在数字位置应用指定权重
                        log_probs = log_probs * weight_mask  # logp 乘以权重
            token_log_prob = torch.gather(  # 从 log_probs 中收集
                log_probs, dim=1, index=masked_input_ids.unsqueeze(1)  # 对应真实 token 的索引
            ).squeeze(1)  # (N,) 得到真实 token 的 logp
            per_token_logps.append(token_log_prob)  # 添加到列表
        
        max_length = max(len(logps) for logps in per_token_logps)  # 找到 batch 中最长的 N (即 mask=True 的数量)
        padded_per_token_logps = []  # 存储填充后的 logp
        padding_masks = []  # 存储填充掩码
        for logps in per_token_logps:  # 遍历 batch
            mask = torch.ones(max_length, device=logps.device)  # 创建全 1 掩码
            mask[len(logps) :] = 0  # 将填充部分设为 0
            padding_masks.append(mask)  # 添加掩码

            # Pad the log probabilities with zeros (用 0 填充 log probabilities)
            padding = torch.zeros(max_length - len(logps), device=logps.device)  # 创建 0 填充
            # Concatenate original logps with padding to maintain gradient flow (拼接以保持梯度流)
            padded_logps = torch.cat([logps, padding])  # 拼接
            padded_per_token_logps.append(padded_logps)  # 添加
        
        per_token_logps = torch.stack(padded_per_token_logps, dim=0)  # 堆叠成 (B, max_N)
        padding_masks = torch.stack(padding_masks, dim=0)  # 堆叠成 (B, max_N)

        return per_token_logps, padding_masks  # 返回填充后的 logp 和掩码

    # Trainer "prepares" the inputs before calling `compute_loss`.
    # (Trainer 在调用 compute_loss 之前会 "准备" 输入。我们重写此方法以跳过该步骤)
    def _prepare_inputs(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs  # 直接返回输入，不做任何处理

    def has_unfinished_sample(  # 检查是否有未完成的样本 (用于分布式)
        self, this_peer_finished: bool, device: torch.device
    ) -> bool:
        this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(  # 0.0=完成, 1.0=未完成
            device
        )
        dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)  # 在所有进程上求和

        if this_peer_finished_flag.item() == 0.0:  # 如果所有进程都完成了
            return False  # 返回 False

        return True  # 否则返回 True

    def create_assistant_response_mask(  # 创建助手回应的掩码
        self, inputs, processor, if_use_weighted: bool = False
    ):
        """
        Create a boolean mask for the assistant's responses based on the chat template format.
        (根据聊天模板格式，为助手的回应创建一个布尔掩码。)
        """
        mask = torch.zeros_like(inputs, dtype=torch.bool)  # 初始化全 False 掩码
        # weighted_mask = torch.zeros_like(inputs, dtype=torch.bool) # (被注释的代码)
        # Get special token IDs (获取特殊 token ID)
        im_start = processor.tokenizer.encode(  # 获取 <|im_start|> (或 Qwen 的等效) 的 ID
            self.vlm_module.get_im_start(), add_special_tokens=False
        )[0]
        im_end = processor.tokenizer.encode(  # 获取 <|im_end|> (或 Qwen 的等效) 的 ID
            self.vlm_module.get_im_end(), add_special_tokens=False
        )[0]
        assistant = processor.tokenizer.encode(  # 获取 "assistant" (或 Qwen 的等效) 的 ID
            self.vlm_module.get_assistant(), add_special_tokens=False
        )[0]
        # answer = processor.tokenizer.encode("answer", add_special_tokens=False)[0] # (被注释的代码)

        # For each sequence in the batch (对 batch 中的每个序列)
        for i in range(inputs.shape[0]):  # 遍历 batch
            sequence = inputs[i]  # 获取单个序列

            # Find all im_start positions (找到所有 im_start 的位置)
            im_start_positions = (sequence == im_start).nonzero().flatten()
            if True:  # (这个 True 条件似乎是固定的，用于特定逻辑：只掩码最后两个 assistant 回应)
                pos = im_start_positions[-2]  # 取倒数第二个 im_start
                if pos + 1 < len(sequence) and sequence[pos + 1] == assistant:  # 检查后面是否是 assistant
                    next_end = sequence[pos:].eq(im_end).nonzero()  # 找到后续的 im_end
                    if len(next_end) > 0:
                        end_pos = pos + next_end[0].item()  # 计算结束位置
                        # Mark the entire response (including the im_start and im_end tokens)
                        # (标记整个回应，包括 im_start 和 im_end)
                        mask[i, pos : end_pos + 1] = True  # 将掩码设为 True
                pos = im_start_positions[-4]  # 取倒数第四个 im_start
                if pos + 1 < len(sequence) and sequence[pos + 1] == assistant:  # 同样检查 assistant
                    next_end = sequence[pos:].eq(im_end).nonzero()
                    if len(next_end) > 0:
                        end_pos = pos + next_end[0].item()
                        # Mark the entire response (including the im_start and im_end tokens)
                        mask[i, pos : end_pos + 1] = True
            else:  # (这段 else 逻辑永远不会执行，因为上面 if True)
                for pos in im_start_positions:
                    # Check if the token after im_start is "assistant"
                    if pos + 1 < len(sequence) and sequence[pos + 1] == assistant:
                        # Find the next im_end
                        next_end = sequence[pos:].eq(im_end).nonzero()
                        if len(next_end) > 0:
                            end_pos = pos + next_end[0].item()
                            # Mark the entire response (including the im_start and im_end tokens)
                            mask[i, pos : end_pos + 1] = True

                        # Debug print (调试打印)

            # if if_use_weighted: ... (被注释的代码块)

        return mask  # 返回掩码

    def compute_loss(  # 定义计算损失的函数 (SFT 的核心)
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:  # SFT trainer 通常不支持返回 outputs
            raise ValueError("The GRPOTrainer does not support returning outputs")

        device = self.accelerator.device  # 获取加速器设备
        message_lists = [x["message_list"] for x in inputs]  # 从输入中提取 message_list

        # z2orcorrects = [x["z2orcorrect"] for x in inputs] # (被注释的代码)

        # Handle both pre-loaded images and image paths (处理预加载图像和图像路径)
        # (这一步在 VLM 模块内部完成)
        messages = message_lists  # 将 message_lists 赋值给 messages

        inputs = self.vlm_module.prepare_from_msg_2_vlm_inputs(  # 使用 VLM 模块准备模型输入
            self.processing_class, messages
        )

        # Generate completions (生成补全)
        # (SFT 不生成，而是计算给定标签的损失)

        # Right pad and stack tensors (右填充和堆叠张量)
        # (这一步由 VLM 模块完成了)
        padded_input_ids = inputs.input_ids  # 获取填充后的 input_ids

        all_logits_to_keep = self.create_assistant_response_mask(  # 创建助手回应的掩码
            padded_input_ids, self.processing_class
        )
        # Concatenate prompt_mask with completion_mask for logit computation
        # (为 logit 计算拼接... - 下面是被注释的代码)
        # attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)
        # pixel_values = prompt_inputs["pixel_values"].repeat(self.num_generations, 1)
        # image_grid_thw = prompt_inputs["image_grid_thw"].repeat_interleave(self.num_generations, dim=0)
        # self.accelerator.wait_for_everyone()
        # print(f"Process {self.accelerator.process_index}: ...")
        # print(f'starting to compute per_token_logps...')

        per_token_logps, completion_mask = self._get_per_token_logps(  # 计算 token 级的 logp
            model, inputs, all_logits_to_keep, self.weight  # 传入模型、输入、掩码和权重
        )
        # print(f'starting to compute ref_per_token_logps...')

        # Compute the KL divergence between the model and the reference model
        # (计算 KL 散度 - 注释不准确，这是 SFT，计算的是交叉熵损失)

        loss = -(per_token_logps * completion_mask).sum() / completion_mask.sum()  # 计算 (加权的) 负对数似然损失 (即交叉熵)

        # Log the metrics (记录指标)
        mode = "train"  # 模式为训练
        self._metrics[mode]["loss"].append(  # 将损失添加到指标字典
            self.accelerator.gather_for_metrics(loss).mean().item()  # 聚合多卡损失
        )

        return loss  # 返回损失

    def eval_prediction_step(self, model, inputs):  # 评估预测步骤
        old_message_lists = [x["message_list"] for x in inputs]  # 获取 message_list
        question_image_paths = [x["question_image_path"] for x in inputs]  # 获取图像路径
        images = []  # 初始化图像列表
        for question_image_path in question_image_paths:  # 遍历路径
            if isinstance(question_image_path, str):  # 如果是字符串 (路径)
                img = PIL.Image.open(question_image_path)  # 打开图像
            # (如果不是 str，假设它已经是 PIL.Image 对象 - 尽管代码没这么写)

            # Ensure minimum dimensions of 28 pixels (确保最小尺寸为 28 像素)
            w, h = img.size  # 获取宽高
            if w < 28 or h < 28:  # 如果太小
                # Calculate new dimensions maintaining aspect ratio (计算保持纵横比的新尺寸)
                if w < h:
                    new_w = 28
                    new_h = int(h * (28 / w))
                else:
                    new_h = 28
                    new_w = int(w * (28 / h))
                img = img.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)  # 缩放图像

            images.append(img)  # 添加到列表

        # Handle both pre-loaded images and image paths (处理...)
        # (此处似乎缺少将图像传递给模型的步骤)

        # Generate completions (生成补全)

        message_lists, answers, messages, images = self.sample_1_response(  # !! 注意：sample_1_response 未在此类中定义
            model,
            self.processing_class,
            old_message_lists,  # 直接用rank作为索引 (原注释)
            images,
        )

        issimple = inputs[0]["is_simple"]  # 检查是否为简单问题
        completions = answers  # 获取生成的答案

        def accuracy_reward(completions, answers):  # 定义计算准确率的函数
            rewards = []
            for completion, answer in zip(completions, answers):
                try:
                    if int(completion) == answer:  # 比较 (转换成整数后)
                        rewards.append(1.0)  # 正确
                    else:
                        rewards.append(0.0)  # 错误
                except:
                    rewards.append(0.0)  # 转换失败则为错误
            return rewards

        answers = [inputs[i]["solution"] for i in range(len(inputs))]  # 获取标准答案
        rewards = accuracy_reward(completions, answers)  # 计算奖励 (即准确率)
        rewards = torch.tensor(  # 转换为 tensor
            rewards, dtype=torch.float32, device=self.accelerator.device
        )
        accuracy_name = "complex" if not issimple else "simple"  # 根据难度命名

        self._metrics["eval"][f"{accuracy_name}"].append(  # 记录指标
            self.accelerator.gather_for_metrics(rewards).mean().item()  # 聚合准确率
        )

        if self.accelerator.is_main_process:  # 如果是主进程
            checkpoint_folder = f"checkpoint-{self.state.global_step}"  # 检查点文件夹名称
            output_dir = os.path.join(self.args.output_dir, checkpoint_folder)  # 输出目录
            os.makedirs(output_dir, exist_ok=True)  # 创建目录
            sample_json = os.path.join(output_dir, f"samples_{accuracy_name}.json")  # 样本 JSON 文件
            if not os.path.exists(sample_json):  # 如果文件不存在
                with open(sample_json, "w", encoding="utf-8") as f:  # 写入
                    json.dump(messages, f, indent=4)  # 保存消息
                os.makedirs(  # 创建图像目录
                    os.path.join(output_dir, f"images_{accuracy_name}"), exist_ok=True
                )
                for i, image in enumerate(images):  # 遍历图像
                    for j, img in enumerate(image):  # (!! 假设 images 是列表的列表)
                        img.save(  # 保存图像
                            os.path.join(
                                output_dir, f"images_{accuracy_name}", f"{i}_{j}.png"
                            )
                        )
        return torch.tensor(0.0, device=self.accelerator.device)  # 返回一个假的损失值 (0.0)

    def prediction_step(  # 重写 prediction_step (用于评估)
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys: Optional[list[str]] = None,
    ):
        inputs = self._prepare_inputs(inputs)  # 准备输入 (实际只是原样返回)
        with torch.no_grad():  # 禁用梯度计算
            with self.compute_loss_context_manager():  # 使用损失计算上下文管理器
                loss = self.eval_prediction_step(model, inputs)  # 调用我们自定义的评估步骤
            loss = loss.mean().detach()  # 获取损失 (虽然返回的是 0.0)
        return loss, None, None  # 返回 (loss, logits, labels)

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:  # 重写 log 方法
        metrics = {}  # 初始化指标字典
        for mode in ["train", "eval"]:  # 遍历训练和评估模式
            metrics[mode] = {  # 计算平均指标
                key: sum(val) / len(val) for key, val in self._metrics[mode].items() if len(val) > 0
            }
            if mode == "eval":  # 如果是评估模式
                metrics[mode] = {  # 添加 "eval_" 前缀
                    f"eval_{key}": val for key, val in metrics[mode].items()
                }
            self._metrics[mode].clear()  # 清空指标缓存
        logs = {**logs, **metrics["train"], **metrics["eval"]}  # 合并日志
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):  # 兼容不同 transformers 版本
            super().log(logs, start_time)  # 新版本调用
        else:  # transformers<=4.46
            super().log(logs)  # 旧版本调用

    def create_model_card(  # 创建模型卡片
        self,
        model_name: Optional[str] = None,  # 模型名称
        dataset_name: Optional[str] = None,  # 数据集名称
        tags: Union[str, list[str], None] = None,  # 标签
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.
        (使用 Trainer 可用的信息创建模型卡片草稿。)
        """
        if not self.is_world_process_zero():  # 仅在主进程 (rank 0) 执行
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(  # 检查基础模型名称
            self.model.config._name_or_path
        ):
            base_model = self.model.config._name_or_path
        else:
            base_model = None  # 未找到基础模型

        tags = tags or []  # 初始化标签列表
        if isinstance(tags, str):  # 如果标签是字符串
            tags = [tags]  # 转换为列表

        if hasattr(self.model.config, "unsloth_version"):  # 检查是否使用 unsloth
            tags.append("unsloth")  # 添加 unsloth 标签

        citation = textwrap.dedent(  # 定义引用的 bibtex (!! 注意：这里引用的是 DeepSeekMath)
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(  # 调用 trl 的工具函数生成模型卡片
            base_model=base_model,  # 基础模型
            model_name=model_name,  # 模型名称
            hub_model_id=self.hub_model_id,  # Hub 上的模型 ID
            dataset_name=dataset_name,  # 数据集名称
            tags=tags,  # 标签
            wandb_url=wandb.run.get_url()  # wandb 链接 (如果可用)
            if is_wandb_available() and wandb.run is not None
            else None,
            comet_url=get_comet_experiment_url(),  # comet 链接 (如果可用)
            trainer_name="GRPO",  # 训练器名称 (!! 硬编码为 GRPO)
            trainer_citation=citation,  # 训练器引用 (!! DeepSeekMath)
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",  # 论文标题
            paper_id="2402.03300",  # 论文 ID
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))  # 保存 README.md
