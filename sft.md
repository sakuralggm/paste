使用sft trainner：[https://connectaman.hashnode.dev/fine-tuning-the-qwen25-7b-vl-instruct-model-a-comprehensive-guide](https://connectaman.hashnode.dev/fine-tuning-the-qwen25-7b-vl-instruct-model-a-comprehensive-guide)

sft trainer 微调 vlm 官方案例：[https://github.com/huggingface/trl/blob/main/examples/scripts/sft_vlm.py](https://github.com/huggingface/trl/blob/main/examples/scripts/sft_vlm.py)

vlm标准数据集格式：[https://huggingface.co/datasets/trl-lib/llava-instruct-mix/viewer/default/train?row=0](https://huggingface.co/datasets/trl-lib/llava-instruct-mix/viewer/default/train?row=0)

要用最新版trl 0.25.0，有对vlm微调的支持



## 微调观察

### 第一次

```python
# 冻结视觉模块
trainable_params_count_before_freeze = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)

print("开始冻结 vision_module 参数...")
for param in model.visual.parameters():
    param.requires_grad = False
print("vision_module 参数已冻结。")

# (可选) 验证冻结
vision_params_count = sum(p.numel() for p in model.visual.parameters())
trainable_params_count_after_freeze = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)

print(f"总可训练参数 (冻结前): {trainable_params_count_before_freeze}") 
print(f"视觉模块 (visual) 总参数: {vision_params_count}")
print(f"总可训练参数 (冻结后): {trainable_params_count_after_freeze}")
'''
开始冻结 vision_module 参数...
vision_module 参数已冻结。
总可训练参数 (冻结前): 8292166656
视觉模块 (visual) 总参数: 676550144
总可训练参数 (冻结后): 7615616512
'''


"""
配置训练参数
"""
sft_config = SFTConfig(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    num_train_epochs=5,
    learning_rate=1e-6,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=1,
    data_seed=49,
    bf16=True,  # Use bfloat16 precision
    save_strategy="epoch",
    save_only_model=True,
    report_to="tensorboard",  # tensorboard --logdir ./tensorboard/sft/
    logging_dir=f"./tensorboard/sft/{datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}",
    output_dir="/mnt/workspace/workgroup/kejia/checkpoint/sft",
)
```

![](https://intranetproxy.alipay.com/skylark/lark/0/2025/png/201056699/1762917198786-7521dd06-84cd-4f5a-be33-2081ea8d4950.png)

![](https://intranetproxy.alipay.com/skylark/lark/0/2025/png/201056699/1762917212555-7b915a9b-3ff0-4cc0-84da-cb7b490940b6.png)

#### 全部代码
```python
from trl import SFTConfig, SFTTrainer
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling
from datasets import load_from_disk
import datetime
import torch
from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

"""
加载Qwen模型
"""
model_id = "/home/chenkejia.ckj/models/Qwen2.5-VL-7B-Instruct"

# 使用Transformers加载模型权重
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="cuda",
    # attn_implementation="flash_attention_2",  # 启用 Flash Attention 2
)
processor = AutoProcessor.from_pretrained(model_id)
collator = DataCollatorForVisionLanguageModeling(processor)

# 冻结视觉模块
trainable_params_count_before_freeze = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)

print("开始冻结 vision_module 参数...")
for param in model.visual.parameters():
    param.requires_grad = False
print("vision_module 参数已冻结。")

# (可选) 验证冻结
vision_params_count = sum(p.numel() for p in model.visual.parameters())
trainable_params_count_after_freeze = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)

print(f"总可训练参数 (冻结前): {trainable_params_count_before_freeze}")
print(f"视觉模块 (visual) 总参数: {vision_params_count}")
print(f"总可训练参数 (冻结后): {trainable_params_count_after_freeze}")

"""
加载数据集
"""
dataset = load_from_disk("/home/chenkejia.ckj/datasets/val_deduplicated_sft")
train_dataset = dataset["train"]

"""
配置训练参数
"""
sft_config = SFTConfig(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    num_train_epochs=5,
    learning_rate=1e-6,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=1,
    data_seed=49,
    bf16=True,  # Use bfloat16 precision
    save_strategy="epoch",
    save_only_model=True,
    report_to="tensorboard",  # tensorboard --logdir ./tensorboard/sft/
    logging_dir=f"./tensorboard/sft/{datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}",
    output_dir="/mnt/workspace/workgroup/kejia/checkpoint/sft",
)


trainer = SFTTrainer(
    model=model, args=sft_config, train_dataset=train_dataset, data_collator=collator
)

"""
开始训练
"""
trainer.train()

"""
保存模型
"""
trainer.save_model("./checkpoint/sft")
```

