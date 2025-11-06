# Pixel Reasoner训练配置

## 资源

[模型和数据合集](https://huggingface.co/collections/TIGER-Lab/pixel-reasoner)

[warmstart模型](https://huggingface.co/TIGER-Lab/PixelReasoner-WarmStart)

[warmstart+rl模型](https://huggingface.co/TIGER-Lab/PixelReasoner-RL-v1)

[warmstart sft数据](https://huggingface.co/datasets/TIGER-Lab/PixelReasoner-SFT-Data)

[rl数据](https://huggingface.co/datasets/TIGER-Lab/PixelReasoner-RL-Data)

[benchmark数据](https://huggingface.co/collections/JasperHaozhe/evaldata-pixelreasoner)

## 第0步 git clone项目

```shell
git clone https://github.com/TIGER-AI-Lab/Pixel-Reasoner.git
```

## Instruction tuning

### conda环境安装

提前下载好`flash_attn-2.7.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`，本地编译很慢。[下载地址](https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl)

**需要注意的是，v100不支持flash_attn，可以不用下载**

```shell
conda create -n warmstart python=3.10 -y
conda activate warmstart
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 # flash-attn最高支持pytorch版本是2.5.xx
pip install -r requirements.txt
pip install wandb==0.18.3
pip install tensorboardx
# 默认是最新的2.8.3，但是报错了，AttributeError: module 'torch.library' has no attribute 'wrap_triton' 
# Install Qwen related packages
pip install git+https://github.com/cjakfskvnad/Qwen-Agent.git
pip install qwen-vl-utils
pip install debugpy
mamba install nvidia/label/cuda-12.4.0::cuda-nvcc
pip install git+https://gitcode.com/GitHub_Trending/tra/transformers.git@89d27fa6fff206c0153e9670ae09e2766eb75cdf
# 直接从whl文件安装flash_attn
pip install flash_attn-2.7.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

修改后的`requirements.txt`：

```shell
# Requirements file equivalent to: pip install -e ".[dev]"
# This includes install_requires + extras["dev"] dependencies


# torch
torch==2.5.1 
torchvision==0.20.1 
torchaudio==2.5.1 

# Core dependencies (install_requires)
accelerate>=1.2.1
bitsandbytes>=0.43.0
einops>=0.8.0
datasets>=3.2.0
deepspeed==0.15.4
hf_transfer>=0.1.4
huggingface-hub[cli]>=0.19.2,<1.0
liger_kernel==0.5.2
packaging>=23.0
safetensors>=0.3.3
sentencepiece>=0.1.99
trl==0.15.0

# Quality tools (extras["quality"])
black>=24.4.2
isort>=5.12.0
flake8>=6.0.0

# Testing tools (extras["tests"])
pytest
parameterized>=0.9.0

# Evaluation tools (extras["eval"])
math-verify 
```

### 下载sft数据集

```shell
cd instruction_tuning
pip install -U huggingface_hub # 最好在另一个环境安装，比如base
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --repo-type dataset --resume-download TIGER-Lab/PixelReasoner-SFT-Data --local-dir PixelReasoner-SFT-Data --local-dir-use-symlinks False
```

**然后将images.zip和videos.zip解压到与sft_tool.py同一目录下，因为程序是通过`images/xxx.jpg`的相对路径访问图片文件的**

### 下载Qwen/Qwen2.5-VL-7B-Instruct

```shell
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download Qwen/Qwen2.5-VL-7B-Instruct --local-dir Qwen2.5-VL-7B-Instruct
```

### 修改sft.sh

注意：**v100不支持bfloat16**，因此命令的选项中要用`--fp16`和`--torch_dtype float16`

根据实际情况调整：

1. `CUDA_VISIBLE_DEVICES`：根据卡的数量调整
2. `--nproc_per_node="卡的数量"`
3. ``--model_name_or_path`：刚才下载的`Qwen2.5-VL-7B-Instruct`模型的路径
4. `--datasetpath`：下载的sft数据集中`release.json`文件的路径

```shell

export WANDB_PROJECT=vlm-r1-grpo-rec

export DEBUG_MODE="true"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# rtx 40系显卡需要把下面的注释去掉
# export NCCL_P2P_DISABLE=1
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1

RUN_NAME=test
export LOG_PATH="./debug_log_$RUN_NAME.txt"

# 把下面路径最前面的/path/to/your/conda_env替换为刚才安装好的conda环境的路径
# 比如我的路径是："/mnt/workspace/workgroup/kejia/miniconda/envs/warmstart/lib/python3.10/site-packages/nvidia/
nvjitlink="/path/to/your/conda_env/lib/python3.10/site-packages/nvidia/nvjitlink/lib"
export LD_LIBRARY_PATH_VALUE=${nvjitlink}:$LD_LIBRARY_PATH
export CUDA_HOME=$CONDA_PREFIX

python -m torch.distributed.run --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12347" \
    sft_tool.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path /mnt/workspace/workgroup/kejia/models/Qwen2.5-VL-7B-Instruct \
    --dataset_name "PixelReasoner-SFT-Data" \
    --datasetpath data/PixelReasoner-SFT-Data/release.json \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --eval_strategy no \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --fp16 \
    --learning_rate 1e-6 \
    --torch_dtype float16 \
    --data_seed 49 \
    --report_to none \
    --gradient_checkpointing true \
    --num_train_epochs 5 \
    --run_name $RUN_NAME \
    --save_strategy epoch \
    --save_steps 100 \
    --save_only_model true \
    --freeze_vision_modules true \
    
    
    
```

### 开始微调

```shell
source sft.sh
```

## Curiosity RL

### conda环境安装

提前下载好`flash_attn-2.7.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`，本地编译很慢。[下载地址](https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl)

**需要注意的是，v100不支持flash_attn，可以不用下载**

```shell
cd curiosity_driven_rl
conda create -n curiosity python=3.10
pip install -e .[vllm]
mamba install nvidia/label/cuda-12.4.0::cuda-nvcc
pip uninstall -y ninja && pip install ninja
pip install click==8.1.3
pip install --force-reinstall git+https://gitcode.com/GitHub_Trending/tra/transformers.git@9985d06add07a4cc691dc54a7e34f54205c04d40
pip install "numpy<2.0.0,>=1.25.0"
pip install deepspeed==0.15.0
# 直接从whl文件安装flash_attn
pip install flash_attn-2.7.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

修改后的`requirements.txt`，替换掉原来的：

```shell
accelerate
bitsandbytes
datasets
deepspeed==0.15
einops
flask
isort
jsonlines
loralib
math-verify
levenshtein
optimum
packaging
peft
pynvml>=12.0.0
qwen_vl_utils
ray[default]==2.42.0
tensorboard
torch==2.5.1
torchmetrics
tqdm
transformers_stream_generator
wandb
wheel
```

#### bug: ImportError: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.32' not found

由于Linux版本不同，这个bug不一定会出现，我的版本是ubuntu20.04，出现了。原因是GLIBC_2.32是更高版本的linux系统才有的系统文件

```shell
sudo vim /etc/apt/sources.list
```

ubuntu20.04的libc6的最高版本是2.30，需要添加一个高级版本系统的源，直接升级libc6。将22.04的源添加到apt源中

```shell
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse
```

然后更新

```shell
sudo apt update
sudo apt install libc6-dev
```

查看是否安装了GLIBC_2.32

```shell
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBC
```

如果没有安装成功，则升级到24.04的源

```shell
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble main restricted universe multiverse
```


### 下载curiosity rl训练集

进入`onestep_evaluation`目录

```shell
cd onestep_evaluation
```

新建以下两个文件：

1. `curiosity_rl.sh`

   ```shell
   export dataname=PixelReasoner-RL-Data
   export hfuser=TIGER-Lab
   export newdataname=release
   cd onestep_evaluation
   bash prepare_rl.sh ${dataname}
   ```

2. `prepare_rl.sh`

   注意，需要修改`working_dir`变量，将该变量指向clone下来的`Pixel-Reasoner/curiosity_driven_rl`

   ```shell
   export HF_ENDPOINT=https://hf-mirror.com
   set -x  # 开启命令回显（调试模式），执行的每条命令都会打印到标准输出
   dataname=${dataname:-"${1}"} # hfname VStar-EvalData-PixelReasoner  # 如果环境变量 dataname 未设置，则使用第一个位置参数
   newdataname=${newdataname:-""} # this is the downloaded parquet file name  # 如果 newdataname 未设置，则默认空串
   hfuser=${hfuser:-"JasperHaozhe"}  # 如果环境变量 hfuser 未设置，则默认 "TIGER-Lab"
   working_dir=/mnt/workspace/workgroup/kejia/projects/pixel-reasoner/curiosity_driven_rl # 工作目录路径，固定写死，需要修改
   
   if [[ ! -f "hfd.sh" ]]; then  # 如果当前目录下不存在 hfd.sh 脚本
       echo "downloading hfd.sh"  # 打印提示
   
       wget https://hf-mirror.com/hfd/hfd.sh  # 下载 hfd.sh
       chmod a+x hfd.sh  # 赋予可执行权限
   else
       echo "hfd.sh already exists."  # 否则提示已存在
   fi
   
   bash hfd.sh ${hfuser}/${dataname} --dataset --tool wget  # 使用 hfd.sh 下载指定 Hugging Face 数据集，使用 wget 工具
   cd ${dataname}  # 进入下载后的数据目录
   unzip images.zip  # 解压 images.zip（包含图片）
   rm images.zip  # 删除压缩包，节省空间
   unzip videos.zip  # 解压 videos.zip（包含视频）
   rm videos.zip  # 删除压缩包，节省空间
   
   # get the benchmark name
   if [[ $(ls *.parquet 2>/dev/null | wc -l) -gt 0 ]]; then  # 如果当前目录下存在任何 .parquet 文件
       parquet_file=$(ls *.parquet | head -1)  # 取第一个 .parquet 文件名
       benchmarkname="${parquet_file%.parquet}"  # 去掉后缀得到基准名字
       echo "benchmark name: $benchmarkname"  # 打印基准名字
   else
       echo "error no *.parquet under this folder" >&2  # 如果没有 .parquet 文件，打印错误到标准错误
       exit 1  # 退出并返回非零状态
   fi
   
   # move the data to the data_folder
   data_folder=${working_dir}/data  # 数据目标文件夹
   mkdir -p ${data_folder}  # 创建目标文件夹（如果不存在）
   mv images ${data_folder}/${benchmarkname}_images  # 将 images 目录移动并重命名为 {benchmarkname}_images
   mv videos ${data_folder}/${benchmarkname}_videos  # 将 images 目录移动并重命名为 {benchmarkname}_images
   mv ${benchmarkname}.parquet ${data_folder}/  # 将 parquet 文件移动到目标数据文件夹
   
   # clear the download cache
   cd ..  # 返回到上级目录
   rm -r ${dataname}  # 删除下载产生的临时目录及其内容
   
   # rename the image path
   python rename_imagepath.py ${working_dir} ${newdataname}  # 运行 Python 脚本，修改 parquet 中的 image 路径（传入工作目录和 newdataname）
   ```

### 下载vstar验证集

进入`onestep_evaluation`目录

```shell
cd onestep_evaluation
```

新建`vstar.sh`

```shell
export dataname=VStar-EvalData-PixelReasoner
export newdataname=vstar
cd onestep_evaluation
bash prepare.sh ${dataname}
```

将项目自带的`prepare.sh`修改如下：

**注意，需要修改`working_dir`变量，将该变量指向clone下来的`Pixel-Reasoner/curiosity_driven_rl`**

```shell
export HF_ENDPOINT=https://hf-mirror.com
set -x  # 开启命令回显（调试模式），执行的每条命令都会打印到标准输出
dataname=${dataname:-"${1}"} # hfname VStar-EvalData-PixelReasoner  # 如果环境变量 dataname 未设置，则使用第一个位置参数
newdataname=${newdataname:-""} # this is the downloaded parquet file name  # 如果 newdataname 未设置，则默认空串
hfuser=${hfuser:-"JasperHaozhe"}  # 如果环境变量 hfuser 未设置，则默认 "TIGER-Lab"
working_dir=/mnt/workspace/workgroup/kejia/projects/pixel-reasoner/curiosity_driven_rl  # 工作目录路径，固定写死，修改这个

if [[ ! -f "hfd.sh" ]]; then  # 如果当前目录下不存在 hfd.sh 脚本
    echo "downloading hfd.sh"  # 打印提示

    wget https://hf-mirror.com/hfd/hfd.sh  # 下载 hfd.sh
    chmod a+x hfd.sh  # 赋予可执行权限
else
    echo "hfd.sh already exists."  # 否则提示已存在
fi

bash hfd.sh ${hfuser}/${dataname} --dataset --tool wget  # 使用 hfd.sh 下载指定 Hugging Face 数据集，使用 wget 工具
cd ${dataname}  # 进入下载后的数据目录
unzip images.zip  # 解压 images.zip（包含图片）
rm images.zip  # 删除压缩包，节省空间

# get the benchmark name
if [[ $(ls *.parquet 2>/dev/null | wc -l) -gt 0 ]]; then  # 如果当前目录下存在任何 .parquet 文件
    parquet_file=$(ls *.parquet | head -1)  # 取第一个 .parquet 文件名
    benchmarkname="${parquet_file%.parquet}"  # 去掉后缀得到基准名字
    echo "benchmark name: $benchmarkname"  # 打印基准名字
else
    echo "error no *.parquet under this folder" >&2  # 如果没有 .parquet 文件，打印错误到标准错误
    exit 1  # 退出并返回非零状态
fi

# move the data to the data_folder
data_folder=${working_dir}/data  # 数据目标文件夹
mkdir -p ${data_folder}  # 创建目标文件夹（如果不存在）
mv images ${data_folder}/${benchmarkname}_images  # 将 images 目录移动并重命名为 {benchmarkname}_images
mv ${benchmarkname}.parquet ${data_folder}/  # 将 parquet 文件移动到目标数据文件夹

# clear the download cache
cd ..  # 返回到上级目录
rm -r ${dataname}  # 删除下载产生的临时目录及其内容

# rename the image path
python rename_imagepath.py ${working_dir} ${newdataname}  # 运行 Python 脚本，修改 parquet 中的 image 路径（传入工作目录和 newdataname）
```

### 下载sft好的模型

这是经过warmstart instruction tuning的模型，按照仓库的README，下载checkpoint-246

```shell
pip install -U huggingface_hub # 最好在另一个环境安装，比如base
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download TIGER-Lab/PixelReasoner-WarmStart --include "checkpoint-246/*" --local-dir PixelReasoner-WarmStart
```

### （可选）修改openrlhf的vllm_engine部分代码

由于**v100不支持bfloat16**，而openrlhf代码中vllm部分的精度是写死的，不能通过运行脚本修改，所以只能修改源码，否则会报错

修改`curiosity_driven_rl/openrlhf/trainer/ray/vllm_engines.py`的`create_vllm_engines`函数，它在return前的最后一条语句是`vllm_engines.append(...)`，将其中的`dtype="bfloat16"`改为`dtype="float16"`（在242行左右）

### 开始训练

#### start_single_node_training.sh

我采用的是单节点的训练方式

首先，新建`start_single_node_training.sh`文件，内容如下：

需要调整的变量：

1. `working_dir`：指向clone下来的`Pixel-Reasoner/curiosity_driven_rl`，用绝对路径
2. `nvjitlink`：指向安装的conda环境中的lib目录下的nvjitlink工具，可参考下面提供的内容修改
3. `policy`：指向下载好的模型地址

训练配置相关参数（看注释修改）：

1. `rbuffer`
2. `bsz`
3. `mbsz`

```shell
benchmark=vstar
export working_dir="/mnt/workspace/workgroup/kejia/projects/pixel-reasoner/curiosity_driven_rl"
# 把/path/to/your/conda_env修改为安装的conda环境的绝对路径，比如我的是/mnt/workspace/workgroup/kejia/miniconda/envs/curiosity/lib/python3.1/site-packages/nvidia/nvjitlink/lib
export nvjitlink="/path/to/your/conda_env/lib/python3.1/site-packages/nvidia/nvjitlink/lib"
export temperature=1.0
export trainver="${working_dir}/data/release.parquet"
export testver="${working_dir}/data/${benchmark}.parquet"
export filter=True # filtering zero advantages
export algo=group # default for grpo
export lr=10
export MAX_PIXELS=4014080 # =[max_image_token]x28x28
export sys=vcot # system prompt version
export mode=train # [no_eval, eval_only, train]
export policy="/mnt/workspace/workgroup/kejia/models/PixelReasoner-WarmStart/checkpoint-246"
export rbuffer=16 # replay buffer size，默认512，为了测试能够跑通改为16
export bsz=8 # global train batch size，默认256，为了测试能够跑通改为8
export evalsteps=1
export mbsz=1 # 默认2，为了测试能够跑通改为8
export tp=1 # vllm tp, 1 for 7B
export repeat=1 # data repeat
export nepoch=3 # data epoch
export logp_bsz=1 # must be 1
export maxlen=10000 # generate_max_len
export tagname=Train
export CUDA_HOME=$CONDA_PREFIX
# export NCCL_P2P_DISABLE=1 # 4000系列不支持P2P, 如果是4000系列则解除注释
# export NCCL_IB_DISABLE=1 # 4000系列不支持IB, 如果是4000系列则解除注释

bash train_vlm_single.sh
```

#### 修改train_vlm_single.sh

##### 分步骤修改

修改`train_vlm_single.sh`如下，分成多部分给出，方便修改：

1. 配置RAY的临时目录

   RAY在跑训练时会产生大量的临时文件，可达上百G，默认保存到`/tmp`，如果根目录不够挂在的硬盘不够大，很容易报 `disk quota error`，可以将临时目录变量配置如下：

   下面的变量，修改`RAY_BASE_DIR`即可。

   ```shell
   set -x
   
   # =============== 新增：设置 Ray 专用目录 ===============
   RAY_BASE_DIR="/mnt/workspace/workgroup/kejia/ray_data"
   mkdir -p "$RAY_BASE_DIR/spill" "$RAY_BASE_DIR/tmp" "$RAY_BASE_DIR/logs"
   
   export RAY_TMPDIR="$RAY_BASE_DIR/tmp" # Ray 运行时临时目录（如 socket、logs）
   export RAY_LOGDIR="$RAY_BASE_DIR/logs" # 由于使用的 Ray 版本不支持 --logs-dir选项，Ray 会自动将日志写入 --temp-dir 下的子目录（如 session_latest/logs）
   
   # 强制 Ray 使用自定义临时目录（通过环境变量）
   # 某些 Ray 组件或底层库（如 vLLM、PyTorch）可能仍使用系统临时目录。你可以通过环境变量引导它们也使用自定义目录
   export TMPDIR="$RAY_BASE_DIR/tmp"
   ```

2. 原仓库自带的`find_interface`函数

   ```shell
   find_interface() {
     local ip_output=$(ip addr show | head -n 10) # Limit to first 10 lines
     local selected_interface=""
   
     # Debug output (can be removed in final version)
     # echo "--- First 10 lines of ip addr show output: ---"
     # echo "$ip_output"
     # echo "--- End of ip addr show output ---"
   
     while IFS= read -r line; do
       # Debug output (can be removed in final version)
       # echo "Processing line: $line"
   
       if [[ "$line" =~ ^[0-9]+:\ ([^:]+):\ \<.*UP.*\> ]]; then
         local interface_name="${BASH_REMATCH[1]}"
         # Debug output (can be removed in final version)
         # echo "  Interface found: $interface_name"
         local interface_up=true
         local is_loopback=false
   
         if [[ "$interface_name" == "lo" ]]; then
           is_loopback=true
           # Debug output (can be removed in final version)
           # echo "  Interface '$interface_name' is loopback. Skipping."
         fi
   
         if $is_loopback; then
           continue # Skip loopback interface
         fi
   
         # Look for inet lines within this interface block
         while IFS= read -r subnet_line; do
           # Debug output (can be removed in final version)
           # echo "  Processing subnet line: $subnet_line"
           if [[ "$subnet_line" =~ inet\ ([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)/([0-9]+)\ .*scope\ ([^ ]+) ]]; then
             local ip_address="${BASH_REMATCH[1]}"
             local scope="${BASH_REMATCH[3]}"
             # Debug output (can be removed in final version)
             # echo "    Found inet line: IP Address: $ip_address, Scope: $scope"
   
             # Exclude loopback IPs and docker0/bridge related IPs by IP range
             if [[ "$ip_address" =~ ^127\. ]]; then
               # Debug output (can be removed in final version)
               # echo "      IP '$ip_address' is loopback. Skipping."
               continue # Skip 127.0.0.0/8 loopback IPs (although 'lo' should already be skipped)
             elif [[ "$ip_address" =~ ^169\.254\. ]]; then
               # Debug output (can be removed in final version)
               # echo "      IP '$ip_address' is link-local (169.254.x.x). Skipping."
               continue # Skip 169.254.0.0/16 link-local IPs (like docker0 often has)
             fi
   
             local is_private_ip=false
             if [[ "$ip_address" =~ ^10\.([0-9]{1,3}\.){2}[0-9]{1,3}$ ]] ||
                [[ "$ip_address" =~ ^172\.(1[6-9]|2[0-9]|3[0-1])\.([0-9]{1,3}\.){1}[0-9]{1,3}$ ]] ||
                [[ "$ip_address" =~ ^192\.168\.([0-9]{1,3}\.){1}[0-9]{1,3}$ ]]; then
               is_private_ip=true
               # Debug output (can be removed in final version)
               # echo "      IP '$ip_address' is a private IP."
             # else
               # Debug output (can be removed in final version)
               # echo "      IP '$ip_address' is NOT a private IP."
             fi
   
             if $is_private_ip || [[ "$scope" == "global" ]]; then # Consider private or global scope interfaces
               selected_interface="$interface_name"
               # Debug output (can be removed in final version)
               # echo "      Interface '$interface_name' with IP '$ip_address' and scope '$scope' is selected."
               # echo "export GLOO_SOCKET_IFNAME=$selected_interface"
               # exit 0 # Exit immediately after finding the first suitable interface for debugging (removed for function)
               break 2 # Found a suitable interface! Break out of both inner and outer loops
             # else
               # Debug output (can be removed in final version)
               # echo "      Interface '$interface_name' with IP '$ip_address' and scope '$scope' is NOT suitable (not private or global)."
             fi
           fi
         done < <(echo "$ip_output" | sed -n "/$interface_name: /,/^[0-9]\+:/p" | sed '$d' ) # Extract lines belonging to current interface block
         if [[ -n "$selected_interface" ]]; then # Check if selected_interface is not empty, if so, interface found and loops broken.
             # Debug output (can be removed in final version)
             # echo "      Selected interface '$selected_interface' already found. Breaking outer loop."
             break # Already found and assigned an interface, break outer loop as well.
         fi
       # else
         # Debug output (can be removed in final version)
         # echo "  Line does not match interface pattern."
       fi
     done < <(echo "$ip_output")
   
     if [[ -n "$selected_interface" ]]; then
       echo "$selected_interface"
     else
       echo "" # Return empty string if no interface is found, so export GLOO_SOCKET_IFNAME=  (empty)
       # echo "No suitable network interface could be automatically identified for GLOO_SOCKET_IFNAME." # No longer print error message to stderr in function context
       # return 1 # Optionally, you could return a non-zero exit code if you need to check for failure.
     fi
   }
   ```

3. 训练变量配置1，不用修改，直接照搬即可

   ```shell
   RAY_MASTER_NODE_ADDRESS="0.0.0.0"
   RAY_MASTER_NODE_PORT=$(shuf -n 1 -i 30000-65535)
   WORLD_SIZE=1
   NODE_RANK=0
   GPUS_PER_NODE=4 # 修改这里以匹配你的GPU数量
   
   MASTER_HOST="$VC_WORKER_HOSTS"
   MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
   # export NCCL_SOCKET_IFNAME=ens2f5
   # export GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}
   export NCCL_NET_PLUGIN=none
   export NCCL_IB_TIMEOUT=22
   export NCCL_IB_RETRY_CNT=15
   export NCCL_DEBUG=INFO
   export CUDA_LAUNCH_BLOCKING=1
   
   export HOST_IP=0.0.0.0
   export VLLM_HOST_IP=0.0.0.0
   
   working_dir=${working_dir=-"/path/to/workdir"}
   cd $working_dir
   export HF_ENDPOINT=https://hf-mirror.com
   export WANDB_API_KEY="apikey"
   export WANDB_MODE="offline"
   nvjitlink=${nvjitlink:-""}
   nnode=$WORLD_SIZE
   tagname=${tagname:-""}
   dataver=${dataver:-"none"}
   tag=qw-vl7b-${tagname}
   rule_reward=${rule:-"none"}
   sys=${sys:-"default"}
   lr=${lr:-"10"}
   algo=${algo:-"group_sft"}
   temperature=${temperature:-"1.0"}
   numref=0
   fmt=${fmt:-"none"}
   bsz=${bsz:-"512"}
   rbuffer=${rbuffer:-"1024"}
   nsamples=${nsamples:-"8"}
   mbsz=${mbsz:-"2"}
   maxlen=${maxlen:-"6144"}
   lossver=${lossver:-"none"}
   mode=${mode:-"none"}
   nactor=${nactor:-"16"}
   nvllm=${nvllm:-"8"}
   filter=${filter:-"None"}
   repeat=${repeat:-"0"}
   nepoch=${nepoch:-"3"}
   logp_bsz=${logp_bsz:-"8"}
   maxtoken=${maxtoken:-"2048"}
   tp=${tp:-"1"}
   aux=${aux:-"0.05"}
   evalsteps=${evalsteps:-"0"}
   save_name="${tag}-${bsz}-lossver${lossver}-samplever${dataver}-fmt${fmt}-${algo}-n${nsamples}-ml${maxlen}-lr${lr}-sys${sys}-${nnode}node" # rbsize 1024->256
   
   DATASET=${trainver:-"/path/to/train.parquet"}
   MODEL_CPK_NAME=${save_name}
   PRETRAIN_MODEL=${policy}
   testdata=${testver:-"/path/to/test.parquet"}
   SAVE_PATH=$working_dir/saves/$save_name
   mkdir -p "${SAVE_PATH}"
   ```

4. **ray集群配置（重点）**

   主要是以下四个变量：

   *   `--actor_num_nodes`: **Actor模型（即正在训练的策略模型）使用的节点数**。单机训练中为1。
   *   `--actor_num_gpus_per_node`: **每个节点上，Actor模型使用的GPU数量**。这是进行模型训练（前向和反向传播）的核心部分。
   *   `--vllm_num_engines`: **vLLM推理引擎的数量**。vLLM用于“rollout”阶段（模型根据提示生成文本）的推理过程。
   *   `--vllm_tensor_parallel_size`: **vLLM的张量并行大小**。如果一个模型大到单张GPU放不下，就需要设置大于1的值来进行模型并行。

   设置要点：

   1. 所有的卡被分成两部分，actor和vllm
   2. `actor_num_nodes * actor_num_gpus_per_node = world size`，即actor进程数，每个actor进程都会独占一个gpu，在该gpu上加载一个完整的policy模型，没有张量并行的选项。实际上真正起作用的是`world size`对应的actor进程数 = actor部分所占的显卡数
   3. `vllm_num_engines * vllm_tensor_parallel_size = vllm所占显卡数`
   4. 要保证两部分所占的显卡数不超过机器上的显卡总数
   5. 要保证actor进程数=vllm engine数，否则在训练时分布式通信会出问题

   ```shell
   post_args=(--ref_num_nodes 0
           --ref_num_gpus_per_node 8
           --actor_num_nodes 1
           --actor_num_gpus_per_node 4
           --vllm_num_engines 4
           --vllm_tensor_parallel_size 1
           --adam_offload
           --micro_train_batch_size ${mbsz} # 在start_single_node_training.sh中配置了
           --train_batch_size ${bsz} # 在start_single_node_training.sh中配置了
           --micro_rollout_batch_size 1
           --rollout_batch_size ${rbuffer} # 在start_single_node_training.sh中配置了
   )
   ```

5. 训练依赖项配置

   其中：

   * `--num-gpus`：根据显卡数调整
   * 需要安装`jq`，`sudo apt install jq`

   ```shell
   CUDA_LIB_PATH="$CONDA_PREFIX/lib"
   
   # :/usr/local/cuda/targets/x86_64-linux/lib
   LD_LIBRARY_PATH_VALUE=${nvjitlink}:${CUDA_LIB_PATH}:$LD_LIBRARY_PATH
   export BNB_CUDA_VERSION=122
   RUNTIME_ENV_JSON="{\"pip\": [\"Qwen-Agent\"], \"env_vars\": {\"LD_LIBRARY_PATH\": \"$LD_LIBRARY_PATH_VALUE\"}}"
   
   # Start Ray head node and capture the output
   # ray_output=$(ray start --head --num-gpus 8) # 修改这里以匹配你的GPU数量
   ray_output=$(ray start --head \
     --num-gpus 8 \
     --temp-dir "$RAY_BASE_DIR/tmp" \
   )
   
   SPILL_CONFIG_OBJ=$(jq -n \
     --arg dir "$RAY_BASE_DIR/spill" \
     '{"type": "filesystem", "params": {"directory_path": $dir}}')
     
   RUNTIME_ENV_JSON=$(jq -n \
       --arg ld_path "$LD_LIBRARY_PATH_VALUE" \
       --arg tmpdir "$RAY_TMPDIR" \
       --argjson spill_config "$SPILL_CONFIG_OBJ" \
       '{
         "pip": ["Qwen-Agent"],
         "env_vars": {
           "LD_LIBRARY_PATH": $ld_path,
           "TMPDIR": $tmpdir
         },
         "_system_config": {
           "object_spilling_config": $spill_config,
           "automatic_object_spilling_enabled": true
         }
       }')
   ```

6. ray启动命令

   ```shell
   ray status
   ray_args=(
     --address="http://127.0.0.1:8265"
     --runtime-env-json="$RUNTIME_ENV_JSON"
     -- python3 -m openrlhf.cli.train_ppo_ray
     --vllm_enable_sleep
     --vllm_gpu_memory_utilization 0.7
     --vllm_sync_backend gloo
     --pretrain "$PRETRAIN_MODEL"
     --save_path "$SAVE_PATH"
     --n_samples_per_prompt "${nsamples}"
     --max_epochs 1
     --num_episodes "${nepoch}"
     --filter "${filter}"
     --prompt_max_len 2048
     --max_out_tokens "${maxtoken}"
     --max_samples 100000
     --generate_max_len "${maxlen}"
     --advantage_estimator "${algo}"
     --zero_stage 3
     --controlled_shuffle "${repeat}"
     --actor_learning_rate "${lr}e-7"
     --rule_reward "${rule_reward}"
     --temperature 1.0
     --val_temperature 0.6
     --top_p 0.95
     --training_mode "${mode}"
     --init_kl_coef 0.0
     --aux_loss_coef "${aux}"
     --entropy_loss_coef 0.0
     --prompt_data "$DATASET"
     --input_key question
     --apply_chat_template
     --normalize_reward
     # --flash_attn # FlashAttention only supports Ampere GPUs or newer. v100不能用flash_attn
     --gradient_checkpointing
     --ckpt_path "$SAVE_PATH"
     --save_steps 3
     --eval_steps "${evalsteps}"
     --max_ckpt_num 3
     --save_hf_ckpt
     --disable_ds_ckpt
     --disable_fast_tokenizer
     --wandb_run_name "$save_name"
     --system_prompt "${sys}"
     --use_kl_estimator_k3
     --wandb_project vlm-rl
     --buffer_norm 0
     --train_vlm
     --eval_data "${testdata}"
     --data_version "${dataver}"
     --loss_version "${lossver}"
     --format "${fmt}"
     "${post_args[@]}"
   )
   
   ray job submit "${ray_args[@]}"
   ```

##### 完整脚本

```shell
set -x

# =============== 新增：设置 Ray 专用目录 ===============
RAY_BASE_DIR="/mnt/workspace/workgroup/kejia/ray_data"
mkdir -p "$RAY_BASE_DIR/spill" "$RAY_BASE_DIR/tmp" "$RAY_BASE_DIR/logs"

export RAY_TMPDIR="$RAY_BASE_DIR/tmp" # Ray 运行时临时目录（如 socket、logs）
export RAY_LOGDIR="$RAY_BASE_DIR/logs" # 由于使用的 Ray 版本不支持 --logs-dir选项，Ray 会自动将日志写入 --temp-dir 下的子目录（如 session_latest/logs）

# 强制 Ray 使用自定义临时目录（通过环境变量）
# 某些 Ray 组件或底层库（如 vLLM、PyTorch）可能仍使用系统临时目录。你可以通过环境变量引导它们也使用自定义目录
export TMPDIR="$RAY_BASE_DIR/tmp"

find_interface() {
  local ip_output=$(ip addr show | head -n 10) # Limit to first 10 lines
  local selected_interface=""

  # Debug output (can be removed in final version)
  # echo "--- First 10 lines of ip addr show output: ---"
  # echo "$ip_output"
  # echo "--- End of ip addr show output ---"

  while IFS= read -r line; do
    # Debug output (can be removed in final version)
    # echo "Processing line: $line"

    if [[ "$line" =~ ^[0-9]+:\ ([^:]+):\ \<.*UP.*\> ]]; then
      local interface_name="${BASH_REMATCH[1]}"
      # Debug output (can be removed in final version)
      # echo "  Interface found: $interface_name"
      local interface_up=true
      local is_loopback=false

      if [[ "$interface_name" == "lo" ]]; then
        is_loopback=true
        # Debug output (can be removed in final version)
        # echo "  Interface '$interface_name' is loopback. Skipping."
      fi

      if $is_loopback; then
        continue # Skip loopback interface
      fi

      # Look for inet lines within this interface block
      while IFS= read -r subnet_line; do
        # Debug output (can be removed in final version)
        # echo "  Processing subnet line: $subnet_line"
        if [[ "$subnet_line" =~ inet\ ([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)/([0-9]+)\ .*scope\ ([^ ]+) ]]; then
          local ip_address="${BASH_REMATCH[1]}"
          local scope="${BASH_REMATCH[3]}"
          # Debug output (can be removed in final version)
          # echo "    Found inet line: IP Address: $ip_address, Scope: $scope"

          # Exclude loopback IPs and docker0/bridge related IPs by IP range
          if [[ "$ip_address" =~ ^127\. ]]; then
            # Debug output (can be removed in final version)
            # echo "      IP '$ip_address' is loopback. Skipping."
            continue # Skip 127.0.0.0/8 loopback IPs (although 'lo' should already be skipped)
          elif [[ "$ip_address" =~ ^169\.254\. ]]; then
            # Debug output (can be removed in final version)
            # echo "      IP '$ip_address' is link-local (169.254.x.x). Skipping."
            continue # Skip 169.254.0.0/16 link-local IPs (like docker0 often has)
          fi

          local is_private_ip=false
          if [[ "$ip_address" =~ ^10\.([0-9]{1,3}\.){2}[0-9]{1,3}$ ]] ||
             [[ "$ip_address" =~ ^172\.(1[6-9]|2[0-9]|3[0-1])\.([0-9]{1,3}\.){1}[0-9]{1,3}$ ]] ||
             [[ "$ip_address" =~ ^192\.168\.([0-9]{1,3}\.){1}[0-9]{1,3}$ ]]; then
            is_private_ip=true
            # Debug output (can be removed in final version)
            # echo "      IP '$ip_address' is a private IP."
          # else
            # Debug output (can be removed in final version)
            # echo "      IP '$ip_address' is NOT a private IP."
          fi

          if $is_private_ip || [[ "$scope" == "global" ]]; then # Consider private or global scope interfaces
            selected_interface="$interface_name"
            # Debug output (can be removed in final version)
            # echo "      Interface '$interface_name' with IP '$ip_address' and scope '$scope' is selected."
            # echo "export GLOO_SOCKET_IFNAME=$selected_interface"
            # exit 0 # Exit immediately after finding the first suitable interface for debugging (removed for function)
            break 2 # Found a suitable interface! Break out of both inner and outer loops
          # else
            # Debug output (can be removed in final version)
            # echo "      Interface '$interface_name' with IP '$ip_address' and scope '$scope' is NOT suitable (not private or global)."
          fi
        fi
      done < <(echo "$ip_output" | sed -n "/$interface_name: /,/^[0-9]\+:/p" | sed '$d' ) # Extract lines belonging to current interface block
      if [[ -n "$selected_interface" ]]; then # Check if selected_interface is not empty, if so, interface found and loops broken.
          # Debug output (can be removed in final version)
          # echo "      Selected interface '$selected_interface' already found. Breaking outer loop."
          break # Already found and assigned an interface, break outer loop as well.
      fi
    # else
      # Debug output (can be removed in final version)
      # echo "  Line does not match interface pattern."
    fi
  done < <(echo "$ip_output")

  if [[ -n "$selected_interface" ]]; then
    echo "$selected_interface"
  else
    echo "" # Return empty string if no interface is found, so export GLOO_SOCKET_IFNAME=  (empty)
    # echo "No suitable network interface could be automatically identified for GLOO_SOCKET_IFNAME." # No longer print error message to stderr in function context
    # return 1 # Optionally, you could return a non-zero exit code if you need to check for failure.
  fi
}

RAY_MASTER_NODE_ADDRESS="0.0.0.0"
RAY_MASTER_NODE_PORT=$(shuf -n 1 -i 30000-65535)
WORLD_SIZE=1
NODE_RANK=0
GPUS_PER_NODE=4 # 修改这里以匹配你的GPU数量

MASTER_HOST="$VC_WORKER_HOSTS"
MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
# export NCCL_SOCKET_IFNAME=ens2f5
# export GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}
export NCCL_NET_PLUGIN=none
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=15
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1

export HOST_IP=0.0.0.0
export VLLM_HOST_IP=0.0.0.0

working_dir=${working_dir=-"/path/to/workdir"}
cd $working_dir
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_API_KEY="apikey"
export WANDB_MODE="offline"
nvjitlink=${nvjitlink:-""}
nnode=$WORLD_SIZE
tagname=${tagname:-""}
dataver=${dataver:-"none"}
tag=qw-vl7b-${tagname}
rule_reward=${rule:-"none"}
sys=${sys:-"default"}
lr=${lr:-"10"}
algo=${algo:-"group_sft"}
temperature=${temperature:-"1.0"}
numref=0
fmt=${fmt:-"none"}
bsz=${bsz:-"512"}
rbuffer=${rbuffer:-"1024"}
nsamples=${nsamples:-"8"}
mbsz=${mbsz:-"2"}
maxlen=${maxlen:-"6144"}
lossver=${lossver:-"none"}
mode=${mode:-"none"}
nactor=${nactor:-"16"}
nvllm=${nvllm:-"8"}
filter=${filter:-"None"}
repeat=${repeat:-"0"}
nepoch=${nepoch:-"3"}
logp_bsz=${logp_bsz:-"8"}
maxtoken=${maxtoken:-"2048"}
tp=${tp:-"1"}
aux=${aux:-"0.05"}
evalsteps=${evalsteps:-"0"}
save_name="${tag}-${bsz}-lossver${lossver}-samplever${dataver}-fmt${fmt}-${algo}-n${nsamples}-ml${maxlen}-lr${lr}-sys${sys}-${nnode}node" # rbsize 1024->256

DATASET=${trainver:-"/path/to/train.parquet"}
MODEL_CPK_NAME=${save_name}
PRETRAIN_MODEL=${policy}
testdata=${testver:-"/path/to/test.parquet"}
SAVE_PATH=$working_dir/saves/$save_name
mkdir -p "${SAVE_PATH}"

post_args=(--ref_num_nodes 0
        --ref_num_gpus_per_node 8
        --actor_num_nodes 1
        --actor_num_gpus_per_node 4
        --vllm_num_engines 4
        --vllm_tensor_parallel_size 1
        --adam_offload
        --micro_train_batch_size ${mbsz} # 在start_single_node_training.sh中配置了
        --train_batch_size ${bsz} # 在start_single_node_training.sh中配置了
        --micro_rollout_batch_size 1
        --rollout_batch_size ${rbuffer} # 在start_single_node_training.sh中配置了
)

CUDA_LIB_PATH="$CONDA_PREFIX/lib"

# :/usr/local/cuda/targets/x86_64-linux/lib
LD_LIBRARY_PATH_VALUE=${nvjitlink}:${CUDA_LIB_PATH}:$LD_LIBRARY_PATH
export BNB_CUDA_VERSION=122
RUNTIME_ENV_JSON="{\"pip\": [\"Qwen-Agent\"], \"env_vars\": {\"LD_LIBRARY_PATH\": \"$LD_LIBRARY_PATH_VALUE\"}}"

# Start Ray head node and capture the output
# ray_output=$(ray start --head --num-gpus 8) # 修改这里以匹配你的GPU数量
ray_output=$(ray start --head \
  --num-gpus 8 \
  --temp-dir "$RAY_BASE_DIR/tmp" \
)

SPILL_CONFIG_OBJ=$(jq -n \
  --arg dir "$RAY_BASE_DIR/spill" \
  '{"type": "filesystem", "params": {"directory_path": $dir}}')
  
RUNTIME_ENV_JSON=$(jq -n \
    --arg ld_path "$LD_LIBRARY_PATH_VALUE" \
    --arg tmpdir "$RAY_TMPDIR" \
    --argjson spill_config "$SPILL_CONFIG_OBJ" \
    '{
      "pip": ["Qwen-Agent"],
      "env_vars": {
        "LD_LIBRARY_PATH": $ld_path,
        "TMPDIR": $tmpdir
      },
      "_system_config": {
        "object_spilling_config": $spill_config,
        "automatic_object_spilling_enabled": true
      }
    }')
    
ray status
ray_args=(
  --address="http://127.0.0.1:8265"
  --runtime-env-json="$RUNTIME_ENV_JSON"
  -- python3 -m openrlhf.cli.train_ppo_ray
  --vllm_enable_sleep
  --vllm_gpu_memory_utilization 0.7
  --vllm_sync_backend gloo
  --pretrain "$PRETRAIN_MODEL"
  --save_path "$SAVE_PATH"
  --n_samples_per_prompt "${nsamples}"
  --max_epochs 1
  --num_episodes "${nepoch}"
  --filter "${filter}"
  --prompt_max_len 2048
  --max_out_tokens "${maxtoken}"
  --max_samples 100000
  --generate_max_len "${maxlen}"
  --advantage_estimator "${algo}"
  --zero_stage 3
  --controlled_shuffle "${repeat}"
  --actor_learning_rate "${lr}e-7"
  --rule_reward "${rule_reward}"
  --temperature 1.0
  --val_temperature 0.6
  --top_p 0.95
  --training_mode "${mode}"
  --init_kl_coef 0.0
  --aux_loss_coef "${aux}"
  --entropy_loss_coef 0.0
  --prompt_data "$DATASET"
  --input_key question
  --apply_chat_template
  --normalize_reward
  # --flash_attn # FlashAttention only supports Ampere GPUs or newer. v100不能用flash_attn
  --gradient_checkpointing
  --ckpt_path "$SAVE_PATH"
  --save_steps 3
  --eval_steps "${evalsteps}"
  --max_ckpt_num 3
  --save_hf_ckpt
  --disable_ds_ckpt
  --disable_fast_tokenizer
  --wandb_run_name "$save_name"
  --system_prompt "${sys}"
  --use_kl_estimator_k3
  --wandb_project vlm-rl
  --buffer_norm 0
  --train_vlm
  --eval_data "${testdata}"
  --data_version "${dataver}"
  --loss_version "${lossver}"
  --format "${fmt}"
  "${post_args[@]}"
)

ray job submit "${ray_args[@]}"
```

