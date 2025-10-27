#!/bin/bash

# ==============================================================================
#  文件名: init.sh (v2)
#  描述:   一个用于自动化环境初始化的脚本，每次开启新终端时都要执行
#  作者:   (由Gemini生成)
# ==============================================================================

# --- 全局设置与辅助函数 ---
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# 打印状态信息的函数
print_status() {
    local type=$1
    local message=$2
    case "$type" in
        "HEADER") echo -e "\n${BLUE}🚀 --- $message --- 🚀${NC}";;
        "INFO") echo -e "[${BLUE}ℹ️${NC}] $message";;
        "SUCCESS") echo -e "[${GREEN}✅${NC}] $message";;
        "WARN") echo -e "[${YELLOW}⚠️${NC}] $message";;
        "ERROR") echo -e "[${RED}❌${NC}] $message";;
    esac
}

# --- 核心功能函数 ---
# 设置代理和APT代理
setup_proxy() {
    print_status "HEADER" "第一步: 设置系统和APT代理"
    local proxy_script="/mnt/nfs/proxy_on"
    if [ -f "$proxy_script" ]; then
        print_status "INFO" "正在加载代理环境变量: source $proxy_script"
        source "$proxy_script"
        export HF_ENDPOINT=https://hf-mirror.com # 增加hf镜像站的环境变量
        ln -sf /usr/lib/x86_64-linux-gnu/libffi.so.7 /mnt/nfs/zzr/conda_envs/qtcls_vllm/lib/libffi.so.7 # 修复curl的bug
        export TZ='Asia/Shanghai' # 设置时区为上海
        export internlm_api_key='sk-Rcr56gIulAkEVsB9b5Qycja4lk5DsDLJ3optiqK918BjMqGu'
        print_status "SUCCESS" "代理环境变量加载完成。"
    else
        print_status "ERROR" "代理脚本 $proxy_script 未找到，跳过此步骤。"
    fi
    local apt_proxy_script="/mnt/nfs/zzr/local/setup_apt_proxy.sh"
    if [ -f "$apt_proxy_script" ]; then
        print_status "INFO" "正在执行APT代理设置脚本: sudo $apt_proxy_script"
        sudo "$apt_proxy_script"
        if [ $? -eq 0 ]; then print_status "SUCCESS" "APT代理设置脚本执行成功。"; else print_status "ERROR" "APT代理设置脚本执行失败。"; fi
    else
        print_status "ERROR" "APT代理脚本 $apt_proxy_script 未找到，跳过此步骤。"
    fi
}

# 配置Conda环境
configure_conda() {
    print_status "HEADER" "第二步: 配置Conda环境目录"
    if ! command -v conda &> /dev/null; then
        print_status "WARN" "Conda命令未找到，现在尝试自动初始化..."
        local conda_executable="/opt/conda/bin/conda"
        if [ -f "$conda_executable" ]; then
            "$conda_executable" init bash
            if [ $? -eq 0 ]; then
                print_status "SUCCESS" "Conda初始化命令执行成功。"
                print_status "INFO" "正在重新加载shell配置以使conda生效..."
                source "$HOME/.bashrc"
            else
                print_status "ERROR" "Conda初始化失败！"
                return
            fi
        else
            print_status "ERROR" "Conda可执行文件 $conda_executable 未找到！"
            return
        fi
    fi

    # 定义目标环境目录，以便在验证步骤中使用
    local envs_dir="/mnt/nfs/zzr/conda_envs"

    print_status "INFO" "正在将完整的 Conda 配置直接写入 ~/.condarc"

# 使用 cat 和 heredoc (EOF) 直接创建或覆盖 .condarc 文件
# 注意：使用 'EOF' 而不是 EOF 可以防止 shell 对文件内容中的 '$' 等特殊字符进行扩展，是更安全的做法。
cat << 'EOF' > ~/.condarc
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.zju.edu.cn/anaconda/pkgs/main
  - https://mirrors.zju.edu.cn/anaconda/pkgs/r
  - https://mirrors.zju.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.zju.edu.cn/anaconda/cloud
  pytorch: https://mirrors.zju.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.zju.edu.cn/anaconda/cloud
  nvidia: https://mirrors.zju.edu.cn/anaconda-r
  msys2: https://mirrors.zju.edu.cn/anaconda/cloud
  bioconda: https://mirrors.zju.edu.cn/anaconda/cloud
  menpo: https://mirrors.zju.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.zju.edu.cn/anaconda/cloud
envs_dirs:
  - /mnt/nfs/zzr/conda_envs
EOF

    # 配置某些conda环境变量
    print_status "INFO" "正在配置 Conda 环境变量..."
    setup_env_variables

    print_status "INFO" "正在验证配置..."

    # 验证逻辑保持不变，它会检查 conda 是否能正确读取和解析新的配置文件
    if conda config --show | grep -q "$envs_dir"; then
        print_status "SUCCESS" "Conda 配置文件 ~/.condarc 写入成功！"
        echo "--- conda config --show 部分输出 ---"
        # 显示 envs_dirs 相关配置以供确认
        conda config --show | grep "envs_dirs" -A 2
        echo "------------------------------------"
    else
        print_status "ERROR" "Conda 配置文件 ~/.condarc 写入失败或配置未生效。"
        # 如果失败，打印出文件内容以帮助调试
        echo "--- ~/.condarc 的当前内容 ---"
        cat ~/.condarc
        echo "-------------------------------"
    fi
}

# 设置conda环境变量
setup_env_variables() {
    local bashrc_file="$HOME/.bashrc"
    # 使用单引号可以防止 $PATH 和 $LD_LIBRARY_PATH 被立即解析
    # local path_line='export PATH="/mnt/nfs/zzr/conda_envs/qtcls/bin:$PATH"'
    local path_line='export PATH="$PATH"'
    local ld_path_line='export LD_LIBRARY_PATH="/mnt/nfs/zzr/conda_envs/qtcls_vllm/lib:$LD_LIBRARY_PATH"'
    
    # 检查第一行配置是否存在，-F 表示按固定字符串搜索，-q 表示静默模式
    if grep -qF "$path_line" "$bashrc_file"; then
        print_status "SUCCESS" "环境变量已存在于 .bashrc，无需再次添加。"
    else
        print_status "INFO" "正在将环境变量添加到 $bashrc_file ..."
        # 使用 echo -e 添加一个空行和注释，然后追加两个 export 语句
        echo -e "\n# 为 qtcls 自定义环境变量" >> "$bashrc_file"
        echo "$path_line" >> "$bashrc_file"
        echo "$ld_path_line" >> "$bashrc_file"
        
        print_status "SUCCESS" "环境变量成功写入 .bashrc！"
        print_status "INFO" "正在重新加载配置使其立即生效..."
        source "$bashrc_file"
    fi
}

# 配置SSH Authorized Keys
setup_ssh_key() {
    print_status "HEADER" "第三步: 配置SSH Authorized Keys"
    local source_key="/mnt/nfs/zzr/config/demo-repo2.pub"
    local target_key="$HOME/.ssh/authorized_keys"
    if [ ! -f "$source_key" ]; then print_status "ERROR" "源公钥文件 $source_key 不存在！"; return; fi
    print_status "INFO" "正在创建 .ssh 目录 (如果不存在)..."; mkdir -p "$HOME/.ssh"
    print_status "INFO" "正在复制公钥到 $target_key"; cp "$source_key" "$target_key"
    print_status "INFO" "正在验证公钥配置..."
    if [ -f "$target_key" ] && diff -q "$source_key" "$target_key" &>/dev/null; then
        print_status "SUCCESS" "SSH公钥配置成功，文件内容一致。"
    else
        print_status "ERROR" "SSH公钥配置失败！目标文件不存在或内容不匹配。"
    fi
}

# 为nfs中个人目录创建一个快捷方式
setup_shell_alias() {
    print_status "HEADER" "第四步: 创建 'me' 命令快捷方式"
    
    local alias_str="alias me='cd /mnt/nfs/zzr'"
    local bashrc_file="$HOME/.bashrc"
    
    if grep -qFx "$alias_str" "$bashrc_file"; then
        print_status "SUCCESS" "别名 'me' 已存在于 .bashrc，无需再次添加。"
    else
        print_status "INFO" "正在将别名 'me' 添加到 $bashrc_file ..."
        echo -e "\n# 自定义别名，用于快速跳转目录\n$alias_str" >> "$bashrc_file"
        print_status "SUCCESS" "别名 'me' 成功写入 .bashrc！"
        source "$HOME/.bashrc"
    fi
}

# 解析SSH连接信息
parse_ssh_info() {
    print_status "HEADER" "第五步: 解析SSH连接信息"
    local addr_file="$HOME/share/ssh-addr-zzr"
    if [ ! -f "$addr_file" ]; then print_status "ERROR" "地址文件 $addr_file 未找到！"; return; fi
    local content; content=$(cat "$addr_file")
    print_status "INFO" "从文件读取到的内容: $content"
    local username ip port
    username=$(echo "$content" | awk -F'[@ ]' '{print $1}'); ip=$(echo "$content" | awk -F'[@ ]' '{print $2}'); port=$(echo "$content" | awk -F'[@ ]' '{print $NF}')
    if [ -n "$username" ] && [ -n "$ip" ] && [ -n "$port" ]; then
        print_status "SUCCESS" "信息解析成功！"
        echo -e "  👤 用户名: ${GREEN}$username${NC}\n  🌐 IP 地址: ${GREEN}$ip${NC}\n  🔌 端 口:  ${GREEN}$port${NC}"
    else
        print_status "ERROR" "解析失败，请检查文件格式是否为 'user@ip -p port'。"
    fi
}

# --- 可选安装 Tmux ---
install_tmux() {
    print_status "HEADER" "第六步: 可选安装Tmux"
    cp /mnt/nfs/zzr/config/.tmux.conf ~/.tmux.conf
    if command -v tmux &> /dev/null; then print_status "SUCCESS" "Tmux 已经安装，版本: $(tmux -V)"; return; fi
    local confirm; read -p "🤔 是否要安装 tmux? (Y/n) " -r confirm
    if [[ -z "$confirm" || "$confirm" =~ ^[Yy]$ ]]; then
        print_status "INFO" "好的，正在准备安装 tmux..."
        sudo apt-get update && sudo apt-get install -y tmux
        if [ $? -eq 0 ]; then print_status "SUCCESS" "Tmux 安装成功！版本: $(tmux -V)"; else print_status "ERROR" "Tmux 安装失败。"; fi
    else
        print_status "WARN" "已取消安装 tmux。"
    fi
}

# --- 可选安装 Tmux, 开发工具, Aria2 及 7-Zip ---
install_tmux_and_dev_tools() {
    print_status "HEADER" "第六步: 可选安装Tmux、开发工具、Aria2及7-Zip"
    
    # 检查 Tmux 是否已安装
    if command -v tmux &> /dev/null; then
        print_status "SUCCESS" "Tmux 已经安装，版本: $(tmux -V)"
    else
        print_status "INFO" "Tmux 未安装。"
    fi
    
    # 检查 build-essential 是否已安装 (通过检查 gcc)
    if command -v gcc &> /dev/null; then
        print_status "SUCCESS" "开发工具 (build-essential) 已安装。"
    else
        print_status "INFO" "开发工具 (build-essential) 未安装。"
    fi

    # 检查 aria2 是否已安装
    if command -v aria2c &> /dev/null; then
        print_status "SUCCESS" "Aria2 已经安装。"
    else
        print_status "INFO" "Aria2 未安装。"
    fi

    # 检查 p7zip-full 是否已安装 (通过检查 7z)
    if command -v 7z &> /dev/null; then
        print_status "SUCCESS" "p7zip-full 已安装。"
    else
        print_status "INFO" "p7zip-full 未安装。"
    fi

    # 检查 git 是否已安装
    if command -v git &> /dev/null; then
        print_status "SUCCESS" "git 已安装。"
    else
        print_status "INFO" "git 未安装。"
    fi

    # 复制配置文件
    print_status "INFO" "正在复制 .tmux.conf 配置文件..."
    # 确保源目录存在
    if [ -f /mnt/nfs/zzr/config/.tmux.conf ]; then
        cp /mnt/nfs/zzr/config/.tmux.conf ~/.tmux.conf
        print_status "SUCCESS" ".tmux.conf 已复制到主目录。"
    else
        print_status "WARN" "源配置文件 /mnt/nfs/zzr/config/.tmux.conf 未找到，跳过复制。"
    fi

    local confirm
    read -p "🤔 是否要安装/更新 tmux, 开发工具 (build-essential), aria2, git 和 p7zip-full? (Y/n) " -r confirm
    
    if [[ -z "$confirm" || "$confirm" =~ ^[Yy]$ ]]; then
        print_status "INFO" "好的，正在准备安装..."
        # 同时安装所有选定的软件包
        sudo apt-get install -y tmux build-essential aria2 p7zip-full git
        git config --global user.name "Index2022"
        git config --global user.email "2033541709@qq.com"
        
        if [ $? -eq 0 ]; then
            print_status "SUCCESS" "选定的工具安装成功！"
            # 打印版本信息以供验证
            if command -v tmux &> /dev/null; then print_status "INFO" "Tmux 版本: $(tmux -V)"; fi
            if command -v gcc &> /dev/null; then print_status "INFO" "GCC 版本: $(gcc --version | head -n1)"; fi
            if command -v aria2c &> /dev/null; then print_status "INFO" "Aria2 版本: $(aria2c --version | head -n1)"; fi
            if command -v git &> /dev/null; then print_status "INFO" "Git 版本: $(git --version | head -n1)"; fi
            # 7z 的版本信息通常在第二行
            if command -v 7z &> /dev/null; then print_status "INFO" "7-Zip 版本: $(7z | head -n2 | tail -n1)"; fi
        else
            print_status "ERROR" "安装过程中发生错误。"
        fi
    else
        print_status "WARN" "已取消安装。"
    fi
}

# --- 主函数 ---
main() {
    clear 
    print_status "HEADER" "环境初始化脚本开始执行 (v1)"
    echo "================================================="
    setup_proxy
    configure_conda
    setup_ssh_key
    setup_shell_alias
    parse_ssh_info
    install_tmux_and_dev_tools
    
    # 🌟 修改：更新了最后的提示信息
    echo -e "\n${GREEN}🎉🎉🎉 所有任务已执行完毕！🎉🎉🎉${NC}"
}

# --- 脚本入口 ---
main
