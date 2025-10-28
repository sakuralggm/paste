#!/bin/bash

echo "开始处理当前目录下的符号链接..."
echo "目标：用源文件/目录 *移动* 替换符号链接。"
echo "-----------------------------------------"

# 使用 find -print0 和 while read -r -d '' 
# 这是在Linux中遍历文件最安全的方法，可以处理空格、换行符等特殊字符
find . -maxdepth 1 -type l -print0 | while IFS= read -r -d '' link_name; do
    
    # [ -e ] 检查文件是否存在。当用于符号链接时，它会自动检查链接的目标。
    # 如果目标不存在（悬空链接），[ ! -e ] 为 true
    if [ ! -e "$link_name" ]; then
        # readlink 不加 -f 来获取原始链接目标，用于显示
        target_name=$(readlink "$link_name")
        echo "警告: 链接 '$link_name' 指向一个不存在的目标 '$target_name'。已跳过。"
        echo "-----------------------------------------"
        continue
    fi
    
    # 获取链接的原始目标路径（可能是相对或绝对路径）
    target_path=$(readlink "$link_name")
    
    echo "处理中: $link_name -> $target_path"
    
    # 步骤 1: 先安全地删除符号链接
    # 这是关键一步，否则 mv 一个目录到指向它自己的链接会失败
    rm "$link_name"
    if [ $? -ne 0 ]; then
        echo "  错误: 无法删除链接 '$link_name'。已跳过。"
        echo "-----------------------------------------"
        continue
    fi
    
    # 步骤 2: 将源文件或源目录移动到原链接的位置和名称
    # -v 选项会显示 mv 正在做什么
    mv -v "$target_path" "$link_name"
    
    if [ $? -eq 0 ]; then
        echo "  成功: 源 '$target_path' 已被移动并重命名为 '$link_name'。"
    else
        echo "  严重错误: 移动 '$target_path' 到 '$link_name' 失败！"
        echo "  注意: 链接 '$link_name' 已被删除，但源文件未移动。"
        # 尝试恢复原链接，避免系统处于不一致状态
        echo "  正在尝试恢复原链接..."
        ln -s "$target_path" "$link_name"
        if [ $? -eq 0 ]; then
            echo "  恢复成功: 原链接 '$link_name' 已被重建。"
        else
            echo "  恢复失败: 无法重建链接。请手动检查 '$target_path'。"
        fi
    fi
    
    echo "-----------------------------------------"
    
done

echo "所有操作完成。"
