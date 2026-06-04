#!/bin/bash
# Paddle API 对齐项目 - 安装脚本
#
# 用法: ./install.sh /path/to/PROJECT_ROOT

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKILLS_DIR="$HOME/.claude/skills"

if [ -z "$1" ]; then
    echo "用法: ./install.sh /path/to/PROJECT_ROOT"
    echo ""
    echo "PROJECT_ROOT 需包含以下子目录："
    echo "  - Paddle/     (Paddle 框架源码)"
    echo "  - PaConvert/  (PyTorch 转换工具)"
    echo "  - docs/       (Paddle 文档)"
    exit 1
fi

PROJECT_ROOT="${1/#\~/$HOME}"

echo "=== Paddle API 对齐项目 - 安装脚本 ==="
echo ""

# 检查必需的子目录
echo "[1/3] 检查项目根目录: $PROJECT_ROOT"
MISSING_DIRS=""
for dir in Paddle PaConvert docs; do
    if [ ! -d "$PROJECT_ROOT/$dir" ]; then
        MISSING_DIRS="$MISSING_DIRS $dir"
    fi
done

if [ -n "$MISSING_DIRS" ]; then
    echo "  ✗ 以下目录不存在:$MISSING_DIRS"
    exit 1
fi
echo "  ✓ 目录检查通过"

# 安装 Skills
echo ""
echo "[2/3] 安装 Skills..."
mkdir -p "$SKILLS_DIR"
cp -r "$SCRIPT_DIR/.claude/skills/"* "$SKILLS_DIR/"
echo "  ✓ Skills 已安装到: $SKILLS_DIR"

# 安装 CLAUDE.md
echo ""
echo "[3/3] 安装 CLAUDE.md..."
cp "$SCRIPT_DIR/.claude/CLAUDE.md" "$PROJECT_ROOT/"
echo "  ✓ 已安装: $PROJECT_ROOT/CLAUDE.md"

echo ""
echo "=== 安装完成 ==="
echo ""
echo "使用方法:"
echo "  cd $PROJECT_ROOT"
echo "  /api-compatibility torch.atan"
