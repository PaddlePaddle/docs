# Paddle API 对齐 PyTorch 项目

基于 Claude Code 的 AI Agent 自动对齐 Paddle API 与 PyTorch API。

## 本目录内容

```
api_compatibility/                  # 本目录
├── README.md                       # 本文件
├── install.sh                      # 安装脚本
└── .claude/                        # Claude Code 配置
    ├── CLAUDE.md                   # 项目背景（自动加载）
    └── skills/                     # Skill 定义
        ├── api-compatibility/      # 总控
        ├── select-solution/        # Step1 选择方案
        ├── python-decorator/       # Step2 Python 装饰器
        ├── cpp-sink/               # Step2 C++下沉
        ├── modify-origin-api/      # Step2 修改原有 API
        ├── add-new-api/            # Step2 新增 API
        ├── add-new-compat-api/     # Step2 新增 compat API
        ├── compatibility-test/     # Step3 兼容测试
        ├── pytorch-test/           # Step4 Pytorch 测试
        ├── update-docs/            # Step5 更新文档
        └── create-pr/              # 提交 PR
```

## 项目根目录要求

项目根目录（PROJECT_ROOT）需提前准备三个仓库：

```
{PROJECT_ROOT}/
├── Paddle/      # Paddle 框架源码
├── PaConvert/   # PyTorch 转换工具
├── docs/        # Paddle 文档
└── CLAUDE.md    # 安装后生成
```

## 安装
PROJECT_ROOT 需提前下载 `Paddle/`、`PaConvert/`、`docs/` 三个子目录。

```bash
./install.sh ${PROJECT_ROOT}
export PYTHONPATH="${PROJECT_ROOT}/Paddle/build/python:${env:PYTHONPATH}"
```

## 使用方式

**总控 Skill（推荐）**：
```bash
/api-compatibility torch.atan torch.asinh
```

**单独调用 Skill**：
```bash
/select-solution torch.atan         # Step1: 选择方案
/cpp-sink torch.atan                # Step2: 代码修改
/compatibility-test torch.atan      # Step3: 兼容测试
/pytorch-test torch.atan            # Step4: Pytorch 测试
/update-docs torch.atan             # Step5: 更新文档
/create-pr torch.atan               # 提交 PR
```

## 工作流程

```
Step1 选择方案 → Step2 代码修改 → Step3 兼容测试 → Step4 Pytorch 测试 → Step5 更新文档
```

## 详细文档
- [项目背景](.claude/CLAUDE.md)
- [总控 Skill 详细流程](.claude/skills/api-compatibility/SKILL.md)
