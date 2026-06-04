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
        ├── api-change-decider/     # Step1：方案决策
        ├── python-decorator/       # Step2：Python 装饰器
        ├── cpp-sink/               # Step2：C++下沉
        ├── modify-origin-api/      # Step2：修改原有 API
        ├── add-new-api/            # Step2：新增 API
        ├── add-new-compat-api/     # Step2：新增 compat API
        ├── add-compatibility-test/ # Step3：兼容性测试
        ├── pytorch-alignment-validator/  # Step4：对齐验证
        ├── api-docs-updater/       # Step5：文档更新
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

```bash
./install.sh /path/to/PROJECT_ROOT
```

PROJECT_ROOT 需包含 `Paddle/`、`PaConvert/`、`docs/` 三个子目录。

## 使用方式

**总控 Skill（推荐）**：
```bash
/api-compatibility torch.atan torch.asinh
```

**单独调用 Skill**：
```bash
/api-change-decider torch.atan    # Step1: 方案决策
/cpp-sink torch.atan              # Step2: 代码修改
/add-compatibility-test torch.atan # Step3: 兼容测试
/pytorch-alignment-validator torch.atan  # Step4: 对齐验证
/api-docs-updater torch.atan      # Step5: 文档更新
/create-pr torch.atan             # 提交 PR
```

## 工作流程

```
Step1 方案决策 → Step2 代码修改 → Step3 兼容测试 → Step4 对齐验证 → Step5 文档更新
```

## 详细文档
- [项目背景](.claude/CLAUDE.md)
- [总控 Skill 详细流程](.claude/skills/api-compatibility/SKILL.md)
