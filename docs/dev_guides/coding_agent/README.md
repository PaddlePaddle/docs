# AI 编程 Agent 规则目录

## 什么是 Agent 规则

Agent 规则是一组预定义的配置文件，用于指导 AI 编程助手（如 Claude Code）在 Paddle 代码库中执行自动化开发任务。

**核心组成**：
- **SKILL.md**：定义 AI 可执行的技能，包含任务流程、操作步骤、注意事项
- **CLAUDE.md**：定义项目背景信息，自动加载到 AI 上下文中

**工作原理**：
1. 用户通过 `/skill-name` 调用技能
2. AI 读取 SKILL.md 中的指令
3. AI 按照预定义流程执行任务

**优势**：
- 任务流程标准化，减少人工干预
- 知识沉淀，经验可持续积累
- 多 Skill 协作，完成复杂任务

## 目录结构

```
coding_agent/
├── README.md                           # 本文件
└── api_compatibility/                  # Paddle API 对齐 PyTorch 项目
    ├── README.md
    ├── install.sh
    └── .claude/
        ├── CLAUDE.md
        └── skills/
```

## 项目列表

| 项目 | 功能 |
|------|------|
| [api_compatibility](api_compatibility/) | 自动对齐 Paddle API 与 PyTorch API |
