---
name: add-new-compat-api
description: 负责《Paddle API 对齐 PyTorch 项目》中 Step2 代码修改，实施『新增 compat 类型 API』方案。在 `paddle.compat` 命名空间下新增 API，实现与 PyTorch API 行为对齐。
context: fork
disable-model-invocation: false
---

# 与 add-new-api 的关系

本文档是 **add-new-api** 的变体，核心流程完全一致，**唯一区别**是新增 API 的存放目录：

| 方案 | API 存放目录 | 适用场景 |
|------|-------------|---------|
| add-new-api | `Paddle/python/paddle/` (常规目录) | 新增 Paddle 原生 API |
| add-new-compat-api | `Paddle/python/paddle/compat/` (compat 目录) | 新增兼容性 API |

**请直接参考 add-new-api 的 SKILL.md 完成开发**，并将所有文件路径中的 `Paddle/python/paddle/` 替换为 `Paddle/python/paddle/compat/`。

---

# 核心差异说明

## 实现位置映射

| add-new-api 路径 | add-new-compat-api 路径 |
|----------------|------------------------|
| `python/paddle/__init__.py` | `python/paddle/compat/__init__.py` |
| `python/paddle/tensor/math.py` | `python/paddle/compat/__init__.py` |
| `python/paddle/tensor/__init__.py` | `python/paddle/compat/__init__.py` |
| `python/paddle/nn/__init__.py` | `python/paddle/compat/nn/__init__.py` |
| `python/paddle/nn/functional/__init__.py` | `python/paddle/compat/nn/functional/__init__.py` |
