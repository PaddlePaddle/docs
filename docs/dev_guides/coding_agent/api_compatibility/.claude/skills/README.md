# 《Paddle API 对齐 PyTorch 项目》完整 SKILL 体系

AI Agent 自动对齐 Paddle API 与 PyTorch API。包含 1 个总 skill 和 6 个分阶段 skill。

## 两种使用方式

### 方式一：一步到位（推荐）

使用总 skill **`/api-compatibility`**，传入待对齐的 PyTorch API 列表：

```
/api-compatibility torch.atan torch.asinh torch.abs_
```

总 skill 会自动执行完整流程（Step1→Step2→Step3→Step4→Step5），返回对齐结果统计表。

### 方式二：分步操作

针对具体任务，单独调用对应 skill：

| 阶段 | Skill | 用途 |
|------|-------|------|
| Step1 | `/api-change-decider` | 分析 API 差异，决策改动方案 |
| Step2 | `/python-decorator` | 用 Python 装饰器修改 API |
| Step2 | `/cpp-sink` | 用 C++下沉修改 API（性能更优） |
| Step3 | `/pytorch-alignment-validator` | 验证 API 对齐 |
| Step4 | `/api-docs-updater` | 更新中文文档 |
| Step5 | `/create-pr` | 提交 PR 到三个仓库 |

## 工作流程

```
输入：待对齐 API 列表
  ↓
Step1：方案决策（api-change-decider）
  ├─ 分析 API 差异
  └─ 决策改动方案（方案 1-6）
  ↓
Step2：代码修改（python-decorator 或 cpp-sink）
  ├─ 方案 1：Python 装饰器
  ├─ 方案 2：C++下沉
  └─ 其他方案：跳过
  ↓
Step3：对齐验证（pytorch-alignment-validator）
  ├─ 更新 API 映射
  ├─ 补充测试用例
  └─ 运行单元测试
  ↓
Step4：文档更新（api-docs-updater）
  └─ 更新中文 API 文档
  ↓
Step5：代码提交（create-pr）
  └─ 提交 PR 到 Paddle/PaConvert/docs
  ↓
输出：对齐结果统计表
```

## 使用示例

**完整对齐一个 API**（推荐方式）：
```
/api-compatibility torch.atan
```
自动完成全流程，得到对齐结果。

**调试单个 API**（需要逐步修改）：
```
/api-change-decider torch.atan  # 确定方案
/cpp-sink torch.atan   # 指定方案修改
/python-decorator torch.atan    # 指定方案修改
/pytorch-alignment-validator torch.atan  # 验证
/api-docs-updater torch.atan  # 更新文档
/create-pr torch.atan  # 提交 PR
```
