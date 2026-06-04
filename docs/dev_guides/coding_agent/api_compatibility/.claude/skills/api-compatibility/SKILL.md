---
name: api-compatibility
description: 开展《Paddle API 对齐 PyTorch 项目》，负责项目整体统筹规划，调用多个 skill，完成输入的 API 对齐
allowed-tools: Read Write Edit Bash Glob Grep Agent Skill WebFetch WebSearch
---

# 一、项目目标

用户提供待对齐的 Pytorch API 列表 $ARGUMENTS，通过调用多个 skill，使 Paddle API 与 PyTorch API 完全对齐，实现：
- 对于任意 PyTorch API 用法，只需将 `torch.*` 替换为 `paddle.*`
- 计算结果完全一致（数值精度、行为逻辑）

# 二、背景知识
## 类方法 API 实现原理

**概念**：类方法 API（如 `torch.Tensor.abs`）和普通 API（如 `torch.abs`）是不同 API，但实现一致，合并处理即可。Paddle 通过 patch 机制将方法动态添加到 Tensor 类上。

**实现方式选择**：

| 场景 | 方式 | 操作 |
|------|------|------|
| 数学运算类方法，直接转发到普通函数 | 方式一：配置 tensor_method_func | 见下方步骤 |
| 自定义实现逻辑、property、魔术方法 | 方式二：修改 patch 文件 | 见下方步骤 |

**方式一：配置 tensor_method_func**

文件：`Paddle/python/paddle/tensor/__init__.py`

```python
# 1. import 所需函数
from .math import trace

# 2. 加入 tensor_method_func 列表（按字母表顺序插入）
tensor_method_func = [
    ...
    'trace',
    ...
]
```

完成。各 patch 文件会自动遍历此列表完成绑定，无需额外修改。

**方式二：修改 patch 文件**

需修改的文件：

| 文件 | 语言 | 适用 |
|------|------|------|
| `python/paddle/base/dygraph/math_op_patch.py` | Python | 动态图 |
| `python/paddle/pir/math_op_patch.py` | Python | PIR 静态图 |
| `python/paddle/base/layers/math_op_patch.py` | Python | 老静态图 |
| `paddle/fluid/pybind/eager_math_op_patch.cc` | C++ | 动态图（性能更优） |

选择规则：
- 动静统一 API：三处 Python patch 文件都改
- 动态图专用 API：仅改动态图 patch 文件
- 追求性能：可用 C++ patch 文件替代 Python

修改方式：
- Python：在 `eager_methods` 列表中添加元组 `('方法名', 方法实现)`
- C++：参考 `eager_math_op_patch.cc` 中已有方法实现

**查找注意事项**：
- ✅ 在 patch 文件中搜索，或搜索对应的普通方法 `def abs(`
- ❌ 不要搜索 `class Tensor`（方法通过 setattr 动态添加，不在类定义中）


## Inplace API 实现原理

**概念说明**：
- 注意要区分**inplace API**和**非 inplace API**，两者是不同的 API，不要混为一谈
- **inplace API**（如`torch.abs_`）：原地操作，直接修改输入 Tensor，其不应有 out 参数，如有 out 需删除
- **非 inplace API**（如`torch.abs`）：返回新 Tensor，不修改输入 Tensor
- Inplace API 无需测试静态图，只需测试动态图

**示例对比**：
```python
y = paddle.abs(x)  # 非 inplace：返回新 Tensor，x 不变
x.abs_()           # inplace：原地修改 x
```

**自动生成机制**：
Paddle 支持自动生成 inplace API，无需在`ops.yaml`中单独配置。当定义了`inplace: (x -> out)`字段后，系统自动生成对应的 inplace 版本，复用原 API 的 Kernel 实现。

**配置示例**：
1. **OP 配置**（`ops.yaml`，第 10-22 行）：
```yaml
- op : abs
  args : (Tensor x)
  output : Tensor(out)
  inplace: (x -> out)  # 关键字段：指定 x 和 out 可以 in-place
  backward : abs_grad
```

2. **Python API 配置**（`python_api_info.yaml`，第 6-9 行）：
- ⚠️ **仅在 C++下沉（方案 2）时需要配置**
```yaml
- op : abs_
  name : [paddle.abs_, paddle.Tensor.abs_]
  args_alias :
    use_default_mapping : True
```

## API 信息获取方式

在开展 API 对齐工作过程中，需要获取 PyTorch API 和 Paddle API 的相关信息。

**参考资源**：

| 资源类型 | 资源位置 | 说明 |
|---------|---------|------|
| PyTorch 官方文档 | https://pytorch.org/docs/stable/ | 了解 API 参数功能定义 |
| PyTorch 源码 | https://github.com/pytorch/pytorch | 参考 PyTorch 底层逻辑（需注意 PyTorch 与 Paddle 架构设计存在差异） |
| Paddle 官方文档 | https://www.paddlepaddle.org.cn/documentation/docs/zh/api/ | 了解 Paddle API 参数功能定义 |
| Paddle 源码 | 见本地仓库 `${ROOT_DIR}/Paddle` | 参考 Paddle 底层逻辑 |

**获取方式**：
1. **查阅官方文档**：优先查阅 PyTorch 和 Paddle 的官方文档，了解 API 的参数定义、功能说明、使用示例等
2. **查阅源码实现**：当文档信息不够详细时，可查阅源码实现
3. **获取 API 签名**：通过 Python 的 `inspect` 模块或 `help()` 函数获取 API 签名
4. **获取文档字符串**：通过 `__doc__` 属性获取 API 的文档字符串
5. **实际测试验证**：编写测试代码，实际运行 API 验证其行为和参数用法

**示例**：
```python
import paddle
import torch

# 获取 API 签名
import inspect
print(inspect.signature(paddle.abs))
print(inspect.signature(torch.abs))

# 获取文档字符串
print(paddle.abs.__doc__)
print(torch.abs.__doc__)

# 实际测试验证
x = paddle.to_tensor([1.0, -2.0, 3.0])
print(paddle.abs(x))
```


# 三、整体工作流程
## 流程概览
```
输入 API 列表 → Step1:所有 API 方案决策 → Step2:所有 API 代码修改 → Step3:所有 API 兼容测试 → Step4:所有 API 对齐验证 → Step5:所有 API 文档更新 → 全部完成（流程全自动推进，不用询问）
```

具体如下：
### Step 1：方案决策（调用 `/api-change-decider` skill）
    Step 1.1: 获取差异信息
    Step 1.2: 提取差异信息
    Step 1.3: 方案决策

### Step 2：代码修改
根据 Step1 的方案决策结果，**按方案分组**，依次调用对应 skill，同一方案的所有 API 在一个 skill 调用中批量处理。

**各方案步骤**：
#### 方案 1：Python 装饰器（调用 `/python-decorator` skill）
    Step 2.1: 差异分析与选择装饰器
    Step 2.2: 应用或开发装饰器
    Step 2.3: 更新函数文档
#### 方案 2：C++下沉（调用 `/cpp-sink` skill）
    Step 2.1: 配置 python_api_info.yaml
    Step 2.2: 迁移文档到_paddle_docs.py
    Step 2.3: 替换 Python 实现
#### 方案 3：修改 API（调用 `/modify-origin-api` skill）
    Step 2.1: 修改 API 签名
    Step 2.2: 修改函数实现逻辑
    Step 2.3: 更新函数文档
#### 方案 4：新增 API（调用 `/add-new-api` skill）
#### 方案 5：新增 compat 类型 API（调用 `/add-new-compat-api` skill）

### Step 3：兼容测试（调用 `/add-compatibility-test` skill）
    Step 3.1: 编写测试用例
    Step 3.2: 编译并运行单测（每次修改代码均需执行编译）

### Step 4：对齐验证（调用 `/pytorch-alignment-validator` skill）
    Step 4.1: 标记已完成的 API
    Step 4.2: 增加测试用例
    Step 4.3: 编译并运行单测（每次修改代码均需执行编译）

### Step 5：文档更新（调用 `/api-docs-updater` skill）

## 流程重要约束
1. 接收用户输入的待对齐 API 列表（如 `torch.argmax`, `torch.log2`, `torch.logsumexp`）
2. **批量处理模式**：依次执行，每个 Step 结束后才进入下一个 Step
   - Step1：对**所有 API**进行方案决策
   - Step2：对**所有 API**进行代码修改
   - Step3：对**所有 API**进行兼容测试
   - Step4：对**所有 API**进行对齐验证
   - Step5：对**所有 API**进行文档更新
3. **流程正向推进原则**
   - 正常情况下必须遵循 Step1 → Step2 → Step3 → Step4 → Step5 的顺序
   - 每个步骤完成后，才能进入下一步骤，禁止跳过任何步骤
4. **异常回退原则**
   - 当 Step3 或 Step4 无法通过时，则需要执行回退（Step3 的通过标准是单测可运行通过，Step4 的通过标准是 API 可配置为 ChangePrefixMatcher）：
     * 若判断为方案选择错误 → 回退到 Step1 重新决策
     * 若判断为代码实现有误 → 回退到 Step2 调整实现方式
   - 回退后需从该步骤重新按流程向前推进，例如回退到 Step2，则重新执行 Step2 → Step3 → Step4 → Step5
5. **允许放弃部分 API**（合理分配精力，最大化成功率）：
   - 当某个 API 异常回退 3 次以上仍无法通过，则放弃该 API，在最终对齐结果统计表中标记该 API 为"未对齐"，并简要说明放弃原因
   - 必须完整回退该 API 的所有修改，确保项目处于干净状态，不得保留任何 API"修改了但没改对"的中间状态
6. 所有 API 都完成 5 个步骤（除被放弃外）后，任务结束

## 工作示例

假设待对齐 API 为 `torch.argmax`：

```
1. Step1: 方案决策 → 得到『方案 2：C++下沉』
2. Step2: 代码修改 → 修改 Paddle 目录文件，将 paddle.argmax 下沉到 C++
3. Step3: 兼容测试 → 在 Paddle 目录添加兼容性单测，编译并运行验证
4. Step4: 对齐验证 → 修改 PaConvert 目录文件，编写 Pytorch 单元测试，对比测试，验证对齐
5. Step5: 文档更新 → 修改 docs 目录文件，更新 paddle.argmax 文档
```

# 四、各 Skill 说明

## 总控 Skill：api-compatibility（本文件）

**功能定位**：
- 本文件（`api-compatibility`）是项目的**总控 skill**，负责整体统筹规划
- 作为用户入口，接收 API 列表输入，协调各子 skill 完成对齐工作
- 遵循「流程正向推进」原则，按 Step1 → Step2 → Step3 → Step4 → Step5 顺序执行

**核心职责**：
1. 解析用户输入的 API 列表
2. 依次调用各步骤对应的子 skill
3. 汇总各步骤执行结果，输出最终对齐统计表

**Skill 调用规范**：
- 本项目专用 skill 为下表所列的子 skill
- 优先使用本项目专用 skill，确保流程一致性和可控性，除非其无法完成任务，否则**尽量不调用其他 skill**

## 项目专用 子 Skill 列表

| Skill | 对应步骤 |
|-------|--------|
| `/api-change-decider` | Step1：方案决策 |
| `/python-decorator` | Step2：方案 1 Python 装饰器 |
| `/cpp-sink` | Step2：方案 2 C++下沉 |
| `/modify-origin-api` | Step2：方案 3 修改原有 API |
| `/add-new-api` | Step2：方案 4 新增 API |
| `/add-new-compat-api` | Step2：方案 5 新增 compat API |
| `/add-compatibility-test` | Step3：兼容测试 |
| `/pytorch-alignment-validator` | Step4：对齐验证 |
| `/api-docs-updater` | Step5：文档更新 |

# 五、自进化机制

**本项目涉及的所有 skill 均具备自进化能力**，通过持续学习和优化来提升工作质量。修改的 SKILL 目录为：`${ROOT_DIR}/docs/docs/dev_guides/coding_agent/api_compatibility/.claude/skills`

**如何自进化**：
- 每次交互结束后，自动复盘分析工作过程中的问题、错误和成功经验
- 结合用户反馈纠偏和运行日志，不断优化描述细节和边界条件
- 识别重复出现的问题模式和最佳实践

**自进化需要修改哪里**：
1. **注意事项**章节：补充新发现的注意事项、工作要求
2. **常见问题处理**章节：补充新发现的问题-解决方案、特殊情况处理方案
3. 修正或补充 SKILL 中新发现的错误或遗漏内容

**修改注意**：
- 所有优化和改进都写入各 skill 的 SKILL.md 文档，确保知识持久化
- 禁止在 SKILL 中添加任何网络代理的内容，以免安全信息泄露
- 禁止在代码中编写任何网络代理的内容，以免安全信息泄露

# 六、常见问题处理

### Q1：API 别名如何管理？

统一放在 `__init__.py` 的 "API alias" 区块，不要在 `math.py` 等实现文件中定义。

**涉及文件**：
- `paddle/tensor/__init__.py`
- `paddle/__init__.py`

**示例**（正确做法）：
```python
# 在 __init__.py 的 "API alias" 区块
fix = trunc
fix_ = trunc_
mod = remainder
```

**注意**：定义别名后，需从 `from .math import` 中移除该别名，避免导入错误。

### Q2：Inplace API 如何实现？

内部所有操作都必须使用 inplace 方法（`scale_`、`add_` 等），不能用 `*`、`+` 等非 inplace 操作。

**错误**：
```python
mv_result = mv_result * alpha  # 创建新 tensor
```

**正确**：
```python
mv_result.scale_(alpha)  # inplace 修改
```
