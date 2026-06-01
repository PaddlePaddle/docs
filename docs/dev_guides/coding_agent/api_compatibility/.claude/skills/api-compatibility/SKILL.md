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

### 2.1 工作目录说明

**ROOT_DIR 变量定义**：
- `${ROOT_DIR}` 表示项目的根工作目录（如 `/workspace`），通常为包含 `Paddle`、`PaConvert`、`docs` 子目录的父目录
- **路径规则**：含 `${ROOT_DIR}` 为绝对路径，不含则为相对于根目录的相对路径，需自行展开

| 工作目录 | 完整路径 | 内容说明 | 对应步骤 |
|---------|---------|---------|---------------|
| Paddle | `${ROOT_DIR}/Paddle` | Paddle 框架源码仓库，包含所有 Paddle API 的实现 | Step2：代码修改 + Step3：兼容测试 |
| PaConvert | `${ROOT_DIR}/PaConvert` | PyTorch 转换工具仓库，包含所有 Pytorch 单元测试 | Step4：对齐验证 |
| docs | `${ROOT_DIR}/docs` | Paddle 文档仓库，包含所有 Paddle API 中文文档 | Step5：文档更新 |


### 2.2 相关文件位置

|**功能模块**|**检索关键字**|**文件路径**|**举例**|**注意**|
|-|-|-|-|-|
|API 中文文档|`{api_name}_cn.rst`|`${ROOT_DIR}/docs/docs/api/paddle/`|tan_cn.rst||
|API 差异文档|`torch.{api_name}.md`|`${ROOT_DIR}/docs/docs/guides/model_convert/convert_from_pytorch/api_difference/`下一级目录|torch.tan.md||
|C++下沉使用|`python_api_info.yaml`、`ops.yaml`|`${ROOT_DIR}/Paddle/paddle/phi/ops/yaml/`|python_api_info.yaml、ops.yaml||
|C++下沉使用|`_paddle_docs.py`|`${ROOT_DIR}/Paddle/python/paddle/`|_paddle_docs.py||
|Paddle API 实现位置|`def {api_name}` 或 `class {api_name}`|`${ROOT_DIR}/Paddle/python/paddle/*/`|`${ROOT_DIR}/Paddle/python/paddle/tensor/math.py`|不要误检索到 sparse 目录下（稀疏 API 位置），本项目与稀疏无关，所有 sparse 相关文件直接忽略|
|Paddle API 兼容性单测位置|`test_api_compatibility_part[1-9]\.py`|`${ROOT_DIR}/Paddle/test/legacy_test/`|test_api_compatibility_part3.py||
|Pytorch API 单测位置|`test_{api_name}.py`|`${ROOT_DIR}/PaConvert/tests/`|test_tan.py||

### 2.3 Paddle API 架构（5 层调用栈）

Paddle API 从上到下由 5 层组成（本项目直接修改第 1、5 层，对于第 2~4 层通常是修改 yaml 配置文件，例如 python_api_info.yaml）：

| 层级 | 名称 | 语言 | 文件位置 | 功能说明 | 是否修改 |
|------|------|------|----------|----------|----------|
| 1 | Python 层 | Python | `*.py` | API 的 Python 接口定义 | ✅ **修改** |
| 2 | Pybind 层 | C++ | 根据`*.yaml`自动生成（`${ROOT_DIR}/Paddle/paddle/fluid/pybind/eager_op_function.cc`）| Python 与 C++的绑定层 | ✅ **修改 yaml 配置来实现修改** |
| 3 | Dygraph 层 | C++ | 根据`*.yaml`自动生成（`${ROOT_DIR}/Paddle/paddle/fluid/eager/.../dygraph_functions.cc`）| 前反向传播组合 | ❌ 通常不改 |
| 4 | C++ API 层 | C++ | 根据`*.yaml`自动生成（`${ROOT_DIR}/Paddle/paddle/phi/api/lib/api.cc`） | Kernel 选择调度 | ❌ 通常不改 |
| 5 | Kernel 层 | C++ | `${ROOT_DIR}/Paddle/paddle/phi/kernels/` | 实际计算逻辑实现 | ✅ **修改** |

**示例 API 层级**：
```python
# Layer 1: Python 层
def atan(x: Tensor, name: str | None = None)

# Layer 2: Pybind 层（根据 ops.yaml 自动生成）
eager_api_abs(PyObject *self, PyObject *args, PyObject *kwargs)

# Layer 3: Dygraph 层（根据 ops.yaml 自动生成）
paddle::Tensor atan_ad_func(const paddle::Tensor& x, ...)

# Layer 4: C++ API 层（根据 ops.yaml 自动生成）
Tensor atan(const Tensor& x, ...)

# Layer 5: Kernel 层
void AtanKernel(const Context& dev_ctx, const DenseTensor& x, DenseTensor* out)
```

### 2.4 专业术语表

| 术语 | 定义 | 备注 |
|------|------|------|
| PyTorch | 深度学习框架，导入模块为`torch`  | - |
| Paddle | 飞桨深度学习框架，导入模块为`paddle` | - |
| API | 应用程序接口 | 既可以是一个 Python 函数，也可以是一个 Python 类 |
| API 完整路径 | API 完整路径 | 如`torch.nn.functional.dropout`、`paddle.nn.functional.dropout`|
| API 路径 | API 完整路径在去掉框架导入模块(torch/paddle)后剩余的部分| 如`nn.functional.dropout` |
| PyTorch API | `torch.*` 系列接口 | 约 2000+个 API，是本项目的**对齐标准**|
| Paddle API | `paddle.*` 系列接口 | 约 2000+个 API，是本项目的**修改对象** |
| API 对齐 | 使两个 API 的行为完全对齐一致 | 对齐包括 API 路径、输入参数、返回值、计算逻辑等|
| API 中文文档 | 中文描述了该 API 的功能与行为 | 位于 `${ROOT_DIR}/docs/docs/api/paddle/` 目录，命名类似 tan_cn.rst  |
| API 差异文档 | 中文描述了 Pytorch API 与 Paddle API 两者的行为差异 | 位于 `${ROOT_DIR}/docs/docs/guides/model_convert/convert_from_pytorch/api_difference/` 下一级子目录，命名类似 torch.tan.md |
| compat 类型 API | 兼容性 API | 为保持后向兼容而添加的 API，能实现除 API 路径之外的完全对齐|

### 2.5 类方法 API 实现原理

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

### 2.6 API 信息获取方式

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

### 2.7 Inplace API 实现原理

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

# 三、整体工作流程

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
   - 当 Step3 或 Step4 无法通过时
   - 需要根据错误信息诊断问题根源：
     * 若判断为方案选择错误 → 回退到 Step1 重新决策
     * 若判断为代码实现有误 → 回退到 Step2 调整实现方式
   - 回退后需从该步骤重新按流程向前推进，例如回退到 Step2，则重新执行 Step2 → Step3 → Step4 → Step5
5. **允许放弃部分 API**（合理分配精力，最大化成功率）：
   - 当某个 API 异常回退 3 次以上仍无法通过，则放弃该 API，在最终对齐结果统计表中标记该 API 为"未对齐"，并简要说明放弃原因
   - ⚠️ 必须完整回退该 API 的所有修改，确保项目处于干净状态，不得保留任何 API"修改了但没改对"的中间状态
6. 所有 API 都完成 5 个步骤（除被放弃外）后，任务结束

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
    Step 3.1: 编写测试用例（仅首次执行）
    Step 3.2: 编译并运行（每次改动均需执行）

### Step 4：对齐验证（调用 `/pytorch-alignment-validator` skill）
    Step 4.1: 标记已完成的 API
    Step 4.2: 增加测试用例
    Step 4.3: 运行单元测试（每次改动均需执行）

### Step 5：文档更新（调用 `/api-docs-updater` skill）

## 工作示例

假设待对齐 API 为 `torch.argmax`：

```
1. Step1: 方案决策 → 得到『方案 2：C++下沉』
2. Step2: 代码修改 → 修改 Paddle 目录文件，将 paddle.argmax 下沉到 C++
3. Step3: 兼容测试 → 在 Paddle 目录添加兼容性单测，编译并运行验证
4. Step4: 对齐验证 → 修改 PaConvert 目录文件，编写 Pytorch 单元测试，对比测试，验证对齐
5. Step5: 文档更新 → 修改 docs 目录文件，更新 paddle.argmax 文档
```

# 四、编程风格指南

- 最小化注释；保持简洁；代码应当自解释、自文档化。
- 注释应有实际价值，例如提醒读者一些非显而易见、无法从局部推断的全局背景。
- 不要为只使用一次的简短逻辑（1-2 行）创建辅助函数，除非能显著提升代码可读性。
- 优先使用清晰的抽象，状态管理应当显式。例如，在 Python 类中管理状态时，应有明确的类定义并列出所有成员，不要在对象上动态 setattr 字段后再动态 getattr。
- 与现有代码风格和架构模式保持一致。
- 假定读者熟悉 Paddle。他们不一定是所读代码的专家，但应具备该领域的一定经验。
- 新增代码注释仅使用 ASCII 字符，不引入 Unicode 字符（如弯引号、破折号、箭头、非 ASCII 字母）。对未改动注释中已有的 Unicode 保持原样，仅对新增或改写的注释执行此规则。
- 如有不确定，选择更简单、更简洁的实现。

# 五、各 Skill 说明

## 5.1 总控 Skill：api-compatibility（本文件）

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

## 5.2 项目专用 Skill 列表

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

# 六、自进化机制

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

# 七、注意事项

1. 严格按标准工作流程执行，杜绝自行臆断和跳过步骤

# 八、常见问题处理

### Q1：为什么有些 API 对齐失败？

**常见原因**：

1. **差异分析阶段失败**：未查询到差异文档或转写配置
2. **方案决策阶段失败**：选择的方案不适用或后向兼容性问题
3. **代码修改阶段失败**：装饰器实现问题或 C++ 下沉编译错误
4. **验证阶段失败**：Paddle API 实现与 PyTorch 计算结果不一致

**处理办法**：
- 查看具体错误信息，返回相应 Step 重新处理
- 使用调试技巧逐步排查问题
