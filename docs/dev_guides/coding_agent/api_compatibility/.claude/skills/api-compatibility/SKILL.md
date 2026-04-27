---
name: api-compatibility
description: 开展《Paddle API 对齐 PyTorch 项目》，负责项目整体统筹规划，调用多个 skill，完成输入的 API 对齐
disable-model-invocation: true
---

# 一、项目目标

**使 Paddle API 与 PyTorch API 完全对齐**，实现：

- 对于任意 PyTorch API 用法，只需将 `torch.*` 替换为 `paddle.*`
- 计算结果完全一致（数值精度、行为逻辑）

**流程概览**：
1. 接收待对齐的 Pytorch API 列表：$ARGUMENTS
2. 为对应的 Paddle API 决策改动方案
3. 执行具体的 Paddle API 修改
4. 验证修改后的 Paddle API 是否对齐 Pytorch
5. 更新修改后的 Paddle API 中文文档
6. 提交代码到相应的仓库
7. 给出 Pytorch API 列表的对齐结果统计

# 二、输入输出规范

## 2.1 输入
用户提供待对齐的 Pytorch API 列表，示例：
```markdown
torch.argmax
torch.log2
torch.logsumexp
```

## 2.2 输出

用户输入列表的对齐情况，格式示例：
```markdown
# API 对齐结果统计

|API 名称|对齐状态|改动方案|备注|
|-|-|-|-|
|torch.argmax|已对齐|方案 2|-|
|torch.log2|已对齐|无需改动|<简介为何无需改动>例如：未查询到差异文件，两者 API 完全一致|
|torch.logsumexp|未对齐|方案 1|<简介失败原因>例如：方案 1 暂不支持修改|
```

# 三、技术背景知识

### 3.1 根目录说明

| 目录名称 | 内容说明 | 对应步骤 |
|---------|---------|---------------|
| Paddle | 包含所有 Paddle API 的实现 | Step2：代码修改 |
| PaConvert | 包含所有 Pytorch 单元测试 | Step3：对齐验证 |
| docs | 包含所有 Paddle API 中文文档 | Step4：文档更新 |
| Paddle、PaConvert、docs | 代码提交对象 | Step5：代码提交 |

### 3.2 相关文件位置

|**功能模块**|**检索关键字**|**文件路径**|**举例**|**注意**|
|-|-|-|-|-|
|API 中文文档|`{api_name}_cn.rst`|docs/docs/api/paddle/|tan_cn.rst||
|API 差异文档|`torch.{api_name}.md`|docs/docs/guides/model_convert/convert_from_pytorch/api_difference/|torch.tan.md||
|C++下沉使用|`python_api_info.yaml`、`ops.yaml`|Paddle/paddle/phi/ops/yaml/|python_api_info.yamlops.yaml||
|C++下沉使用|`_paddle_docs.py`|Paddle/python/paddle/|_paddle_docs.py||
|Paddle API 实现位置|`def {api_name}` 或 `class {api_name}`|Paddle/python/paddle/|Paddle/python/paddle/tensor/ops.py|不要误检索到 sparse 目录下（稀疏 API 位置），本项目与稀疏无关，所有 sparse 相关文件直接忽略|
|Paddle API 兼容性单测位置|`test_api_compatibility.py`|Paddle/test/legacy_test/test_api_compatibility.py||
|Pytorch API 单测位置|`test_{api_name}.py`|PaConvert/tests/|PaConvert/tests/test_tan.py||

### 3.3 Paddle API 架构（5 层调用栈）

Paddle API 从上到下由 5 层组成（本项目直接修改第 1、5 层，对于第 2~4 层通常是修改 yaml 配置文件，例如 python_api_info.yaml）：

| 层级 | 名称 | 语言 | 文件位置 | 功能说明 | 是否修改 |
|------|------|------|----------|----------|----------|
| 1 | Python 层 | Python | `*.py` | API 的 Python 接口定义 | ✅ **修改** |
| 2 | Pybind 层 | C++ | 根据`*.yaml`自动生成（`paddle/fluid/pybind/eager_op_function.cc`）| Python 与 C++的绑定层 | ✅ **修改 yaml 配置来实现修改** |
| 3 | Dygraph 层 | C++ | 根据`*.yaml`自动生成（`paddle/fluid/eager/.../dygraph_functions.cc`）| 前反向传播组合 | ❌ 通常不改 |
| 4 | C++ API 层 | C++ | 根据`*.yaml`自动生成（`paddle/phi/api/lib/api.cc`） | Kernel 选择调度 | ❌ 通常不改 |
| 5 | Kernel 层 | C++ | `paddle/phi/kernels/` | 实际计算逻辑实现 | ✅ **修改** |

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

### 3.4 专业术语表

| 术语 | 定义 | 备注 |
|------|------|------|
| PyTorch | 深度学习框架，导入模块为`torch`  | - |
| Paddle | 飞桨深度学习框架，导入模块为`paddle` | - |
| API | 应用程序接口 | 既可以是一个 Python 函数，也可以是一个 Python 类 |
| API 完整路径 | API 完整路径 | 如`torch.nn.functional.dropout`、`paddle.nn.functional.dropout`|
| API 相对引用路径 | API 完整路径在去掉框架导入模块(torch/paddle)后剩余的部分| 如`nn.functional.dropout` |
| PyTorch API | `torch.*` 系列接口 | 约 2000+个 API，是本项目的**对齐标准**|
| Paddle API | `paddle.*` 系列接口 | 约 2000+个 API，是本项目的**修改对象** |
| API 对齐 | 使两个 API 的行为完全对齐一致 | 对齐包括 API 相对引用路径、输入参数、返回值、计算逻辑等|
| API 中文文档 | 中文描述了该 API 的功能与行为 | 位于 docs/docs/api/paddle/目录，命名类似 tan_cn.rst  |
| API 差异文档 | 中文描述了 Pytorch API 与 Paddle API 两者的行为差异 | 位于 docs/docs/guides/model_convert/convert_from_pytorch/api_difference/下的一级子目录，命名类似 torch.tan.md |
| compat 类型 API | 兼容性 API | 为保持后向兼容而添加的 API，能实现除 API 相对引用路径之外的完全对齐，实现之后差异分类将成为『仅 API 调用方式不一致』|

### 3.5 类方法 API 实现原理

**概念说明**：
- 注意要区分**类方法 API**和**普通 API**，两者是不同的 API，不要混为一谈
- **类方法 API**（如`torch.Tensor.abs`）：`torch.Tensor`类方法
- **普通 API**（如`torch.abs`）：普通方法

Paddle 的 Tensor 类方法通过**patch 机制**实现，即将普通方法动态添加到`paddle.Tensor`（即`core.eager.Tensor`）类上成为类方法。

**实现机制**（参考`Paddle/python/paddle/base/dygraph/math_op_patch.py`）：

```python
# 从 paddle.tensor 模块获取方法定义
import paddle.tensor

# 将普通方法 patch 到 core.eager.Tensor 类上
for method_name in paddle.tensor.tensor_method_func:
    if hasattr(core.eager.Tensor, method_name):
        continue
    method_impl = getattr(paddle.tensor, method_name, None)
    if method_impl:
        setattr(core.eager.Tensor, method_name, method_impl)
```

**查找类方法实现时的注意事项**：
- ✅ **正确做法**：直接搜索对应的普通方法实现，如搜索`def abs(`或`def atan(`
- ❌ **错误做法**：不要搜索`class Tensor`或在 Tensor 类定义中查找方法
- 原因：Tensor 类方法是通过`setattr`动态添加的，不在类定义的源码中直接体现

### 3.6 Inplace API 实现原理

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

# 四、标准工作流程

## 4.1 流程概览

```
输入 API 列表 → Step1:所有 API 方案决策 → Step2:所有 API 代码修改 → Step3:所有 API 对齐验证 → Step4:所有 API 文档更新 → Step5:代码提交 → 全部完成（流程全自动推进，不用询问）
```

具体如下：
### Step 1：方案决策（调用 `/api-change-decider` skill）
    Step 1.1: 分析差异文档
    Step 1.2: 方案决策

### Step 2：代码修改
#### 方案 1：Python 装饰器（调用 `/python-decorator` skill）
    Step 2.1: 差异分析与选择装饰器
    Step 2.2: 应用或开发装饰器
    Step 2.3: 添加 out 参数支持
    Step 2.4: 更新函数文档字符串
    Step 2.5: 添加测试用例
    Step 2.6: 编译并运行
#### 方案 2：C++下沉（调用 `/cpp-sink` skill）
    Step 2.1: 配置 python_api_info.yaml
    Step 2.2: 迁移文档到_paddle_docs.py
    Step 2.3: 替换 Python 实现
    Step 2.4: 添加测试用例
    Step 2.5: 编译并运行

### Step 3：对齐验证（调用 `/pytorch-alignment-validator` skill）
    Step 3.1: 标记已对齐的 API
    Step 3.2: 补充测试用例
    Step 3.3: 运行单元测试

### Step 4：文档更新（调用 `/api-docs-updater` skill）
    Step 4.1: 获取代码变更信息
    Step 4.2: 更新 API 中文文档

### Step 5：代码提交（调用 `/create-pr` skill）

**执行逻辑**：
1. 接收用户输入的待对齐 API 列表（如 `torch.argmax`, `torch.log2`, `torch.logsumexp`）
2. **批量处理模式**：按 Step 顺序依次执行，每个 Step 处理完所有 API 后才进入下一个 Step
   - Step1：对**所有 API**进行方案决策，记录每个 API 的方案类型
   - Step2：对**所有 API**进行代码修改
   - Step3：对**所有 API**进行对齐验证
   - Step4：对**所有 API**更新文档
   - Step5：对**所有 API**进行代码提交
3. **流程推进的豁免与放弃条件**：
   - **豁免条件**：
     * 每一个执行步骤均需调用相应的 skill 来执行
     * 当前仅支持方案 1/2，若决策为其他方案，则该 API 跳过，只需记录决策结果即可，不要自行处理
   - **放弃策略**（合理分配精力，最大化成功率）：
     * **放弃判断标准**：
       - 当某个 API 在 Step2 或 Step3 经过多次尝试（建议 3 次）仍无法通过验证
       - 经分析判断短期内难以解决，继续投入时间成本过高
     * **放弃执行要求（必须严格遵守）**：
       - ⚠️ 必须完整回退该 API 在 Step2 和 Step3 中的所有代码修改
       - ⚠️ 确保项目处于干净状态，不得保留任何"修改了但没改对"的中间状态
       - 在最终的对齐结果统计表中标记该 API 为"未对齐"，并简要说明放弃原因
     * **整体策略原则**：
       - 目标是最大化 API 列表的整体对齐成功率，而非执着于单个 API
       - 优先处理更可能成功的 API，避免在困难 API 上消耗过多时间
       - 放弃是为了提高整体效率的理性决策，不是逃避问题
4. 所有 API 都完成 5 个步骤（除被豁免或放弃外）后，任务结束


## 4.2 详细步骤

### Step 1: 方案决策 ⚙️

**目标**：确定每个 API 的改动方案

**执行步骤**：
1. 输入：需要对齐的 PyTorch API 列表（如 `torch.atan`、`torch.asinh`）
2. 调用 `/api-change-decider` skill
3. 输出：方案类型、对应 Paddle API、差异分类、决策依据

**方案类型**：
- 无需改动
- 方案 1：Python 装饰器
- 方案 2：C++下沉
- 方案 3：修改 API
- 方案 4：新增 API
- 方案 5：新增 compat 类型 API

### Step 2: 代码修改 💻

**目标**：根据方案修改 Paddle API 代码

**执行步骤**：
1. 输入：方案类型、对应 Paddle API（如 `paddle.atan`、`paddle.asinh`）、差异分类、决策依据
2. 根据方案类型，调用对应的子智能体，每个子智能体批量处理其所负责的 API：
   - 方案 1 → 调用 `/python-decorator` skill
   - 方案 2 → 调用 `/cpp-sink` skill
   - 方案 3 → `/python-decorator`和`/cpp-sink`skill 支持新增 out 参数，其他修改无对应 skill 支持，豁免
   - 方案 4 → 无对应 skill 支持，豁免
   - 方案 5 → 无对应 skill 支持，豁免
3. 输出：是否代码修改无误（即单测运行通过）

**异常处理**：
- 本步骤多次调试仍异常时，主控智能体根据报错信息评估，是否需要回退到前序步骤：
  - 是否 Step1 中有 API 的方案决策错误？


### Step 3: 对齐验证 ✅ **（金标准）**

**目标**：验证修改后的 Paddle API 能与 PyTorch API 完全对齐

**执行步骤**：
1. 输入：PyTorch API 列表（如 `torch.atan`、`torch.asinh`）
2. 调用 `/pytorch-alignment-validator` skill
3. 输出：是否通过对齐验证（即单测运行通过）

**异常处理**：
- 本步骤多次调试仍异常时，主控智能体根据报错信息评估，是否需要回退到前序步骤：
  - 是否 Step1 中有 API 的方案决策错误？
  - 是否 Step2 中有 API 的代码实现有误？

### Step 4: 文档更新 📝

**目标**：更新 Paddle API 中文文档

**执行步骤**：
1. 调用 `/api-docs-updater` skill

**异常处理**：
- 文档更新必须与 Step2 中的代码修改保持同步
- 确保文档准确反映代码的最新行为

### Step 5: 代码提交 📤

**目标**：提交修改后的代码到 Paddle、PaConvert、Docs 仓库

**执行步骤**：
1. 调用 `/create-pr` skill
2. 在对应的三个仓库分别创建或更新 Pull Request
3. 包含代码修改、测试用例和文档更新

## 4.3 重要约束 ⚠️

1. **流程正向推进原则**
   - 正常情况下必须遵循 Step1 → Step2 → Step3 → Step4 → Step5 的顺序
   - 每个步骤完成并验证通过后，才能进入下一步骤
   - 禁止跳过任何步骤（特别是 Step3 对齐验证步骤和 Step5 代码提交步骤）

2. **异常回溯调整原则**
   - 当 Step2(代码修改)或 Step3(对齐验证)多次尝试仍无法通过时
   - 主智能体需要根据错误信息诊断问题根源：
     * 若判断为方案选择错误 → 返回 Step1 重新决策
     * 若判断为代码实现有误 → 在 Step2 调整实现方式
   - 回溯后需从该步骤重新按流程向前推进，例如回溯到 Step2，则重新 Step2 → Step3 → Step4 → Step5

3. **完整性保障**
   - 每个 API 的 5 个步骤都必须完整执行（除被豁免或放弃外）
   - 即使需要回溯调整，最终也要确保走完全流程
   - 所有步骤的产出物（代码、测试、文档、PR）必须齐全


## 4.4 工作示例

假设待对齐 API 为 `torch.argmax`：

```
1. Step1: 方案决策 → 得到『方案 2：C++下沉』
2. Step2: 代码修改 → 修改 Paddle 目录，将 paddle.argmax 下沉到 C++
3. Step3: 对齐验证 → 修改 PaConvert 目录，编写 Pytorch 单元测试，对比测试，验证对齐
4. Step4: 文档更新 → 修改 docs 目录，更新 paddle.argmax 文档
5. Step5: 代码提交 → 调用 /create-pr skill，在 Paddle、PaConvert、Docs 三个仓库分别创建或更新 PR
```


# 五、代码修改规范

## 5.1 代码注释规范
- ✅ 改动位置如果方便注释，可以注释`# Edit by AI Agent`，但只能注释 1 次

## 5.2 代码质量规范
- ✅ 保持代码风格与项目一致
- ✅ 不破坏现有功能
- ✅ 确保向后兼容性

## 5.3 测试规范
- ✅ 修改后必须通过原有测试
- ✅ 必须添加新的单元测试
- ✅ 必须通过 PyTorch 对齐验证

## 5.4 文档规范

- ✅ 同步更新中文文档
- ✅ 文档格式符合 Paddle 规范
- ✅ 准确描述 API 功能和参数

# 六、注意事项

1. 复盘记忆中的历史易错点，避免重复犯错
2. 严格按标准工作流程执行，杜绝自行臆断和跳过步骤
