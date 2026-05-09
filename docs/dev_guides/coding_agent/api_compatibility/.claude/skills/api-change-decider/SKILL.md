---
name: api-change-decider
description: 负责《Paddle API 对齐 PyTorch 项目》中 Step1：方案决策，分析 PyTorch API 与 Paddle API 之间的差异，制定合适的 API 改动方案
disable-model-invocation: false
---

# 一、输入输出规范

## 1.1 输入
需要对齐的 PyTorch API 列表（如 `torch.atan`、`torch.asinh`）

## 1.2 输出
以表格形式展示，列说明如下：

| Pytorch API | 方案类型 | Paddle API | 差异分类 | 决策依据 |
|-|-|-|-|-|
| 需对齐的 PyTorch API | 从方案 1~6 中选择 | 需改动的 Paddle API 完整路径 | 差异分类 | 总结差异分析过程和选择理由 |

以下为示例表格：

| Pytorch API | 方案类型 | Paddle API | 差异分类 | 决策依据 |
|-|-|-|-|-|
| torch.atan | 方案 2 | paddle.atan | torch 参数更多 | 仅参数名不同(input→x)+仅多 out 参数，Python 实现仅有一次`_C_ops.atan(x)`调用，满足 C++下沉条件，性能最优 |
| torch.select_scatter | 方案 1 | paddle.select_scatter | 仅参数名不一致 | input→x, src→values, dim→axis 参数名不同。底层调用 `_C_ops.set_value_with_tensor`（非 `_C_ops.select_scatter`），不满足方案 2 条件 1，降级为方案 1（Python 装饰器） |
| torch.Tensor.select_scatter | 方案 1 | paddle.Tensor.select_scatter | 仅参数名不一致 | 与 paddle.select_scatter 共享实现（patch 机制），随主 API 一并修改 |
| torch.logspace | 方案 3 + 方案 1 | paddle.logspace | 参数名不一致 + torch 参数更多 | 全文有两类差异：①end→stop/steps→num 参数名不同；②torch 多 out/device/requires_grad 参数。方案 3 末尾新增 out=None/device=None/requires_grad=False（后向兼容）；方案 1 装饰器添加 end/stop、steps/num 别名 |


# 二、候选方案

## 方案 1：Python 装饰器
**适用场景**：
- 支持多种参数重载情况：参数名不同、参数顺序不同、参数个数不同、参数类型不同
- 重载条件：能通过输入参数特征（名称、类型、个数）区分两套签名

**工作原理**：
根据输入参数的名称、类型、个数判断是 PyTorch 签名还是 Paddle 签名，针对两套签名分别适配功能，既保留原有 Paddle 功能也实现了 PyTorch 对齐

**核心限制**：
- **重载条件**：能通过输入参数特征（名称、类型、个数）区分两套签名
- **无法重载则不适用**：如果无法通过输入参数特征区分两套签名，则不适用

**优点**：灵活性强，兼容性好
**缺点**：性能低于 C++ 下沉实现

## 方案 2：C++ 下沉
**适用场景**：
- 仅涉及参数名不同或仅多 out 参数的情况

**适用条件**（必须全部满足）：
1. API 名称和 `_C_ops.xxx` 的 OP 名称一致
  - ✅ 正确示例：`paddle.atan` 调用 `_C_ops.atan`
  - ❌ 错误示例：`paddle.select_scatter` 调用 `_C_ops.set_value_with_tensor`（名称不一致）
2. API 差异**仅为参数名不同或仅多 out 参数**（不涉及参数顺序或个数差异；"仅多 out 参数"的精确含义见 2.1 关键概念）
3. Python 实现中**仅有一次** `_C_ops.xxx` 调用
4. Python 实现中不涉及其他 Paddle API 调用
  - ❌ **禁止**：调用任何其他 paddle API（如 `paddle.where`、`paddle.abs`、`paddle.flatten` 等）
  - ❌ **禁止**：调用 Tensor 方法（如 `x.flatten()`、`x.reshape()`、`x.unsqueeze()` 等，这些等同于调用 paddle API）
  - ❌ **禁止**：使用 `paddle.full_like`、`paddle.zeros_like`、`paddle.cast` 等
  - ✅ **允许**：Python 内置函数（如 `len()`、`isinstance()`、`list()`、`range()` 等）
  - ✅ **允许**：简单的属性访问（如 `x.shape`、`x.dtype`、`x.ndim` 等）
  - ✅ **允许**：简单的算术运算（如 `index + 1`、`axis < 0` 等）
5. `_C_ops.xxx` 前面**无复杂前处理逻辑**，前处理逻辑（如存在）容易改写为 C++
  - ❌ **复杂前处理**（不满足条件）：
    - 涉及多个 paddle API 调用的逻辑（如 `fill_constant`、`paddle.where`、`paddle.cast` 等）
    - 复杂的 shape 处理逻辑（如 `reshape`、`transpose`、`flatten` 等）
    - 条件分支中的 paddle API 调用
    - 循环中的 paddle API 调用
  - ✅ **简单前处理**（满足条件）：
    - 仅做参数校验（如 `isinstance()`、类型检查、范围检查）
    - 仅做参数类型转换（如 `convert_np_dtype_to_dtype_`）
    - 仅做简单参数处理（如负索引转正索引、默认值设置）
    - 仅做属性获取（如 `x.shape`、`x.dtype`）

**注意**：
- 读取 Python 实现逻辑时，忽略静态图部分（LayerHelper 分支代码不再维护）

**优点**：性能最优
**缺点**：限制条件很多，需严格满足

## 方案 3：修改 API
**适用场景**：
- API 相对引用路径一致，但存在可通过修改原有实现来对齐的差异
- 包括以下情况：
  1. **新增参数**：向现有 API 添加新参数（如添加 out 参数）
  2. **参数默认值调整**：修改参数默认值以对齐 PyTorch
  3. **扩展参数功能**：对已有参数扩展新功能同时保留原有功能
  4. **参数用法调整**：修改参数的处理逻辑以支持不同用法

**适用条件**（必须全部满足）**：
1. API 相对引用路径一致
2. 修改必须保持后向兼容性，例如：
   - 在 API 参数末尾添加有默认值的参数
   - 对已有参数保留原有功能时，扩展新功能

**不适用条件**（❌ 禁止）**：
- 改变已有参数顺序
- 改变已有参数名称
- 修改返回值类型
- 删除现有参数
- 修改现有参数的默认值

## 方案 4：新增 API
**适用场景**：
- 无法通过修改现有 API 实现对齐，需要新增 API 来实现功能
- 具体包括以下情况：
  1. **API 相对引用路径不一致**：API 调用方式、模块路径与 PyTorch 不同
  2. **仅 API 调用方式不一致**：API 相对引用路径不一致，但参数完全相同
  3. **组合替代实现**：PyTorch API 需多个 Paddle API 组合实现
  4. **API 别名**：为 PyTorch 别名 API 新增对应的 Paddle 别名 API
  5. **功能缺失**：Paddle 暂无等效实现，需新增完整 API

## 方案 5：新增 compat 类型 API
**适用场景**：
- 无法通过其他方案实现对齐的退而求其次方案
- 具体包括以下情况：
  1. **无法原地修改**：修改现有 API 会引入严重的后向兼容性问题
  2. **无法新增 API**：PyTorch 对应的 API 路径在 Paddle 中已被占用
  3. **返回值不一致**：修改返回值类型会破坏现有调用
  4. **无法重载**：通过装饰器无法根据输入参数特征区分两套签名

**注意**：
- 新增对标 PyTorch 的兼容 API（位于 `paddle.compat.*` 路径下），不影响原 Paddle API
- 新增 API 需与 PyTorch 完全一致（路径、参数、返回值、行为）


# 三、标准工作流程

## Step 1: 差异分析

### 1.1 获取差异信息

> **⚠️ 短路逻辑（必须严格遵守）**：
> - 按**优先级 1 → 2 → 3 的顺序**依次尝试
> - **一旦从某个来源获取到完整差异信息，立即停止，不再查找其他来源**
> - 这是"三选一"而非"三者都要"，严禁同时查找多个来源

#### 优先级 1：查阅 API 差异文档 ⭐
- 查阅 PyTorch API 对应的 API 差异文档（`torch.{api_name}.md`，位于 `${ROOT_DIR}/docs/docs/guides/model_convert/convert_from_pytorch/api_difference/` 目录下）
- 如果文档存在且信息完整（包含对应 Paddle API、转写示例），则以此为准 ✅ **→ 停止查找，进入下一步**
- 如果文档不存在或信息不完整 → 继续优先级 2

> **⚠️ 注意：差异文档标题仅反映主要分类，须阅读全文**
> - 差异文档的标题头（如 `[ torch 参数更多 ]`、`[ 仅参数名不一致 ]`）只选取了**最显著的一类差异**作为标题
> - 一个 API 往往同时涉及多种差异，例如既有参数名不同，又有 torch 多出 out 参数
> - 必须通读差异文档的**完整参数映射表**，提取所有差异点，不能仅以文件标题作为差异分类的唯一依据

#### 优先级 2：查询转写配置 ⭐⭐
- 查询 PyTorch API 对应的转写配置（位于 `${ROOT_DIR}/PaConvert/paconvert/api_mapping.json` 或 `${ROOT_DIR}/PaConvert/paconvert/attribute_mapping.json`）
- 根据 Matcher 类型和字段内容分析差异分类和对应的 Paddle API
- 如果能从中获取到差异信息 ✅ **→ 停止查找，进入下一步**
- 如果转写配置不存在或无法获取信息 → 继续优先级 3
- **转写配置字段说明**：

| 字段 | 说明 | 对应差异分类 |
|------|------|------------|
| `Matcher: "ChangePrefixMatcher"` | API 完全一致，仅框架前缀不同（torch→paddle），无需任何参数转换 | API 完全一致 |
| `Matcher: "ChangeAPIMatcher"` | API 调用方式不同，但功能一致，无需参数转换 | 仅 API 调用方式不一致 |
| `Matcher: "NumelMatcher"` | 特定 API（如 torch.numel）的专用转换器 | 仅 API 调用方式不一致 |
| `Matcher: "TensorFunc2PaddleFunc"` | Tensor 类方法转为 Paddle 函数，如 `x.func()` → `paddle.func(x)` | 仅 API 调用方式不一致（无其他差异时） |
| `Matcher: "Func2Attribute"` | 函数调用转为属性访问，如 `x.func()` → `x.attr` | 仅 API 调用方式不一致（无其他差异时） |
| `Matcher: "Attribute2Func"` | 属性访问转为函数调用，如 `x.attr` → `x.func()` | 仅 API 调用方式不一致（无其他差异时） |
| `Matcher: "GenericMatcher"` | 通用转换器，适用于可直接映射的 API | 仅参数名不一致 / 仅 paddle 参数更多 / 仅参数默认值不一致 / torch 参数更多|
| `paddle_api` | 对应的 Paddle API 完整路径，如 `"paddle.transpose"` | 表示 API 映射关系|
| `kwargs_change` | 参数名映射关系，如 `{"input": "x", "dims": "perm"}` | 仅参数名不一致 |
| `unsupport_args` | 不支持的参数列表，如 `["stride"]` | torch 参数更多（部分不支持） |
| `paddle_default_kwargs` | Paddle 需要设置的默认参数，如 `{"axis": -1}` | 参数默认值不一致 / paddle 参数更多 |
| 其他自定义 `Matcher` | 除上述 Matcher 外的其他自定义 Matcher 名称 | 参数用法/类型不一致 / 组合替代实现等复杂情况 |

- **注意**：
- 当 `TensorFunc2PaddleFunc`、`Func2Attribute`、`Attribute2Func` 同时包含 `unsupport_args`、`kwargs_change` 或 `paddle_default_kwargs` 字段时，说明除了调用方式差异外，还存在其他差异，不能归类为"仅 API 调用方式不一致"
- 更多的分类映射关系则需要自行分析，可结合 `${ROOT_DIR}/PaConvert/paconvert/api_matcher.py` 中对应 Matcher 的实现逻辑进行差异分类


#### 优先级 3：网络搜索 API 文档 ⭐⭐⭐
- 从网络中分别搜索：
  1. PyTorch 官方 API 文档（https://pytorch.org/docs/stable/）
  2. Paddle 官方 API 文档（https://www.paddlepaddle.org.cn/documentation/docs/zh/api/）
- 对比两者的 API 文档，手动分析差异，获取 API 的签名、参数、返回值等信息


### 1.2 确定差异分类
差异分类共 13 类，具体如下：

| 序号 | 差异分类 | 说明 |
|:---:|:---|:---|
| 1 | API 完全一致 | 无差异，无需改动 |
| 2 | 仅 API 调用方式不一致 | API 相对引用路径不一致，但参数完全相同 |
| 3 | 仅参数名不一致 | 参数功能相同但参数名称不同 |
| 4 | paddle 参数更多 | Paddle 提供更多可选参数 |
| 5 | 参数默认值不一致 | 参数默认值不同 |
| 6 | torch 参数更多 | PyTorch 提供更多参数 |
| 7 | 输入参数用法不一致 | 参数处理方式不同 |
| 8 | 输入参数类型不一致 | 参数类型要求不同 |
| 9 | 返回参数类型不一致 | 返回值类型或结构不同 |
| 10 | 组合替代实现 | PyTorch API 需多个 Paddle API 组合实现 |
| 11 | 可删除 | PyTorch API 在 Paddle 中可直接删除 |
| 12 | API 别名 | PyTorch API 是其他 API 的别名 |
| 13 | 功能缺失 | Paddle 暂无等效实现 |

### 1.3 提取差异信息

总结相关差异信息，提供给 Step2 进行方案决策：

| 项目 | 内容  |
|------|------|
| 差异分类 | 具体差异分类 |
| API 映射 | PyTorch API vs 对应的 Paddle API（可能为空）|
| 参数映射 | 参数对应关系和差异说明 |
| 转写示例 | 代码转换示例 |

注意以下参数直接忽略，不视作差异信息：
1. 忽略第 1 列的 generator、memory_format、layout 参数
2. 忽略第 2 列的 name 参数

> 只能忽略上述明确列出的参数，**其他参数不可忽略**。例如 **device、out、requires_grad、dtype** 等不可忽略，必须处理。

## Step 2: 方案决策

### 2.1 关键概念

**API 相对引用路径**：API 完整路径在去掉框架导入模块(torch/paddle)后剩余的部分
- 示例：`torch.nn.functional.dropout` 的 API 相对引用路径是 `nn.functional.dropout`
- 示例：`paddle.nn.functional.dropout` 的 API 相对引用路径是 `nn.functional.dropout`
- 示例：`torch.Tensor.tile` 的 API 相对引用路径是 `Tensor.tile`

**仅多 out 参数**：torch 比 paddle **恰好只多一个参数**，且该参数名为 `out`，其余所有参数完全一致
- ✅ 符合：torch 签名比 paddle 签名仅多 `out` 一个参数，其余参数完全一致
- ❌ 不符合：torch 比 paddle 多两个及以上参数（即使其中包含 `out`，如同时多了 `out` 和 `requires_grad`）
- ❌ 不符合：torch 多的参数不是 `out`（如仅多 `dtype`）
- ❌ 不符合：除参数个数差异外，还存在参数名不同等其他差异（此时差异分类应合并为"torch 参数更多 + 参数名不一致"，不满足"仅多 out 参数"）

### 2.2 关键原则

> **⚠️ 严格遵循**：决策时必须严格遵守以下原则，不得主观臆断：
>
> 1. **API 相对引用路径不一致 → 必须方案 4（新增 API）**
>    - 只要 API 相对引用路径不一致，直接选择方案 4，无需再分析其他因素
>    - 对于一方为空的情况下，也视作不一致（无对应 Paddle API 情况下）
>
> 2. **严格按流程图和规则判断**
>    - 必须按照决策流程图的路径执行
>    - 严格按照各方案的适用条件判断
>    - 不得因"可以"、"应该"等主观判断偏离规则定义

### 2.3 决策流程图

```
开始
  │
  ├───→ API 相对引用路径是否一致？ ──────┐
  │                                  │
  │是                                │否
  ↓                                  ↓
具体有哪些差异？              方案 4（新增 API）→结束
  │
  ├──→ 1. API 完全一致 → 方案 6（无需改动）→结束
  │
  ├──→ 2. 仅 API 调用方式不一致 → 方案 4（新增 API）→结束
  │
  ├──→ 3. 仅参数名不一致 → 方案 2（C++下沉）→ 不适用则方案 1→结束
  │
  ├──→ 4. paddle 参数更多 → 是否影响对齐？─┬→否→方案 6（无需改动）→结束
  │                                     └→是→方案 3（修改 API）→导致后向不兼容则方案 5→结束
  │
  ├──→ 5. 参数默认值不一致 → 是否影响对齐？─┬→否→方案 6（无需改动）→结束
  │                                     └→是→方案 3（修改 API）→导致后向不兼容则方案 5→结束
  │
  ├──→ 6. torch 参数更多 → 仅多 out 参数（见 2.1 定义）？─┬→是→方案 2（C++下沉）→不适用则方案 1→结束
  │                                                  └→否→方案 3（修改 API）→导致后向不兼容则方案 1→无法重载则方案 5→结束
  │
  ├──→ 7. 输入参数用法不一致 → 方案 3（修改 API）→导致后向不兼容则方案 1→无法重载则方案 5→结束
  │
  ├──→ 8. 输入参数类型不一致 → 方案 3（修改 API）→导致后向不兼容则方案 1→无法重载则方案 5→结束
  │
  ├──→ 9. 返回参数类型不一致 → 方案 5（新增 compat 类型 API）→结束
  │
  ├──→ 10. 组合替代实现 → 方案 4（新增 API）→结束
  │
  ├──→ 11. 可删除 →  方案 6（无需改动）→结束
  │
  ├──→ 12. API 别名 → 方案 4（新增 API）→结束
  │
  └──→ 13. 功能缺失 → 方案 4（新增 API）→结束
```

### 2.4 详细决策规则

**前置判断**：分类 3~10 判定规则均基于**API 相对引用路径一致**的前提。只要 API 相对引用路径不一致，直接选择**方案 4（新增 API）**，无需进行如下判断。

#### 1. API 完全一致
- **决策**：方案 6（无需改动）

#### 2. 仅 API 调用方式不一致
- **决策**：方案 4（新增 API）
- **说明**：API 相对引用路径不一致，但参数完全相同，需要新增 API 来实现路径对齐

#### 3. 仅参数名不一致
- **优先级 1**：方案 2（C++下沉）
  - **前置检查**：确认满足方案 2 的全部适用条件
  - **任一不满足** → **优先级 2**
- **优先级 2**：方案 1（Python 装饰器）
  - 当方案 2 不满足时，使用方案 1

#### 4. paddle 参数更多
- **判断**：额外参数是否影响对齐
  - **否**（例如默认参数，Paddle 保持默认即可）→ 方案 6（无需改动）
  - **是** → 方案 3（修改 API）
    - **检查兼容性**：修改是否会引入后向不兼容问题
    - **会导致后向不兼容** → 方案 5（新增 compat 类型 API）

#### 5. 参数默认值不一致
- **判断**：不一致的默认值是否影响对齐
  - **否**（例如不影响结果的参数）→ 方案 6（无需改动）
  - **是** → 方案 3（修改 API）
    - **检查兼容性**：修改是否会引入后向不兼容问题
    - **会导致后向不兼容** → 方案 5（新增 compat 类型 API）

#### 6. torch 参数更多
- **判断**：是否仅多 out 参数（详见 2.1 定义，必须满足"恰好只多一个参数且该参数名为 out"）
  - **是**：
    - **优先级 1**：方案 2（C++下沉）
      - **前置检查**：确认满足方案 2 的全部 5 个适用条件
      - **任一不满足** → **优先级 2**
    - **优先级 2**：方案 1（Python 装饰器）
  - **否**：
    - **优先级 1**：方案 3（修改 API）
      - **检查兼容性**：修改是否会引入后向不兼容问题
      - **会导致后向不兼容** → **优先级 2**
    - **优先级 2**：方案 1（Python 装饰器）
      - **检查能否区分**：能否根据输入参数特征区分两套签名
      - **无法重载** → **优先级 3**
    - **优先级 3**：方案 5（新增 compat 类型 API）

#### 7. 输入参数用法不一致
- **优先级 1**：方案 3（修改 API）
  - **检查兼容性**：修改是否会引入后向不兼容问题
  - **会导致后向不兼容** → **优先级 2**
- **优先级 2**：方案 1（Python 装饰器）
  - **检查能否区分**：能否根据输入参数特征区分两套签名
  - **无法重载** → **优先级 3**
- **优先级 3**：方案 5（新增 compat 类型 API）

#### 8. 输入参数类型不一致
- **优先级 1**：方案 3（修改 API）
  - **检查兼容性**：修改是否会引入后向不兼容问题
  - **会导致后向不兼容** → **优先级 2**
- **优先级 2**：方案 1（Python 装饰器）
  - **检查能否区分**：能否根据输入参数特征区分两套签名
  - **无法重载** → **优先级 3**
- **优先级 3**：方案 5（新增 compat 类型 API）

#### 9. 返回参数类型不一致
- **唯一决策**：方案 5（新增 compat 类型 API）
- **原因**：
  - ❌ 方案 3：修改返回值必然存在兼容性问题
  - ❌ 方案 1：只能根据输入参数特征来实现重载，返回值不一致不满足重载条件
  - ✅ 方案 5：在 compat 路径下新增，不影响原 API

#### 10. 组合替代实现
- **决策**：方案 4（新增 API）
- **说明**：PyTorch API 需要多个 Paddle API 组合实现，通过新增 API 实现组合计算逻辑

#### 11. 可删除
- **决策**：方案 6（无需改动）
- **说明**：PyTorch API 在 Paddle 中可直接删除，无需开发

#### 12. API 别名
- **决策**：方案 4（新增 API）
- **说明**：为 PyTorch 别名 API 新增对应的 Paddle 别名 API

#### 13. 功能缺失
- **决策**：方案 4（新增 API）
- **说明**：Paddle 暂无等效实现，需要新增 API 来实现该功能

### 2.5 方案组合说明

> **重要提醒**：一个 API 可能涉及多种差异分类，需要综合分析所有差异点，可以组合多种方案来消除所有差异点。

**示例 1**：`torch.logspace` 差异文档标题为 `[ torch 参数更多 ]`，但全文实际涉及两类差异：
- 参数名不一致：`end→stop`、`steps→num`
- torch 参数更多：额外 `out`、`device`、`requires_grad` 参数
- **组合方案**：方案 3 + 方案 1
  - 方案 3：在 `paddle.logspace` 末尾新增 `out=None`、`device=None`、`requires_grad=False` 参数（后向兼容），处理"torch 参数更多"差异
  - 方案 1：Python 装饰器添加参数别名 `end/stop`、`steps/num`，处理"参数名不一致"差异

**示例 2**：`torch.slice_scatter` 差异文档标题为 `[ 输入参数用法不一致 ]`，但全文实际涉及两类差异：
- 参数名不一致：`input→x`、`src→value`、`dim→axes`、`start→starts`、`end→ends`、`step→strides`
- 输入参数类型不一致：torch 各参数为 `int`，Paddle 对应参数为 `list of int`
- **组合方案**：方案 3 + 方案 1
  - 方案 3：扩展 `paddle.slice_scatter` 的 `axes`/`starts`/`ends`/`strides` 参数，使其同时支持 `int` 和 `list of int`（后向兼容），处理"输入参数类型不一致"差异
  - 方案 1：Python 装饰器添加参数别名 `input/x`、`src/value`、`dim/axes`、`start/starts`、`end/ends`、`step/strides`，处理"参数名不一致"差异


# 四、常见问题处理

### Q1：如何判断方案 3 是否会导致后向不兼容？

此问题仅适用于**流程图中标注"导致后向不兼容则方案 1/5"的分支**（如差异分类 6/7/8）。

**判断方法**：严格按照方案 3 的"不适用条件"判断：
- ❌ 改变已有参数顺序 → 后向不兼容
- ❌ 改变已有参数名称（旧名不可用）→ 后向不兼容
- ❌ 修改返回值类型 → 后向不兼容
- ❌ 删除现有参数 → 后向不兼容
- ❌ 修改现有参数的默认行为 → 后向不兼容

**示例**：
- `torch.tril_indices` 多 `device` 参数 → 在末尾添加 `device=None` → 不违反上述条件 → **后向兼容** → **方案 3**
- `torch.slice_scatter` 扩展参数类型（int → int|list）→ 不违反上述条件 → **后向兼容** → **方案 3**

# 五、注意事项

1. 严格按标准工作流程执行，杜绝自行臆断和跳过步骤
2. 所有路径使用 `${ROOT_DIR}` 变量表示根目录，需自行替换为实际路径
3. 决策前检查 Paddle 代码的实际状态，差异文档可能滞后，代码反映真实情况（有可能已经完成了 Paddle 代码修改但未更正映射文档）
