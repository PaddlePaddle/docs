---
name: api-change-decider
description: 负责《Paddle API 对齐 PyTorch 项目》中 Step1：方案决策，分析 PyTorch API 与 Paddle API 之间的差异，制定合适的 API 改动方案
context: fork
agent: Explore
disable-model-invocation: false
---

# 一、候选方案

## 方案 1：Python 装饰器
**适用场景**：
参数顺序不同、参数个数不同、参数类型不同、参数名不同

**适用条件**（必须满足）：
1. API 路径一致
2. 重载条件：两套签名必须存在参数特征差异（个数、类型、名称至少一项不同），且根据调用时传入的参数能够唯一判定应该使用哪个签名：
- 参数个数不同：如 func(a) vs func(a, b)，根据传入的参数个数不同，可以唯一确定调用的是哪个函数。（默认参数可不传入，不计入个数）
- 参数类型不同：如 func(x: Tensor, y: Tensor) vs func(x: Tensor, y: int)，根据传入的第二个位置参数的类型不同，可以唯一确定调用的是哪个函数。
- 参数名称不同：如 func(a, b) vs func(a, c)，根据调用时使用的关键字参数名不同（如 `func(a=1, b=2)` vs `func(a=1, c=2)`），可以唯一确定调用的是哪个函数。

**优点**：灵活性强，兼容性好
**缺点**：性能低于 C++ 下沉实现

## 方案 2：C++ 下沉
**适用场景**：
仅涉及参数名不同或仅多 out 参数的情况

**适用条件**（必须全部满足）：
1. API 路径一致
2. API 名称和 `_C_ops.xxx` 的 OP 名称一致
   - ✅ 正确示例：`paddle.atan` 调用 `_C_ops.atan`
   - ❌ 错误示例：`paddle.select_scatter` 调用 `_C_ops.set_value_with_tensor`（名称不一致）
3. API 差异**仅为参数名不同或仅多 out 参数**（不涉及参数顺序或个数差异）
   - torch 比 paddle 恰好只多一个参数 `out`，其余参数完全一致
   - ✅ 符合：仅多 `out` 一个参数
   - ❌ 不符合：多两个及以上参数、多的参数不是 `out`、还存在参数名不同等其他差异
4. Python 实现中**仅有一次** `_C_ops.xxx` 调用
5. Python 实现中不涉及其他 Paddle API 调用
   - ❌ **禁止**：调用任何其他 paddle API（如 `paddle.where`、`paddle.abs`、`paddle.flatten` 等）
   - ❌ **禁止**：调用 Tensor 方法（如 `x.flatten()`、`x.reshape()`、`x.unsqueeze()` 等，这些等同于调用 paddle API）
   - ❌ **禁止**：使用 `paddle.full_like`、`paddle.zeros_like`、`paddle.cast` 等
   - ✅ **允许**：Python 内置函数（如 `len()`、`isinstance()`、`list()`、`range()` 等）
   - ✅ **允许**：简单的属性访问（如 `x.shape`、`x.dtype`、`x.ndim` 等）
   - ✅ **允许**：简单的算术运算（如 `index + 1`、`axis < 0` 等）
6. `_C_ops.xxx` 前面**无复杂前处理逻辑**，前处理逻辑（如存在）容易改写为 C++
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

**优点**：性能最优
**缺点**：限制条件很多，需严格满足

## 方案 3：修改原有 API
**适用场景**：
同方案 1（参数顺序、个数、类型、名称差异），但通过直接修改原 API 实现而非装饰器

**适用条件**（必须全部满足）：
1. API 路径一致
2. 修改必须能保持后向兼容性

**不适用条件**（❌ 禁止）：
- 改变已有参数顺序
- 改变已有参数名称
- 修改返回值类型
- 删除现有参数
- 修改现有参数的默认行为

**修改模式**：
- 新增参数：向现有 API 添加新参数
- 扩展参数类型/功能：对已有参数扩展新功能同时保留原有功能

## 方案 4：新增 API
**适用场景**：
- API 路径不一致
- 仅 API 调用方式不一致
- 组合替代实现
- API 别名
- 功能缺失

**修改模式**：
1. 新增 API 别名：PyTorch API 与 Paddle API 功能完全一致，仅 API 路径/名称不同
2. 新增 Python 层 API：需在 Python 层实现新 API
3. 新增 OP：需在 C++ 层新增底层算子

## 方案 5：新增 compat 类型 API
**适用条件**（必须全部满足）：
1. API 路径一致
2. 不满足方案 1/2/3 的适用条件

**典型场景**：
- API 路径一致，但修改原有 API 会破坏后向兼容性（如修改返回值类型）
- API 路径一致，但不满足装饰器的重载条件

**不适用条件**（❌ 禁止）：
- API 路径不一致

**实现要点**：
- 新增兼容 API 位于 `paddle.compat.*` 路径下，不影响原 Paddle API


# 二、标准工作流程

## Step 1: 获取差异信息

按优先级依次查找差异信息，一旦获取完整信息立即停止，这是"三选一"而非"三者都要"。

### 优先级 1：查阅 API 差异文档
查阅 `${ROOT_DIR}/docs/docs/guides/model_convert/convert_from_pytorch/api_difference/torch.{api_name}.md`。如果文档存在且完整（包含对应 Paddle API、转写示例），则以此为准并停止查找；否则继续优先级 2。

**注意**：
- 文档可能有细微错误，当发现问题可自行分析，或结合源码进一步确认
- 一个 API 可能涉及多种差异，差异文档标题仅反映主要分类，必须通读完整参数映射表提取所有差异，不能仅看标题
- 忽略参数：torch 侧 `generator`、`memory_format`、`layout`；paddle 侧 `name`。其他参数不可忽略

### 优先级 2：查询转写配置
查询 `${ROOT_DIR}/PaConvert/paconvert/api_mapping.json` 或 `attribute_mapping.json`，根据 Matcher 类型分析差异分类。如果能直接从配置中提取完整差异信息，则以此为准并停止查找；否则继续优先级 3。

| 字段 | 对应差异分类 |
|------|------------|
| `ChangePrefixMatcher` | API 完全一致 |
| `ChangeAPIMatcher` / `NumelMatcher` / `TensorFunc2PaddleFunc` / `Func2Attribute` / `Attribute2Func` | 仅 API 调用方式不一致（若同时包含 `unsupport_args`、`kwargs_change` 字段，则存在多重差异） |
| `GenericMatcher` | 参数名不一致 / paddle 参数更多 / 参数默认值不一致 / torch 参数更多 |
| `paddle_api` | API 映射关系 |
| `kwargs_change` | 参数名不一致 |
| `unsupport_args` | torch 参数更多（部分不支持） |
| `paddle_default_kwargs` | 参数默认值不一致 / paddle 参数更多 |
| 其他自定义 Matcher | 参数用法/类型不一致 / 组合替代实现 |

**注意**：
- 更多 Matcher 可参考 `api_matcher.py` 分析。

### 优先级 3：自行获取 API 信息

当优先级 1 和优先级 2 均无法获取完整差异信息时，通过多种方式自行获取 PyTorch API 和 Paddle API 的信息，进行对比分析。

获取方式请参考`api-compatibility/SKILL.md` 中的「3.6 API 信息获取方式」章节。

## Step 2：提取差异信息

根据获取到的 PyTorch API 与 Paddle API 信息，提取以下内容：

| 项目 | 内容  |
|------|------|
| API 映射 | PyTorch API vs 对应的 Paddle API（可能为空）|
| 差异分类 | 可包含多种差异分类 |
| 参数映射 | 参数对应关系和差异说明 |
| 转写示例 | 代码转换示例 |

候选差异分类为：

| 序号 | 差异分类 | 说明 |
|:---:|:---|---|
| 1 | API 完全一致 | 无差异，无需改动 |
| 2 | 仅 API 调用方式不一致 | API 路径不一致，但参数完全相同 |
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

## Step 3: 方案决策

### 3.1 判断 API 路径一致性

**定义**：API 路径 = 去掉框架前缀后的调用路径（包括名称）。

**判断方法**：执行 `python -c "import paddle; paddle.{path}"` 验证是否存在。

| PyTorch API | API 路径 | Paddle 对应 | 路径一致性 |
|-------------|---------|-------------|:---:|
| `torch.trunc` | `trunc` | `paddle.trunc` | ✅ 一致 |
| `torch.histc` | `histc` | `paddle.histogram` | ❌ 不一致 |
| `torch.Tensor.addmv_` | `Tensor.addmv_` | 无 | ❌ 不一致 |
| `torch.autograd.enable_grad` | `autograd.enable_grad` | `paddle.autograd.enable_grad` | ✅ 一致 |

**示例**：

```bash
python -c "import paddle; paddle.trunc"           # 存在 → 路径一致
python -c "import paddle; paddle.addmv"           # AttributeError → 路径不一致
python -c "import paddle; paddle.Tensor.addmv_"   # AttributeError → 路径不一致
```

**输出**：记录路径一致性结果（一致/不一致），用于选择决策表。


### 3.2 查表决策

根据 3.1 的路径一致性结果选择决策表：**路径不一致** → 决策表 A；**路径一致** → 决策表 B。

**方案切换说明**：`→` 表示「不满足当前条件则切换到下一个备选方案」。
- 方案 2 → 方案 1：不满足 C++ 下沉条件
- 方案 3 → 方案 1：修改会破坏后向兼容
- 方案 1 → 方案 5：不满足装饰器重载条件

**决策表 A：路径不一致**（必须包含方案 4）

| 差异分类 | 决策链（→ 备选） |
|----------|------------------|
| 分类 2, 10, 12, 13 | 方案 4（新增 Python 层 API 或 C++ 算子） |
| 分类 3 | 方案 2 + 方案 4（新增 API 别名）→ 方案 1 + 方案 4（新增 API 别名） |
| 分类 4, 5 | 不影响对齐 → 无需修改；否则 → 方案 3 + 方案 4（新增 API 别名）→ 方案 4（新增 Python 层 API） |
| 分类 6 | 仅多 out 且满足方案 2 条件 → 方案 2 + 方案 4（新增 API 别名）；否则 → 方案 3 + 方案 4（新增 API 别名）→ 方案 1 + 方案 4（新增 API 别名）→ 方案 4（新增 Python 层 API） |
| 分类 7, 8 | 方案 3 + 方案 4（新增 API 别名）→ 方案 1 + 方案 4（新增 API 别名）→ 方案 4（新增 Python 层 API） |
| 分类 9 | 方案 4（新增 Python 层 API） |
| 分类 11 | 无需修改 |

**决策表 B：路径一致**

| 差异分类 | 决策链（→ 备选） |
|----------|------------------|
| 分类 1, 11 | 无需修改 |
| 分类 3 | 方案 2 → 方案 1 |
| 分类 4, 5 | 不影响对齐 → 无需修改；否则 → 方案 3 → 方案 5 |
| 分类 6 | 仅多 out 且满足方案 2 条件 → 方案 2；否则 → 方案 3 → 方案 1 → 方案 5 |
| 分类 7, 8 | 方案 3 → 方案 1 → 方案 5 |
| 分类 9 | 方案 5 |


### 3.3 多差异分类组合决策

当 API 存在多个差异分类时，**逐个差异分类独立查表，再合并方案**。

**步骤 1**：遍历每个差异分类，根据决策表独立查表得到对应方案。

**步骤 2**：对多个方案执行合并。

| 规则 | 判断条件 | 操作 |
|------|----------|------|
| 去重 | 多个差异分类对应相同方案 | 只保留一个 |
| 互斥 | 方案 2 与方案 1 同时出现 | 仅保留方案 2 |
| 组合 | 不同方案且不互斥 | 按编号升序，用 `+` 连接 |

**步骤 3**：输出最终组合方案。

**示例**：

| API | 差异分类 | 查表结果 | 合并结果 |
|-----|----------|----------|----------|
| torch.atan | 分类 3 + 分类 6 | 分类 3 → 方案 2；分类 6 → 方案 2 | 方案 2（去重） |
| torch.logspace | 分类 3 + 分类 6 | 分类 3 → 方案 1；分类 6 → 方案 3 | 方案 3 + 方案 1 |
| torch.histc | 分类 2 + 分类 6 + 分类 9 | 分类 2 → 方案 4；分类 6 → 方案 4；分类 9 → 方案 4 | 方案 4（去重） |


# 三、输出要求

**重要**：本 SKILL 以 agent 模式执行，上下文不共享。完成决策后，必须按以下格式输出完整信息，供后续步骤使用。

## 输出格式

```
### [序号] PyTorch API 名称

**基本信息**
- PyTorch API：`完整签名`
- Paddle API：`完整签名`（如无对应则填"无"）
- 实现位置：`文件路径:行号`（如未获取则填"未获取"）

**差异分析**
- API 路径一致性：一致 / 不一致
- 差异分类：`分类名称`（多个分类用" + "连接）
- 具体差异：
  - 参数名映射：`torch 参数 → paddle 参数`（如有）
  - 额外参数：torch 多出 `xxx, yyy`（如有）
  - 参数类型：`参数名: torch 类型 vs paddle 类型`（如有）

**方案决策**
- 选择方案：`方案 N（方案名称）` 或 `方案 N1 + 方案 N2`
- 决策依据：`简述选择理由`
- 组合说明：`分别说明各方案解决的差异点`（仅组合方案需要）
```

## 示例

**示例 1：单方案**

```
### [1] torch.atan

**基本信息**
- PyTorch API：`torch.atan(input, *, out=None)`
- Paddle API：`paddle.atan(x, name=None, *, out=None)`
- 实现位置：`python/paddle/tensor/math.py:45`

**差异分析**
- API 路径一致性：一致
- 差异分类：参数名不一致 + torch 参数更多
- 具体差异：
  - 参数名映射：`input → x`
  - 额外参数：torch 多出 `out`

**方案决策**
- 选择方案：方案 2（C++ 下沉）
- 决策依据：API 路径一致；仅参数名不同 + 多 out 参数，且满足 C++ 下沉全部条件。
```

**示例 2：组合方案**

```
### [2] torch.logspace

**基本信息**
- PyTorch API：`torch.logspace(start, end, steps, base=10.0, *, out=None, dtype=None, layout=None, device=None, requires_grad=False)`
- Paddle API：`paddle.logspace(start, stop, num, base=10.0, dtype=None, name=None)`
- 实现位置：`python/paddle/tensor/creation.py:557-749`

**差异分析**
- API 路径一致性：一致
- 差异分类：参数名不一致 + torch 参数更多
- 具体差异：
  - 参数名映射：`end → stop, steps → num`
  - 额外参数：torch 多出 `out, device, requires_grad`

**方案决策**
- 选择方案：方案 3 + 方案 1
- 决策依据：API 路径一致；新增 out/device/requires_grad 参数可保持后向兼容；参数名差异需通过装饰器适配。
- 组合说明：方案 3 处理 torch 参数更多；方案 1 处理参数名不一致。
```

# 四、注意事项

1. **严格按标准工作流程执行**，杜绝自行臆断和跳过步骤
2. 决策前务必检查 Paddle 代码实际状态，差异文档可能滞后，代码反映真实情况（有可能已经完成了 Paddle 代码修改但未更正映射文档）
3. **读取或修改 Python 实现时，忽略静态图部分**（LayerHelper 分支代码不再维护）

# 五、常见问题处理

### Q1：为什么 torch.histc 不能选择方案 5？

**答**：方案 5（新增 compat 类型 API）要求 API 路径一致。torch.histc 对应的 Paddle API 是 `paddle.histogram`，API 路径不一致（`histc` vs `histogram`），所以必须选择方案 4。

### Q2：API 路径不一致时，方案 1/2/3 还能用吗？

**答**：可以组合使用，但执行顺序很重要：

1. **先执行方案 1/2/3**：修改原 Paddle API（如新增参数、适配参数名等）
2. **再执行方案 4**：新增 API 别名指向修改后的原 API

**示例**：`torch.histc` vs `paddle.histogram`
- **步骤 1**：方案 3 修改 `paddle.histogram`，新增 `out` 参数支持
- **步骤 2**：方案 4 新增 `paddle.histc` 作为 `paddle.histogram` 的别名

**输出方案**：方案 3 + 方案 4（新增 API 别名）
