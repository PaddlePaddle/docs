---
name: cpp-sink
description: 负责《Paddle API 对齐 PyTorch 项目》中 Step2：API 代码修改，实施『C++下沉』方案。通过将 Python API 下沉至 C++层，可以减少 Python 装饰器带来的性能开销，提升 API 调度效率。
disable-model-invocation: false
---

# 一、标准工作流程

根据 API 的复杂度，将下沉场景分为三类：

## 场景一: 仅参数名不一致（最简单）

**适用条件（需全部满足）：**
- Paddle Python API 与`ops.yaml`中参数**类型或数量一致**
- Paddle API 和 PyTorch API 仅参数名称不一致
- Python 端调用`_C_ops`前**无任何前处理逻辑**

**典型示例：** `paddle.log2`

### Step 1：配置 `python_api_info.yaml`

添加参数别名映射（按 op name 的字典序添加）：

```yaml
- op : log2
  name : [paddle.log2, paddle.Tensor.log2]
  args_alias :
    use_default_mapping : True  # 启用默认映射：x->input
```

**说明：**
- `name`：指定对应的 Python API 名称（函数 API 和 Tensor 方法）
- `use_default_mapping : True`：自动配置常用字段映射（x→input, y→other, axis→dim, keepdims→keepdim）
- 不在默认映射范围时，可配置自定义映射：`y: [exponent]`（Paddle 参数名为 key，PyTorch 参数名为 value）

### Step 2：迁移文档到 `_paddle_docs.py`

使用 `add_doc_and_signature` 函数迁移文档，关键要点：

1. **Alias 说明格式**：在 Args 部分为有别名的参数添加 `Alias: ``alias_name```
2. **out 参数处理**：keyword-only 参数放在 `Keyword Args:` 部分，位置参数放在 `Args:` 部分，如下：
```python
# out 为 keyword-only 参数
"""
Args:
    ...
Keyword Args:
    out (Tensor|optional): The output tensor. Default: None.
"""

# out 为位置参数
"""
Args:
    out (Tensor, optional): The output Tensor. Default: None.
"""
```

**示例**：
```python
add_doc_and_signature(
    "log2",
    r"""
    Calculates the log to the base 2 of the given input tensor, element-wise.
    ...
    Args:
        x (Tensor): Input tensor. Alias: ``input``.
        ...
    Keyword Args:
        out (Tensor, optional): The output tensor. Default: None.
    Returns:
        ...
""",
    """
def log2(
    x: Tensor,
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor
""",
)
```

**注意**：
- Tensor 类方法（如 paddle.Tensor.abs）没有文档，无需处理，请勿与普通方法（如 paddle.abs）混淆
- Inplace 方法（如 paddle.abs_等下划线 API），只需要更新 API 签名，不需要修改文档
- 注意文档格式规范，需与 `_paddle_docs.py` 现有文档格式保持一致

### Step 3：替换 Python 实现

找到 API 的 Python 实现位置（如 `python/paddle/tensor/math.py`，注意不要误检索到稀疏 API 位置）：

1. 在文件上方合适位置**导入**C++实现
2. 直接**删除**原有的 Python 函数实现

```python
# 在文件最上方合适位置导入（不要添加注释）
from paddle._C_ops import log2  # noqa: F401

# 以下内容全部删除（不要添加注释）
# def log2(x: Tensor, name: str | None = None) -> Tensor:
#     ...
```

## 场景二: 具有前处理逻辑（中等复杂度）

**适用条件（全部满足）：**
- Paddle Python API 与`ops.yaml`中参数**类型或数量一致**
- Paddle API 和 PyTorch API 仅参数名称不一致
- Python 端在调用`_C_ops`前**有其他前处理逻辑**

**典型示例：** `paddle.logsumexp`

### Step 1：配置 `python_api_info.yaml`

```yaml
- op : logsumexp
  name : [paddle.logsumexp, paddle.Tensor.logsumexp]
  args_alias:
    use_default_mapping : True  # x->input, axis->dim
  pre_process:
    func : LogsumexpPreProcess(x, axis, reduce_all)  # 前处理函数
```

**关键点：**
- `name`：指定对应的 Python API 名称
- `args_alias.use_default_mapping`：启用默认参数映射
- `pre_process.func`：指定前处理函数名称及其参数列表


### Step 2：实现前处理函数

将 Python 端在调用`_C_ops`前的**其他前处理逻辑**，修改为 C++的实现。在 `paddle/fluid/pybind/arg_pre_process.h` 声明，在 `arg_pre_process.cc` 实现。

**关键要点：**
- 必须同时实现**动态图版本**（参数类型为 `Tensor*`）和**静态图版本**（参数类型为 `pir::Value*`）
- 函数通过指针修改参数值
- 参考 `arg_pre_process.cc` 中已有实现，保持风格一致

**声明示例**：
```cpp
void LogsumexpPreProcess(Tensor *x, std::vector<int> *axis, bool *reduce_all);      // 动态图
void LogsumexpPreProcess(pir::Value *x, std::vector<int> *axis, bool *reduce_all);  // 静态图
```

### Step 3：迁移文档到 `_paddle_docs.py`

参考场景一的 Step 2 执行相同操作。

### Step 4：替换 Python 实现

参考场景一的 Step 3 执行相同操作。

## 场景三: 复杂参数映射（最复杂）

**适用条件（全部满足）：**
- Paddle Python API 与`ops.yaml`中参数**类型或数量不一致**
- 需要复杂的自定义参数解析逻辑，来匹配 Paddle Python API 与`ops.yaml`的参数差异

**典型示例：** `paddle.argmax`/`paddle.argmin`

### Step 1：配置 `python_api_info.yaml`

```yaml
- op : argmax
  name : [paddle.argmax, paddle.Tensor.argmax]
  args_mapper :
    func : ArgMaxMinMapper  # 自定义参数映射函数
```

**关键点：**
- `name`：指定对应的 Python API 名称
- `args_mapper.func`：指定自定义 Mapper 函数名称
- 当使用`args_mapper`时，**不会生成默认参数解析代码**
- Mapper 需要手动解析所有参数，包括参数别名支持

### Step 2：实现自定义 Mapper

将 Python API 与`ops.yaml`中参数**类型或数量不一致**的差异，通过 C++自定义 Mapper 来进行匹配与转换。在 `paddle/fluid/pybind/args_mapper.h` 声明，在 `args_mapper.cc` 实现。

**关键要点：**
- 必须同时实现**动态图版本**和**静态图版本**
- 使用 `args_mapper` 时不会生成默认参数解析代码，需手动解析所有参数
- 参考 `args_mapper.cc` 中已有实现，保持风格一致

**核心工具函数：**
| 函数 | 用途 |
|------|------|
| `GetTensorFromArgsOrKWArgs` | 解析 Tensor 参数，支持别名列表 |
| `GetItemFromArgsOrKWArgs` | 获取通用 Python 对象 |
| `CastPyArg2Scalar/Boolean/DataType` | 类型转换 |
| `CheckParamsCount` / `CheckRemainingParamsValidity` | 参数校验 |

**静态图特殊处理：** 使用 `pir::Value` 代替 `Tensor`，常量需通过 `paddle::dialect::full` 转换为 Value 类型。

**声明示例**：
```cpp
void ArgMaxMinMapper(PyObject* args, PyObject* kwargs,
                     Tensor* x, Scalar* axis, bool* keepdims, bool* flatten, DataType* dtype);  // 动态图
void ArgMaxMinMapper(PyObject* args, PyObject* kwargs,
                     pir::Value* x, pir::Value* axis, ...);  // 静态图
```

### Step 3: 迁移文档到 `_paddle_docs.py`

除了完成场景一的文档迁移操作之外，场景三文档迁移还需要额外注意：

如果使用了自定义 Mapper，则 API 有可能支持了签名重载，需要分别描述两种签名，如下：
> 注：只需在文档正文中阐述两种签名（Paddle 在前，Pytorch 在后），文档其他位置如 Args/Returns 仍以 Paddle 风格签名为准

```python
"""
This API has two signatures:

1. ``paddle.sum(x, axis=None, dtype=None, keepdim=False, name=None, *, out=None)`` (Paddle-style)
2. ``paddle.sum(input, dim=None, keepdim=False, dtype=None, *, out=None)`` (PyTorch-style)

Args:
    ...

Returns:
    ...
"""
```

### Step 4：替换 Python 实现

参考场景一的 Step 3 执行相同操作。


# 二、注意事项

1. 严格按标准工作流程执行，杜绝自行臆断和跳过步骤
2. 若 Python API 参数顺序与`_C_ops` API 不同，属于特殊情况，Cpp 下沉方案无法实现，需要使用 Python 装饰器方案
3. 代码中不允许提交中文，代码注释采用英文
4. 若 API 需支持`out`参数，必须修改`add_doc_and_signature`中的字符串，增加 out 参数
5. 不要修改`generated_tensor_methods_patch.py`，该文件是自动生成的，修改没有意义，如无法对齐可考虑放弃 C++下沉方案而不是改动该文件
6. 示例代码若涉及多种数据类型，可能触发类型检查误报，添加注释忽略：
```python
  .. code-block:: pycon
      >>> # type: ignore
      >>> import paddle
      >>> x = paddle.to_tensor([1.0, 2.0])
```

# 三、常见问题处理

## Q1：静态图报错"must be Value, but got Variable"

**错误现象**：
```python
TypeError: (InvalidType) all(): argument (position 1) must be Value, but got Variable
```

**解决方法**：
1. 删除过时的测试文件：
   ```bash
   rm -rf test/deprecated/test_xxx.py
   ```
2. 删除`CMakeLists.txt`中涉及的单测配置
3. 检查并更新所有引用这些测试的代码

---

## Q2：参数解析报错"got an unexpected keyword argument"

**错误现象**：
```python
TypeError: argmax() got an unexpected keyword argument 'invalid_param'
```

**解决方法**：
1. 检查参数名称拼写
2. 确认是否支持该参数
