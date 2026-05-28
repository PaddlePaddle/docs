---
name: python-decorator
description: 负责《Paddle API 对齐 PyTorch 项目》中 Step2：API 代码修改，实施『Python 装饰器』方案。通过 Python 装饰器，在 Python 层为 Paddle API 实现参数名称、参数顺序、参数类型和参数用法的重载，实现 PyTorch 风格的 API 调用，并保持 Paddle API 的向后兼容性。
disable-model-invocation: false
---

# 一、现有装饰器体系

Paddle 现有装饰器统一位于 `${ROOT_DIR}/Paddle/python/paddle/utils/decorator_utils.py`，按功能分为两类：

## 1.1 通用别名装饰器

| 装饰器 | 支持参数数量 | 性能 | 使用场景 | 优先级 |
|---------|-------------|------|----------|--------|
| `param_one_alias` | 1 个 | 高 | 单个参数别名（如 x↔input） | ⭐⭐⭐ |
| `param_two_alias` | 2 个 | 高 | 两个参数别名（如 x↔input, axis↔dim） | ⭐⭐⭐ |
| `ParamAliasDecorator` | 3 个及以上 | 中 | 复杂参数映射场景 | ⭐⭐ |
| `param_two_alias_one_default` | 2 个+默认值 | 低 | 需要默认值（median 专用） | ⭐（不推荐）|

## 1.2 专用装饰器

| 装饰器 | 功能描述 | 关键特性 |
|--------|----------|----------|
| `index_select_decorator` | 参数别名 + 参数顺序转换 | 检测第 2 个位置参数是否为 int 来判断不同参数顺序 |
| `index_add_decorator` | 参数别名 + 参数顺序转换 | 检测第 2 个位置参数是否为 int 来判断不同参数顺序 |
| `transpose_decorator` | 参数别名 + 参数用法转换 | 通过 dim0/dim1 两个 int 来自动构造 perm 列表 |
| `size_args_decorator` | 参数别名 + 可变参数 | 合并全部 int 位置参数为 shape 列表 |
| `view_decorator` | 参数别名 + 可变参数 | 合并全部 int 位置参数为 shape 列表 |
| `reshape_decorator` | 参数别名 + 可变参数 | 第 1 个参数别名，合并后面多个 int 位置参数为 shape 列表 |
| `expand_decorator` | 参数别名 + 可变参数 |  第 1 个参数别名，合并后面多个 int 位置参数为 shape 列表 |
| `legacy_reduction_decorator` | 参数别名 + 报错信息增强 | 检测到 size_average/reduce 用法后，增强报错信息 |
| `lp_pool_function_decorator` | 参数别名 + 参数顺序转换 | 检测第 5 个位置参数是否为 bool 来判断不同参数顺序 |

# 二、标准工作流程

**整体流程**：Step 1 差异分析与选择装饰器 → Step 2 应用或开发装饰器 → Step 3 更新函数文档

## Step 1: 差异分析与选择装饰器

根据 PyTorch API 与 Paddle API 的**差异分析**来区分不同场景，选择合适的装饰器方案。

---

### 场景决策表

| 差异类型 | 参数顺序 | 参数个数 | 参数用法 | 推荐方案 |
|---------|--------|--------|--------|--------|
| 仅参数名不同 | 相同 | 相同 | 相同 | ✅ 方式一（通用别名装饰器） |
| 参数名+参数顺序不同 | 不同 | 相同 | 相同 | ❌ 需要使用方式二（专用装饰器） |
| 参数名+参数个数不同 | 相同 | 不同 | 相同 | ❌ 需要使用方式二（专用装饰器） |
| 参数名+参数用法不同 | 相同 | 相同 | 不同 | ❌ 需要使用方式二（专用装饰器） |
| 其他复杂情况 | 其他 | 其他 | 其他 | ❌ 需要使用方式二（专用装饰器） |

**判断方法**：若上表中任何一列出现"不同"，则需使用方式二开发专用装饰器。

---

### 1. 仅参数名不同（参数顺序相同）

根据需要映射的参数数量，选择对应的通用别名装饰器：
- 1 个参数 → `param_one_alias`
- 2 个参数 → `param_two_alias`
- 3 个及以上参数 → `ParamAliasDecorator`

**使用示例**：
```python
# 单个参数别名
## torch.deg2rad(input)
## paddle.deg2rad(x)
from paddle.utils.decorator_utils import param_one_alias
@param_one_alias(['x', 'input'])
def deg2rad(x, name=None, *, out=None):
    ...

# 两个参数别名
## torch.squeeze(input, dim)
## paddle.squeeze(x, axis)
from paddle.utils.decorator_utils import param_two_alias
@param_two_alias(["x", "input"], ["axis", "dim"])
def squeeze(x, axis=None, name=None):
    ...

# 多个参数别名
## torch.nn.functional.normalize(input, p, dim, eps)
## paddle.nn.functional.normalize(x, p, axis, epsilon)
from paddle.utils.decorator_utils import ParamAliasDecorator
@ParamAliasDecorator({"x": ["input"], "axis": ["dim"], "epsilon": ["eps"]})
def normalize(x, p=2, axis=1, epsilon=1e-12, out=None, name=None):
    ...
```

### 2. 参数名不同 + 可变参数

开发新的专用装饰器，可参考`size_args_decorator`、`reshape_decorator`、`expand_decorator`、`view_decorator`

**使用示例**：
```python
# torch.reshape(x, 2, 5)  # shape 支持可变参数用法
# paddle.reshape(x, [2, 5])
from paddle.utils.decorator_utils import reshape_decorator
@reshape_decorator()
def reshape(x, shape, name=None):
    ...
```

### 3. 参数名不同 + 参数顺序不同

开发新的专用装饰器

**示例**：
```python
# torch.index_select(input, dim, index)
# paddle.index_select(x, index, axis)
from paddle.utils.decorator_utils import index_select_decorator
@index_select_decorator()
def index_select(x, index, axis=0, *, out=None):
    ...
```

### 4. 参数名不同 + 参数用法不同

开发新的专用装饰器

**示例**：
```python
# torch.transpose(x, dim0, dim1)  # 交换两个维度，
# paddle.transpose(x, perm)
from paddle.utils.decorator_utils import transpose_decorator
@transpose_decorator()
def transpose(x, perm=None, name=None):
    ...
```

## Step 2: 应用或开发装饰器

根据**Step1**中的不同场景：
- **场景 1**（仅参数名不同）：请进入 **方式一（通用别名装饰器）**。
- **场景 2 & 3 & 4**（可变参数、参数顺序不同、参数用法不同及其他情况）：请进入 **方式二（开发新的专用装饰器）**。

### 方式一：通用别名装饰器

1. 在 API 函数文件中导入装饰器
2. 在 API 函数定义前添加装饰器

**示例**：
```python
from paddle.utils.decorator_utils import param_two_alias

@param_two_alias(["x", "input"], ["axis", "dim"])
def cumsum(x, axis=None, dtype=None, name=None):
    ...
```

### 方式二：开发新的专用装饰器

1. 在`${ROOT_DIR}/Paddle/python/paddle/utils/decorator_utils.py`中定义新装饰器
2. 按照以下模板和要点实现
3. 在 API 函数上使用新装饰器

**装饰器模板**：
```python
import functools
import inspect

def custom_decorator():
    """
    装饰器功能说明

    Usage Example:
        PyTorch: torch.api(arg1, arg2)
        Paddle: paddle.api(arg2, arg1)  # 或其他映射关系
    """

    def decorator(func):
        @functools.wraps(func)  # 保持原函数的__name__, __doc__等元信息
        def wrapper(*args, **kwargs):
            # 1. 参数别名映射
            if "input" in kwargs and "x" not in kwargs:
                kwargs["x"] = kwargs.pop("input")

            # 2. 参数顺序转换或其他特殊处理
            # 根据需要调整 args

            # 3. 调用原函数
            return func(*args, **kwargs)

        wrapper.__signature__ = inspect.signature(func)  # 保持函数签名
        return wrapper

    return decorator
```

**关键实现要点**：

1. **关键字参数别名**
```python
if "input" in kwargs and "x" not in kwargs:
    kwargs["x"] = kwargs.pop("input")
```

2. **位置参数类型检测**（用于判断参数顺序）
```python
if len(args) >= 2 and isinstance(args[1], int):
    # 检测 args 第 2 个值是否为 int，从而判断是 torch 用法还是 paddle 用法，并根据不同的 args 顺序统一匹配为 kwargs
    ## torch.index_select(input, dim, index)
    ## paddle.index_select(x, index, axis)
    kwargs["x"] = args[0]
    kwargs["axis"] = args[1]
    args = args[2:]
```

3. **可变参数处理**
```python
# 合并多个 int 位置参数为列表
if len(args) >= 2 and all(isinstance(arg, int) for arg in args[1:]):
    kwargs["shape"] = list(args[1:])
    args = args[:1]
```


**完整示例**：
```python
def index_select_decorator():
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 1. 参数别名映射
            if "input" in kwargs and "x" not in kwargs:
                kwargs["x"] = kwargs.pop("input")
            if "dim" in kwargs and "axis" not in kwargs:
                kwargs["axis"] = kwargs.pop("dim")

            # 2. PyTorch 参数顺序匹配：识别不同的 args 顺序，统一处理为 kwargs
            if len(args) >= 2 and isinstance(args[1], int):
                # PyTorch 顺序: (input, dim, index) → Paddle 顺序: (x, index, axis)
                kwargs["x"] = args[0]
                kwargs["axis"] = args[1]
                if len(args) > 2:
                    kwargs["index"] = args[2]
                args = ()

            return func(*args, **kwargs)

        wrapper.__signature__ = inspect.signature(func)
        return wrapper

    return decorator
```

**注意事项**：
1. 尽可能参考 `${ROOT_DIR}/Paddle/python/paddle/utils/decorator_utils.py` 中已有的专用装饰器来实现，在风格和逻辑上保持尽可能一致
2. 如果两者 API 对应参数的顺序不同，则装饰器需要通过位置参数(args)类型检测来区分两者，并分别匹配不同的参数顺序
3. 专用装饰器应该尽可能逻辑简单（单个函数控制在 30 行以内），只假定存在 Paddle 签名+PyTorch 签名两种用法，其他情况无需判断，提升性能
4. overload 注解：专用装饰器需添加 overload 注解（通用别名装饰器无需注解），需针对 Paddle 签名、PyTorch 签名分别添加 overload 注解（Paddle 在前，PyTorch 在后）。导入方式：`from typing import overload`
```python
@overload
def gather(
    x: Tensor,
    index: Tensor,
    axis: Tensor | int | None = None,
    name: str | None = None,
    out: Tensor | None = None,
) -> Tensor: ...

@overload
def gather(
    input: Tensor,
    dim: int,
    index: Tensor,
    out: Tensor | None = None,
) -> Tensor: ...
```

> **注意**：`out` 参数由方案 3（修改原有 API）负责，本方案不处理 out 参数的新增。

## Step 3: 更新函数文档字符串

如果使用的是通用别名装饰器，则在文档的 Args 部分为有别名的参数添加 Alias Support 说明，如下：
> 注：Alias 说明应放在该参数描述的末尾，格式为: Alias: ``alias_name`` ，多个 Alias 描述为: Alias: ``alias_name1`` or ``alias_name2``

```python
@param_two_alias(["x", "input"], ["axis", "dim"])
def fft(
    x: Tensor,
    n: int | None = None,
    axis: int = -1,
    norm: _NormalizeMode = "backward",
    name: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    """
    Calculate one-dimensional discrete Fourier transform.

    Args:
        x (Tensor): The input data. It's a Tensor type. It's a complex.
            Alias: ``input``.
        n (int|None, optional): The length of the output transform axis.
        axis (int, optional): Axis used to calculate FFT.
            Alias: ``dim``.
        norm (str, optional): Indicates which direction to scale the `forward` or `backward` transform
            pair and what normalization factor to use.
        name (str|None, optional): The default value is None.

    Keyword args:
        out(Tensor, optional): The output tensor.
    """
```

如果使用的是专用装饰器，则表明 API 支持了签名重载，需要分别描述两种签名，可以参考代码中的@overload 注解，如下：
> 注：只需在文档正文中阐述两种签名（Paddle 在前，Pytorch 在后），文档其他位置如 Args/Returns 仍以 Paddle 风格签名为准

```python
@overload
def broadcast_tensors(input: Sequence[Tensor], name: str | None = None) -> list[Tensor]: ...

@overload
def broadcast_tensors(*tensors: Tensor) -> list[Tensor]: ...

@variadic_tensor_decorator('input')
def broadcast_tensors(input: Sequence[Tensor], name: str | None = None) -> list[Tensor]:
    """
    Note:
        This API has two signatures:
        1. ``paddle.broadcast_tensors(input, name=None)`` (Paddle-style):
            Broadcast a list of tensors following broadcast semantics.
        2. ``paddle.broadcast_tensors(*tensors)`` (PyTorch-style):
            Broadcast variadic tensor arguments following broadcast semantics.

    Args:
        ...

    Returns:
        ...
    """
```

如果支持了 out 参数，必须在 API 文档中描述 out 参数，out 为 keyword-only 参数（*后面）时注意增加`Keyword Args:`，并在此部分描述。out 为位置参数时直接在 Args 部分描述。如下：

```python
# out 为 keyword-only 参数
def func(x, name=None, *, out=None):
    """
    ...

    Args:
        ...

    Keyword Args:
        out (Tensor|optional): The output tensor. Default: None.

    Returns:
        ...
    """
```

**注意事项**：
- Tensor 类方法（如 paddle.Tensor.abs）没有文档，无需处理，请勿与普通方法（如 paddle.abs）混淆
- Inplace 方法（如 paddle.abs_等下划线 API），只需要更新 API 签名，不需要修改文档

# 三、背景知识

## 3.1 Paddle API 分层结构

**Paddle API 架构（5 层）**：
1. **Python 层**：Python 函数定义（本方案修改层）
2. **Pybind 层**：Python 与 C++绑定（自动生成）
3. **Dygraph 层**：动态图前反向传播组合（自动生成）
4. **C++ API 层**：Kernel 选择调度（自动生成）
5. **Kernel 层**：实际计算逻辑实现（C++）

**本方案修改范围**：
- ✅ 仅修改第 1 层（Python 层）
- ❌ 第 2~4 层通过 yaml 配置自动生成，无需手动修改
- ❌ 第 5 层涉及 C++实现，不在本方案范围内

## 3.2 Python 装饰器原理

### 装饰器基本结构

```python
import functools
import inspect

def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 参数处理逻辑
        return func(*args, **kwargs)

    wrapper.__signature__ = inspect.signature(func)
    return wrapper
```

**关键点**：
- `@functools.wraps(func)`：保持原函数的`__name__`、`__doc__`等元信息
- `wrapper.__signature__`：保持函数签名，支持 IDE 的参数提示
- 装饰器必须返回 wrapper 函数

### 参数处理模式

**kwargs 别名映射**
```python
if "input" in kwargs and "x" not in kwargs:
    kwargs["x"] = kwargs.pop("input")
```

**位置参数类型检测**
```python
if len(args) >= 2 and isinstance(args[1], int):
    kwargs["x"] = args[0]
    kwargs["shape"] = list(args[1:])
    args = ()
```

## 3.3 装饰器特点与注意事项

### 装饰器特点
- **零侵入性**：无需修改 API 的实现代码
- **适用面广**：支持灵活处理各种 API 签名重载情况，如参数名不同、参数顺序不同、参数个数不同、参数类型不同、参数用法不同等
- **向后兼容**：保持 Paddle 原有 API 调用方式
- **开发效率**：相比 C++下沉方案，修改更快速直接

# 四、注意事项

1. 严格按标准工作流程执行，杜绝自行臆断和跳过步骤
2. 不要修改 sparse 目录下的 API
3. 确保不破坏现有功能，保持向后兼容性
4. 开发专用装饰器时参考现有实现
5. 代码中不允许提交中文，代码注释采用英文
6. 复盘记忆中的历史易错点，避免重复犯错

# 五、常见问题处理

## Q1：类型转换错误

**错误现象**：
```python
TypeError: expected Tensor as argument, got numpy.ndarray
```

**解决方法**：
```python
# 确保输入是 Tensor 类型
tensor_input = paddle.to_tensor(numpy_input)
paddle.api(tensor_input, ...)
```

---

## Q2：装饰器参数处理错误

**错误现象**：装饰器内参数转换失败或逻辑异常

**解决方法**：
1. 检查位置参数（args）与关键字参数（kwargs）的转换逻辑是否正确
2. 确保参数别名映射完整（包括所有别名情况）
3. 使用调试技巧进行追踪：

在装饰器中添加日志进行调试：

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 在装饰器中添加日志
def wrapper(*args, **kwargs):
    logging.debug(f"Before: args={args}, kwargs={kwargs}")
    # 处理逻辑...
    logging.debug(f"After: args={args}, kwargs={kwargs}")
    return func(*args, **kwargs)

# 或使用打印调试
def wrapper(*args, **kwargs):
    print(f"[DEBUG] args={args}, kwargs={kwargs}")
    # ...
```

---
