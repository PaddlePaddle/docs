---
name: modify-origin-api
description: 负责《Paddle API 对齐 PyTorch 项目》中 Step2：API 代码修改，实施『修改原有 API』方案。通过修改原有 Paddle API 的实现（新增参数、扩展参数类型/功能），使 Paddle API 与 PyTorch API 行为对齐，同时保持后向兼容性。
disable-model-invocation: false
---

# 一、适用场景

根据差异分析结果，本方案适用于以下四类场景：

| 场景 | 适用情况 | 典型示例 |
|------|---------|---------|
| 新增参数 | torch 比 paddle 多出某些参数（不含参数名/顺序差异） | `paddle.randn` 新增 `out/device/requires_grad/pin_memory` |
| 扩展参数类型/功能 | 参数支持的类型范围不同（torch 支持 paddle 不支持的类型） | `paddle.eye` 的 `num_rows` 从 `int` 扩展为 `int \| paddle.Tensor` |

# 二、标准工作流程

## Step 1：修改 API 代码

若尚未获得 PyTorch API 的相关信息，则自行获取，获取方式请参考`api-compatibility/SKILL.md` 中的「3.6 API 信息获取方式」章节。

然后分析 PyTorch API 的功能和行为，以及与 Paddle API 的差异。在 `${ROOT_DIR}/Paddle/python/paddle/` 目录下定位到对应的 API 实现位置，修改 Paddle API 的签名和实现代码。

**修改原则**：
1. **后向兼容**：修改不得破坏现有调用，禁止以下操作：
   - 改变已有参数顺序
   - 改变已有参数名称
   - 修改返回值类型
   - 删除现有参数
   - 修改现有参数的默认行为
2. **与 PyTorch 一致**：参数名、类型、顺序与 PyTorch 相同。若与原则 1 冲突，则放宽为：Paddle 签名需包含 PyTorch 签名的全部用法（例如：PyTorch 的某参数仅支持关键字传参，而 Paddle 可同时支持位置传参和关键字传参）

若不满足以上两个原则，则无法适用本方案，需考虑其他方案。

### 修改 API 签名

修改**现有 API 签名**：

```python
# 修改前
def randn(
    shape: ShapeLike,
    dtype: DTypeLike | None = None,
    name: str | None = None,
) -> Tensor:

# 修改后：新增 out/device/requires_grad 为 keyword-only 参数
def randn(
    shape: ShapeLike,
    dtype: DTypeLike | None = None,
    name: str | None = None,
    *,
    out: paddle.Tensor | None = None,
    device: PlaceLike | None = None,
    requires_grad: bool = False,
) -> Tensor:
```

### 修改函数实现逻辑

以下给出了几个常用参数的实现方式：

#### 新增 `out` 参数

**前置检查**：若 PyTorch API 有 `out` 参数，必须先确认其性质：
- **Keyword-only**（`*` 后）：`torch.api(input, *, out=None)` → Paddle 实现中 `out` 也应为 keyword-only（`*, out=None`）
- **位置参数**：`torch.api(input, out=None)` → Paddle 实现中 `out` 可为位置参数

**方式一：直接指定 out（推荐）**

适用条件：
1. 情况 1：API 最后一个逻辑是调用`_C_ops`；情况 2：API 调用了其他 API，调用的最后一个其他 API 也支持 out
2. out 参数只有一个 Tensor

```python
# 情况 1：API 最后一个逻辑是调用`_C_ops`
def less_than(x, y, name=None, *, out=None) -> Tensor:
    if in_dynamic_or_pir_mode():
        return _C_ops.less_than(x, y, out=out)
    else:
        ...

# 情况 2：API 调用的最后一个其他 API 也支持 out
def fft(x, n=None, axis=-1, norm="backward", name=None, *, out=None) -> Tensor:
    if is_integer(x) or is_floating_point(x):
        return fft_r2c(
            x, n, axis, norm, forward=True, onesided=False, name=name, out=out
        )
    else:
        return fft_c2c(x, n, axis, norm, forward=True, name=name, out=out)
```

**方式二：通过 assign 实现**

适用条件：不符合方式一的情况

```python
def func(x, axis=None, name=None, *, out: Tensor | None = None):
    # case1: 只有 1 个 out 的情况
    ret = <计算逻辑>
    if out is not None:
        paddle.assign(ret, out)
        return out
    return ret

    # case2: 有多个 out 的情况
    ret1, ret2 = <计算逻辑>
    if out is not None:
        paddle.assign(ret1, out[0])
        paddle.assign(ret2, out[1])
        return out
    return ret1, ret2
```

**注意事项**：
1. 需在 API 签名中增加 out 参数，`out`参数需与 Pytorch 用法一致，一般情况下 out 均是 keyword-only 参数（使用`*,`分隔），少数情况下 out 是位置参数
2. 处理 out 参数时，仅需处理 in_dynamic_or_pir_mode()分支下的逻辑，老静态图（LayerHelper）分支无需处理 out 参数

#### 新增 `device` 参数

替换原有的 `_current_expected_place()` 调用，根据 `device` 是否为 None 来决定使用哪个设备：

```python
# 修改前
place = _current_expected_place()

# 修改后
place = (
    _current_expected_place()
    if device is None
    else _get_paddle_place(device)
)
```

需确保 `_get_paddle_place` 已在文件中导入：
```python
from ..framework import (
    _get_paddle_place,
    ...
)
```

#### 新增 `requires_grad` 参数

在返回 tensor 前设置 `stop_gradient`：

```python
tensor = _C_ops.xxx(...)
if requires_grad is True:
    tensor.stop_gradient = False
return tensor
```

#### 新增 `pin_memory` 参数

`pin_memory` 参数已封装为 `_to_pinned_place` 函数，使用方式如下：

需确保 `_to_pinned_place` 已在文件中导入：
```python
from paddle.framework import (
    _to_pinned_place,
    ...
)
```

实现逻辑：

```python
# 第一处：处理 device 后，在调用 _C_ops 创建 tensor 前添加
if pin_memory and in_dynamic_mode() and device is not None:
    device = _to_pinned_place(device)

# ... 调用 _C_ops 创建 tensor ...

# 第二处：返回前
if requires_grad is True:
    tensor.stop_gradient = False
if pin_memory and in_dynamic_mode():
    tensor = tensor.pin_memory()
return tensor
```

#### 扩展参数类型

若现有实现已能通过 `_C_ops` 接受新类型（通常情况下 C++ 底层支持 `Scalar` 类型可同时接受 int 和 Tensor），则无需改动实现逻辑。

若底层不支持，添加类型转换分支：

```python
# 若底层需要 int，先将 Tensor 转换
if isinstance(num_rows, paddle.Tensor):
    num_rows = int(num_rows)
if isinstance(num_columns, paddle.Tensor):
    num_columns = int(num_columns)
```

## Step 2：更新函数文档

在 docstring 的 Args 部分（或 Keyword Args 部分，取决于参数是否为 keyword-only）添加新参数说明。

**文档更新模板**：

```python
"""
...
Args:
    shape (ShapeLike): ...
    dtype (DTypeLike|None, optional): ...
    name (str|None, optional): ...

Keyword Args:
    out (Tensor|None, optional): ...
    device (PlaceLike|None, optional): ...
    requires_grad (bool, optional): ...

Returns:
    ...
"""
```

**文档规范**：
- keyword-only 参数须放在 `Keyword Args:` 下描述
- Tensor 类方法（如 paddle.Tensor.abs）没有文档，无需处理
- Inplace 方法（如 paddle.abs_ 等下划线 API），只需更新 API 签名，不需要修改文档

# 三、背景知识

## 3.1 常用工具函数

```python
# 获取当前期望的 place
from paddle.framework import _current_expected_place
place = _current_expected_place()

# 将 device 字符串/对象转为 Place
from paddle.framework import _get_paddle_place
place = _get_paddle_place("cpu")    # -> core.CPUPlace()
place = _get_paddle_place("gpu:0")  # -> core.CUDAPlace(0)
place = _get_paddle_place("cuda:0") # -> core.CUDAPlace(0)（扩展后支持）

# 将 device 转为 pinned place（用于 pin_memory 参数）
from paddle.framework import _to_pinned_place
if pin_memory and in_dynamic_mode() and device is not None:
    device = _to_pinned_place(device)

# 判断运行模式
from paddle.framework import in_dynamic_mode, in_dynamic_or_pir_mode
if in_dynamic_mode(): ...
if in_dynamic_or_pir_mode(): ...
```

## 3.2 `pin_memory` 参数机制

`pin_memory=True` 将 Tensor 分配在锁页内存（pinned memory），提升 CPU->GPU 数据传输速度。

Paddle 已封装 `_to_pinned_place` 函数（参见 PR #78823），简化实现：

```python
from paddle.framework import _to_pinned_place

# 使用方式
if pin_memory and in_dynamic_mode() and device is not None:
    device = _to_pinned_place(device)

# 创建 tensor 后
if pin_memory and in_dynamic_mode():
    tensor = tensor.pin_memory()
```

## 3.3 `PlaceLike` 类型导入

若函数签名中需要使用 `PlaceLike` 类型，需在文件顶部的 `TYPE_CHECKING` 块中添加导入：

```python
if TYPE_CHECKING:
    from paddle import Tensor
    from paddle._typing import DTypeLike, PlaceLike, ShapeLike  # 添加 PlaceLike
```

# 四、注意事项

1. 严格按标准工作流程执行，杜绝自行臆断和跳过步骤
2. **后向兼容性是红线**：任何修改都不得破坏现有调用，新参数必须有合理默认值
3. 若新参数需要透传（如 `randn -> standard_normal -> gaussian`），必须在整个调用链上都添加该参数
4. `math_op_patch.py` 中的 Tensor 方法必须与对应的普通函数保持参数同步，且需同时修改 dygraph 版和 pir 版两个文件

# 五、常见问题处理

## Q1：新增参数后，`_C_ops` 调用报错 "unexpected keyword argument 'out'"

**错误现象**：
```
TypeError: xxx() got an unexpected keyword argument 'out'
```

**解决方法**：
该 op 对应的 `_C_ops` 接口不支持 `out` 参数。需确认底层 op 是否有 `out` 支持，若没有则不能使用 C++ 侧 out 注入方式，改为手动复制结果：
```python
tensor = _C_ops.xxx(...)
if out is not None:
    paddle.assign(tensor, output=out)
    return out
return tensor
```

## Q2：`pin_memory` 测试时报 "Pinning memory is not supported"

**错误现象**：
```
RuntimeError: Pinning memory is not supported for Place(cpu)
```

**解决方法**：
`pin_memory=True` 仅在 GPU/XPU 设备上有意义。确保测试逻辑中正确添加了环境判断：
```python
if paddle.device.is_compiled_with_cuda() or paddle.device.is_compiled_with_xpu():
    x = paddle.xxx([2], device="gpu", pin_memory=True)
    self.assertTrue("pinned" in str(x.place))
```

## Q3：修改 `math_op_patch.py` 后静态图测试失败

**错误现象**：
```
AttributeError: 'OpResult' object has no attribute 'pin_memory'
```

**解决方法**：
`pin_memory` 调用（`tensor.pin_memory()`）需在 `in_dynamic_mode()` 保护下执行，PIR 静态图中 `tensor` 是 `Value` 对象，不支持该方法。确认实现中有正确的模式判断：
```python
if pin_memory and in_dynamic_mode():
    tensor = tensor.pin_memory()
```

## Q4：扩展参数类型（如 `int | Tensor`）后运行报错

**错误现象**：
```
TypeError: eye(): argument 'num_rows' must be int, not Tensor
```

**解决方法**：
底层 `_C_ops` 接口尚不支持该类型。在 Python 层添加显式转换：
```python
if isinstance(num_rows, paddle.Tensor):
    num_rows = int(num_rows)
```
