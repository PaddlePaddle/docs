---
name: add-new-api
description: 负责《Paddle API 对齐 PyTorch 项目》中 Step2：API 代码修改，实施『新增 API』方案。通过新增 Paddle API（新增 API 别名、新增 Python 层 API、新增 C++算子），覆盖 Pytorch API 调用路径，实现与 PyTorch API 行为对齐。
context: fork
disable-model-invocation: false
---

# 一、适用场景

根据 API 的性质，将新增场景分为三种模式：

**场景一：新增 API 别名**
- 适用：目标 API 已有对应 Paddle 实现，仅名称/路径不同
- 是否需要编译：不需要
- 修改文件数：1~5 个 .py
- 实现难度：简单
- 典型示例：ne、cat、paddle.optim、paddle.distributions

**场景二：新增 Python 层 API**
- 适用：需要新 API 逻辑，可纯 Python 实现（不需要开发新的 C++ 算子，可用其他 Python API 组合实现）
- 是否需要编译：不需要
- 修改文件数：3~6 个 .py
- 实现难度：中等
- 典型示例：argwhere、from_numpy

**场景三：新增 C++ 算子**
- 适用：需要新的底层计算逻辑，涉及 C++/CUDA（需要开发新的 C++ 算子，用 C++ 开发算子实现代码，再封装 Python API）
- 是否需要编译：必须重新编译
- 修改文件数：10~18 个（.py/.h/.cc/.cu/.yaml）
- 实现难度：复杂
- 典型示例：aminmax、addcmul

# 二、标准工作流程

根据实际需求，选择场景一、场景二或场景三执行。所有场景完成后，统一执行新增 API 英文文档和编译运行测试。

## 场景一：新增 API 别名

新增的 API 别名应统一在 `__init__.py` 中定义，便于统一管理，避免分散在各个实现文件中，与现有别名管理保持一致（位于 Paddle/python/paddle/__init__.py、Paddle/python/paddle/tensor/__init__.py、Paddle/python/paddle/nn/functional/__init__.py、Paddle/python/paddle/nn/__init__.py 等文件中）。

### 类型一：函数/类别名

适用条件：目标 PyTorch API 与某个已有 Paddle API 功能完全一致，仅名称不同。

典型示例：paddle.ne（对齐 torch.ne）、paddle.cat（对齐 torch.cat）、paddle.nn.SiLU（对齐 torch.nn.SiLU）

请严格按以下步骤依次执行：

#### Step 1：在模块文件中添加别名赋值

找到原始 API 的实现位置（如 python/paddle/tensor/math.py），在文件末尾添加别名赋值：

```python
# python/paddle/tensor/math.py 末尾
ne = not_equal
lt = less_than
less = less_than
le = less_equal
greater = gt
ge = greater_equal
```

若属于 paddle.nn 命名空间（如 SiLU = Silu），则在对应的 nn/__init__.py 中添加：

```python
# python/paddle/nn/__init__.py
SiLU = Silu
```

#### Step 2：在 __init__.py 中导出别名

在顶层 python/paddle/__init__.py 的对应 from paddle.tensor import (...) 块内，按字母序插入新别名：

```python
# python/paddle/__init__.py
from paddle.tensor import (
    ...
    ge,
    greater,
    le,
    less,
    lt,
    ne,
    ...
)
```

#### Step 3（如需）：添加 Tensor 方法别名

若需要以 paddle.Tensor.xxx 形式调用（对齐 torch.Tensor.xxx），在 python/paddle/tensor/__init__.py 中同步处理：

```python
# python/paddle/tensor/__init__.py
# 1. 在 tensor_method_func 列表中按字母序添加方法名
tensor_method_func = [
    ...
    'ne',
    'lt',
    'less',
    ...
]

# 2. 在文件末尾与模块导出一起声明别名（作为 Tensor 方法注入来源）
ne = not_equal
lt = less_than
```

Paddle 通过 patch 机制将 paddle.tensor 模块中的函数动态 setattr 到 paddle.Tensor，因此只需在此处声明即可自动注入为 Tensor 方法，无需修改 class Tensor 定义。

### 类型二：类方法/属性别名

适用条件：需要对齐类的方法或属性，根据已有实现与目标形式的性质差异，分为三种情况处理。

#### 情况一：方法 → 方法 或 属性 → 属性（性质相同）

当已有实现与目标形式性质相同时（都是方法或都是属性），直接使用别名赋值即可。

典型示例：
```python
# 方法 → 方法
get_submodule = get_sublayer
set_submodule = set_sublayer

# 属性 → 属性
grad = _grad
```

操作步骤：在目标类中直接添加别名赋值。

#### 情况二：方法 → 属性

当已有实现是方法（需调用），目标形式是属性（直接访问）时，需使用 `@property` 装饰器包装原有方法。

典型示例：PyLayerContext.saved_tensors（对齐 torch.saved_tensors，原有 saved_tensor() 需加括号调用）

```python
# python/paddle/autograd/py_layer.py

class PyLayerContext:
    def saved_tensor(self):
        # original method implementation
        ...

    @property
    def saved_tensors(self):
        return self.saved_tensor()  # delegate to existing method
```

#### 情况三：属性 → 方法

当已有实现是属性（直接访问），目标形式是方法（需调用）时，需定义新方法返回属性值。

典型示例：假设 Paddle 有属性 `x.shape`，需对齐 PyTorch 方法 `x.get_shape()`：

```python
class Tensor:
    @property
    def shape(self):
        # original property implementation
        ...

    def get_shape(self):
        """New method that delegates to the existing property."""
        return self.shape
```

若目标类是 `paddle.Tensor`，由于 Python 层无对应 class 定义，还需额外通过 patch 机制动态注入上述别名。

**新增 Tensor 方法的详细说明请参考**：主 SKILL 文档的「3.5 类方法 API 实现原理」章节。

### 类型三：命名空间别名

适用条件：PyTorch 拥有某个 Paddle 没有的命名空间，需创建新包将已有 API 重新组织暴露。

典型示例：paddle.optim（对齐 torch.optim）、paddle.utils.data（对齐 torch.utils.data）、paddle.distributions（对齐 torch.distributions）

（详细步骤待补充）

## 场景二：新增 Python 层 API

适用条件：PyTorch 有某个 API，Paddle 没有，但可以基于已有 Paddle API 组合实现，无需新增 C++ 算子。

典型示例：paddle.argwhere（= paddle.nonzero(input, as_tuple=False)）、paddle.from_numpy、paddle.asarray

详细开发流程请参考：开发 API Python 端（references/new_python_api.md）

## 场景三：新增 C++ 算子

适用条件：PyTorch 有某个计算 OP，Paddle 完全没有对应实现，需要从底层开始完整新增算子。

典型示例：paddle.aminmax（对齐 torch.aminmax）、paddle.addcmul（对齐 torch.addcmul）

详细开发流程请参考：开发 C++ 算子（references/new_cpp_op.md）

# 三、背景知识

代码文件存放位置应尽可能与已有 API 的目录保持一致。下方介绍了 `python/paddle/tensor/` 目录的组织结构，供开发者参考。

大部分常用的数组运算 API（在 NumPy 中有功能相似的 numpy.xxx API）都放在 python/paddle/tensor 目录下：
- array.py：TensorArray 相关操作
- attribute.py：Tensor 元数据操作：is_complex, is_integer, shape, rank 等
- creation.py：Tensor 创建：to_tensor, ones, full_like 等
- einsum.py：einsum 运算
- linalg.py：线性代数：matmul, norm, det
- logic.py：逻辑运算：logical_and, allclose, greater_than
- manipulation.py：数组元素操作：concat, stack, transpose 等
- math.py：算术运算：加减乘除、三角函数、sum、cumsum 等
- random.py：随机数：randn, uniform
- search.py：搜索排序：argsort, argmin
- stat.py：统计：mean, var, std
- to_string.py：Tensor 打印功能

python/paddle/nn/functional 目录中包含用于神经网络的函数，如 batch_norm、conv2d，这些在 NumPy 中通常没有直接对应的函数。


# 四、注意事项

1. 严格按标准工作流程执行，杜绝自行臆断和跳过步骤
2. 代码中不允许提交中文，代码注释与 docstring 均使用英文
3. 别名添加需注意顺序：在 __init__.py 中按字母序插入，保持代码风格一致
4. Inplace 别名注意：mul_ 等 inplace 方法需确保原方法支持对应调用形式；inplace API 无需测试静态图
5. 新命名空间不要遗漏注册：创建新包后，务必在 setup.py.in 和 setup.py 中同步注册


# 五、常见问题处理

## Q1：Tensor 方法别名未生效（AttributeError）

错误现象：
```python
x = paddle.to_tensor([1, 2, 3])
x.ne(y)  # AttributeError: 'Tensor' object has no attribute 'ne'
```

解决方法：
1. 检查 python/paddle/tensor/__init__.py 中 tensor_method_func 列表是否包含 'ne'
2. 检查 python/paddle/tensor/math.py 末尾是否有 ne = not_equal
