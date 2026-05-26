# 新增 Python 层 API

本文档介绍如何在飞桨框架中开发 Python API，主要包括 API 代码开发、单元测试编写等内容。

## 一、获取并分析 PyTorch API 信息

若尚未获得 PyTorch API 的相关信息，则自行获取，获取方式请参考`api-compatibility/SKILL.md` 中的「3.6 API 信息获取方式」章节。

然后分析 PyTorch API 的功能和行为，在 Paddle 中新增对应的 API，使其与 PyTorch API 保持一致。具体包括：API 名称、调用路径、参数名及参数功能等。

## 二、开发 Python API 代码

Python API 的代码开发主要有两种方式：组合其他 Python API 实现和调用 C++ 算子接口实现。

### 2.1 方式一：组合其他 Python API

这种方式适用于可以通过组合已有 Python API 来实现的新 API。以 zeros 函数为例，它通过组合 fill_constant 实现：

```python
def zeros(
    shape: ShapeLike,
    dtype: DTypeLike | None = None,
    name: str | None = None,
) -> paddle.Tensor:
    if dtype is None:
        dtype = paddle.get_default_dtype()
    return fill_constant(value=0.0, shape=shape, dtype=dtype, name=name)
```

完整代码请参考 Paddle/python/paddle/tensor/creation.py。

### 2.2 方式二：调用 C++ 算子接口

如果 API 的实现需要调用 C++ 算子，则需要调用 C++ 算子对应的 Python C 函数。以 paddle.trace API 为例：

```python
def trace(
    x: Tensor,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
    name: str | None = None,
) -> Tensor:
    # 动静统一分支，直接调用算子对应的 Python C 函数
    if in_dynamic_or_pir_mode():
        return _C_ops.trace(x, offset, axis1, axis2)
```

完整代码请参考 Paddle/python/paddle/tensor/math.py。

关键要点：
- _C_ops：位于 Paddle/python/paddle/_C_ops.py，是从 Paddle 编译得到的二进制文件中导入的 Python C 函数，与 C++ 算子对应
- trace：Python API 对应的 C++算子名
- 参数顺序：(x, offset, axis1, axis2) 需按照 YAML 配置文件（paddle/phi/ops/yaml/ops.yaml）中定义的顺序传入

注意：目前飞桨动态图与 PIR 模式已统一，使用 in_dynamic_or_pir_mode() 进行判断即可，新增 API 时无需添加老静态图分支代码。

## 三、将 API 绑定为 Tensor 方法

Paddle 中的许多计算函数，既能够作为独立函数使用，也能作为 Tensor 的方法使用：

```python
x = paddle.randn([2, 3])
paddle.abs(x) # 与 x.abs() 等价
paddle.sin(paddle.abs(x)) # 与 x.abs().sin() 等价
paddle.sum(x, axis=0) # 与 x.sum(axis=0) 等价
```

当作为 Tensor 方法调用时，相当于自动把该 Tensor 作为独立函数的第一个参数传入。

**新增 Tensor 方法的详细说明请参考**：主 SKILL 文档的「3.5 类方法 API 实现原理」章节。

## 四、设置 API 公开名称与别名

API 开发完成后需关注：
- 公开 API 需加入对应目录 __init__.py 的 __all__ 列表
- 常用 API 可在更高层级建立别名，如 paddle.tensor 下的 API 可在 paddle 根目录建立别名

以 paddle.trace API 为例，其 trace 函数定义在 Paddle/python/paddle/tensor/math.py 中，又在 Paddle/python/paddle/tensor/__init__.py 中被 import，并且也在 python/paddle/__init__.py 中被 import。

```python
# Paddle/python/paddle/tensor/math.py
def trace(...):
    ...

# Paddle/python/paddle/tensor/__init__.py
from .math import trace

# Paddle/python/paddle/__init__.py
from .tensor.math import trace
```

以 paddle.trace 为例，设置 paddle.trace 为正式名称：
- 仅在 Paddle/python/paddle/__init__.py 的 __all__ 中加入 'trace'
- 不在 Paddle/python/paddle/tensor/__init__.py 和 Paddle/python/paddle/tensor/math.py 的 __all__ 中加入

说明：当出现类似把一个元素放入一个集中管理的列表的操作时，可以考虑按照字母表顺序插入列表中的合适位置。如果有多人同时新增 API 时，这样的方式比直接加在末尾更不容易出现冲突。

## 五、添加单元测试

### 5.1 Python API 单元测试

单测文件存放路径和命名方式：在 Paddle/test/legacy_test/ 目录下，以 test_xxx.py 的形式命名（假设 Python API 名为 xxx）。

Python API 的单元测试继承 unittest.TestCase，用 NumPy/SciPy 对应功能作为参考基准进行测试。参考示例：Paddle/test/legacy_test/test_activation_op.py。

开发步骤：
1. 用 NumPy/SciPy 实现用于对比结果的计算函数（NumPy/SciPy 有现成函数时可跳过这一步）
2. 在 setUp 函数中定义输入等相关属性参数
3. 实现动态图以及 PIR 分支单元测试代码

示例代码：

```python
# 使用 numpy 实现对比函数（NumPy 无现成函数时需自行实现）
def ref_hardtanh(x, min=-1.0, max=1.0):
    out = np.minimum(np.maximum(x, min), max)
    return out

class TestHardtanhAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(1024)
        self.x_np = np.random.uniform(-3, 3, [10, 12]).astype('float32')
        self.place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()

    # 静态图单测
    def test_static_api(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('X', [10, 12], dtype="float32")
                out1 = F.hardtanh(x)
                exe = paddle.static.Executor(self.place)
                res = exe.run(feed={'X': self.x_np}, fetch_list=[out1])
            np.testing.assert_allclose(ref_hardtanh(self.x_np), res[0], rtol=1e-05)

    # 动态图单测
    def test_dygraph_api(self):
        with dynamic_guard():
            x = paddle.to_tensor(self.x_np)
            out = F.hardtanh(x)
            np.testing.assert_allclose(ref_hardtanh(self.x_np), out.numpy(), rtol=1e-05)

    # 错误处理测试（可选）
    def test_errors(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                # 测试非法输入类型
                self.assertRaises(TypeError, F.hardtanh, 1)
```

开发要点：
- 必须添加动态图和静态图测试用例，确保对应情况工作正常
- 通常无需测试反向计算（C++ 算子单测已覆盖反向算子功能）
- 数值对比用 numpy.testing.assert_allclose(actual, desired) 或 numpy.allclose(actual, desired)
- 每个 case 开头需用 static_guard() 或 dynamic_guard() 显式切换运行模式

### 5.2 编译并运行单测（每次修改均需执行）

单测编写完成后，按以下命令验证执行：

```bash
cd ${ROOT_DIR}/Paddle/build
cmake .. && make -j$(nproc)
python test_xxx.py
```

根据报错信息修改代码，确保所有测试用例通过。每次修改后均需要重新执行本步骤。

**编译注意事项**：
- 无需重装，直接生效（勿执行 setup/install 等安装操作）
- 勿删除 build 目录（否则增量编译失效，编译时间极长）

## 六、新增 API 英文文档

完成 API 代码开发后，需要新增 API 英文文档。详细规范请参考：API 文档书写规范（references/api_docs_guidelines.md）
