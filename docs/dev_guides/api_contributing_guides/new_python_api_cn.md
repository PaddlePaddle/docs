# 开发 API Python 端

## 新增飞桨 API 的场景与意义
API 作为用户使用飞桨框架的接口，承接着实现用户模型开发需求的重要作用。虽然目前飞桨已经提供了一千多个 API 用于支持各类场景下的模型开发，但在某些前沿邻域模型的探索中仍然可能会遇到框架已提供的 API 不足以支撑开发需求的情况，此时就可以通过在飞桨框架中新增 API 来解决这类问题。
开发飞桨 API 可以加深对深度学习框架底层架构的理解，提升技术视野，同时也是在为深度学习框架开源社区的发展提供助力，让更多的 AI 开发者享受到 AI 基础设施带来的便利。

新增飞桨 API 主要包含两种情况：

1. 不需要开发新的 C++ 算子，可以用其他 Python API 组合得到新的 API，只写 Python 代码即可。
2. 需要开发新的 C++ 算子，需要用 C++ 开发算子实现代码、再封装 Python API 代码。

两种情况下均有 Python 端的开发工作。本文将介绍开发新的飞桨 API 时，需要完成的 Python 端开发内容以及注意事项。

## 一、开发前准备

开发代码前请确认：

- 已签署 [贡献者许可协议（Contributor License Agreement，CLA）](https://cla-assistant.io/PaddlePaddle/Paddle)；

- 已阅读 [代码贡献流程](..\code_contributing_path_cn.html)、[贡献前阅读](read_before_contributing_cn.html) 和相关代码规范；

- 已根据 [API 设计和命名规范](api_design_guidelines_standard_cn.html) 确定了新增 API 的名称和存放位置；

- 已提交 [API 设计文档](read_before_contributing_cn.html#apiDesignDoc) 并通过评审；

- 已将 [PaddlePaddle/Paddle](https://github.com/PaddlePaddle/Paddle) 仓库的代码获取到本地，准备好了 Paddle 开发环境。

## 二、 开发 Python API 代码

### 2.1 确定文件位置和 API 名称

提交飞桨 API 设计文档时，就需要参考 [API 设计和命名规范](api_design_guidelines_standard_cn.html) 确定 Python API 的代码文件存放位置和 API 名称。按照已有设计，在 [python/paddle](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle) 目录下的相应子目录中添加新的 `.py` 代码文件，遵循相似功能的 API 放在同一文件夹的原则。

比如，大部分常用的数组运算 API（在 numpy 中有功能相似的 `numpy.***` API ）都放在 `python/paddle/tensor` 目录下。具体的功能细分如下：

| **文件**        | **功能**                                                     |
| --------------- | ------------------------------------------------------------ |
| array.py        | TensorArray 相关的操作                                       |
| attribute.py    | Tensor 元数据相关的操作，比如数据类型判断，`is_complex`, `is_integer`, 元数据获取，`shape`, `rank` 等 |
| creation.py     | Tensor 创建类，比如 `to_tensor`, `ones`, `full_like` 等      |
| einsum.py       | einsum 运算                                                  |
| linalg.py       | 线性代数类运算，比如 `matmul`, `norm`, `det`                 |
| logic.py        | 逻辑类运算，比如 `logical_and`, `allclose`, `greater_than`   |
| manipulation.py | 非算术运算类的数组元素操作，比如拼接 `concat`，堆叠`stack`，转置`transpose` 等 |
| math.py         | 逐元素算术运算，比如加减乘除，三角函数等；规约类算术运算，比如 `sum`；扫描类算术运算，比如 `cumsum` |
| random.py       | 随机数发生类函数，比如 `randn`, `uniform`，注意和 creation 中的区别 |
| search.py       | 搜索，排序，比如 `argsort`, `argmin`                         |
| stat.py         | 统计类，比如 `mean`, `var`, `std`                            |
| to_string.py    | Tensor 的打印相关功能                                        |

与 `paddle/tensor` 功能类似，`paddle.nn.functional` 中也包含许多用于操作 tensor 的函数，但是这里主要是放一些更常用于神经网络中的函数，比如 `batch_norm`, `conv2d`，这些往往可能在 numpy 中没有直接对应的函数。

> 说明：写新的 API 时可以参考该 API 的功能和哪一类更为相似，如果有不确定的情况，请 [新建 ISSUE](https://github.com/PaddlePaddle/Paddle/issues/new?assignees=&labels=type%2Ffeature-request%2Cstatus%2Fnew-issue&template=2_feature-request.yml) 说明。

### 2.2 Python API 的代码开发示例

先看一个简单的 Python API 的代码样例，如图 1 所示，可以看到主要包括以下几部分：

- **函数定义**：定义 Python 接口函数。
- **英文文档**：API 的英文文档直接写在 `.py` 代码文件中，如下图所示；API 的中文文档则写到 [PaddlePaddle/docs](https://github.com/PaddlePaddle/docs) 仓库中。
- **代码示例**：该 API 的使用示例代码。
- **函数主体代码**：包括输入参数的检查、调用算子的执行逻辑等内容。

<center><img src="https://github.com/PaddlePaddle/docs/blob/develop/docs/dev_guides/api_contributing_guides/images/zeros_python_api.png?raw=true" width="900px" ></center>

<center>图 1 Python API 代码样例</center>

接下来通过两个代码示例，介绍 Python API 的函数主体代码开发的一些惯例，以及用到的主要函数类的接口。

> 说明：因为飞桨框架同时支持动态图和静态图，因此通常情况下，飞桨 API 需要实现动态图分支和静态图分支，不同分支下的行为是保持一致的，并且对外统一成一个 API 接口。

#### 2.2.1 代码示例一（组合其他 Python API ）

如图 1 所示，zeros 函数是通过组合 fill_constant 实现的，并且 fill_constant 里已经处理了动态图和静态图的情况，所以直接调用即可。这就是组合其他 Python API 实现的例子。

```python
def zeros(shape, dtype=None, name=None):
    # 为了突出重点，省略中间的文档和示例部分
    if dtype is None:
        dtype = 'float32'
    return fill_constant(value=0.0, shape=shape, dtype=dtype, name=name)
```
【代码仓库链接】

- [zeros 示例代码](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/creation.py#L612)
- [fill_constant 示例代码](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/layers/tensor.py#L718)

#### 2.2.2 代码示例二（调用 C++ 算子接口）

如果 API 的实现中需要调用 C++ 算子，则需要分别实现动态图分支和静态图分支的代码（由于飞桨框架同时支持动态图和静态图两种训练模式，动态图和静态图在执行逻辑上有所差异，需要在 Python 端根据当前的运行模式选择进入到对应的执行分支去处理）。

接下来以 [paddle.trace](../../api/paddle/trace_cn.html) API 的实现代码为例，分别介绍动态图分支和静态图分支的开发要点。

【代码仓库链接】[trace 示例代码](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/math.py#L2910)


```python
def trace(x, offset=0, axis1=0, axis2=1, name=None):
    # 为了突出重点，省略部分代码
    # 动态图分支，直接调用算子对应的 Python C 函数
    if in_dygraph_mode():
        return _C_ops.trace( x, offset, axis1, axis2 )

    # 静态图分支
    ## 输入参数检查
    __check_input(x, offset, axis1, axis2)

    ## 构造输出，添加 op，返回输出
    helper = LayerHelper('trace', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type='trace',
        inputs={'Input': [x]},
        attrs={'offset': offset,
               'axis1': axis1,
               'axis2': axis2},
        outputs={'Out': [out]})
    return out
```

**（1）动态图分支**

截取上面示例中动态图相关代码如下：

```python
    # 动态图分支，直接调用算子对应的 Python C 函数
    if in_dygraph_mode():
        return _C_ops.trace( x, offset, axis1, axis2 )
```

动态图分支的写法一般是调用 C++ 算子对应的 Python C 函数，示例中调用名为 `trace` 的 算子，使用 `_C_ops.trace`，然后传入参数。

  - `_C_ops` 是 [python/paddle/_C_ops.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/_C_ops.py)，其实现了从 Paddle 编译得到的二进制文件中 import  C++ 算子对应的 Python C 函数。
  - `trace` 是算子的 Python C 函数名。Python C 函数的命名直接采用算子名。
  - 参数 `( x, offset, axis1, axis2 )`需按照 [YAML 配置文件](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/api/yaml/ops.yaml#L185) 中定义的输入参数顺序传入，C++ 算子的输入、输出和属性等描述是通过 YAML 配置文件定义的，具体可参见 [开发 C++ 算子](new_cpp_op_cn.html) 章节介绍。

> 注意：由于目前飞桨动态图仅支持新动态图，通过 `in_dygraph_mode()` 去使用，`_in_legacy_dygraph()`为旧动态图开关已被遗弃，**在新增算子时无需添加旧动态图分支代码**。

**（2）静态图分支**

截取上面示例中静态图相关代码如下：

```python
    # 静态图分支
    ## 输入参数检查
    __check_input(x, offset, axis1, axis2)

    ## 构造输出，添加 OP，返回输出
    # LayerHelper 是一个用于创建 OP 输出变量、向静态图 program 中添加 OP 的辅助工具类
    helper = LayerHelper('trace', **locals())
    # 创建输出 Tensor
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    # 将输入 Tensor，输出 Tensor, 非 Tensor 的 attributes 以三个字典的形式，作为参数添加 OP
    helper.append_op(
        type='trace',
        inputs={'Input': [x]},
        attrs={'offset': offset,
               'axis1': axis1,
               'axis2': axis2},
        outputs={'Out': [out]})
    return out
```

对于静态图，一般分为输入参数检查、创建输出 Tensor、添加 OP 几个步骤。

- **输入参数检查：** 包括必要的类型检查、值检查，以及输入 Tensor 的 shape、dtype 等检查，确保组网能正常运行等，这里的参数检查可以帮助用户尽早的暴露问题并修正，从而降低模型的开发调试成本。
  - 输入参数的检查一般仅在静态图分支中使用。主要原因是静态图下该函数仅在模型组网时执行一次，运行期不会再执行；而动态图下该函数会被多次执行，Python 端过多的输入检查会影响执行效率。并且由于动态图即时执行的优势，如果发生错误也可以通过分析 C++ 端的报错信息定位问题。
  - 示例中输入参数检查的代码逻辑比较复杂但仅用于 `trace` 函数，因此在该函数内定义一个检查输入参数的函数 `__check_input`，代码见下文。
- **创建输出 Tensor ，添加 OP：**
  - 先创建 LayerHelper 对象，再使用 LayerHelper 对象创建输出 Tensor（[LayerHelper](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/layer_helper.py) 是一个用于创建 OP 输出变量、向 静态图 Program 中添加 OP 的辅助工具类）。
  - 在 `append_op` 添加 `inputs` 和 `outputs` 项，其中的 key 值（静态图中变量名）一般与 Python 接口中定义的输入输出 Tensor 变量名的命名相同。（注意：这里 `trace` 中的 `Input` 没有与 Python 接口中 `x` 命名直接对应是由于为了兼容旧算子体系下 `trace` 算子的定义实现而做了额外的映射，新增算子时无需考虑这种情况。）

输入参数检查的 `__check_input` 函数代码如下所示，其中检测 Tensor 的数据类型可以用 [check_variable_and_dtype](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/data_feeder.py#L80) 或 [check_type](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/data_feeder.py#L128) 函数进行检测。

```python
def __check_input(x, offset, axis1, axis2):
        # 检查输入 x 的 dtype 是否在要求范围内
        check_dtype(x.dtype, 'Input',
                    ['int32', 'int64', 'float16', 'float32', 'float64'],
                    'trace')
        # 检查输入 x 的维度信息
        input_shape = list(x.shape)
        assert len(input_shape) >= 2,                     \
                "The x must be at least 2-dimensional, "   \
                "But received Input x's dimensional: %s.\n" %  \
                len(input_shape)

        axis1_ = axis1 if axis1 >= 0 else len(input_shape) + axis1
        axis2_ = axis2 if axis2 >= 0 else len(input_shape) + axis2
        # 检查参数值是否有效
        assert ((0 <= axis1_) and (axis1_ < len(input_shape))),     \
            "The argument axis1 is out of range (expected to be in range of [%d, %d], but got %d).\n"  \
            % (-(len(input_shape)), len(input_shape) - 1, axis1)

        assert ((0 <= axis2_) and (axis2_ < len(input_shape))),   \
            "The argument axis2 is out of range (expected to be in range of [%d, %d], but got %d).\n"   \
            % (-(len(input_shape)), len(input_shape) - 1, axis2)


        assert  axis1_ != axis2_,   \
               "axis1 and axis2 cannot be the same axis." \
                "But received axis1 = %d, axis2 = %d\n"%(axis1, axis2)
```

### 2.3 将 API 绑定为 Tensor 的方法

**（1）背景介绍**

Paddle 中的许多计算函数，既能够作为独立函数使用，也能作为 `Tensor` 的方法使用。作为 `Tensor` 方法使用则可以更方便地链式调用。例子如下：

```python
x = paddle.randn([2, 3])

paddle.abs(x) # 与 x.abs() 等价
paddle.sin(paddle.abs(x)) # 与 x.abs().sin() 等价

paddle.sum(x, axis=0) # 与 x.sum(axis=0) 等价
```

这两种使用方式的对应规则是，当作为 `Tensor` 方法调用时，相当于自动把该 Tensor 作为独立函数的第一个参数传入，其余参数的传入则和独立函数的使用一致。目前 [python/paddle/tensor](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/tensor) 子目录下的许多 API 都支持这样的调用方式。

**（2）具体做法**

如需让新增的函数支持作为 `Tensor`  方法调用，则需要将函数名添加到 `Python/paddle/tensor/__init__.py` 中的 `tensor_method_func` 列表中。具体的做法是：


  1. 在 [Python/paddle/tensor/__init__.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/__init__.py) 中 import 所需的函数；
  2. 然后将其名字加入 `tensor_method_func` 列表。


```plain
# import 所需函数
from .math import trace
# 加入 tensor_method_func 列表
tensor_method_func = [
    'trace',
]
```

### 2.4 将 API 加入公开 API 列表并设置正式名称

**（1）背景介绍**

根据 [API 设计和命名规范](api_design_guidelines_standard_cn.html)，API 的代码开发完成并加入对应目录/文件中后，还有两个开发要点需关注：

  - 新开发的 API 如果需要公开，需加入公开 API 列表，一般添加到对应目录下 `__init__.py`文件的`__all__` 列表中；非公开 API 不能添加到 `__all__` 列表中。
  - 常用的 API 可以在更高层级建立别名，比如： `paddle.tensor` 目录下的 API，均在 `paddle` 根目录建立别名，其他所有 API 在 `paddle` 根目录下均没有别名。并且有多个别名时需设置一个推荐的名称，作为正式名称。

建立别名的方法可以参考如下 Python 的用法。在 Python 中，如果模块 `a` 中导入了模块 `b` 提供的函数或者类 `f`，那么开发者想要使用 `f`，既可以从模块 `a` 中导入，也可以从模块 `b` 中导入。

```python
# b.py，模块 b 中定义了 f
def f():
  pass

# a.py，模板 a 中导入了 b 提供的 f
from b import f

# client.py，在使用时，既可以从 a 中导入 f 也可以从 b 中导入 f
from b import f # it's ok
from a import f # it's ok, too
```

**（2）具体做法**

  - 一些常用的 Paddle API 可先参考上述方法建立别名，比如前文示例中  `paddle.trace `  API 的 `trace` 函数定义在 [Python/paddle/tensor/math.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/math.py#L2784) 中，又在 [Python/paddle/tensor/__init__.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/__init__.py) 中被 import，并且也在 [Python/paddle/__init__.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/__init__.py) 中被 import。

```python
# Python/paddle/tensor/math.py
def trace(...):
        ...

# Python/paddle/tensor/__init__.py
from .math import trace

# Python/paddle/__init__.py
from .tensor.math import trace
```

如此设置，`import paddle` 之后，可以通过 `paddle.trace`, `paddle.tensor.trace` 和 `paddle.tensor.math.trace` 多个名称来调用这个函数，即该 API 有多个名称，但是推荐使用 `paddle.trace`这个更简洁的名称作为正式名称。

  - 设置 `paddle.trace` 作为正式名称，具体做法是：
    - 仅在 `Python/paddle/__init__.py` 文件的 `__all__` 列表中加入 `'trace'`；
    - 不在 `Python/paddle/tensor/__init__.py` 和 `Python/paddle/tensor/math.py` 的 `__all__` 列表中加入 `'trace'`。

> 说明：当出现类似把一个元素放入一个集中管理的列表的操作时，可以考虑按照字母表顺序插入列表中的合适位置。因为如果有多人同时新增 API 时，这样的方式比直接加在末尾更不容易出现冲突。

## 三、开发单元测试代码

### 3.1 添加 C++ 算子单元测试

**（1）文件存放路径和命名方式**

在 [python/paddle/fluid/tests/unittests](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests) 目录下，一般以 `test_xxx_op.py` 的形式命名（假设算子名为`xxx`），与 Python API 的单元测试文件命名为相同的前缀。

**（2）C++ 算子单元测试的开发指导**

相关的开发指导和规范可以参考：

  - [C++ 算子开发指南-添加单元测试](new_cpp_op_cn.html#tianjiadanyuanceshi)
  - [Op 开发手册(Operator Development Manual)](https://github.com/PaddlePaddle/Paddle/wiki/Operator-Development-Manual-Index)

在此不作展开，本文主要讲述 Python API 的单元测试。

### 3.2 添加 Python API 单元测试

**（1）文件存放路径和命名方式**

在 [python/paddle/fluid/tests/unittests](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests) 目录下，一般以 `test_xxx.py` 的形式命名（假设算子名为`xxx`）。

如果为这个 API 也开发了对应的 C++ 算子，那么也可以把对 Python API 的单元测试和 C++ 算子的单元测试写在同一个文件中，一般以 `test_xxx_op.py` 的形式命名。

**（2）Python API 单测开发及验收规范**

请遵循飞桨的 [API 单测开发及验收规范](api_accpetance_criteria_cn.html)，需提前阅读。

> 特别注意：单元测试要求新增代码单元测试行覆盖率达到 90%，可在 [CI 测试](../git_guides/paddle_ci_manual_cn.html) 的 PR-CI-Coverage 测试项中查看覆盖率。

**（3）Python API 单元测试的开发指导**

Python API 的单元测试直接继承 Python 内置的 `UnitTest.TestCase` 类，一般来说需要用 NumPy/SciPy 中的对应功能作为参考，如果 NumPy/SciPy 中没有现成的对应函数，可以用 NumPy/SciPy 实现一个作为参考，并以这个为基准对新增的 Python API 进行测试，如 [test_activation_op.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/unittests/test_activation_op.py#L845) 中 paddle.nn.Hardtanh API 的单元测试，代码如下所示。

**开发步骤：**

   1. 用 NumPy/SciPy 实现用于对比结果的计算函数（NumPy/SciPy 有现成函数时可跳过这一步）；
   2. 在 `setUp` 函数中定义输入等相关属性参数；
   3. 实现静态图单元测试代码；
   4. 实现动态图单元测试代码。

```python
# 使用 numpy 实现 hardtanh 函数，用于对比结果
def ref_hardtanh(x, min=-1.0, max=1.0):
    out = np.copy(x)
    out[np.abs(x - min) < 0.005] = min + 0.02
    out[np.abs(x - max) < 0.005] = max + 0.02
    out = np.minimum(np.maximum(x, min), max)
    return out

class TestHardtanhAPI(unittest.TestCase):
    # test paddle.nn.Hardtanh, paddle.nn.functional.hardtanh
    def setUp(self):
        np.random.seed(1024)
        self.x_np = np.random.uniform(-3, 3, [10, 12]).astype('float32')
        self.place=paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    # 静态图单测
    def test_static_api(self):
        # 开启静态图模式
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', [10, 12])
            out1 = F.hardtanh(x)
            m = paddle.nn.Hardtanh()
            out2 = m(x)
            exe = paddle.static.Executor(self.place)
            # 计算静态图结果
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
        out_ref = ref_hardtanh(self.x_np)
        for r in res:
            # 对比静态图与 numpy 实现函数计算结果是否相同
            self.assertEqual(np.allclose(out_ref, r), True)

    # 动态图单测
    def test_dygraph_api(self):
        # 关闭静态图模式
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        # 测试动态图 F.hardtanh 和 paddle.nn.Hardtanh 计算结果
        out1 = F.hardtanh(x)
        m = paddle.nn.Hardtanh()
        out2 = m(x)
        out_ref = ref_hardtanh(self.x_np)
        for r in [out1, out2]:
            self.assertEqual(np.allclose(out_ref, r.numpy()), True)

        out1 = F.hardtanh(x, -2.0, 2.0)
        m = paddle.nn.Hardtanh(-2.0, 2.0)
        out2 = m(x)
        out_ref = ref_hardtanh(self.x_np, -2.0, 2.0)
        for r in [out1, out2]:
            self.assertEqual(np.allclose(out_ref, r.numpy()), True)
        paddle.enable_static()
```

**开发要点：**


  - 无论是用其他 Python API 组合得到新的 API，还是封装新开发 C++ 算子得到的新 API，都必须添加动态图和静态图的测试用例，确保对应情况工作正常，结果符合预期。
  - 通常情况下新增 Python API 的单元测试，可以不必测试反向计算功能，因为在 C++ 算子的单元测试中会包含反向算子功能的测试。
  - 用 NumPy/SciPy 的实现对比时，一般用 `self.assertTrue(numpy.allclose(actual, desired))` 或者 `numpy.testing.assert_allclose(actual, desired)` 来进行数值对比。其中，`numpy.testing.assert_allclose` 相对误差和绝对误差是 `rtol=1e-07, atol=0`；`numpy.allclose` 的相对误差和绝对误差是 `rtol=1e-05, atol=1e-08`，前者比后者更严格。一般进行单元测试的时候，都使用默认的误差阈值，如需设置自定义的阈值，需要说明原因。
  - 因为单元测试各个 case 的运行次序是不确定的，为了保证不同的测试 case 运行在正确的运行模式（动态图/静态图）上，常见的做法有：
    - 在每个测试 case 的起始部分，显式切换 paddle 的运行模式，用`paddle.enable_static` 和 `paddle.disable_static` 分别激活和取消静态图模式。如前文代码所示，在 `test_static_api` 和 `test_dygraph_api` 的开头分别切换了状态。

    - 将静态图和动态图测试定义为不以 `test` 开头的函数（如 [test_l1_loss.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/unittests/test_l1_loss.py#L77) 中定义为 `run_imperative`、`run_static` 函数)，然后定义一个 test 开头的函数，切换不同的状态去运行它。


      ```python
       def test_cpu(self):
           # 关闭静态图模式，测试动态图模式
           paddle.disable_static(place=paddle.fluid.CPUPlace())
           self.run_imperative()
           # 开启静态图模式，测试静态图模式
           paddle.enable_static()

           with fluid.program_guard(fluid.Program()):
               self.run_static()
      ```

    - 将动态图和静态图的测试 case 分在不同的 Python 文件中，`import paddle` 后在模块级别设置 paddle 的运行模式。比如 [test_rnn_cells.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/unittests/rnn/test_rnn_cells.py) 和 [test_rnn_cells_static.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/unittests/rnn/test_rnn_cells_static.py) 的做法。

    - 在测试模块级别设定 paddle 的运行模式为静态图（一般是在一个模块的开始，而不是写在 `if __name__=="__main__":` 里)。然后在需要使用动态图的 case 里，将动态图部分的代码至于 `dygraph.guard` 上下文管理器内。这是老式的写法，目前不再推荐这么写，但已有的代码库中也存在这样的模式。


### 3.3 运行单元测试

代码开发完成后，需要从源码编译 Paddle，并调试开发的功能。

**(1) 本地编译 Paddle**

编译方法请参见 [从源码编译](../../install/compile/fromsource.html) 章节，推荐使用 Docker 编译的方式。Docker 环境中已预装好编译 Paddle 需要的各种依赖，相较本机编译更便捷。

> 注意：编译必须打开 WITH_TESTING 选项（`-DWITH_TESTING=ON`），以确保新增的单元测试文件（python/paddle/fluid/tests/unittests/ 目录下 test_*.py 文件）自动加入工程进行编译。

运行单元测试需要在 `build` 目录下，以 `ctest ${test_name}` 的命令运行。其中 `test_name` 指的是所需运行测试 target 的名字，和上述添加的单元测试文件名字相同，但不带 `.py` 后缀。

**(2) 执行单元测试**

编译成功后，在 `build` 目录下执行 `ctest ${test_name}` 命令来运行单元测试，并确保单元测试通过。其中 `test_name` 指的是所需运行测试 `target` 的名字，和上述添加的单元测试文件名字相同，但不带 .py 后缀。

比如运行 `python/paddle/fluid/tests/unittests/test_logsumexp.py` 的命令如下：

```plain
ctest -R test_logsumexp
```

> 注意：执行单测一定要用 `ctest` 命令，不可直接 `python test_*.py`。

对于需要开发 C++ 算子的 API，可以把 C++ 算子的单元测试与 Python API 的单元测试写在一个文件中。

`ctest` 还可以批量运行名字匹配某个正则表达式的测试 `target`, 通过 `-R` 参数传入正则表达式。比如通过 `ctest -R test_logsumexp` 就可以运行所有以 `test_logsumexp` 开头的单测 target.

此外，需要单元测试输出更详细的信息以便 debug 时，可以在运行 `ctest` 时传入 `-V` 或者 `-VV` 选项以查看更详细的输出，如 `ctest -V -R test_logsumexp`。

## 四、写作 API 文档

前文中说到英文文档直接与 Python API 的代码写在一起，中文文档则写到 [PaddlePaddle/docs](https://github.com/PaddlePaddle/docs) 仓库中。写作指导和规范要求如下：

- 文档写作的详细指导可以参考 [文档贡献指南](../docs_contributing_guides_cn.html)，包括文件存放位置、文档修改和提交方法等。
- 文档写作的规范可以参考 [API 文档书写规范](api_docs_guidelines_cn.html)，包括中英文 API 文档的模板、写作规范、测试要求等。

提前 PR 后，GitHub 上的 paddle-bot 会给出根据所提交的中文文档所生成的官网文档的预览链接，可以点进去查看新增的文档所渲染出的页面效果，看是否符合预期。尤其需要注意检查是否有错别字、数学公式、示例代码渲染是否正确等问题。例如：

https://github.com/PaddlePaddle/docs/pull/4418

<center><img src="https://github.com/PaddlePaddle/docs/blob/develop/docs/dev_guides/api_contributing_guides/images/docs-review.png?raw=true" width="1000px" ></center>

## 五、确保通过 CI 测试

提交 PR 后会触发 CI（Continuous Integration，持续集成）测试，并且之后每提交一次代码合入（`git push`）都会触发一次 CI 测试。CI 测试可尽可能保障代码质量，详细测试内容可参见 [Paddle CI 测试详解](../git_guides/paddle_ci_manual_cn.html)，包括 CI 失败的一些处理建议。

当添加新的 API 时需要通过 CI 中所有的 `Required` 的测试项通过才能合入代码。

> 注意：其中 `PR-CI-APPROVAL`  和 `PR-CI-Static-Check` 这两个 CI 测试项需要飞桨相关开发者 approve 才能通过，除了这两个之外的 CI 测试项通过后，可以联系飞桨开发者提醒他们评审代码。

## 六、其他注意事项

### 6.1 调试 Python 代码时减少重编译的方法

- 如果你的修改不涉及 C++ 代码，那么一般不需要重新编译就可以重新运行测试，以验证刚发生的修改是否解决了问题。

Paddle 编译过程中，对于 Python 代码的处理方式是，先把它们拷贝到 build 目录，对于 Python API 和 Python 单元测试所在的文件都是如此处理。比如： `Python/paddle/fluid/tests/unittests/test_bmm_op.py` 拷贝到 build 目录后位置是 `build/Python/paddle/fluid/tests/unittests/test_bmm_op.py`。并且通过 `ctest` 运行单元测试时，会把 `build/Python` 这个目录加入 `PYTHONPATH`，因此它所调用的单元测试文件 和 Python API 代码文件也是 build 目录里的那一份。

- 如果你的修改没有涉及任何 C++ 文件，那么你也可以直接在 build 目录下修改对应的文件，直到问题解决，然后把文件拷贝回去覆盖 `Paddle` 目录的对应文件。

> 特别提醒：不要忘记拷贝回去这一步，因为重新 build 的时候，会再次从 `Paddle` 目录拷贝 Python 文件，如果最后忘了拷贝回 `Paddle` 目录，那么你的修改会因为再次的编译而被覆盖。

## 七、参考资料

- [Op 开发手册(Operator Development Manual)](https://github.com/PaddlePaddle/Paddle/wiki/Operator-Development-Manual-Index)
- [API 的设计和命名规范](api_docs_guidelines_cn.html)
- [API 单测开发及验收规范](api_accpetance_criteria_cn.html)
- [文档贡献指南](../docs_contributing_guides_cn.html)
- [API 文档书写规范](api_docs_guidelines_cn.html)
