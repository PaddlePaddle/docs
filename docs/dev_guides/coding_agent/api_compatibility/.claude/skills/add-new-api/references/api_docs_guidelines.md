# API 文档书写规范

路径说明：本文档中的路径描述如无特别说明，均相对于 ${ROOT_DIR}/Paddle 目录。

## 一、概述

本文档介绍飞桨框架 API 文档的书写规范，包括文档的基本要求、各模块的写作说明、典型案例以及文档测试等内容。

**基本要求：**

1. **至关重要**：API 文档对该 API 的描述，一定要与 API 的行为保持一致。中英文文档的内容要严格一致
2. **API 文档的字段**：API 名称、API 功能描述、API 参数、API 返回、API 代码示例、API 属性（class）、API 方法（methods）等。API 抛出异常的情况，不需要在文档中体现
3. **API 功能描述**：请注意，看文档的用户没有和开发同学一样的知识背景。因此，请提示用户在什么场景下使用该 API。请使用深度学习领域通用的词汇和说法
4. **API 参数**：写清楚对输入参数的要求，写清楚在不同情况下的行为区别（如默认值时的行为）。同类性质的参数（如：输入 Tensor `x`，每个 API 中的 `name` 参数）
5. **API 代码示例**：中英文文档当中的代码示例完全一致（注释可不用翻译），中文文档建议使用 COPY-FROM 的方式与英文文档做同步。代码示例尽量不用随机输入，并给出输出值。构造输入数据时，尽量使用 paddle 提供的 API，如 `paddle.zeros`、`paddle.ones`、`paddle.full`、`paddle.arange`、`paddle.rand`、`paddle.randn`、`paddle.randint`、`paddle.normal`、`paddle.uniform`，尽量不要引入第三方库（如 NumPy）
6. **其他**：对于 `Variable`、`DenseTensor`、`Tensor` 等描述，统一使用 `Tensor`
7. 对于 `Linear`、`Conv2D`、`L1Loss` 这些 class 形式的 API，需要写清楚被调用时输入输出的形状（如 `forward` 方法的参数）。位置放在 `Parameters` / `参数` block 后面，具体为：

中文时：
    形状:
          - **input** (Tensor)：形状为（批大小，通道数，高度，宽度），即，NCHW 格式的 4-D Tensor。
          - **output** (Tensor)：形状为（批大小，卷积核个数，输出图像的高度，输出图像的高度）的 4-D Tensor。

英文时：
    Shape:
          - input: 4-D tensor with shape: (batch, num_channels, height, width), i.e.: NCHW.
          - output: 4-D tensor with shape: (batch, num_filters, new_height, new_width).

## 二、典型案例

- paddle.concat：python/paddle/tensor/manipulation.py
- paddle.split：python/paddle/tensor/manipulation.py
- paddle.squeeze：python/paddle/tensor/manipulation.py
- paddle.full_like：python/paddle/tensor/creation.py
- paddle.ones：python/paddle/tensor/creation.py
- paddle.ones_like：python/paddle/tensor/creation.py

## 三、英文模板

    def add(x, y, name=None):
        """

        Add two tensors element-wise. The equation is:

        .. math::
            out = x + y

        Note:
            ``paddle.add`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting`.

        Args:
            x (Tensor): The input tensor, it's data type should be float32, float64, int32, int64.
            y (Tensor): The input tensor, it's data type should be float32, float64, int32, int64.
            name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

        Returns:
            N-D Tensor. A location into which the result is stored. It's dimension equals with :attr:`x`.

        Examples:
            .. code-block:: pycon

                >>> import paddle

                >>> x = paddle.to_tensor([2, 3, 4], 'float64')
                >>> y = paddle.to_tensor([1, 5, 2], 'float64')
                >>> z = paddle.add(x, y)
                >>> print(z)
                Tensor(shape=[3], dtype=float64, place=Place(cpu), stop_gradient=True,
                [3., 8., 6.])

        """

## 四、中文模板

    .. _cn_api_paddle_add:

    add
    -------------------------------

    .. py:function:: paddle.add(x, y, name=None)

    输入 :attr:`x` 与输入 :attr:`y` 逐元素相加，并将各个位置的输出元素保存到返回结果中。计算公式为：

    .. math::
        out = x + y

    .. note::
       ``paddle.add`` 遵守广播机制，如您想了解更多，请参见 :ref:`cn_user_guide_broadcasting`。

    参数
    :::::::::
        - **x** (Tensor) - 输入的 Tensor，数据类型为 float32、float64、int32 或 int64。
        - **y** (Tensor) - 输入的 Tensor，数据类型为 float32、float64、int32 或 int64。
        - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

    返回
    :::::::::
    ``Tensor``，维度和数据类型都与 :attr:`x` 相同，存储运算后的结果。

    代码示例
    ::::::::::

    COPY-FROM: paddle.add

## 五、API 文档各模块写作说明

### 5.1 API 标签

标签 `api_label` 一般用于文档间的引用。英文 API 文档的标签是自动生成的，而中文 API 文档的标签则需要在文档第一行手动编写。

如 paddle.add：
    .. _cn_api_paddle_add:

其中 `api_label` 是 `cn_api_paddle_add`，但在中文文档中，需要在标签 `cn_api_paddle_add` 的前面加上 `.. _` 、后面加上 `:` （固定格式）

**api_label 设定规范**：
1. 英文 api_label：`api_` + <完整的 API 名称，把 `.` 替换成 `_` >，如 `paddle.add` 对应 `api_paddle_add`
2. 中文 api_label：`cn_` + 英文 api_label，如 `paddle.add` 对应 `cn_api_paddle_add`

### 5.2 API 名称

API 名称直接写 API 的名字即可，不需要写全完整路径。如 paddle.add：
    add
    ---------

### 5.3 API 声明

API 的声明部分，要给出 API 的声明信息。

function：如 paddle.add
    .. py:function:: paddle.add(x, y, name=None)

class：如 paddle.nn.Conv2D
    .. py:class:: paddle.nn.Conv2D(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', weight_attr=None, bias_attr=None, data_format='NCHW')

注意：此处的参数名称需要与后文 API 参数板块中的严格保持一致。

### 5.4 API 功能描述

API 功能描述部分只需要尽可能简单的描述出 API 的功能作用即可，要让用户能快速看懂。可以拆解为 3 个部分：功能作用 + 计算公式 + 注解部分。

- 功能作用：描述该 API 的功能作用。由于用户没有对应的背景，所以需要补充必要的细节，比如是不是逐元素的
- 计算公式：给出该 API 的计算公式，由于公式中每个变量都对应 API 的参数，所以不需要做额外的说明
- 注解部分：如果 API 有需要特殊说明的部分，可以在注解部分给出

**注意事项**：
1. 写作 API 文档时，请使用深度学习领域通用的词汇和说法
2. 文档中的前后说明要一致，比如维度的说明，统一使用 4-D Tensor 的格式，不确定的写"多维"
3. 功能描述中涉及到的专有数据结构如 `Tensor`、`DenseTensor` 和 `Variable`，中英文都统一使用 `Tensor`，无需翻译
4. 如果涉及到一些通用的知识，如广播机制，可以用注解的方式写出来

中文：
```
.. note::
    ``paddle.add`` 遵守广播机制，如您想了解更多，请参见 :ref:`cn_user_guide_broadcasting`。
```

英文：
```
Note:
    ``paddle.add`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting`.
```

### 5.5 API 参数（重要）

**注意**：
- 一些通用的参数说明，直接复制 docs/templates/common_docs.py（位于 docs 仓库）中的描述即可
- 若当前 API 无参数，则不需要填写该板块

API 参数部分，要解释清楚每个参数的意义和使用场景。**需要注意以下两点**：

1. 对于有默认值的参数，至少要讲清楚在默认值下的逻辑，而不仅仅是介绍这个参数是什么以及默认值是什么。

如 stop_gradient 的对比：
```python
# 错误写法
stop_gradient (bool，可选) - 提示是否应该停止计算梯度，默认值为 True。

# 正确写法：需添加默认值为 True 的行为，即表示停止计算梯度
stop_gradient (bool，可选) - 提示是否应该停止计算梯度，默认值为 True，表示停止计算梯度。
```

或如 return_numpy：需要分别描述 True 和 False 两种情况：
```python
# 错误写法
return_numpy (bool) – 该变量表示是否将 fetched tensor 转换为 NumPy 数据。默认值为 True。

# 正确写法
return_numpy (bool) – 该参数表示是否将返回的计算结果转化为 NumPy 数据；如果为 False，则每个变量返回的类型为 Tensor，否则返回变量的类型为 numpy.ndarray。默认为：True。
```

2. 在讲清楚每个 API 参数是什么的同时，还需要描述清楚每个参数的具体作用是什么。

### 5.6 API 返回

先描述 API 返回值的类型，然后描述 API 的返回值及其含义。如 paddle.add：
    返回
    :::::::::
    ``Tensor``，维度和数据类型都与 :attr:`x` 相同，存储运算后的结果。

### 5.7 API 抛出异常

API 抛出异常部分，由于历史原因写在文档中，建议在源码的 warning 中做提示，不在文档中展开。

### 5.8 API 代码示例（重要）

代码示例是 API 文档的核心部分之一，应该对 API 使用的各种场景尽可能在一个示例中给出，并给出对应的结果。

**书写规范**

书写示例代码如同在 Python 的标准交互界面 REPL 中编程一样：
- `>>> ` 表示单行语句
- `... ` 表示多行或复合语句

```python
>>> import paddle
>>> x = paddle.to_tensor([[1, 2], [3, 4]])
>>> y = paddle.to_tensor([[5, 6], [7, 8]])
>>> res = paddle.multiply(x, y)
```

为保证示例代码正确性，CI 环境会对其进行检查。更多规范请参考 Python 文档示例代码书写规范（code_example_writing_specification_cn.md）。

**注意事项**
- 中英文示例代码保持完全一致，中文文档建议使用 COPY-FROM 同步
- 原则上所有 API 都需提供示例代码，class member methods、abstract API、callback 等特殊情况可通过白名单审核
- 仅 GPU 环境的 API，在 CPU 上运行时给出含 "Not compiled with CUDA" 的错误提示即可

英文 API 代码示例格式：
```python
def api():
    """
    Examples:
        .. code-block:: pycon

            >>> import paddle
            # ... 示例代码 ...
    """
```

中文文档格式：
```
代码示例
::::::::::

COPY-FROM: paddle.add
```

### 5.9 API 属性

API 的属性用来描述 API 所包含的属性。如果 API 有属性，每个属性需要分为以下部分描述：
- 名称：属性名称直接写属性的名字即可，不需要将全路径写全
- 注意：列举出使用该属性时应注意的一些问题，如果没有可以不填
- 描述：与 API 功能描述部分要求一致
- 返回：与 API 返回部分要求一致
- 代码示例：与 API 代码示例部分要求一致

### 5.10 API 方法

API 的方法用来描述 API 所包含的方法，一些类的 API 会有这个内容，没有方法的 API 可以不写此模块。如果有，每个方法需要分为六个部分描述：
- 名称：方法名称直接写方法的名字即可，不需要将全路径写全
- 声明：与 API 声明的要求一致
- 参数：与 API 参数的要求一致
- 描述：与 API 功能描述的要求一致
- 返回：与 API 返回的要求一致
- 代码示例：与 API 代码示例部分要求一致

### 5.11 注解

注解部分描述用户使用该 API 时需要额外注意的事项。

例 1：使用注意事项（paddle.sqrt）
- 中文：`.. note:: 请确保输入中的数值是非负数。`
- 英文：`Note: Input value must be greater than or equal to zero.`

### 5.12 警告

警告部分用于不推荐用户使用的 API 或计划废弃的 API。

例 1：计划废弃的 API（paddle.fluid.clip.set_gradient_clip）
- 中文：`.. warning:: 此 API 对位置使用的要求较高...不推荐使用。推荐在 optimizer 初始化时设置梯度裁剪。`
- 英文：`Warning: This API must be used after building network...It is recommended to set grad_clip when initializing the optimizer...`

## 六、注意事项

- 中文文档、英文文档齐全，内容一一对应
- 文档清晰可读，易于用户使用
- 给出易于理解的 API 介绍，包括文字描述和公式描述
- 参数命名通俗易懂无歧义，明确给出传参类型，对参数含义以及使用方法进行详细说明，对返回值进行详细说明
- 示例代码需要做到复制粘贴即可运行，并且需要明确给出预期运行结果（如果可以）
- 阅读无障碍：无错别字、上下文连贯、内容清晰易懂、链接可正常跳转、图片公式显示正常
