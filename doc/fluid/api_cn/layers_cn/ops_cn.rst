========
ops
========


.. _cn_api_fluid_layers_abs:

abs
-------------------------------

.. py:function:: paddle.fluid.layers.abs(x, name=None)

绝对值激活函数。

.. math::
    out = |x|

参数:

    - **x** - abs算子的输入
    - **use_cudnn** (BOOLEAN) – （bool，默认为false）是否仅用于cudnn核，需要安装cudnn

返回：        abs算子的输出。

**代码示例**：

.. code-block:: python

        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.abs(data)


.. _cn_api_fluid_layers_acos:

acos
-------------------------------

.. py:function:: paddle.fluid.layers.acos(x, name=None)

arccosine激活函数。

.. math::
    out = cos^{-1}(x)

参数:
    - **x** - acos算子的输入

返回：        acos算子的输出。

**代码示例**：

.. code-block:: python

        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.acos(data)


.. _cn_api_fluid_layers_asin:

asin
-------------------------------

.. py:function:: paddle.fluid.layers.asin(x, name=None)

arcsine激活函数。

.. math::
    out = sin^{-1}(x)

参数:
    - **x** - asin算子的输入

返回：        asin算子的输出。

**代码示例**：

.. code-block:: python

        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.asin(data)



.. _cn_api_fluid_layers_atan:

atan
-------------------------------

.. py:function:: paddle.fluid.layers.atan(x, name=None)

arctanh激活函数。

.. math::
    out = tanh^{-1}(x)

参数:
    - **x** - atan算子的输入

返回：       atan算子的输出。

**代码示例**：

.. code-block:: python

        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.atan(data)





.. _cn_api_fluid_layers_ceil:

ceil
-------------------------------

.. py:function:: paddle.fluid.layers.ceil(x, name=None)

向上取整运算激活函数。

.. math::
    out = \left \lceil x \right \rceil



参数:

    - **x** - Ceil算子的输入
    - **use_cudnn** (BOOLEAN) – （bool，默认为false）是否仅用于cudnn核，需要安装cudnn

返回：        Ceil算子的输出。

**代码示例**：

.. code-block:: python

        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.ceil(data)









.. _cn_api_fluid_layers_cos:

cos
-------------------------------

.. py:function:: paddle.fluid.layers.cos(x, name=None)

Cosine余弦激活函数。

.. math::

    out = cos(x)



参数:

    - **x** - cos算子的输入
    - **use_cudnn** (BOOLEAN) – （bool，默认为false）是否仅用于cudnn核，需要安装cudnn


返回：        Cos算子的输出

**代码示例**：

.. code-block:: python

        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.cos(data)








.. _cn_api_fluid_layers_cumsum:

cumsum
-------------------------------

.. py:function:: paddle.fluid.layers.cumsum(x,axis=None,exclusive=None,reverse=None)

沿给定轴的元素的累加和。默认结果的第一个元素和输入的第一个元素一致。如果exlusive为真，结果的第一个元素则为0。

参数：
    - **x** -累加操作符的输入
    - **axis** (INT)-需要累加的维。-1代表最后一维。[默认 -1]。
    - **exclusive** (BOOLEAN)-是否执行exclusive累加。[默认false]。
    - **reverse** (BOOLEAN)-若为true,则以相反顺序执行累加。[默认 false]。

返回：累加器的输出

**代码示例**：

.. code-block:: python

    data = fluid.layers.data(name="input", shape=[32, 784])
    result = fluid.layers.cumsum(data, axis=0)









.. _cn_api_fluid_layers_exp:

exp
-------------------------------

.. py:function:: paddle.fluid.layers.exp(x, name=None)

Exp激活函数(Exp指以自然常数e为底的指数运算)。

.. math::
    out = e^x

参数:

    - **x** - Exp算子的输入
    - **use_cudnn** (BOOLEAN) – （bool，默认为false）是否仅用于cudnn核，需要安装cudnn


返回：       Exp算子的输出

**代码示例**：

.. code-block:: python

        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.exp(data)









.. _cn_api_fluid_layers_floor:

floor
-------------------------------

.. py:function:: paddle.fluid.layers.floor(x, name=None)


向下取整运算激活函数。

.. math::
    out = \left \lfloor x \right \rfloor


参数:

    - **x** - Floor算子的输入
    - **use_cudnn** (BOOLEAN) – （bool，默认为false）是否仅用于cudnn核，需要安装cudnn


返回：        Floor算子的输出。

**代码示例**：

.. code-block:: python

        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.floor(data)










.. _cn_api_fluid_layers_hard_shrink:

hard_shrink
-------------------------------

.. py:function:: paddle.fluid.layers.hard_shrink(x,threshold=None)

HardShrink激活函数(HardShrink activation operator)


.. math::

  out = \begin{cases}
        x, \text{if } x > \lambda \\
        x, \text{if } x < -\lambda \\
        0,  \text{otherwise}
      \end{cases}

参数：
    - **x** - HardShrink激活函数的输入
    - **threshold** (FLOAT)-HardShrink激活函数的threshold值。[默认：0.5]

返回：HardShrink激活函数的输出

**代码示例**：

.. code-block:: python

    data = fluid.layers.data(name="input", shape=[784])
    result = fluid.layers.hard_shrink(x=data, threshold=0.3)









.. _cn_api_fluid_layers_logsigmoid:

logsigmoid
-------------------------------

.. py:function:: paddle.fluid.layers.logsigmoid(x, name=None)

Logsigmoid激活函数。


.. math::

    out = \log \frac{1}{1 + e^{-x}}


参数:
    - **x** - LogSigmoid算子的输入
    - **use_cudnn** (BOOLEAN) – （bool，默认为false）是否仅用于cudnn核，需要安装cudnn

返回：        LogSigmoid算子的输出

**代码示例**：

.. code-block:: python

        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.logsigmoid(data)









.. _cn_api_fluid_layers_reciprocal:

reciprocal
-------------------------------

.. py:function:: paddle.fluid.layers.reciprocal(x, name=None)

Reciprocal（取倒数）激活函数


.. math::
    out = \frac{1}{x}

参数:

    - **x** - reciprocal算子的输入
    - **use_cudnn** (BOOLEAN) – （bool，默认为false）是否仅用于cudnn核，需要安装cudnn

返回：        Reciprocal算子的输出。

**代码示例**：

.. code-block:: python

        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.reciprocal(data)












.. _cn_api_fluid_layers_round:

round
-------------------------------

.. py:function:: paddle.fluid.layers.round(x, name=None)

Round取整激活函数。


.. math::
     out = [x]


参数:

    - **x** - round算子的输入
    - **use_cudnn** (BOOLEAN) – （bool，默认为false）是否仅用于cudnn核，需要安装cudnn

返回：        Round算子的输出。

**代码示例**：

.. code-block:: python

        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.round(data)



.. _cn_api_fluid_layers_rsqrt:

rsqrt
-------------------------------

.. py:function:: paddle.fluid.layers.rsqrt(x, name=None)

rsqrt激活函数

请确保输入合法以免出现数字错误。

.. math::
    out = \frac{1}{\sqrt{x}}


参数:

    - **x** - rsqrt算子的输入 
    - **use_cudnn** (BOOLEAN) – （bool，默认为false）是否仅用于cudnn核，需要安装cudnn

返回：     rsqrt运算输出

**代码示例**：

.. code-block:: python

        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.rsqrt(data)



.. _cn_api_fluid_layers_sigmoid:

sigmoid
-------------------------------

.. py:function:: paddle.fluid.layers.sigmoid(x, name=None)

sigmoid激活函数

.. math::
    out = \frac{1}{1 + e^{-x}}


参数:

    - **x** - Sigmoid算子的输入
    - **use_cudnn** (BOOLEAN) – （bool，默认为false）是否仅用于cudnn核，需要安装cudnn

返回：     Sigmoid运算输出.

**代码示例**：

.. code-block:: python

        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.sigmoid(data)












.. _cn_api_fluid_layers_sin:

sin
-------------------------------

.. py:function:: paddle.fluid.layers.sin(x, name=None)

正弦sine激活函数。

.. math::
     out = sin(x)


参数:

    - **x** - sin算子的输入
    - **use_cudnn** (BOOLEAN) – （bool，默认为false）是否仅用于cudnn核，需要安装cudnn


返回：        Sin算子的输出。

**代码示例**：

.. code-block:: python

        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.sin(data)













.. _cn_api_fluid_layers_softplus:

softplus
-------------------------------

.. py:function:: paddle.fluid.layers.softplus(x,name=None)

softplus激活函数。

.. math::
    out = \ln(1 + e^{x})

参数：
    - **x** - Softplus操作符的输入
    - **use_cudnn** (BOOLEAN) – （bool，默认为false）是否仅用于cudnn核，需要安装cudnn

返回：Softplus操作后的结果

**代码示例**：

.. code-block:: python

        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.softplus(data)











.. _cn_api_fluid_layers_softshrink:

softshrink
-------------------------------

.. py:function:: paddle.fluid.layers.softshrink(x, name=None)

Softshrink激活算子

.. math::
        out = \begin{cases}
                    x - \lambda, \text{if } x > \lambda \\
                    x + \lambda, \text{if } x < -\lambda \\
                    0,  \text{otherwise}
              \end{cases}

参数：
        - **x** - Softshrink算子的输入
        - **lambda** （FLOAT）- 非负偏移量。

返回：       Softshrink算子的输出

**代码示例**：

.. code-block:: python

        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.softshrink(data)












.. _cn_api_fluid_layers_softsign:

softsign
-------------------------------

.. py:function:: paddle.fluid.layers.softsign(x,name=None)


softsign激活函数。

.. math::
    out = \frac{x}{1 + |x|}

参数：
    - **x** : Softsign操作符的输入
    - **use_cudnn** (BOOLEAN) – （bool，默认为false）是否仅用于cudnn核，需要安装cudnn


返回：Softsign操作后的结果

**代码示例**：

.. code-block:: python

        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.softsign(data)











.. _cn_api_fluid_layers_sqrt:

sqrt
-------------------------------

.. py:function:: paddle.fluid.layers.sqrt(x, name=None)

算数平方根激活函数。

请确保输入是非负数。有些训练当中，会出现输入为接近零的负值，此时应加上一个小值epsilon（1e-12）将其变为正数从而正确运算并进行后续的操作。


.. math::
    out = \sqrt{x}

参数:

    - **x** - Sqrt算子的输入
    - **use_cudnn** (BOOLEAN) – （bool，默认为false）是否仅用于cudnn核，需要安装cudnn

返回：       Sqrt算子的输出。

**代码示例**：

.. code-block:: python

        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.sqrt(data)













.. _cn_api_fluid_layers_square:

square
-------------------------------

.. py:function:: paddle.fluid.layers.square(x,name=None)

取平方激活函数。

.. math::
    out = x^2

参数:
    - **x** : 平方操作符的输入
    - **use_cudnn** (BOOLEAN) – （bool，默认为false）是否仅用于cudnn核，需要安装cudnn

返回：平方后的结果

**代码示例**：

.. code-block:: python

        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.square(data)











.. _cn_api_fluid_layers_tanh:

tanh
-------------------------------

.. py:function:: paddle.fluid.layers.tanh(x, name=None)




tanh 激活函数。

.. math::
    out = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}


参数:

    - **x** - Tanh算子的输入
    - **use_cudnn** (BOOLEAN) – （bool，默认为false）是否仅用于cudnn核，需要安装cudnn

返回：     Tanh算子的输出。

**代码示例**：

.. code-block:: python

        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.tanh(data)













.. _cn_api_fluid_layers_tanh_shrink:

tanh_shrink
-------------------------------

.. py:function:: paddle.fluid.layers.tanh_shrink(x, name=None)

tanh_shrink激活函数。

.. math::
    out = x - \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}

参数:

    - **x** - TanhShrink算子的输入
    - **use_cudnn** (BOOLEAN) – （bool，默认为false）是否仅用于cudnn核，需要安装cudnn

返回：     tanh_shrink算子的输出

**代码示例**：

.. code-block:: python

        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.tanh_shrink(data)









.. _cn_api_fluid_layers_thresholded_relu:

thresholded_relu
-------------------------------

.. py:function:: paddle.fluid.layers.thresholded_relu(x,threshold=None)

ThresholdedRelu激活函数

.. math::

  out = \left\{\begin{matrix}
      x, &if x > threshold\\
      0, &otherwise
      \end{matrix}\right.

参数：
- **x** -ThresholdedRelu激活函数的输入
- **threshold** (FLOAT)-激活函数threshold的位置。[默认1.0]。

返回：ThresholdedRelu激活函数的输出

**代码示例**：

.. code-block:: python

  data = fluid.layers.data(name="input", shape=[1])
  result = fluid.layers.thresholded_relu(data, threshold=0.4)









.. _cn_api_fluid_layers_uniform_random:

uniform_random
-------------------------------

.. py:function:: paddle.fluid.layers.uniform_random(shape, dtype='float32', min=-1.0, max=1.0, seed=0)
该操作符初始化一个张量，该张量的值是从均匀分布中抽样的随机值

参数：
    - **shape** (LONGS)-输出张量的维
    - **dtype** (np.dtype|core.VarDesc.VarType|str) – 数据的类型, 例如float32, float64。 默认: float32.
    - **min** (FLOAT)-均匀随机分布的最小值。[默认 -1.0]
    - **max** (FLOAT)-均匀随机分布的最大值。[默认 1.0]
    - **seed** (INT)-随机种子，用于生成样本。0表示使用系统生成的种子。注意如果种子不为0，该操作符每次都生成同样的随机数。[默认 0]


**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    result = fluid.layers.uniform_random(shape=[32, 784])











