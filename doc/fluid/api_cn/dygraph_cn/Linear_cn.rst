.. _cn_api_fluid_dygraph_Linear:

Linear
-------------------------------

.. py:class:: paddle.fluid.dygraph.Linear(input_dim, output_dim, param_attr=None, bias_attr=None, act=None, dtype='float32')

:alias_main: paddle.nn.Linear
:alias: paddle.nn.Linear,paddle.nn.layer.Linear,paddle.nn.layer.common.Linear
:old_api: paddle.fluid.dygraph.Linear




**线性变换层：**

.. math::

        \\Out = Act({XW + b})\\

其中，:math:`X` 为输入的 Tensor， :math:`W` 和 :math:`b` 分别为权重和偏置。

Linear 层只接受一个 Tensor 的输入。
Linear 层将输入 Tensor 与权重矩阵 :math:`W` 相乘，然后生成形状为 :math:`[N，*，output_dim]` 的输出张量，
其中 :math:`N` 是批量大小，:math:`*` 表示任意数量的附加尺寸。
如果 bias_attr 不是 None，则将创建一个 bias 变量并将其添加到输出中。
最后，如果激活 act 不是 None，则相应激活函数也将应用于输出上。

参数:
  - **input_dim** (int) – 线性变换层输入单元的数目。
  - **output_dim** (int) – 线性变换层输出单元的数目。
  - **param_attr** (ParamAttr, 可选) – 指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
  - **bias_attr** (ParamAttr, 可选) – 指定偏置参数属性的对象，若 `bias_attr` 为bool类型，如果设置为False，表示不会为该层添加偏置；如果设置为True，表示使用默认的偏置参数属性。默认值为None，表示使用默认的偏置参数属性。默认的偏置参数属性将偏置参数的初始值设为0。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
  - **act** (str, 可选) – 应用于输出上的激活函数，如tanh、softmax、sigmoid，relu等，支持列表请参考 :ref:`api_guide_activations` ，默认值为None。
  - **dtype** (str, 可选) – 权重的数据类型，可以为float32或float64。默认为float32。

返回：无

**代码示例**

..  code-block:: python

    from paddle.fluid.dygraph.base import to_variable
    import paddle.fluid as fluid
    from paddle.fluid.dygraph import Linear
    import numpy as np

    data = np.random.uniform( -1, 1, [30, 10, 32] ).astype('float32')
    with fluid.dygraph.guard():
        linear = Linear(32, 64)
        data = to_variable(data)
        res = linear(data)  # [30, 10, 64]

属性
::::::::::::
.. py:attribute:: weight

本层的可学习参数，类型为 ``Parameter``

.. py:attribute:: bias

本层的可学习偏置，类型为 ``Parameter``

