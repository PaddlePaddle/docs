.. _cn_api_fluid_dygraph_LayerNorm:

LayerNorm
-------------------------------

.. py:class:: paddle.fluid.dygraph.LayerNorm(normalized_shape, scale=True, shift=True, epsilon=1e-05, param_attr=None, bias_attr=None, act=None, dtype="float32")




该接口用于构建 ``LayerNorm`` 类的一个可调用对象，具体用法参照 ``代码示例``。其中实现了层归一化层（Layer Normalization Layer）的功能，其可以应用于小批量输入数据。更多详情请参考：`Layer Normalization <https://arxiv.org/pdf/1607.06450v1.pdf>`_

计算公式如下

.. math::
            \\\mu=\frac{1}{H}\sum_{i=1}^{H}x_i\\

            \\\sigma=\sqrt{\frac{1}{H}\sum_i^H{(x_i-\mu)^2} + \epsilon}\\

             \\y=f(\frac{g}{\sigma}(x-\mu) + b)\\

- :math:`x`：该层神经元的向量表示
- :math:`H`：层中隐藏神经元个数
- :math:`\epsilon`：添加较小的值到方差中以防止除零
- :math:`g`：可训练的比例参数
- :math:`b`：可训练的偏差参数


参数
::::::::::::

    - **normalized_shape** (int 或 list 或 tuple) – 需规范化的 shape，期望的输入 shape 为 ``[*, normalized_shape[0], normalized_shape[1], ..., normalized_shape[-1]]``。如果是单个整数，则此模块将在最后一个维度上规范化（此时最后一维的维度需与该参数相同）。
    - **scale** (bool，可选) - 指明是否在归一化后学习自适应增益 ``g``。默认值：True。
    - **shift** (bool，可选) - 指明是否在归一化后学习自适应偏差 ``b``。默认值：True。
    - **epsilon** (float，可选) - 指明在计算过程中是否添加较小的值到方差中以防止除零。默认值：1e-05。
    - **param_attr** (ParamAttr，可选) - 指定权重参数属性的对象。默认值为 None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **bias_attr** (ParamAttr，可选) - 指定偏置参数属性的对象。默认值为 None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **act** (str，可选) - 应用于输出上的激活函数，如 tanh、softmax、sigmoid，relu 等，支持列表请参考 :ref:`api_guide_activations`，默认值为 None。
    - **dtype** (str，可选) - 输出 Tensor 的数据类型，数据类型必须为：float32 或 float64，默认为 float32。


返回
::::::::::::
无

代码示例
::::::::::::

.. code-block:: python

    import paddle.fluid as fluid
    from paddle.fluid.dygraph.base import to_variable
    import numpy

    x = numpy.random.random((3, 32, 32)).astype('float32')
    with fluid.dygraph.guard():
        x = to_variable(x)
        layerNorm = fluid.LayerNorm([32, 32])
        ret = layerNorm(x)
