.. _cn_api_paddle_nn_LayerNorm:

LayerNorm
-------------------------------

.. py:class:: paddle.nn.LayerNorm(normalized_shape, epsilon=1e-05, weight_attr=None, bias_attr=None, name=None)

构建 ``LayerNorm`` 类的一个可调用对象，具体用法参照 ``代码示例``。其中实现了层归一化层（Layer Normalization Layer）的功能，其可以应用于小批量输入数据。更多详情请参考：`Layer Normalization <https://arxiv.org/pdf/1607.06450v1.pdf>`_ 。

计算公式如下：

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

    - **normalized_shape** (int|list|tuple) – 需规范化的 shape，期望的输入 shape 为 ``[*, normalized_shape[0], normalized_shape[1], ..., normalized_shape[-1]]``。如果是单个整数，则此模块将在最后一个维度上规范化（此时最后一维的维度需与该参数相同）。
    - **epsilon** (float，可选) - 指明在计算过程中是否添加较小的值到方差中以防止除零。默认值：1e-05。
    - **weight_attr** (ParamAttr|bool|None，可选) - 指定权重参数属性的对象。如果为 False 固定为 1，不进行学习。默认值为 None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_paddle_ParamAttr` 。
    - **bias_attr** (ParamAttr|None，可选) - 指定偏置参数属性的对象。如果为 False 固定为 0，不进行学习。默认值为 None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_paddle_ParamAttr` 。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
::::::::::::

    - input: 2-D, 3-D, 4-D 或 5D 的 Tensor。
    - output：和输入形状一样。

代码示例
::::::::::::

COPY-FROM: paddle.nn.LayerNorm
