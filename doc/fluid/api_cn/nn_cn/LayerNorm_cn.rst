.. _cn_api_nn_LayerNorm:

LayerNorm
-------------------------------

.. py:class:: paddle.nn.LayerNorm(normalized_shape, epsilon=1e-05, weight_attr=None, bias_attr=None, name=None)

该接口用于构建 ``LayerNorm`` 类的一个可调用对象，具体用法参照 ``代码示例`` 。其中实现了层归一化层（Layer Normalization Layer）的功能，其可以应用于小批量输入数据。更多详情请参考：`Layer Normalization <https://arxiv.org/pdf/1607.06450v1.pdf>`_

计算公式如下

.. math::
            \\\mu=\frac{1}{H}\sum_{i=1}^{H}x_i\\

            \\\sigma=\sqrt{\frac{1}{H}\sum_i^H{(x_i-\mu)^2} + \epsilon}\\

             \\y=f(\frac{g}{\sigma}(x-\mu) + b)\\

- :math:`x` : 该层神经元的向量表示
- :math:`H` : 层中隐藏神经元个数
- :math:`\epsilon` : 添加较小的值到方差中以防止除零
- :math:`g` : 可训练的比例参数
- :math:`b` : 可训练的偏差参数


参数：
    - **normalized_shape** (int 或 list 或 tuple) – 需规范化的shape，期望的输入shape为 ``[*, normalized_shape[0], normalized_shape[1], ..., normalized_shape[-1]]`` 。如果是单个整数，则此模块将在最后一个维度上规范化（此时最后一维的维度需与该参数相同）。
    - **epsilon** (float, 可选) - 指明在计算过程中是否添加较小的值到方差中以防止除零。默认值：1e-05。
    - **weight_attr** (ParamAttr|bool, 可选) - 指定权重参数属性的对象。如果为False固定为1，不进行学习。默认值为None, 表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **bias_attr** (ParamAttr, 可选) - 指定偏置参数属性的对象。如果为False固定为0，不进行学习。默认值为None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **name** (string, 可选) – LayerNorm的名称, 默认值为None。更多信息请参见 :ref:`api_guide_Name` 。

返回：无

形状：
    - input: 2-D, 3-D, 4-D或5D 的Tensor。
    - output: 和输入形状一样。

**代码示例**

.. code-block:: python

    import paddle
    import numpy as np

    paddle.disable_static()
    np.random.seed(123)
    x_data = np.random.random(size=(2, 2, 2, 3)).astype('float32')
    x = paddle.to_tensor(x_data) 
    layer_norm = paddle.nn.LayerNorm(x_data.shape[1:])
    layer_norm_out = layer_norm(x)

    print(layer_norm_out.numpy())

