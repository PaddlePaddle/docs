.. _cn_api_paddle_compat_nn_Linear:

Linear
-------------------------------

.. py:class:: paddle.compat.nn.Linear(in_features, out_features, bias, device=None, dtype=None)

PyTorch 兼容的 :ref:`cn_api_paddle_nn_Linear` 版本：

    - 与 PyTorch 一致的数学意义：

    .. math::

        Out = XW^T + b

    其中 :math:`W` 是权重张量， :math:`b` 是偏置张量， :math:`X` 是输入张量

    - 与 PyTorch 一致的默认可视化方法： Kaiming normal 用于权重初始化，平均分布用于偏置初始化
    - 支持设定 ``Linear`` 的数据类型以及运行设备

使用前请详细参考：`【仅参数名不一致】torch.nn.Linear`_ 以确定是否使用此模块。

.. _【仅参数名不一致】torch.nn.Linear: https://www.paddlepaddle.org.cn/documentation/docs/en/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Linear.html

参数
:::::::::::

        - **in_features** (int) - 输入单元数。
        - **out_features** (int) - 输出单元数。
        - **bias** (bool, 可选) - 如果为 True，将创建偏置（一个形状为 :math:`[out\_features]` 的一维张量）并添加到输出。默认值 True。
        - **device** (PlaceLike, 可选) - 创建参数所在的设备。默认值为 None，表示默认 Paddle 设备。
        - **dtype** (DTypeLike, 可选) - 创建参数的数据类型。默认值为 None，并设置为 Linear 的默认数据类型（float32）。

形状
:::::::::

        - **输入** : 多维张量，形状为 :math:`[*, in\_features]` 。数据类型为 float16、float32、float64，默认为 float32。
        - **输出** : 多维张量，形状为 :math:`[*, out\_features]` 。数据类型与输入相同。



代码示例
::::::::::::

COPY-FROM: paddle.compat.nn.Linear
