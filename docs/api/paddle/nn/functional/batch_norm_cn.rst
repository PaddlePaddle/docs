.. _cn_api_paddle_nn_functional_batch_norm:

batch_norm
-------------------------------

.. py:class:: paddle.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, training=False, momentum=0.9, epsilon=1e-05, data_format='NCHW', use_global_stats=None, name=None)

推荐使用 nn.BatchNorm1D，nn.BatchNorm2D, nn.BatchNorm3D，由内部调用此方法。

详情见 :ref:`cn_api_paddle_nn_BatchNorm1D` 。

参数
::::::::::::

    - **x** (int) - 输入，数据类型为 float32, float64。
    - **running_mean** (Tensor) - 均值的 Tensor。
    - **running_var** (Tensor) - 方差的 Tensor。
    - **weight** (Tensor) - 权重的 Tensor。
    - **bias** (Tensor) - 偏置的 Tensor。
    - **training** (bool，可选) – 当该值为 True 时，表示为训练模式（train mode），即使用批数据计算并在训练期间跟踪全局均值和方差。为 False 时，表示使用推理模式（inference mode），即使用训练期间计算出的全局均值及方差计算。默认值为 False。
    - **momentum** (float，可选) - 此值用于计算 ``moving_mean`` 和 ``moving_var``。默认值：0.9。更新公式如上所示。
    - **epsilon** (float，可选) - 为了数值稳定加在分母上的值。默认值：1e-05。
    - **data_format** (str，可选) - 指定输入数据格式。数据格式可以为 ``"NC"``、``"NCL"``、``"NCHW"`` 或者 ``"NCDHW"``，其中 N 是批大小，C 是通道数，D 是特征深度，H 是特征高度，W 是特征宽度，L 是特征长度。默认值为 ``"NCHW"``。
    - **use_global_stats** (bool|None，可选) - 指示是否使用全局均值和方差。若设置为 False，则使用一个 mini-batch 的统计数据。若设置为 True 时，将使用全局统计数据。若设置为 None，则会在测试阶段使用全局统计数据，在训练阶段使用一个 mini-batch 的统计数据。默认值为 None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
::::::::::::
无


代码示例
::::::::::::

COPY-FROM: paddle.nn.functional.batch_norm
