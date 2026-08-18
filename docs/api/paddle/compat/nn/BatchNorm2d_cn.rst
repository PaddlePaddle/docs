.. _cn_api_paddle_compat_nn_BatchNorm2d:

BatchNorm2d
-------------------------------

.. py:class:: paddle.compat.nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)

PyTorch 兼容的二维批归一化层，参数含义与 :ref:`cn_api_paddle_compat_nn_BatchNorm1d` 相同。

参数
:::::::::

    - **num_features** (int) - 输入的通道数。
    - **eps** (float，可选) - 数值稳定项。默认值：1e-5。
    - **momentum** (float|None，可选) - 运行统计量的更新系数。默认值：0.1。
    - **affine** (bool，可选) - 是否创建可学习的缩放和偏置参数。默认值：True。
    - **track_running_stats** (bool，可选) - 是否跟踪运行均值和方差。默认值：True。
    - **device** (PlaceLike|None，可选) - 参数设备。默认值：None。
    - **dtype** (DTypeLike|None，可选) - 参数数据类型。默认值：None。
