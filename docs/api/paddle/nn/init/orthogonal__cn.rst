.. _cn_api_paddle_nn_init_orthogonal_:

orthogonal\_
-------------------------------

.. py:class:: paddle.nn.init.orthogonal_(tensor, gain=1)

将输入张量的值设置为截断的正态分布的随机数，该操作会直接修改输入张量。

参数
::::::::::::

    - **tensor** (Tensor) - 输入张量。
    - **gain** (float，可选) - 比例因子，默认值为 1.0。
