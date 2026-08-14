.. _cn_api_paddle_nn_init_xavier_normal_:

xavier_normal\_
-------------------------------

.. py:function:: paddle.nn.init.xavier_normal_(tensor, gain=1.0, fan_in=None, fan_out=None)

将输入张量的值设置为 Xavier 正常分布的随机数，该操作会直接修改输入张量。

参数
::::::::::::

    - **tensor** (Tensor) - 输入张量。
    - **gain** (float，可选) - 比例因子，默认值为 1.0。
    - **fan_in** (float|None，可选) - Xavier 初始化的 ``fan_in``。默认从 Tensor 推断。默认值为 None。
    - **fan_out** (float|None，可选) - Xavier 初始化的 ``fan_out``。默认从 Tensor 推断。默认值为 None。
