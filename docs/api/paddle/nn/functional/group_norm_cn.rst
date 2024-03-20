.. _cn_api_paddle_nn_functional_group_norm:

group_norm
-------------------------------

.. py:function:: paddle.nn.functional.group_norm(x, num_groups, epsilon=1e-05, weight=None, bias=None, data_format='NCHW', name=None)

推荐使用 nn.GroupNorm。

详情见 :ref:`cn_api_paddle_nn_GroupNorm` 。

参数
::::::::::::

    - **x** (Tensor) - 输入 Tensor，形状为 [批大小，通道数，\*]。
    - **num_groups** (int) - 从通道中分离出来的 ``group`` 的数目。
    - **epsilon** (float，可选) - 为防止方差除零，增加一个很小的值。默认值：1e-05。
    - **weight** (Tensor，可选) - 权重的 Tensor，形状为 [通道数]，默认为 None。
    - **bias** (Tensor，可选) - 偏置的 Tensor，形状为 [通道数]，默认为 None。
    - **data_format** (string，可选) - 只支持 “NCHW” [num_batches，channels，height，width] 格式。默认值：“NCHW”。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
::::::::::::
    ``Tensor``，输出形状与 ``x`` 一致。

代码示例
::::::::::::

COPY-FROM: paddle.nn.functional.group_norm
