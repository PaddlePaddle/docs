.. _cn_api_paddle_nn_functional_maxout:

maxout
-------------------------------

.. py:function:: paddle.nn.functional.maxout(x, groups, axis=1, name=None)

maxout 激活层。

假设输入形状为(N, Ci, H, W)，输出形状为(N, Co, H, W)，则 :math:`Co=Ci/groups` 运算公式如下：

.. math::

    &out_{si+j} = \max_{k} x_{gsi + sk + j} \\
    &g = groups \\
    &s = \frac{input.size}{num\_channels} \\
    &0 \le i < \frac{num\_channels}{groups} \\
    &0 \le j < s \\
    &0 \le k < groups

参数
::::::::::::

::::::::::
    - **x** (Tensor) - 输入是形状为 :math:`[N, C, H, W]` 或 :math:`[N, H, W, C]` 的 4-D Tensor，N 是批尺寸，C 是通道数，H 是特征高度，W 是特征宽度，数据类型为 float32 或 float64。
    - **groups** (int) - 指定将输入 Tensor 的 channel 通道维度进行分组的数目。输出的通道数量为通道数除以组数。
    - **axis** (int，可选) - 指定通道所在维度的索引。当数据格式为 NCHW 时，axis 应该被设置为 1，当数据格式为 NHWC 时，axis 应该被设置为-1 或者 3。默认值为 1。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
    ``Tensor``，数据类型同 ``x`` 一致。

代码示例
::::::::::

COPY-FROM: paddle.nn.functional.maxout
