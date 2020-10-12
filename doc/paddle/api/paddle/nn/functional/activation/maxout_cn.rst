.. _cn_api_nn_cn_maxout:

maxout
-------------------------------

.. py:function:: paddle.nn.functional.maxout(x, groups, axis=1, name=None)

maxout激活层.

假设输入形状为(N, Ci, H, W)，输出形状为(N, Co, H, W)，则 :math:`Co=Ci/groups` 运算公式如下:

.. math::

    &out_{si+j} = \max_{k} x_{gsi + sk + j} \\
    &g = groups \\
    &s = \frac{input.size}{num\_channels} \\
    &0 \le i < \frac{num\_channels}{groups} \\
    &0 \le j < s \\
    &0 \le k < groups

参数:
::::::::::
    - x (Tensor) - 输入是形状为 :math:`[N, C, H, W]` 或 :math:`[N, H, W, C]` 的4-D Tensor，N是批尺寸，C是通道数，H是特征高度，W是特征宽度，数据类型为float32或float64。
    - groups (int) - 指定将输入张量的channel通道维度进行分组的数目。输出的通道数量为通道数除以组数。
    - axis (int, 可选) - 指定通道所在维度的索引。当数据格式为NCHW时，axis应该被设置为1，当数据格式为NHWC时，axis应该被设置为-1或者3。默认值为1。
    - name (str, 可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
::::::::::
    ``Tensor`` ，数据类型同 ``x`` 一致。

代码示例
::::::::::

.. code-block:: python

    import paddle
    import paddle.nn.functional as F

    x = paddle.rand([1, 2, 3, 4])
    # [[[[0.5002636  0.22272532 0.17402348 0.2874594 ]
    #    [0.95313174 0.6228939  0.7129065  0.7087491 ]
    #    [0.02879342 0.88725346 0.61093384 0.38833922]]
    #   [[0.5231306  0.03807496 0.91661984 0.15602879]
    #    [0.666127   0.616567   0.30741522 0.24044901]
    #    [0.7142536  0.7351477  0.31588817 0.23782359]]]]
    out = F.maxout(x, groups=2)
    # [[[[0.5231306  0.22272532 0.91661984 0.2874594 ]
    #    [0.95313174 0.6228939  0.7129065  0.7087491 ]
    #    [0.7142536  0.88725346 0.61093384 0.38833922]]]]
