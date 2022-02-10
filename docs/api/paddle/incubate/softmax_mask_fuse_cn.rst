.. _cn_api_incubate_softmax_mask_fuse:

softmax_mask_fuse 
-------------------------------

.. py:function:: paddle.incubate.softmax_mask_fuse(x, mask, name=None)

该op是对输入 ``x`` 进行被输入 ``mask`` mask后的softmax操作。该op主要针对加速Transformer架构而设计。将 ``tmp = x + mask, rst = softmax(tmp)`` 两个操作合为一个操作。计算公式为：

.. math::
    out = softmax(x + mask)

.. note::
    该API只可在GPU上运行

参数
:::::::::
    - x (4-D Tensor) - 输入的Tensor，必须为4D的shape，数据类型为：float16、float32。x的第四维必须大于等于32，并且小于8192。
    - mask (4-D Tensor) - 输入的Tensor，必须为4D的shape，数据类型为：float16、float32。mask的第二维必须为1，其余维度必须与x的对应维度相同。
    - name (str, 可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref: `api_guide_Name` 。

返回
:::::::::
``Tensor``，维度和数据类型都与 ``x`` 相同，存储运算后的结果


代码示例
::::::::::

.. code-block:: python

    # required: gpu
    import paddle
    import paddle.incubate as incubate
    x = paddle.rand([2, 8, 8, 32])
    mask = paddle.rand([2, 1, 8, 32])
    rst = incubate.softmax_mask_fuse(x, mask)
    # [[[[0.02404429, 0.04658398, 0.02746007, ..., 0.01489375, 0.02397441, 0.02851614] ... ]]]
