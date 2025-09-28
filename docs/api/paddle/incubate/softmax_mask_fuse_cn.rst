.. _cn_api_paddle_incubate_softmax_mask_fuse:

softmax_mask_fuse
-------------------------------

.. py:function:: paddle.incubate.softmax_mask_fuse(x, mask, name=None)

对输入 ``x`` 进行使用输入 ``mask`` mask 后的 softmax 操作。
该 API 主要针对加速 Transformer 架构而设计。将 ``tmp = x + mask, rst = softmax(tmp)`` 两个操作合为一个操作。计算公式为：

.. math::
    out = softmax(x + mask)

.. note::
    该 API 只可在 GPU 上运行
参数
:::::::::
    - x (4-D Tensor) - 输入的 Tensor，必须为 4D 的 shape，数据类型为：float16、float32。x 的第四维必须大于等于 32，并且小于 8192。
    - mask (4-D Tensor) - 输入的 Tensor，必须为 4D 的 shape，数据类型为：float16、float32。mask 的第二维必须为 1，其余维度必须与 x 的对应维度相同。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
``Tensor``，维度和数据类型都与 ``x`` 相同，存储运算后的结果


代码示例
::::::::::

COPY-FROM: paddle.incubate.softmax_mask_fuse
