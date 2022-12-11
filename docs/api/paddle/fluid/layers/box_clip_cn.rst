.. _cn_api_fluid_layers_box_clip:

box_clip
-------------------------------

.. py:function:: paddle.fluid.layers.box_clip(input, im_info, name=None)




将检测框框剪切为 ``im_info`` 给出的大小。对于每个输入框，公式如下：

::

    xmin = max(min(xmin, im_w - 1), 0)
    ymin = max(min(ymin, im_h - 1), 0)
    xmax = max(min(xmax, im_w - 1), 0)
    ymax = max(min(ymax, im_h - 1), 0)

其中 im_w 和 im_h 是通过 im_info 计算的：

::

    im_h = round(height / scale)
    im_w = round(weight / scale)


参数
::::::::::::

    - **input** (Variable)  – 维度为[N_1, N_2, ..., N_k, 4]的多维 Tensor，其中最后一维为 box 坐标维度。数据类型为 float32 或 float64。
    - **im_info** (Variable)  – 维度为[N, 3]的 2-D Tensor，N 为输入图片个数。具有（高度 height，宽度 width，比例 scale）图像的信息，其中高度和宽度是输入大小，比例是输入大小和原始大小的比率。数据类型为 float32 或 float64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
 表示剪切后的检测框的 Tensor，数据类型为 float32 或 float64，形状与输入检测框相同

返回类型
::::::::::::
Variable


代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.box_clip
