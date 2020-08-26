.. _cn_api_fluid_layers_box_clip:

box_clip
-------------------------------

.. py:function:: paddle.fluid.layers.box_clip(input, im_info, name=None)

:alias_main: paddle.nn.functional.box_clip
:alias: paddle.nn.functional.box_clip,paddle.nn.functional.vision.box_clip
:old_api: paddle.fluid.layers.box_clip



将检测框框剪切为 ``im_info`` 给出的大小。对于每个输入框，公式如下：

::

    xmin = max(min(xmin, im_w - 1), 0)
    ymin = max(min(ymin, im_h - 1), 0)
    xmax = max(min(xmax, im_w - 1), 0)
    ymax = max(min(ymax, im_h - 1), 0)

其中im_w和im_h是通过im_info计算的：

::

    im_h = round(height / scale)
    im_w = round(weight / scale)


参数：
    - **input** (Variable)  – 维度为[N_1, N_2, ..., N_k, 4]的多维Tensor，其中最后一维为box坐标维度。数据类型为float32或float64。
    - **im_info** (Variable)  – 维度为[N, 3]的2-D Tensor，N为输入图片个数。具有（高度height，宽度width，比例scale）图像的信息，其中高度和宽度是输入大小，比例是输入大小和原始大小的比率。数据类型为float32或float64。
    - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回： 表示剪切后的检测框的Tensor或LoDTensor，数据类型为float32或float64，形状与输入检测框相同

返回类型：Variable


**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    boxes = fluid.data(
        name='boxes', shape=[None, 8, 4], dtype='float32', lod_level=1)
    im_info = fluid.data(name='im_info', shape=[None, 3])
    out = fluid.layers.box_clip(
        input=boxes, im_info=im_info)
