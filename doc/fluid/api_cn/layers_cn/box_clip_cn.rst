.. _cn_api_fluid_layers_box_clip:

box_clip
-------------------------------

.. py:function:: paddle.fluid.layers.box_clip(input, im_info, name=None)

将box框剪切为 ``im_info`` 给出的大小。对于每个输入框，公式如下：

::

    xmin = max(min(xmin, im_w - 1), 0)
    ymin = max(min(ymin, im_h - 1), 0)
    xmax = max(min(xmax, im_w - 1), 0)
    ymax = max(min(ymax, im_h - 1), 0)

其中im_w和im_h是从im_info计算的：

::

    im_h = round(height / scale)
    im_w = round(weight / scale)


参数：
    - **input (variable)**  – 输入框，最后一个维度为4
    - **im_info (variable)**  – 具有（高度height，宽度width，比例scale）排列的形为[N，3]的图像的信息。高度和宽度是输入大小，比例是输入大小和原始大小的比率
    - **name (str)**  – 该层的名称。 为可选项

返回：剪切后的tensor

返回类型： Variable


**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    boxes = fluid.layers.data(
        name='boxes', shape=[8, 4], dtype='float32', lod_level=1)
    im_info = fluid.layers.data(name='im_info', shape=[3])
    out = fluid.layers.box_clip(
        input=boxes, im_info=im_info, inplace=True)










