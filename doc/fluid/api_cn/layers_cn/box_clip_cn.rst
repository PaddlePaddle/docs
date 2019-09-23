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

其中im_w和im_h是通过im_info计算的：

::

    im_h = round(height / scale)
    im_w = round(weight / scale)


参数：
    - **input (Tensor|LoDTensor)**  – 数据类型为float，double的Tensor或者LoDTensor。输入检测框，最后一个维度为4
    - **im_info (Tensor)**  – 数据类型为float，double的Tensor。格式为[N, 3]，N为输入图片个数。具有（高度height，宽度width，比例scale）图像的信息，其中高度和宽度是输入大小，比例是输入大小和原始大小的比率
    - **name (str|None)**  – 该层的名称。 为可选项，默认为None

返回： Variable（Tensor|LoDTensor），数据类型为float，double的Tensor或者LoDTensor。剪切后的检测框，形状与输入检测框相同


**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    boxes = fluid.layers.data(
        name='boxes', shape=[8, 4], dtype='float32', lod_level=1)
    im_info = fluid.layers.data(name='im_info', shape=[3])
    out = fluid.layers.box_clip(
        input=boxes, im_info=im_info)