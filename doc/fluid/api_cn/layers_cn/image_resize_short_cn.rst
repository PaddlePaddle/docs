.. _cn_api_fluid_layers_image_resize_short:

image_resize_short
-------------------------------

.. py:function:: paddle.fluid.layers.image_resize_short(input, out_short_len, resample='BILINEAR')

调整一批图片的大小。输入图像的短边将被调整为给定的out_short_len 。输入图像的长边按比例调整大小，最终图像的长宽比保持不变。

参数:
        - **input** (Variable) -  图像调整图层的输入张量，这是一个4维的形状张量(num_batch, channels, in_h, in_w)。
        - **out_short_len** (int) -  输出图像的短边长度。
        - **resample** (str) - resample方法，默认为双线性插值。

返回： 4维张量，shape为(num_batches, channels, out_h, out_w)

返回类型: 变量（variable）

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    input = fluid.layers.data(name="input", shape=[3,6,9], dtype="float32")
    out = fluid.layers.image_resize_short(input, out_short_len=3)

