.. _cn_api_fluid_layers_image_resize_short:

image_resize_short
-------------------------------

.. py:function:: paddle.fluid.layers.image_resize_short(input, out_short_len, resample='BILINEAR')




该 OP 用于调整一批图片的大小。输入图像的短边将被调整为给定的 out_short_len。输入图像的长边按比例调整大小，最终图像的长宽比保持不变。

参数
::::::::::::

        - **input** (Variable) -  图像调整图层的输入 Tensor，这是一个维度为[num_batch, channels, in_h, in_w]的 4-D Tensor。
        - **out_short_len** (int) -  输出图像的短边长度。
        - **resample** (str) - resample 方法，默认为双线性插值。

返回
::::::::::::
 4 维 Tensor，shape 为(num_batches, channels, out_h, out_w)

返回类型
::::::::::::
 变量（variable）

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.image_resize_short
