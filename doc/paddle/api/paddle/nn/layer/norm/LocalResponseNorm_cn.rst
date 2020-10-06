.. _cn_api_nn_LocalResponseNorm:

LocalResponseNorm
-------------------------------

.. py:function:: paddle.nn.LocalResponseNorm(size, alpha=0.0001, beta=0.75, k=1.0, data_format="NCHW", name=None)

局部响应正则化（Local Response Normalization）用于对局部输入区域进行正则化，执行一种侧向抑制（lateral inhibition）。更多详情参考： `ImageNet Classification with Deep Convolutional Neural Networks <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`_

.. note::
   对应的 `functional方法` 请参考: :ref:`cn_api_nn_functional_local_response_norm` 。

参数
:::::::::
 - **size** (int） - 累加的通道数。
 - **alpha** （float，可选）- 缩放参数，正数。默认值为1e-4。
 - **beta** （float，可选）- 指数，正数。默认值为0.75。
 - **k** （float，可选）- 位移，正数。默认值为1.0。
 - **data_format** (str，可选) - 指定输入的数据格式，输出的数据格式将与输入保持一致。如果输入是三维 `Tensor` ,该参数可以是"NCL"或"NLC"，其中N是批尺寸，C是通道数，L是特征长度。如果输入是四维 `Tensor` ,该参数可以是"NCHW"或"NHWC"，其中N是批尺寸，C是通道数，H是特征高度，W是特征宽度。如果输入是五维 `Tensor` ,该参数可以是"NCDHW"或"NDHWC"，其中N是批尺寸，C是通道数，D是特征深度，H是特征高度，W是特征宽度。默认值："NCHW"。
 - **name** (str，可选): 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

形状
:::::::::
 - **输入**  - 三维/四维/五维 `Tensor` 。
 - **输出**  - 数据类型及维度和输入相同的 `Tensor` 。

代码示例
:::::::::

.. code-block:: python

    import paddle

    x = paddle.rand(shape=(3, 3, 112, 112), dtype="float32")
    m = paddle.nn.LocalResponseNorm(size=5)
    y = m(x)
    print(y.shape)  # [3, 3, 112, 112]
