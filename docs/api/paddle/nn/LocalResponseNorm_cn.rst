.. _cn_api_paddle_nn_LocalResponseNorm:

LocalResponseNorm
-------------------------------

.. py:function:: paddle.nn.LocalResponseNorm(size, alpha=0.0001, beta=0.75, k=1.0, data_format="NCHW", name=None)

局部响应正则化（Local Response Normalization）用于对局部输入区域进行正则化，执行一种侧向抑制（lateral inhibition）。更多详情参考：`ImageNet Classification with Deep Convolutional Neural Networks <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`_

.. note::
   对应的 `functional 方法` 请参考：:ref:`cn_api_paddle_nn_functional_local_response_norm` 。

参数
:::::::::
 - **size** (int） - 累加的通道数。
 - **alpha** （float，可选）- 缩放参数，正数。默认值为 1e-4。
 - **beta** （float，可选）- 指数，正数。默认值为 0.75。
 - **k** （float，可选）- 位移，正数。默认值为 1.0。
 - **data_format** (str，可选) - 指定输入的数据格式，输出的数据格式将与输入保持一致。如果输入是 3-D `Tensor`，该参数可以是"NCL"或"NLC"，其中 N 是批尺寸，C 是通道数，L 是特征长度。如果输入是 4-D `Tensor`，该参数可以是"NCHW"或"NHWC"，其中 N 是批尺寸，C 是通道数，H 是特征高度，W 是特征宽度。如果输入是 5-D `Tensor`，该参数可以是"NCDHW"或"NDHWC"，其中 N 是批尺寸，C 是通道数，D 是特征深度，H 是特征高度，W 是特征宽度。默认值："NCHW"。
 - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
:::::::::
 - **input** ：3-D/4-D/5-D Tensor。
 - **output** ：数据类型及维度和输入相同的 Tensor。

代码示例
:::::::::

COPY-FROM: paddle.nn.LocalResponseNorm
