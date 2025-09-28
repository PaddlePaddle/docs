.. _cn_api_paddle_nn_functional_local_response_norm:

local_response_norm
-------------------------------

.. py:function:: paddle.nn.functional.local_response_norm(x, size, alpha=1e-4, beta=0.75, k=1., data_format="NCHW", name=None)

局部响应正则化（Local Response Normalization）用于对局部输入区域进行正则化，执行一种侧向抑制（lateral inhibition）。更多详情参考：`ImageNet Classification with Deep Convolutional Neural Networks <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`_

其中 ``input`` 是 mini-batch 的输入特征。计算过程如下：

.. math::

    Output(i,x,y) = Input(i,x,y)/\left ( k+\alpha \sum_{j=max(0,i-size/2)}^{min(C-1,i+size/2)}(Input(j,x,y))^2 \right )^\beta

在以上公式中：

  - :math:`size`：累加的通道数
  - :math:`k`：位移
  - :math:`\alpha`：缩放参数
  - :math:`\beta`：指数参数

参数
:::::::::
 - **x** （Tensor）- 输入的 3-D/4-D/5-D `Tensor`，数据类型为：float16 或 float32。
 - **size** (int） - 累加的通道数。
 - **alpha** （float，可选）- 缩放参数，正数。默认值为 1e-4。
 - **beta** （float，可选）- 指数，正数。默认值为 0.75。
 - **k** （float，可选）- 位移，正数。默认值为 1.0。
 - **data_format** (str，可选) - 指定输入的数据格式，输出的数据格式将与输入保持一致。如果输入是 3-D `Tensor`，该参数可以是"NCL"或"NLC"，其中 N 是批尺寸，C 是通道数，L 是特征长度。如果输入是 4-D `Tensor`，该参数可以是"NCHW"或"NHWC"，其中 N 是批尺寸，C 是通道数，H 是特征高度，W 是特征宽度。如果输入是 5-D `Tensor`，该参数可以是"NCDHW"或"NDHWC"，其中 N 是批尺寸，C 是通道数，D 是特征深度，H 是特征高度，W 是特征宽度。默认值："NCHW"。
 - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
局部响应正则化得到的输出特征，数据类型及维度和 input 相同的 `Tensor` 。

代码示例
:::::::::

COPY-FROM: paddle.nn.functional.local_response_norm
