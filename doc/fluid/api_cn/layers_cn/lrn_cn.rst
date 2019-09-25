.. _cn_api_fluid_layers_lrn:

lrn
-------------------------------

.. py:function:: paddle.fluid.layers.lrn(input, n=5, k=1.0, alpha=0.0001, beta=0.75, name=None)


该OP实现了局部响应正则化层（Local Response Normalization Layer），用于对局部输入区域正则化，执行一种侧向抑制（lateral inhibition）。更多详情参考： `ImageNet Classification with Deep Convolutional Neural Networks <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`_

其中 ``input`` 是mini-batch的输入特征。计算过程如下：

.. math::

    Output(i,x,y) = Input(i,x,y)/\left ( k+\alpha \sum_{j=max(0,i-n/2)}^{min(C-1,i+n/2)}(Input(j,x,y))^2 \right )^\beta

在以上公式中：
  - :math:`n` ：累加的通道数
  - :math:`k` ：位移
  - :math:`\alpha` ： 缩放参数
  - :math:`\beta` ： 指数参数


参数：
    - **input** （Variable）- 输入特征，维度为[N,C,H,W]的4D-Tensor，其中N为batch大小，C为输入通道数，H为特征高度，W为特征宽度。必须包含4个维度，否则会抛出 ``ValueError`` 的异常。数据类型为float32。
    - **n** (int，可选） - 累加的通道数，默认值为5。
    - **k** （float，可选）- 位移，正数。默认值为1.0。
    - **alpha** （float，可选）- 缩放参数，正数。默认值为1e-4。
    - **beta** （float，可选）- 指数，正数。默认值为0.75。
    - **name** (None|str) – 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。


返回：局部响应正则化得到的输出特征，数据类型及维度和input相同。

返回类型：Variable

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.data(
        name="data", shape=[3, 112, 112], dtype="float32")
    lrn = fluid.layers.lrn(input=data)
    print(lrn.shape)  # [-1, 3, 112, 112]
    print(lrn.dtype)  # float32
