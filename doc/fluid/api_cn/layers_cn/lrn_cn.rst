.. _cn_api_fluid_layers_lrn:

lrn
-------------------------------

.. py:function:: paddle.fluid.layers.lrn(input, n=5, k=1.0, alpha=0.0001, beta=0.75, name=None)

局部响应正则层（Local Response Normalization Layer）

该层对局部输入区域正则化，执行一种侧向抑制（lateral inhibition）。

公式如下：

.. math::

    Output(i,x,y) = Input(i,x,y)/\left ( k+\alpha \sum_{j=max(0,i-n/2)}^{min(C-1,i+n/2)}(Input(j,x,y))^2 \right )^\beta

在以上公式中：
  - :math:`n` ：累加的通道数
  - :math:`k` ：位移（避免除数为0）
  - :math:`\alpha` ： 缩放参数
  - :math:`\beta` ： 指数参数

参考 : `ImageNet Classification with Deep Convolutional Neural Networks <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`_

参数：
    - **input** （Variable）- 该层输入张量，输入张量维度必须为4
    - **n** (int，默认5） - 累加的通道数
    - **k** （float，默认1.0）- 位移（通常为正数，避免除数为0）
    - **alpha** （float，默认1e-4）- 缩放参数
    - **beta** （float，默认0.75）- 指数
    - **name** （str，默认None）- 操作符名

抛出异常:
  - ``ValueError`` - 如果输入张量的阶不为4

返回：张量，存储转置结果

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.data(
        name="data", shape=[3, 112, 112], dtype="float32")
    lrn = fluid.layers.lrn(input=data)











