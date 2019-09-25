.. _cn_api_fluid_layers_npair_loss:

npair_loss
-------------------------------

.. py:function:: paddle.fluid.layers.npair_loss(anchor, positive, labels, l2_reg=0.002)

**Npair Loss Layer**

参考阅读 `Improved Deep Metric Learning with Multi class N pair Loss Objective <http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf>`_

NPair损失需要成对的数据。NPair损失分为两部分：第一部分是对嵌入向量进行L2正则化；第二部分是每一对数据的相似性矩阵的每一行和映射到ont-hot之后的标签的交叉熵损失的和。

参数:
    - **anchor** (Variable) -  锚点图像的嵌入Tensor，形状为[batch_size, embedding_dims]的2-D Tensor。数据类型：float32和float64。
    - **positive** (Variable) -  正例图像的嵌入Tensor，形状为[batch_size, embedding_dims]的2-D Tensor。数据类型：float32和float64。
    - **labels** (Variable) - 标签向量，形状为[batch_size]的1-DTensor。数据类型：float32、float64和int64。
    - **l2_reg** (float) - 嵌入向量的L2正则化系数，默认：0.002。

返回： Tensor。经过npair loss计算之后的结果，是一个值。

返回类型：Variable

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np
    anchor = fluid.layers.data(
              name = 'anchor', shape = [18, 6], dtype = 'float32', append_batch_size=False)
    positive = fluid.layers.data(
              name = 'positive', shape = [18, 6], dtype = 'float32', append_batch_size=False)
    labels = fluid.layers.data(
              name = 'labels', shape = [18], dtype = 'float32', append_batch_size=False)

    res = fluid.layers.npair_loss(anchor, positive, labels, l2_reg = 0.002)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    a = np.random.rand(18, 6).astype("float32")
    p = np.random.rand(18, 6).astype("float32")
    l = np.random.rand(18).astype("float32")
    output = exe.run(feed={"anchor": a, "positive": p, "labels": l}, fetch_list=[res])
    print(output)






