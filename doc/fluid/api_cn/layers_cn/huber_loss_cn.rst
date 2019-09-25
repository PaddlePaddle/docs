.. _cn_api_fluid_layers_huber_loss:

huber_loss
-------------------------------

.. py:function:: paddle.fluid.layers.huber_loss(input, label, delta)


该OP计算输入（input）与标签（label）之间的Huber损失。Huber损失是常用的回归损失之一，相较于平方误差损失，Huber损失减小了对异常点的敏感度，更具鲁棒性。

当输入与标签之差的绝对值大于delta时，计算线性误差:

.. math::
        huber\_loss = delta * (label - input) - 0.5 * delta * delta

当输入与标签之差的绝对值小于delta时，计算平方误差:

.. math::
        huber\_loss = 0.5 * (label - input) * (label - input)


参数:
  - **input** （Variable） - 输入的预测数据，维度为[batch_size, 1] 的2D-Tensor，且最后一维必须是1。数据类型为float32或float64。
  - **label** （Variable） - 输入的真实标签，维度为[batch_size, 1] 的2D-Tensor，且最后一维必须是1。数据类型为float32或float64。
  - **delta** （float） -  Huber损失的阈值参数，用于控制Huber损失对线性误差或平方误差的侧重。数据类型为float32。

返回： 计算出的Huber损失，维度为[batch_size, 1] 的二维Tensor，数据类型与input相同。

返回类型: Variable



**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    import numpy as np

    DATATYPE='float32'
    input_data = np.array([[1.],[2.],[3.],[4.]]).astype(DATATYPE)
    label_data = np.array([[3.],[3.],[4.],[4.]]).astype(DATATYPE)

    x = fluid.layers.data(name='input', shape=[1], dtype=DATATYPE)
    y = fluid.layers.data(name='label', shape=[1], dtype=DATATYPE)
    loss = fluid.layers.huber_loss(input=x, label=y, delta=1.0)

    place = fluid.CPUPlace()
    #place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    HuberLoss, = exe.run(feed={'input':input_data ,'label':label_data}, fetch_list=[loss.name])
    print(HuberLoss)  #[[1.5], [0.5], [0.5], [0. ]], dtype=float32
