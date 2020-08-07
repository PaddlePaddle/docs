MSELoss
-------------------------------

.. py:function:: paddle.nn.loss.MSELoss(reduction='mean')

该OP用于计算预测值和目标值的均方差误差。

对于预测值input和目标值label：

当reduction为'none'时：

.. math::
    Out = (input - label)^2

当`reduction`为`'mean'`时：

.. math::
    Out = \operatorname{mean}((input - label)^2)

当`reduction`为`'sum'`时：

.. math::
    Out = \operatorname{sum}((input - label)^2)

参数：
    - **reduction** (str, 可选) - 约简方式，可以是 'none' | 'mean' | 'sum'。设为'none'时不使用约简，设为'mean'时返回loss的均值，设为'sum'时返回loss的和。

形状:
    - **input** (Variable) - 预测值，维度为 :math:`[N_1, N_2, ..., N_k]` 的多维Tensor。数据类型为float32或float64。
    - **label** (Variable) - 目标值，维度为 :math:`[N_1, N_2, ..., N_k]` 的多维Tensor。数据类型为float32或float64。
    

返回：预测值和目标值的均方差

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    import numpy as np
    import paddle
    from paddle import fluid
    import paddle.fluid.dygraph as dg

    mse_loss = paddle.nn.loss.MSELoss()
    input = fluid.data(name="input", shape=[1])
    label = fluid.data(name="label", shape=[1])
    place = fluid.CPUPlace()
    input_data = np.array([1.5]).astype("float32")
    label_data = np.array([1.7]).astype("float32")

    # declarative mode
    output = mse_loss(input,label)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    output_data = exe.run(
        fluid.default_main_program(),
        feed={"input":input_data, "label":label_data},
        fetch_list=[output],
        return_numpy=True)
    print(output_data)
    # [array([0.04000002], dtype=float32)]

    # imperative mode
    with dg.guard(place) as g:
        input = dg.to_variable(input_data)
        label = dg.to_variable(label_data)
        output = mse_loss(input, label)
        print(output.numpy())
        # [0.04000002]
