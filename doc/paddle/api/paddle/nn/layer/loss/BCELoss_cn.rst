.. _cn_api_paddle_nn_BCELoss:

BCELoss
-------------------------------

.. py:function:: paddle.nn.BCELoss(input, label, weight=None, reduction='mean')

该接口用于创建一个BCELoss的可调用类，用于计算输入和标签之间的二值交叉熵损失值。二值交叉熵损失函数公式如下：

当 `weight` 不为空时，公式为：

.. math::
  Out = -1 * weight * (label * log(input) + (1 - label) * log(1 - input))

当 `weight` 为空时，公式为：

.. math::
  Out = -1 * (label * log(input) + (1 - label) * log(1 - input))

当 `reduction` 为 `none` 时，最终的输出结果为：

.. math::
  Out = Out

当 `reduction` 为 `sum` 时，最终的输出结果为：

.. math::
  Out = MEAN(Out)

当 `reduction` 为 `sum` 时，最终的输出结果为：

.. math::
  Out = SUM(Out)


**注意：输入数据一般是 `fluid.layers.sigmoid` 的输出。因为是二分类，所以标签值应该是0或者1。

输入input和标签label的维度是[N, *], 其中N是batch_size， `*` 是任意其他维度。
如果 :attr:`reduction` 是 ``'none'``, 则输出的维度为 [N, *], 与输入input的形状相同。
如果 :attr:`reduction` 是 ``'mean'`` 或 ``'sum'``, 则输出的维度为 [1]。

参数：
  - **weight(Variable, optional)**：- 手动指定每个batch二值交叉熵的权重，如果指定的话，维度必须是一个batch的数据的维度。数据类型是float32, float64。默认是：None。
  - **reduction(str, optional)**：- 指定应用于输出结果的计算方式，可选值有: ``'none'``, ``'mean'``, ``'sum'`` 。默认为 ``'mean'``，计算 `BCELoss` 的均值；设置为 ``'sum'`` 时，计算 `BCELoss` 的总和；设置为 ``'none'`` 时，则返回BCELoss。

返回：返回计算BCELoss的可调用对象。

**代码示例**

.. code-block:: python

    # declarative mode
    import paddle.fluid as fluid
    import numpy as np
    import paddle
    input = fluid.data(name="input", shape=[3, 1], dtype='float32')
    label = fluid.data(name="label", shape=[3, 1], dtype='float32')
    bce_loss = paddle.nn.loss.BCELoss()
    output = bce_loss(input, label)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    
    input_data = np.array([0.5, 0.6, 0.7]).astype("float32")
    label_data = np.array([1.0, 0.0, 1.0]).astype("float32")
    output_data = exe.run(fluid.default_main_program(),
            feed={"input":input_data, "label":label_data},
            fetch_list=[output],
            return_numpy=True)
    
    print(output_data)  # [array([0.65537095], dtype=float32)]
    
    # imperative mode
    import paddle.fluid.dygraph as dg
    with dg.guard(place) as g:
        input = dg.to_variable(input_data)
        label = dg.to_variable(label_data)
        output = bce_loss(input, label)
        print(output.numpy())  # [0.65537095]
