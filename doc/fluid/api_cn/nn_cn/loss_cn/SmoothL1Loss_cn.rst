SmoothL1Loss
-------------------------------

.. py:class:: paddle.nn.loss.CrossEntropyLoss(reduction='mean')

该OP计算输入x和标签label间的SmoothL1损失，如果逐个元素的绝对误差低于1，则创建使用平方项的条件
，否则为L1损失。在某些情况下，它可以防止爆炸梯度, 也称为Huber损失,该损失函数的数学计算公式如下：

    .. math::
         loss(x,y)=\\frac{1}{n}\\sum_{i}z_i

`z_i`的计算公式如下：

    .. math::

         \\mathop{z_i}=\\left\\{\\begin{array}{rcl}
        0.5(x_i - y_i)^2 & & {if |x_i - y_i| > 1} \\\\
        |x_i - y_i| - 0.5 & & {otherwise}
        \\end{array} \\right.


参数
::::::::::
    - **reduction** (string, 可选): - 指定应用于输出结果的计算方式，数据类型为string，可选值有: `none`, `mean`, `sum` 。默认为 `mean` ，计算 `mini-batch` loss均值。设置为 `sum` 时，计算 `mini-batch` loss的总和。设置为 `none` 时，则返回loss Tensor。
    
调用参数
::::::::::
    - **x** (Tensor): 输入 `Tensor`， 数据类型为float32或float64。其形状为 :math:`[N, C]` , 其中 `C` 为类别数。对于多维度的情形下，它的形状为 :math:`[N, C, d_1, d_2, ..., d_k]`，k >= 1。
    - **label** (Tensor): 输入x对应的标签值，数据类型为float32。数据类型和x相同。



返回：返回计算 `SmoothL1Loss` 后的损失值。

返回类型：Tensor

**代码示例**

..  code-block:: python

            # declarative mode
            import paddle
            import paddle.fluid as fluid
            import numpy as np
            x = fluid.layers.data(name="x", shape=[-1, 3], dtype="float32")
            label = fluid.layers.data(name="label", shape=[-1, 3], dtype="float32")
            loss = paddle.nn.SmoothL1Loss()
            result = loss(x,label)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            x = np.random.rand(3,3).astype("float32")
            label = np.random.rand(3,3).astype("float32")
            output= exe.run(feed={"x": x, "label": label},
                            fetch_list=[result])
            print(output)
            # imperative mode
            import paddle.fluid.dygraph as dg
            with dg.guard(place) as g:
                x = dg.to_variable(input_data)
                label = dg.to_variable(label_data)
                weight = dg.to_variable(weight_data)
                loss = paddle.nn.SmoothL1Loss()
                output = loss(x, label)
                print(output.numpy())