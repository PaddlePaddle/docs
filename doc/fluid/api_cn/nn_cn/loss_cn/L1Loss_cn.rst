L1Loss
-------------------------------

.. py:function:: paddle.nn.loss.L1Loss(reduction='mean')

:alias_main: paddle.nn.L1Loss
:alias: paddle.nn.L1Loss,paddle.nn.layer.L1Loss,paddle.nn.layer.loss.L1Loss



该接口用于创建一个L1Loss的可调用类，L1Loss计算输入input和标签label间的 `L1 loss` 损失。

该损失函数的数学计算公式如下：

当 `reduction` 设置为 ``'none'`` 时，
    
    .. math::
        Out = |input - label|

当 `reduction` 设置为 ``'mean'`` 时，

    .. math::
       Out = MEAN(|input - label|)

当 `reduction` 设置为 ``'sum'`` 时，
    
    .. math::
       Out = SUM(|input - label|)

输入input和标签label的维度是[N, *], 其中N是batch_size， `*` 是任意其他维度。
如果 :attr:`reduction` 是 ``'none'``, 则输出Loss的维度为 [N, *], 与输入input相同。
如果 :attr:`reduction` 是 ``'mean'`` 或 ``'sum'``, 则输出Loss的维度为 [1]。

参数：
    - **reduction** (string, 可选): - 指定应用于输出结果的计算方式，可选值有: ``'none'``, ``'mean'``, ``'sum'`` 。默认为 ``'mean'``，计算 `L1Loss` 的均值；设置为 ``'sum'`` 时，计算 `L1Loss` 的总和；设置为 ``'none'`` 时，则返回L1Loss。数据类型为string。

返回：返回计算L1Loss的可调用对象。

**代码示例**

.. code-block:: python

        # declarative mode
        import paddle.fluid as fluid
        import numpy as np
        import paddle
        input = fluid.data(name="input", shape=[1])
        label = fluid.data(name="label", shape=[1])
        l1_loss = paddle.nn.loss.L1Loss(reduction='mean')
        output = l1_loss(input,label)
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())

        input_data = np.array([1.5]).astype("float32")
        label_data = np.array([1.7]).astype("float32")
        output_data = exe.run(fluid.default_main_program(),
                feed={"input":input_data, "label":label_data},
                fetch_list=[output],
                return_numpy=True)

        print(output_data)  # [array([0.2], dtype=float32)]
        
        # imperative mode
        import paddle.fluid.dygraph as dg
        with dg.guard(place) as g:
            input = dg.to_variable(input_data)
            label = dg.to_variable(label_data)
            l1_loss = paddle.nn.loss.L1Loss(reduction='mean')
            output = l1_loss(input,label)
            print(output.numpy())  # [0.2]

