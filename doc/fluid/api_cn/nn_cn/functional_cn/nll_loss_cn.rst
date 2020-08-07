nll_loss
-------------------------------
.. py:function:: paddle.nn.functional.nll_loss(x, label, weight=None, ignore_index=-100, reduction='mean')

该接口返回`negative log likelihood`。可在`paddle.nn.loss.NLLLoss`查看详情。

参数
:::::::::
    - **x** (Tensor): - 输入 `Tensor`, 其形状为 :math:`[N, C]` , 其中 `C` 为类别数。但是对于多维度的情形下，它的形状为 :math:`[N, C, d_1, d_2, ..., d_K]` 。数据类型为float32或float64。
    - **label** (Tensor): - 输入x对应的标签值。其形状为 :math:`[N,]` 或者 :math:`[N, d_1, d_2, ..., d_K]`, 数据类型为int64。
    - **weight** (Tensor, 可选): - 手动指定每个类别的权重。其默认为 `None` 。如果提供该参数的话，长度必须为 `num_classes` 。数据类型为float32或float64。
    - **ignore_index** (int64, 可选): - 指定一个忽略的标签值，此标签值不参与计算。默认值为-100。数据类型为int64。
    - **reduction** (string, 可选): - 指定应用于输出结果的计算方式，可选值有: `none`, `mean`, `sum` 。默认为 `mean` ，计算 `mini-batch` loss均值。设置为 `sum` 时，计算 `mini-batch` loss的总和。设置为 `none` 时，则返回loss Tensor。数据类型为string。

返回
:::::::::
返回存储表示 `negative log likihood loss` 的损失值。

代码示例
:::::::::

.. code-block:: python

        import paddle
        import numpy as np
        from paddle.nn.functional import nll_loss
        log_softmax = paddle.nn.LogSoftmax(axis=1)
        
        x_np = np.random.random(size=(10, 10)).astype(np.float32)
        label_np = np.random.randint(0, 10, size=(10,)).astype(np.int64)
        
        place = paddle.CPUPlace()
        
        # imperative mode
        paddle.enable_imperative(place)
        x = paddle.imperative.to_variable(x_np)
        log_out = log_softmax(x)
        label = paddle.imperative.to_variable(label_np)
        imperative_result = nll_loss(log_out, label)
        print(imperative_result.numpy())
        
        # declarative mode
        paddle.disable_imperative()
        prog = paddle.Program()
        startup_prog = paddle.Program()
        with paddle.program_guard(prog, startup_prog):
            x = paddle.nn.data(name='x', shape=[10, 10], dtype='float32')
            label = paddle.nn.data(name='label', shape=[10], dtype='int64')
            log_out = log_softmax(x)
            res = nll_loss(log_out, label)
        
            exe = paddle.Executor(place)
            declaritive_result = exe.run(
                prog,
                feed={"x": x_np,
                      "label": label_np},
                fetch_list=[res])
        print(declaritive_result)
