NLLLoss
-------------------------------

.. py:function:: paddle.fluid.dygraph.NLLLoss(weight=None, reduction='mean', ignore_index=-100)

该OP计算输入input和标签label间的 `negative log likelihood loss` 损失 ，可用于训练一个 `n` 类分类器。

如果提供 `weight` 参数的话，它是一个 `1-D` 的tensor, 里面的值对应类别的权重。当你的训练集样本
不均衡的话，使用这个参数是非常有用的。

该损失函数的数学计算公式如下：

当 `reduction` 设置为 `none` 时，损失函数的数学计算公式为：

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_{y_n} x_{n,y_n}, \quad
        w_{c} = \text{weight}[c] \cdot \mathbb{1}\{c \not= \text{ignore_index}\},

其中 `N` 表示 `batch_size` 。如果 `reduction` 的值不是 `none` (默认为 `mean`)，那么此时损失函数
的数学计算公式为：

    .. math::
        \ell(x, y) = \begin{cases}
            \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n}} l_n, &
            \text{if reduction} = \text{'mean';}\\
            \sum_{n=1}^N l_n,  &
            \text{if reduction} = \text{'sum'.}
        \end{cases}

参数：
    - **input** (Variable): - 输入 `Tensor`, 其形状为 :math:`[N, C]` , 其中 `C` 为类别数。但是对于多维度的情形下，它的形状为 :math:`[N, C, d_1, d_2, ..., d_K]` 。数据类型为float32或float64。
    - **label** (Variable): - 输入input对应的标签值。其形状为 :math:`[N,]` 或者 :math:`[N, d_1, d_2, ..., d_K]`, 数据类型为int64。
    - **weight** (Variable, 可选): - 手动指定每个类别的权重。其默认为 `None` 。如果提供该参数的话，长度必须为 `num_classes` 。数据类型为float32或float64。
    - **reduction** (string, 可选): - 指定应用于输出结果的计算方式，可选值有: `none`, `mean`, `sum` 。默认为 `mean` ，计算 `mini-batch` loss均值。设置为 `sum` 时，计算 `mini-batch` loss的总和。设置为 `none` 时，则返回loss Tensor。数据类型为string。
    - **ignore_index** (int64, 可选): - 指定一个忽略的标签值，此标签值不参与计算。默认值为-100。数据类型为int64。

返回：返回存储表示 `negative log likihood loss` 的损失值。

返回类型：Variable

**代码示例**

..  code-block:: python

            # declarative mode
            import paddle.fluid as fluid
            import numpy as np
            import paddle
            input_np = np.random.random(size=(10, 10)).astype(np.float32)
            label_np = np.random.randint(0, 10, size=(10,)).astype(np.int64)
            prog = fluid.Program()
            startup_prog = fluid.Program()
            place = fluid.CPUPlace()
            with fluid.program_guard(prog, startup_prog):
                input = fluid.data(name='input', shape=[10, 10], dtype='float32')
                label = fluid.data(name='label', shape=[10], dtype='int64')
                nll_loss = fluid.dygraph.NLLLoss()
                res = nll_loss(input, label)
                exe = fluid.Executor(place)
                static_result = exe.run(
                    prog,
                    feed={"input": input_np,
                          "label": label_np},
                    fetch_list=[res])
            print(static_result)
            
            # imperative mode
            import paddle.fluid.dygraph as dg
            with dg.guard(place) as g:
                input = dg.to_variable(input_np)
                label = dg.to_variable(label_np)
                output = nll_loss(input, label)
                print(output.numpy())

