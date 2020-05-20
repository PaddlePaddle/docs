.. _cn_api_fluid_dygraph_Sequential:

Sequential
-------------------------------

.. py:class:: paddle.fluid.dygraph.Sequential(*layers)




顺序容器。子Layer将按构造函数参数的顺序添加到此容器中。传递给构造函数的参数可以Layers或可迭代的name Layer元组。

参数：
    - **layers** (tuple) - Layers或可迭代的name Layer对。

返回：无

**代码示例**

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np
    data = np.random.uniform(-1, 1, [30, 10]).astype('float32')
        # 使用 iterable Layers 创建 Sequential 容器
    with paddle.imperative.guard():
        data = paddle.imperative.to_variable(data)
        # 使用 iterable Layers 创建 Sequential 容器
        model1 = paddle.nn.Sequential(fluid.Linear(10, 1), fluid.Linear(1, 2))
        model1[0]
        res1 = model1(data)
        # 使用 iterable name Layer 对创建 Sequential 容器
        model2 = paddle.nn.Sequential(('l1', fluid.Linear(10, 2)), ('l2', fluid
            .Linear(2, 3)))
        model2['l1']
        model2.add_sublayer('l3', fluid.Linear(3, 3))
        res2 = model2(data)

