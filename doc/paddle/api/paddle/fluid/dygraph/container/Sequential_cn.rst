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

    import paddle.fluid as fluid
    import numpy as np
    data = np.random.uniform(-1, 1, [30, 10]).astype('float32')
    with fluid.dygraph.guard():
        data = fluid.dygraph.to_variable(data)
        # 使用 iterable Layers 创建 Sequential 容器
        model1 = fluid.dygraph.Sequential(
            fluid.Linear(10, 1), fluid.Linear(1, 2)
        )
        model1[0]  # 访问第一个子层
        res1 = model1(data)  # 顺序执行
        # 使用 iterable name Layer 对创建 Sequential 容器
        model2 = fluid.dygraph.Sequential(
            ('l1', fluid.Linear(10, 2)),
            ('l2', fluid.Linear(2, 3))
        )
        model2['l1']  # 访问 l1 子层
        model2.add_sublayer('l3', fluid.Linear(3, 3))  # 添加子层
        res2 = model2(data)  # 顺序执行


