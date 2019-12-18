.. _cn_api_fluid_dygraph_Sequential:

Sequential
-------------------------------

.. py:class:: paddle.fluid.dygraph.Sequential(layers)

顺序容器。子 Layer 将按构造函数中 layers 参数的顺序添加到此容器中。
传递给构造函数的参数可以是可迭代的 Layer 或可迭代的 name Layer 对。

参数：
    - **layers** (iterable) - 可迭代的 Layer 或可迭代的 name Layer 对。

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
            fluid.FC('fc1', 2),
            fluid.FC('fc2', 3)
        )
        model1[0]  # 访问 fc1 子层
        res1 = model1(data)  # 顺序执行
        # 使用 iterable name-Layer 对创建 Sequential 容器
        model2 = fluid.dygraph.Sequential(
            ('l1', fluid.FC('l1', 2)),
            ('l2', fluid.FC('l2', 3))
        )
        model2['l1']  # 访问 fc1 子层
        model2.add_sublayer('l3', fluid.FC('l3', 3))  # 添加子层
        print([l.full_name() for l in model2.sublayers()])  # ['l1/FC_0', 'l2/FC_0', 'l3/FC_0']
        res2 = model2(data)  # 顺序执行


