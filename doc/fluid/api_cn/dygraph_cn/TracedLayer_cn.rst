.. _cn_api_fluid_dygraph_TracedLayer:

TracedLayer
-------------------------------

.. py:class:: paddle.fluid.dygraph.TracedLayer(program, parameters, feed_names, fetch_names)

TracedLayer是一个由动态图模型转换而来的callable对象。TracedLayer内部会将动态图模型转换为静态图模型，并使用 ``Executor`` 和 ``CompiledProgram``
运行静态图模型。转换后的静态图模型与原动态图模型共享参数。

所有的TracedLayer对象均不应通过构造函数创建，而应通过调用静态方法 ``TracedLayer.trace(layer, inputs)`` 创建。

TracedLayer只能用于将data independent的动态图模型转换为静态图模型，即待转换的动态图模型不应随tensor数据或维度的变化而变化。

.. py:staticmethod:: trace(layer, inputs)

创建TracedLayer对象的唯一接口，该接口会调用 ``layer(*inputs)`` 方法运行动态图模型并将其转换为静态图模型。

参数:
    - **layer** (dygraph.Layer) - 待追踪的动态图layer对象。
    - **inputs** (list(Variable)) - 动态图layer对象的输入变量列表。

返回: 包含2个元素的tuple，其中第一个元素是 ``layer(*inputs)`` 的输出结果，第二个元素是转换后得到的TracedLayer对象。

返回类型: tuple

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    from paddle.fluid.dygraph import FC, to_variable, TracedLayer
    import numpy as np

    class ExampleLayer(fluid.dygraph.Layer):
        def __init__(self, name_scope):
            super(ExampleLayer, self).__init__(name_scope)
            self._fc = FC(self.full_name(), 10)

        def forward(self, input):
            return self._fc(input)

    with fluid.dygraph.guard():
        layer = ExampleLayer("example_layer")
        in_np = np.random.random([2, 3]).astype('float32')
        in_var = to_variable(in_np)
        out_dygraph, static_layer = TracedLayer.trace(layer, inputs=[in_var])
        out_static_graph = static_layer([in_var])
        print(len(out_static_graph)) # 1
        print(out_static_graph[0].shape) # (2, 10)

.. py:method:: set_strategy(build_strategy=None, exe_strategy=None)

设置构建和执行静态图模型的相关策略。

参数:
    - **build_strategy** (BuildStrategy, 可选) - TracedLayer内部 ``CompiledProgram`` 的构建策略。
    - **exec_strategy** (ExecutionStrategy, 可选) - TracedLayer内部 ``CompiledProgram`` 的执行策略。

返回: 无

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    from paddle.fluid.dygraph import FC, to_variable, TracedLayer
    import numpy as np

    class ExampleLayer(fluid.dygraph.Layer):
        def __init__(self, name_scope):
            super(ExampleLayer, self).__init__(name_scope)
            self._fc = FC(self.full_name(), 10)

        def forward(self, input):
            return self._fc(input)

    with fluid.dygraph.guard():
        layer = ExampleLayer("example_layer")
        in_np = np.random.random([2, 3]).astype('float32')
        in_var = to_variable(in_np)

        out_dygraph, static_layer = TracedLayer.trace(layer, inputs=[in_var])

        build_strategy = fluid.BuildStrategy()
        build_strategy.enable_inplace = True

        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.num_threads = 2

        static_layer.set_strategy(build_strategy=build_strategy, exec_strategy=exec_strategy)
        out_static_graph = static_layer([in_var])

.. py:method:: save_inference_model(dirname, feed=None, fetch)

将TracedLayer保存为用于预测部署的模型。保存的预测模型可被C++预测接口加载。

参数:
    - **dirname** (str) - 预测模型的保存目录。
    - **feed** (list(int), 可选) - 预测模型输入变量的索引。若为None，则TracedLayer的所有输入变量均会作为预测模型的输入。默认值为None。
    - **fetch** (list(int), 可选) - 预测模型输出变量的索引。若为None，则TracedLayer的所有输出变量均会作为预测模型的输出。默认值为None。

返回: fetch变量的名称列表

返回类型: list(str)

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    from paddle.fluid.dygraph import FC, to_variable, TracedLayer
    import numpy as np

    class ExampleLayer(fluid.dygraph.Layer):
        def __init__(self, name_scope):
            super(ExampleLayer, self).__init__(name_scope)
            self._fc = FC(self.full_name(), 10)

        def forward(self, input):
            return self._fc(input)

    with fluid.dygraph.guard():
        layer = ExampleLayer("example_layer")
        in_np = np.random.random([2, 3]).astype('float32')
        in_var = to_variable(in_np)
        out_dygraph, static_layer = TracedLayer.trace(layer, inputs=[in_var])
        fetch_var_names = static_layer.save_inference_model(
                    './saved_infer_model', feed=[0], fetch=[0])
        print(fetch_var_names) # [u'save_infer_model/scale_0']