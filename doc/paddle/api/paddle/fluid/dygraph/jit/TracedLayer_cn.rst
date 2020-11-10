.. _cn_api_fluid_dygraph_TracedLayer:

TracedLayer
-------------------------------


.. py:class:: paddle.jit.TracedLayer(program, parameters, feed_names, fetch_names)




TracedLayer用于将前向动态图模型转换为静态图模型，主要用于将动态图保存后做在线C++预测。除此以外，用户也可使用转换后的静态图模型在Python端做预测，通常比原先的动态图性能更好。

TracedLayer使用 ``Executor`` 和 ``CompiledProgram`` 运行静态图模型。转换后的静态图模型与原动态图模型共享参数。

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

    import paddle

    class ExampleLayer(paddle.nn.Layer):
        def __init__(self):
            super(ExampleLayer, self).__init__()
            self._fc = paddle.nn.Linear(3, 10)

        def forward(self, input):
            return self._fc(input)

    layer = ExampleLayer()
    in_var = paddle.uniform(shape=[2, 3], dtype='float32')
    out_dygraph, static_layer = paddle.jit.TracedLayer.trace(layer, inputs=[in_var])

    # 内部使用Executor运行静态图模型
    out_static_graph = static_layer([in_var])
    print(len(out_static_graph)) # 1
    print(out_static_graph[0].shape) # (2, 10)

    # 将静态图模型保存为预测模型
    static_layer.save_inference_model(dirname='./saved_infer_model')

.. py:method:: set_strategy(build_strategy=None, exec_strategy=None)

设置构建和执行静态图模型的相关策略。

参数:
    - **build_strategy** (BuildStrategy, 可选) - TracedLayer内部 ``CompiledProgram`` 的构建策略。
    - **exec_strategy** (ExecutionStrategy, 可选) - TracedLayer内部 ``CompiledProgram`` 的执行策略。

返回: 无

**代码示例**

.. code-block:: python

    import paddle

    class ExampleLayer(paddle.nn.Layer):
        def __init__(self):
            super(ExampleLayer, self).__init__()
            self._fc = paddle.nn.Linear(3, 10)

        def forward(self, input):
            return self._fc(input)

    layer = ExampleLayer()
    in_var = paddle.uniform(shape=[2, 3], dtype='float32')

    out_dygraph, static_layer = paddle.jit.TracedLayer.trace(layer, inputs=[in_var])

    build_strategy = paddle.static.BuildStrategy()
    build_strategy.enable_inplace = True

    exec_strategy = paddle.static.ExecutionStrategy()
    exec_strategy.num_threads = 2

    static_layer.set_strategy(build_strategy=build_strategy, exec_strategy=exec_strategy)
    out_static_graph = static_layer([in_var])

.. py:method:: save_inference_model(dirname, feed=None, fetch=None)

将TracedLayer保存为用于预测部署的模型。保存的预测模型可被C++预测接口加载。

参数:
    - **dirname** (str) - 预测模型的保存目录。
    - **feed** (list(int), 可选) - 预测模型输入变量的索引。若为None，则TracedLayer的所有输入变量均会作为预测模型的输入。默认值为None。
    - **fetch** (list(int), 可选) - 预测模型输出变量的索引。若为None，则TracedLayer的所有输出变量均会作为预测模型的输出。默认值为None。

返回: 无

**代码示例**

.. code-block:: python

    import numpy as np
    import paddle

    class ExampleLayer(paddle.nn.Layer):
        def __init__(self):
            super(ExampleLayer, self).__init__()
            self._fc = paddle.nn.Linear(3, 10)

        def forward(self, input):
            return self._fc(input)

    save_dirname = './saved_infer_model'
    in_np = np.random.random([2, 3]).astype('float32')
    in_var = to_variable(in_np)
    layer = ExampleLayer()
    out_dygraph, static_layer = paddle.jit.TracedLayer.trace(layer, inputs=[in_var])
    static_layer.save_inference_model(save_dirname, feed=[0], fetch=[0])

    paddle.enable_static()
    place = paddle.CPUPlace()
    exe = paddle.static.Executor(place)
    program, feed_vars, fetch_vars = paddle.static.load_inference_model(save_dirname,
                                        exe)

    fetch, = exe.run(program, feed={feed_vars[0]: in_np}, fetch_list=fetch_vars)
    print(fetch.shape) # (2, 10)
