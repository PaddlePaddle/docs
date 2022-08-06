.. _cn_api_fluid_dygraph_TracedLayer:

TracedLayer
-------------------------------


.. py:class:: paddle.jit.TracedLayer(program, parameters, feed_names, fetch_names)




TracedLayer 用于将前向动态图模型转换为静态图模型，主要用于将动态图保存后做在线 C++预测。除此以外，用户也可使用转换后的静态图模型在 Python 端做预测，通常比原先的动态图性能更好。

TracedLayer 使用 ``Executor`` 和 ``CompiledProgram`` 运行静态图模型。转换后的静态图模型与原动态图模型共享参数。

所有的 TracedLayer 对象均不应通过构造函数创建，而应通过调用静态方法 ``TracedLayer.trace(layer, inputs)`` 创建。

TracedLayer 只能用于将 data independent 的动态图模型转换为静态图模型，即待转换的动态图模型不应随 tensor 数据或维度的变化而变化。

方法
::::::::::::

**static** trace(layer, inputs)
'''''''''

创建 TracedLayer 对象的唯一接口，该接口会调用 ``layer(*inputs)`` 方法运行动态图模型并将其转换为静态图模型。

**参数**

    - **layer** (dygraph.Layer) - 待追踪的动态图 layer 对象。
    - **inputs** (list(Variable)) - 动态图 layer 对象的输入变量列表。

**返回**

tuple，包含 2 个元素，其中第一个元素是 ``layer(*inputs)`` 的输出结果，第二个元素是转换后得到的 TracedLayer 对象。


**代码示例**

COPY-FROM: paddle.jit.TracedLayer.trace

set_strategy(build_strategy=None, exec_strategy=None)
'''''''''

设置构建和执行静态图模型的相关策略。

**参数**

    - **build_strategy** (BuildStrategy，可选) - TracedLayer 内部 ``CompiledProgram`` 的构建策略。
    - **exec_strategy** (ExecutionStrategy，可选) - TracedLayer 内部 ``CompiledProgram`` 的执行策略。

**返回**

 无。

**代码示例**

COPY-FROM: paddle.jit.TracedLayer.set_strategy

save_inference_model(path, feed=None, fetch=None)
'''''''''

将 TracedLayer 保存为用于预测部署的模型。保存的预测模型可被 C++预测接口加载。

``path`` 是存储目标的前缀，存储的模型结构 ``Program`` 文件的后缀为 ``.pdmodel``，存储的持久参数变量文件的后缀为 ``.pdiparams``。

**参数**

    - **path** (str) - 存储模型的路径前缀。格式为 ``dirname/file_prefix`` 或者 ``file_prefix`` 。
    - **feed** (list(int)，可选) - 预测模型输入变量的索引。若为 None，则 TracedLayer 的所有输入变量均会作为预测模型的输入。默认值为 None。
    - **fetch** (list(int)，可选) - 预测模型输出变量的索引。若为 None，则 TracedLayer 的所有输出变量均会作为预测模型的输出。默认值为 None。

**返回**

 无。

**代码示例**

COPY-FROM: paddle.jit.TracedLayer.save_inference_model
