.. _cn_api_fluid_dygraph_jit_SaveLoadConfig:

SaveLoadConfig
-------------------------------

.. py:class:: paddle.fluid.dygraph.jit.SaveLoadConfig()

用于配置接口 :ref:`cn_api_fluid_dygraph_jit_save` 和 :ref:`cn_api_fluid_dygraph_jit_load` 存储载入 :ref:`cn_api_fluid_dygraph_TranslatedLayer` 时的附加选项。

**示例代码：**

    1. 在存储模型时使用 ``SaveLoadConfig``

    .. code-block:: python

        import numpy as np
        import paddle.fluid as fluid
        from paddle.fluid.dygraph import Linear
        from paddle.fluid.dygraph import declarative
        class SimpleNet(fluid.dygraph.Layer):
            def __init__(self, in_size, out_size):
                super(SimpleNet, self).__init__()
                self._linear = Linear(in_size, out_size)
            @declarative
            def forward(self, x):
                y = self._linear(x)
                z = self._linear(y)
                return z
        # 开启命令式编程模式
        fluid.enable_dygraph() 
        # 训练模型
        net = SimpleNet(8, 8)
        adam = fluid.optimizer.AdamOptimizer(learning_rate=0.1, parameter_list=net.parameters())
        x = fluid.dygraph.to_variable(np.random.random((4, 8)).astype('float32'))
        for i in range(10):
            out = net(x)
            loss = fluid.layers.mean(out)
            loss.backward()
            adam.minimize(loss)
            net.clear_gradients()
        # 在存储模型时使用SaveLoadConfig
        model_path = "simplenet.example.model"
        configs = fluid.dygraph.jit.SaveLoadConfig()
        configs.model_filename = "__simplenet__"
        fluid.dygraph.jit.save(
            layer=net,
            model_path=model_path,
            input_spec=[x],
            configs=configs)

    2. 在载入模型时使用 ``SaveLoadConfig``

    .. code-block:: python

        import numpy as np
        import paddle.fluid as fluid
        # 开启命令式编程模式
        fluid.enable_dygraph() 
        # 在载入模型时使用SaveLoadconfig
        model_path = "simplenet.example.model"
        configs = fluid.dygraph.jit.SaveLoadConfig()
        configs.model_filename = "__simplenet__"
        infer_net = fluid.dygraph.jit.load(model_path, configs=configs)
        # 预测
        x = fluid.dygraph.to_variable(np.random.random((4, 8)).astype('float32'))
        pred = infer_net(x)

属性
::::::::::::

.. py:attribute:: output_spec

选择保存模型（ :ref:`cn_api_fluid_dygraph_TranslatedLayer` ）的输出变量，通过指定的这些变量能够使模型仅计算特定的结果。
默认情况下，原始 :ref:`cn_api_fluid_dygraph_Layer` 的forward方法的所有返回变量都将配置为存储后模型 :ref:`cn_api_fluid_dygraph_TranslatedLayer` 的输出变量。

``output_spec`` 属性类型需要是 ``list[Variable]``。如果输入的 ``output_spec`` 列表不是原始 :ref:`cn_api_fluid_dygraph_Layer` 的forward方法的所有返回变量，
将会依据输入的 ``output_spec`` 列表对存储的模型进行裁剪。

.. note::
    ``output_spec`` 属性仅在存储模型时使用。

**示例代码：**
    .. code-block:: python

        import numpy as np
        import paddle.fluid as fluid
        from paddle.fluid.dygraph import Linear
        from paddle.fluid.dygraph import declarative
        class SimpleNet(fluid.dygraph.Layer):
            def __init__(self, in_size, out_size):
                super(SimpleNet, self).__init__()
                self._linear = Linear(in_size, out_size)
            @declarative
            def forward(self, x):
                y = self._linear(x)
                z = self._linear(y)
                loss = fluid.layers.mean(z)
                return z, loss
        # 开启命令式编程模式
        fluid.enable_dygraph() 
        # 训练模型
        net = SimpleNet(8, 8)
        adam = fluid.optimizer.AdamOptimizer(learning_rate=0.1, parameter_list=net.parameters())
        x = fluid.dygraph.to_variable(np.random.random((4, 8)).astype('float32'))
        for i in range(10):
            out, loss = net(x)
            loss.backward()
            adam.minimize(loss)
            net.clear_gradients()
        # 使用SaveLoadconfig.output_spec
        model_path = "simplenet.example.model.output_spec"
        configs = fluid.dygraph.jit.SaveLoadConfig()
        # 仅在存储模型中保留预测结果，丢弃loss
        configs.output_spec = [out]
        fluid.dygraph.jit.save(
            layer=net,
            model_path=model_path,
            input_spec=[x],
            configs=configs)
        infer_net = fluid.dygraph.jit.load(model_path, configs=configs)
        x = fluid.dygraph.to_variable(np.random.random((4, 8)).astype('float32'))
        # 仅有预测结果输出
        pred = infer_net(x)


.. py:attribute:: model_filename

存储转写 :ref:`cn_api_fluid_dygraph_Layer` 模型结构 ``Program`` 的文件名称。默认文件名为 ``__model__``。

**示例代码**
    .. code-block:: python

        import numpy as np
        import paddle.fluid as fluid
        from paddle.fluid.dygraph import Linear
        from paddle.fluid.dygraph import declarative
        class SimpleNet(fluid.dygraph.Layer):
            def __init__(self, in_size, out_size):
                super(SimpleNet, self).__init__()
                self._linear = Linear(in_size, out_size)
            @declarative
            def forward(self, x):
                y = self._linear(x)
                z = self._linear(y)
                return z
        # 开启命令式编程模式
        fluid.enable_dygraph() 
        # 训练模型
        net = SimpleNet(8, 8)
        adam = fluid.optimizer.AdamOptimizer(learning_rate=0.1, parameter_list=net.parameters())
        x = fluid.dygraph.to_variable(np.random.random((4, 8)).astype('float32'))
        for i in range(10):
            out = net(x)
            loss = fluid.layers.mean(out)
            loss.backward()
            adam.minimize(loss)
            net.clear_gradients()
        model_path = "simplenet.example.model.model_filename"
        configs = fluid.dygraph.jit.SaveLoadConfig()
        configs.model_filename = "__simplenet__"
        # 配置configs.model_filename存储模型
        fluid.dygraph.jit.save(
            layer=net,
            model_path=model_path,
            input_spec=[x],
            configs=configs)
        # [结果] 存储模型目录文件包括:
        # __simplenet__  __variables__  __variables.info__
        # 配置configs.model_filename载入模型
        infer_net = fluid.dygraph.jit.load(model_path, configs=configs)
        x = fluid.dygraph.to_variable(np.random.random((4, 8)).astype('float32'))
        pred = infer_net(x)


.. py:attribute:: params_filename

存储转写 :ref:`cn_api_fluid_dygraph_Layer` 所有持久参数（包括 ``Parameters`` 和持久的 ``Buffers``）的文件名称。默认文件名称为 ``__variable__``。

**示例代码**
    .. code-block:: python

        import numpy as np
        import paddle.fluid as fluid
        from paddle.fluid.dygraph import Linear
        from paddle.fluid.dygraph import declarative
        class SimpleNet(fluid.dygraph.Layer):
            def __init__(self, in_size, out_size):
                super(SimpleNet, self).__init__()
                self._linear = Linear(in_size, out_size)
            @declarative
            def forward(self, x):
                y = self._linear(x)
                z = self._linear(y)
                return z
        # 开启命令式编程模式
        fluid.enable_dygraph() 
        # 训练模型
        net = SimpleNet(8, 8)
        adam = fluid.optimizer.AdamOptimizer(learning_rate=0.1, parameter_list=net.parameters())
        x = fluid.dygraph.to_variable(np.random.random((4, 8)).astype('float32'))
        for i in range(10):
            out = net(x)
            loss = fluid.layers.mean(out)
            loss.backward()
            adam.minimize(loss)
            net.clear_gradients()
        model_path = "simplenet.example.model.params_filename"
        configs = fluid.dygraph.jit.SaveLoadConfig()
        configs.params_filename = "__params__"
        # 配置configs.params_filename存储模型
        fluid.dygraph.jit.save(
            layer=net,
            model_path=model_path,
            input_spec=[x],
            configs=configs)
        # [结果] 存储模型目录文件包括:
        # __model__  __params__  __variables.info__
        # 配置configs.params_filename载入模型
        infer_net = fluid.dygraph.jit.load(model_path, configs=configs)
        x = fluid.dygraph.to_variable(np.random.random((4, 8)).astype('float32'))
        pred = infer_net(x)


.. py:attribute:: separate_params

配置是否将 :ref:`cn_api_fluid_dygraph_Layer` 的参数存储为分散的文件。
（这是为了兼容接口 :ref:`cn_api_fluid_io_save_inference_model` 的行为）

如果设置为 ``True`` ，每个参数将会被存储为一个文件，文件名为参数名，同时``SaveLoadConfig.params_filename`` 指定的文件名将不会生效。默认为 ``False``。

**示例代码**
    .. code-block:: python

        import numpy as np
        import paddle.fluid as fluid
        from paddle.fluid.dygraph import Linear
        from paddle.fluid.dygraph import declarative
        class SimpleNet(fluid.dygraph.Layer):
            def __init__(self, in_size, out_size):
                super(SimpleNet, self).__init__()
                self._linear = Linear(in_size, out_size)
            @declarative
            def forward(self, x):
                y = self._linear(x)
                z = self._linear(y)
                return z
        # 开启命令式编程模式
        fluid.enable_dygraph() 
        # 训练模型
        net = SimpleNet(8, 8)
        adam = fluid.optimizer.AdamOptimizer(learning_rate=0.1, parameter_list=net.parameters())
        x = fluid.dygraph.to_variable(np.random.random((4, 8)).astype('float32'))
        for i in range(10):
            out = net(x)
            loss = fluid.layers.mean(out)
            loss.backward()
            adam.minimize(loss)
            net.clear_gradients()
        model_path = "simplenet.example.model.separate_params"
        configs = fluid.dygraph.jit.SaveLoadConfig()
        configs.separate_params = True
        # 配置configs.separate_params存储模型
        fluid.dygraph.jit.save(
            layer=net,
            model_path=model_path,
            input_spec=[x],
            configs=configs)
        # [结果] 存储模型目录文件包括:
        # linear_0.b_0  linear_0.w_0  __model__  __variables.info__
        # 配置configs.params_filename载入模型
        infer_net = fluid.dygraph.jit.load(model_path, configs=configs)
        x = fluid.dygraph.to_variable(np.random.random((4, 8)).astype('float32'))
        pred = infer_net(x)
