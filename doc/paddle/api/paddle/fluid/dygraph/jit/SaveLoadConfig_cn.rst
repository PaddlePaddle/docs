.. _cn_api_fluid_dygraph_jit_SaveLoadConfig:

SaveLoadConfig
--------------

.. py:class:: paddle.SaveLoadConfig()

用于配置接口 :ref:`cn_api_fluid_dygraph_jit_save` 和 :ref:`cn_api_fluid_dygraph_jit_load` 存储载入 :ref:`cn_api_fluid_dygraph_TranslatedLayer` 时的附加选项。

**示例代码：**

    1. 在存储模型时使用 ``SaveLoadConfig``

    .. code-block:: python

        import paddle
        import paddle.nn as nn
        import paddle.optimizer as opt

        class SimpleNet(nn.Layer):
            def __init__(self, in_size, out_size):
                super(SimpleNet, self).__init__()
                self._linear = nn.Linear(in_size, out_size)

            @paddle.jit.to_static
            def forward(self, x):
                y = self._linear(x)
                z = self._linear(y)
                return z

        # enable dygraph mode
        paddle.disable_static() 

        # train model
        net = SimpleNet(8, 8)
        adam = opt.Adam(learning_rate=0.1, parameters=net.parameters())
        x = paddle.randn([4, 8], 'float32')
        for i in range(10):
            out = net(x)
            loss = paddle.tensor.mean(out)
            loss.backward()
            adam.step()
            adam.clear_grad()

        # use SaveLoadconfig when saving model
        model_path = "simplenet.example.model"
        config = paddle.SaveLoadConfig()
        config.model_filename = "__simplenet__"
        paddle.jit.save(
            layer=net,
            model_path=model_path,
            config=config)

    2. 在载入模型时使用 ``SaveLoadConfig``

    .. code-block:: python

        import paddle

        # enable dygraph mode
        paddle.disable_static() 

        # use SaveLoadconfig when loading model
        model_path = "simplenet.example.model"
        config = paddle.SaveLoadConfig()
        config.model_filename = "__simplenet__"
        infer_net = paddle.jit.load(model_path, config=config)
        # inference
        x = paddle.randn([4, 8], 'float32')
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

        import paddle
        import paddle.nn as nn
        import paddle.optimizer as opt

        class SimpleNet(nn.Layer):
            def __init__(self, in_size, out_size):
                super(SimpleNet, self).__init__()
                self._linear = nn.Linear(in_size, out_size)

            @paddle.jit.to_static
            def forward(self, x):
                y = self._linear(x)
                z = self._linear(y)
                loss = paddle.tensor.mean(z)
                return z, loss

        # enable dygraph mode
        paddle.disable_static() 

        # train model
        net = SimpleNet(8, 8)
        adam = opt.Adam(learning_rate=0.1, parameters=net.parameters())
        x = paddle.randn([4, 8], 'float32')
        for i in range(10):
            out, loss = net(x)
            loss.backward()
            adam.step()
            adam.clear_grad()

        # use SaveLoadconfig.output_spec
        model_path = "simplenet.example.model.output_spec"
        config = paddle.SaveLoadConfig()
        config.output_spec = [out]
        paddle.jit.save(
            layer=net,
            model_path=model_path,
            config=config)

        infer_net = paddle.jit.load(model_path)
        x = paddle.randn([4, 8], 'float32')
        pred = infer_net(x)



.. py:attribute:: model_filename

存储转写 :ref:`cn_api_fluid_dygraph_Layer` 模型结构 ``Program`` 的文件名称。默认文件名为 ``__model__``。

**示例代码**
    .. code-block:: python

        import paddle
        import paddle.nn as nn
        import paddle.optimizer as opt

        class SimpleNet(nn.Layer):
            def __init__(self, in_size, out_size):
                super(SimpleNet, self).__init__()
                self._linear = nn.Linear(in_size, out_size)

            @paddle.jit.to_static
            def forward(self, x):
                y = self._linear(x)
                z = self._linear(y)
                return z

        # enable dygraph mode
        paddle.disable_static() 

        # train model
        net = SimpleNet(8, 8)
        adam = opt.Adam(learning_rate=0.1, parameters=net.parameters())
        x = paddle.randn([4, 8], 'float32')
        for i in range(10):
            out = net(x)
            loss = paddle.tensor.mean(out)
            loss.backward()
            adam.step()
            adam.clear_grad()

        # saving with configs.model_filename
        model_path = "simplenet.example.model.model_filename"
        config = paddle.SaveLoadConfig()
        config.model_filename = "__simplenet__"
        paddle.jit.save(
            layer=net,
            model_path=model_path,
            config=config)

        # loading with configs.model_filename
        infer_net = paddle.jit.load(model_path, config=config)
        x = paddle.randn([4, 8], 'float32')
        pred = infer_net(x)


.. py:attribute:: params_filename

存储转写 :ref:`cn_api_fluid_dygraph_Layer` 所有持久参数（包括 ``Parameters`` 和持久的 ``Buffers``）的文件名称。默认文件名称为 ``__variable__``。

**示例代码**
    .. code-block:: python

        import paddle
        import paddle.nn as nn
        import paddle.optimizer as opt

        class SimpleNet(nn.Layer):
            def __init__(self, in_size, out_size):
                super(SimpleNet, self).__init__()
                self._linear = nn.Linear(in_size, out_size)

            @paddle.jit.to_static
            def forward(self, x):
                y = self._linear(x)
                z = self._linear(y)
                return z

        # enable dygraph mode
        paddle.disable_static() 

        # train model
        net = SimpleNet(8, 8)
        adam = opt.Adam(learning_rate=0.1, parameters=net.parameters())
        x = paddle.randn([4, 8], 'float32')
        for i in range(10):
            out = net(x)
            loss = paddle.tensor.mean(out)
            loss.backward()
            adam.step()
            adam.clear_grad()

        model_path = "simplenet.example.model.params_filename"
        config = paddle.SaveLoadConfig()
        config.params_filename = "__params__"

        # saving with configs.params_filename
        paddle.jit.save(
            layer=net,
            model_path=model_path,
            config=config)

        # loading with configs.params_filename
        infer_net = paddle.jit.load(model_path, config=config)
        x = paddle.randn([4, 8], 'float32')
        pred = infer_net(x)


.. py:attribute:: separate_params

配置是否将 :ref:`cn_api_fluid_dygraph_Layer` 的参数存储为分散的文件。
（这是为了兼容接口 :ref:`cn_api_fluid_io_save_inference_model` 的行为）

如果设置为 ``True`` ，每个参数将会被存储为一个文件，文件名为参数名，同时``SaveLoadConfig.params_filename`` 指定的文件名将不会生效。默认为 ``False``。

**示例代码**
    .. code-block:: python

        import paddle
        import paddle.nn as nn
        import paddle.optimizer as opt

        class SimpleNet(nn.Layer):
            def __init__(self, in_size, out_size):
                super(SimpleNet, self).__init__()
                self._linear = nn.Linear(in_size, out_size)

            @paddle.jit.to_static
            def forward(self, x):
                y = self._linear(x)
                z = self._linear(y)
                return z

        # enable dygraph mode
        paddle.disable_static() 

        # train model
        net = SimpleNet(8, 8)
        adam = opt.Adam(learning_rate=0.1, parameters=net.parameters())
        x = paddle.randn([4, 8], 'float32')
        for i in range(10):
            out = net(x)
            loss = paddle.tensor.mean(out)
            loss.backward()
            adam.step()
            adam.clear_grad()

        model_path = "simplenet.example.model.separate_params"
        config = paddle.jit.SaveLoadConfig()
        config.separate_params = True

        # saving with configs.separate_params
        paddle.jit.save(
            layer=net,
            model_path=model_path,
            config=config)
        # [result] the saved model directory contains:
        # linear_0.b_0  linear_0.w_0  __model__  __variables.info__

        # loading with configs.params_filename
        infer_net = paddle.jit.load(model_path, config=config)
        x = paddle.randn([4, 8], 'float32')
        pred = infer_net(x)


.. py:attribute:: keep_name_table
    
配置是否保留 ``paddle.load`` 载入结果中 ``structured_name`` 到真实的参数变量名的映射表。这个映射表是调用 ``paddle.save`` 时存储的，一般仅用于调试，移除此映射表不影响真实的训练和预测。默认情况下不会保留在 ``paddle.load`` 的结果中。默认值为False。

.. note::
    该配置仅用于 ``paddle.load`` 和 ``paddle.fluid.load_dygraph`` 方法。

**示例代码**
    .. code-block:: python

        import paddle
            
        paddle.disable_static()

        linear = paddle.nn.Linear(5, 1)

        state_dict = linear.state_dict()
        paddle.save(state_dict, "paddle_dy.pdparams")

        config = paddle.SaveLoadConfig()
        config.keep_name_table = True
        para_state_dict = paddle.load("paddle_dy.pdparams", config)

        print(para_state_dict)
        # the name_table is 'StructuredToParameterName@@'
        # {'bias': array([0.], dtype=float32), 
        #  'StructuredToParameterName@@': 
        #     {'bias': u'linear_0.b_0', 'weight': u'linear_0.w_0'}, 
        #  'weight': array([[ 0.04230034],
        #     [-0.1222527 ],
        #     [ 0.7392676 ],
        #     [-0.8136974 ],
        #     [ 0.01211023]], dtype=float32)}
