.. _cn_api_fluid_dygraph_Layer:

Layer
-------------------------------

.. py:class:: paddle.nn.Layer(name_scope=None, dtype="float32")




基于OOD实现的动态图Layer，包含该Layer的参数、前序运行的结构等信息。

参数：
    - **name_scope** (str，可选) - 为Layer内部参数命名而采用的名称前缀。如果前缀为“mylayer”，在一个类名为MyLayer的Layer中，参数名为“mylayer_0.w_n”，其中w是参数的名称，n为自动生成的具有唯一性的后缀。如果为None，前缀名将为小写的类名。默认值为None。
    - **dtype** (str可选) - Layer中参数数据类型。如果设置为str，则可以是“bool”，“float16”，“float32”，“float64”，“int8”，“int16”，“int32”，“int64”，“uint8”或“uint16”。默认值为 "float32"。

返回：无

.. py:method:: train()

将此层及其所有子层设置为训练模式。这只会影响某些模块，如Dropout和BatchNorm。

返回：无

**代码示例**

.. code-block:: python

    import paddle

    class MyLayer(paddle.nn.Layer):
        def __init__(self):
            super(MyLayer, self).__init__()
            self._linear = paddle.nn.Linear(1, 1)
            self._dropout = paddle.nn.Dropout(p=0.5)

        def forward(self, input):
            temp = self._linear(input)
            temp = self._dropout(temp)
            return temp

    x = paddle.randn([10, 1], 'float32')
    mylayer = MyLayer()
    mylayer.eval()  # set mylayer._dropout to eval mode
    out = mylayer(x)
    mylayer.train()  # set mylayer._dropout to train mode
    out = mylayer(x)

.. py:method:: eval()

将此层及其所有子层设置为预测模式。这只会影响某些模块，如Dropout和BatchNorm。

返回：无

**代码示例**

.. code-block:: python

    import paddle

    class MyLayer(paddle.nn.Layer):
        def __init__(self):
            super(MyLayer, self).__init__()
            self._linear = paddle.nn.Linear(1, 1)
            self._dropout = paddle.nn.Dropout(p=0.5)

        def forward(self, input):
            temp = self._linear(input)
            temp = self._dropout(temp)
            return temp

    x = paddle.randn([10, 1], 'float32')
    mylayer = MyLayer()
    mylayer.eval()  # set mylayer._dropout to eval mode
    out = mylayer(x)
    print(out)

.. py:method:: full_name()

Layer的全名。组成方式为： ``name_scope`` + “/” + MyLayer.__class__.__name__ 。

返回：str， Layer的全名

**代码示例**

.. code-block:: python

    import paddle

    class LinearNet(paddle.nn.Layer):
        def __init__(self):
            super(LinearNet, self).__init__(name_scope = "demo_linear_net")
            self._linear = paddle.nn.Linear(1, 1)

        def forward(self, x):
            return self._linear(x)

    linear_net = LinearNet()
    print(linear_net.full_name())   # demo_linear_net_0

.. py:method:: register_forward_pre_hook(hook)

为Layer注册一个 ``forward pre-hook`` 函数，该 ``hook`` 函数将会在 ``forward`` 函数调用之前被调用。

``hook`` 函数具有以下形式：它的 ``input`` 是 ``Layer`` 的 ``input`` ，并且可以返回一个元组或者单个修改值；如果返回单个修改值，则将值包装到一个元组中。用户可以使用该函数来查看或修改 ``Layer`` ``forward`` 函数的输入。

hook(Layer, input) -> None or modified input

参数：
    - **hook** (function) - 被注册为 ``forward pre-hook`` 的函数

返回：HookRemoveHelper，可通过调用 ``hook_remove_helper.remove()`` 来删除注册的hook函数。

**代码示例**

.. code-block:: python

    import paddle
    import numpy as np

    # the forward_post_hook change the input of the layer: input = input * 2
    def forward_pre_hook(layer, input):
        # user can use layer and input for information statistis tasks
        # change the input
        input_return = (input[0] * 2)
        return input_return

    linear = paddle.nn.Linear(13, 5)
    # register the hook
    forward_pre_hook_handle = linear.register_forward_pre_hook(forward_pre_hook)
    value0 = np.arange(26).reshape(2, 13).astype("float32")
    in0 = paddle.to_tensor(value0)
    out0 = linear(in0)

    # remove the hook
    forward_pre_hook_handle.remove()
    value1 = value0 * 2
    in1 = paddle.to_tensor(value1)
    out1 = linear(in1)

    # hook change the linear's input to input * 2, so out0 is equal to out1.
    assert (out0.numpy() == out1.numpy()).any()

.. py:method:: register_forward_post_hook(hook)

为Layer注册一个 ``forward post-hook`` 函数，该 ``hook`` 函数将会在 ``forward`` 函数调用之后被调用。

``hook`` 函数具有以下形式，它的 ``input`` 和 ``output`` 是 ``Layer`` 的 ``input`` 和 ``output`` 。用户可以用该函数来查看和修改 ``Layer`` ``forward`` 函数的输出。

hook(Layer, input, output) -> None or modified output

参数：
    - **hook** (function) - 被注册为 ``forward post-hook`` 的函数

返回：HookRemoveHelper，可通过调用 ``hook_remove_helper.remove()`` 来删除注册的hook函数。

**代码示例**

.. code-block:: python

    import paddle
    import numpy as np

    # the forward_post_hook change the output of the layer: output = output * 2
    def forward_post_hook(layer, input, output):
        # user can use layer, input and output for information statistis tasks
        # change the output
        return output * 2

    linear = paddle.nn.Linear(13, 5)
    # register the hook
    forward_post_hook_handle = linear.register_forward_post_hook(forward_post_hook)
    value1 = np.arange(26).reshape(2, 13).astype("float32")
    in1 = paddle.to_tensor(value1)
    out0 = linear(in1)

    # remove the hook
    forward_post_hook_handle.remove()
    out1 = linear(in1)

    # hook change the linear's output to output * 2, so out0 is equal to out1 * 2.
    assert (out0.numpy() == (out1.numpy()) * 2).any()
                
.. py:method:: create_parameter(shape, attr=None, dtype="float32", is_bias=False, default_initializer=None)

为Layer创建参数。

参数：
    - **shape** (list) - 参数的形状。列表中的数据类型必须为int。
    - **attr** (ParamAttr，可选) - 指定权重参数属性的对象，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。默认值为None。
    - **dtype** (str|core.VarDesc.VarType, 可选) - Layer中参数数据类型。如果设置为str，则可以是“bool”，“float16”，“float32”，“float64”，“int8”，“int16”，“int32”，“int64”，“uint8”或“uint16”。默认值为“float32”。
    - **is_bias** (bool, 可选) - 是否是偏置参数。默认值：False。
    - **default_initializer** (Initializer, 可选) - 默认的参数初始化方法。如果设置为None，则设置非bias参数的初始化方式为 paddle.nn.initializer.Xavier ，设置bias参数的初始化方式为 paddle.nn.initializer.Constant 。默认值：None。

返回：Tensor， 创建的参数变量

**代码示例**

.. code-block:: python

    import paddle

    class MyLayer(paddle.nn.Layer):
        def __init__(self):
            super(MyLayer, self).__init__()
            self._linear = paddle.nn.Linear(1, 1)
            w_tmp = self.create_parameter([1,1])
            self.add_parameter("w_tmp", w_tmp)

        def forward(self, input):
            return self._linear(input)

    mylayer = MyLayer()
    for name, param in mylayer.named_parameters():
        print(name, param)      # will print w_tmp,_linear.weight,_linear.bias

.. py:method:: create_variable(name=None, persistable=None, dtype=None)

为Layer创建变量。

参数：
    - **name** (str, 可选) - 变量名。默认值：None。
    - **persistable** (bool, 可选) - 是否为持久性变量，后续会被移出。默认值：None。
    - **dtype** (str, 可选) - Layer中参数数据类型。如果设置为str，则可以是“bool”，“float16”，“float32”，“float64”，“int8”，“int16”，“int32”，“int64”，“uint8”或“uint16”。默认值为 "float32" 。

返回：Tensor， 返回创建的 ``Tensor`` 

**代码示例**

.. code-block:: python

    import paddle

    class MyLinear(paddle.nn.Layer):
        def __init__(self,
                    in_features,
                    out_features):
            super(MyLinear, self).__init__()
            self.linear = paddle.nn.Linear( 10, 10)
                
            self.back_var = self.create_variable(name = "linear_tmp_0", dtype=self._dtype)
        
        def forward(self, input):
            out = self.linear(input)
            paddle.assign( out, self.back_var)
            
            return out

.. py:method:: create_tensor(name=None, persistable=None, dtype=None)

为Layer创建变量。

参数：
    - **name** (str, 可选) - 变量名。默认值：None。
    - **persistable** (bool, 可选) - 是否为持久性变量，后续会被移出。默认值：None。
    - **dtype** (str, 可选) - Layer中参数数据类型。如果设置为str，则可以是“bool”，“float16”，“float32”，“float64”，“int8”，“int16”，“int32”，“int64”，“uint8”或“uint16”。默认值为 "float32" 。

返回：Tensor， 返回创建的 ``Tensor`` 

**代码示例**

.. code-block:: python

    import paddle

    class MyLinear(paddle.nn.Layer):
        def __init__(self,
                    in_features,
                    out_features):
            super(MyLinear, self).__init__()
            self.linear = paddle.nn.Linear( 10, 10)
                
            self.back_var = self.create_tensor(name = "linear_tmp_0", dtype=self._dtype)
        
        def forward(self, input):
            out = self.linear(input)
            paddle.assign( out, self.back_var)
            
            return out


.. py:method:: parameters(include_sublayers=True)

返回一个由当前层及其子层的所有参数组成的列表。

参数：
    - **include_sublayers** (bool, 可选) - 是否返回子层的参数。如果为True，返回的列表中包含子层的参数。默认值：True。

返回：list， 一个由当前层及其子层的所有参数组成的列表，列表中的元素类型为Parameter(Tensor)。

**代码示例**

.. code-block:: python

    import paddle

    linear = paddle.nn.Linear(1,1)
    print(linear.parameters())  # print linear_0.w_0 and linear_0.b_0

.. py:method:: children()

返回所有子层的迭代器。

返回：iterator， 子层的迭代器。

**代码示例**

.. code-block:: python

    import paddle

    linear1 = paddle.nn.Linear(10, 3)
    linear2 = paddle.nn.Linear(3, 10, bias_attr=False)
    model = paddle.nn.Sequential(linear1, linear2)

    layer_list = list(model.children())

    print(layer_list)   # [<paddle.nn.layer.common.Linear object at 0x7f7b8113f830>, <paddle.nn.layer.common.Linear object at 0x7f7b8113f950>]

.. py:method:: named_children()

返回所有子层的迭代器，生成子层名称和子层的元组。

返回：iterator， 产出子层名称和子层的元组的迭代器。

**代码示例**

.. code-block:: python

    import paddle

    linear1 = paddle.nn.Linear(10, 3)
    linear2 = paddle.nn.Linear(3, 10, bias_attr=False)
    model = paddle.nn.Sequential(linear1, linear2)
    for prefix, layer in model.named_children():
        print(prefix, layer)
        # ('0', <paddle.nn.layer.common.Linear object at 0x7fb61ed85830>)
        # ('1', <paddle.nn.layer.common.Linear object at 0x7fb61ed85950>)

.. py:method:: sublayers(include_sublayers=True)

返回一个由所有子层组成的列表。

参数：
    - **include_sublayers** (bool, 可选) - 是否返回子层中各个子层。如果为True，则包括子层中的各个子层。默认值：True。

返回： list， 一个由所有子层组成的列表，列表中的元素类型为Layer。

**代码示例**

.. code-block:: python

    import paddle

    class MyLayer(paddle.nn.Layer):
        def __init__(self):
            super(MyLayer, self).__init__()
            self._linear = paddle.nn.Linear(1, 1)
            self._dropout = paddle.nn.Dropout(p=0.5)

        def forward(self, input):
            temp = self._linear(input)
            temp = self._dropout(temp)
            return temp

    mylayer = MyLayer()
    print(mylayer.sublayers())  # [<paddle.nn.layer.common.Linear object at 0x7f44b58977d0>, <paddle.nn.layer.common.Dropout object at 0x7f44b58978f0>]

.. py:method:: clear_gradients()

清除该层所有参数的梯度。

返回：无

**代码示例**

.. code-block:: python

    import paddle
    import numpy as np

    value = np.arange(26).reshape(2, 13).astype("float32")
    a = paddle.to_tensor(value)
    linear = paddle.nn.Linear(13, 5)
    adam = paddle.optimizer.Adam(learning_rate=0.01,
                                parameters=linear.parameters())
    out = linear(a)
    out.backward()
    adam.step()
    linear.clear_gradients()

.. py:method:: named_parameters(prefix='', include_sublayers=True)

返回层中所有参数的迭代器，生成名称和参数的元组。

参数：
    - **prefix** (str, 可选) - 在所有参数名称前加的前缀。默认值：''。
    - **include_sublayers** (bool, 可选) - 是否返回子层的参数。如果为True，返回的列表中包含子层的参数。默认值：True。

返回：iterator， 产出名称和参数的元组的迭代器。

**代码示例**

.. code-block:: python

    import paddle

    fc1 = paddle.nn.Linear(10, 3)
    fc2 = paddle.nn.Linear(3, 10, bias_attr=False)
    model = paddle.nn.Sequential(fc1, fc2)
    for name, param in model.named_parameters():
        print(name, param)

.. py:method:: named_sublayers(prefix='', include_sublayers=True, include_self=False, layers_set=None)

返回层中所有子层上的迭代器，生成名称和子层的元组。重复的子层只产生一次。

参数：
    - **prefix** (str, 可选) - 在所有参数名称前加的前缀。默认值：''。
    - **include_sublayers** (bool, 可选) - 是否返回子层中各个子层。如果为True，则包括子层中的各个子层。默认值：True。
    - **include_self** (bool, 可选) - 是否包含该层自身。默认值：False。
    - **layers_set** (set, 可选): 记录重复子层的集合。默认值：None。

返回：iterator， 产出名称和子层的元组的迭代器。

**代码示例**

.. code-block:: python

    import paddle

    fc1 = paddle.nn.Linear(10, 3)
    fc2 = paddle.nn.Linear(3, 10, bias_attr=False)
    model = paddle.nn.Sequential(fc1, fc2)
    for prefix, layer in model.named_sublayers():
        print(prefix, layer)

.. py:method:: register_buffer(name, tensor, persistable=True)

将一个Tensor注册为buffer。

buffer是一个不可训练的变量，不会被优化器更新，但在评估或预测阶段可能是必要的状态变量。比如 ``BatchNorm`` 中的均值和方差。

注册的buffer默认是可持久性的，会被保存到 ``state_dict`` 中。如果指定 ``persistable`` 参数为False，则会注册一个非持久性的buffer，即不会同步和保存到 ``state_dict`` 中。

参数：
    - **name** (str) - 注册buffer的名字。可以通过此名字来访问已注册的buffer。
    - **tensor** (Tensor) - 将被注册为buffer的变量。
    - **persistable** (bool, 可选) - 注册的buffer是否需要可持久性地保存到 ``state_dict`` 中。

返回：None

**代码示例**

.. code-block:: python

    import numpy as np
    import paddle
    
    linear = paddle.nn.Linear(10, 3)
    value = np.array([0]).astype("float32")
    buffer = paddle.to_tensor(value)
    linear.register_buffer("buf_name", buffer, persistable=True)
    # get the buffer by attribute.
    print(linear.buf_name)

.. py:method:: buffers(include_sublayers=True)

返回一个由当前层及其子层的所有buffers组成的列表。

参数：
    - **include_sublayers** (bool, 可选) - 是否返回子层的buffers。如果为True，返回的列表中包含子层的buffers。默认值：True。

返回：list， 一个由当前层及其子层的所有buffers组成的列表，列表中的元素类型为Tensor。

**代码示例**

.. code-block:: python

    import numpy as np
    import paddle

    linear = paddle.nn.Linear(10, 3)
    value = np.array([0]).astype("float32")
    buffer = paddle.to_tensor(value)
    linear.register_buffer("buf_name", buffer, persistable=True)

    print(linear.buffers())     # == print([linear.buf_name])

.. py:method:: named_buffers(prefix='', include_sublayers=True)

返回层中所有buffers的迭代器，生成名称和buffer的元组。

参数：
    - **prefix** (str, 可选) - 在所有buffer名称前加的前缀。默认值：''。
    - **include_sublayers** (bool, 可选) - 是否返回子层的buffers。如果为True，返回的列表中包含子层的buffers。默认值：True。

返回：iterator， 产出名称和buffer的元组的迭代器。

**代码示例**

.. code-block:: python

    import numpy as np
    import paddle

    fc1 = paddle.nn.Linear(10, 3)
    buffer1 = paddle.to_tensor(np.array([0]).astype("float32"))
    # register a tensor as buffer by specific `persistable`
    fc1.register_buffer("buf_name_1", buffer1, persistable=True)

    fc2 = paddle.nn.Linear(3, 10)
    buffer2 = paddle.to_tensor(np.array([1]).astype("float32"))
    # register a buffer by assigning an attribute with Tensor.
    # The `persistable` can only be False by this way.
    fc2.buf_name_2 = buffer2

    model = paddle.nn.Sequential(fc1, fc2)

    # get all named buffers
    for name, buffer in model.named_buffers():
        print(name, buffer)

.. py:method:: forward(*inputs, **kwargs)

定义每次调用时执行的计算。应该被所有子类覆盖。

参数：
    - **\*inputs** (tuple) - 解包后的tuple参数。
    - **\*\*kwargs** (dict) - 解包后的dict参数。

返回： 无

.. py:method:: add_sublayer(name, sublayer)

添加子层实例。可以通过self.name访问该sublayer。

参数：
    - **name** (str) - 子层名。
    - **sublayer** (Layer) - Layer实例。

返回：Layer， 添加的子层

**代码示例**

.. code-block:: python

    import paddle

    class MySequential(paddle.nn.Layer):
        def __init__(self, *layers):
            super(MySequential, self).__init__()
            if len(layers) > 0 and isinstance(layers[0], tuple):
                for name, layer in layers:
                    self.add_sublayer(name, layer)
            else:
                for idx, layer in enumerate(layers):
                    self.add_sublayer(str(idx), layer)

        def forward(self, input):
            for layer in self._sub_layers.values():
                input = layer(input)
            return input

    fc1 = paddle.nn.Linear(10, 3)
    fc2 = paddle.nn.Linear(3, 10, bias_attr=False)
    model = MySequential(fc1, fc2)
    for prefix, layer in model.named_sublayers():
        print(prefix, layer)


.. py:method:: add_parameter(name, parameter)

添加参数实例。可以通过self.name访问该parameter。

参数：
    - **name** (str) - 参数名。
    - **parameter** (Parameter) - Parameter实例。

返回：Parameter， 传入的参数实例

**代码示例**

.. code-block:: python

    import paddle

    class MyLayer(paddle.nn.Layer):
        def __init__(self):
            super(MyLayer, self).__init__()
            self._linear = paddle.nn.Linear(1, 1)
            w_tmp = self.create_parameter([1,1])
            self.add_parameter("w_tmp", w_tmp)

        def forward(self, input):
            return self._linear(input)

    mylayer = MyLayer()
    for name, param in mylayer.named_parameters():
        print(name, param)      # will print w_tmp,_linear.weight,_linear.bias


.. py:method:: state_dict(destination=None, include_sublayers=True)

获取当前层及其子层的所有参数和可持久性buffers。并将所有参数和buffers存放在dict结构中。

参数：
    - **destination** (dict, 可选) - 如果提供 ``destination`` ，则所有参数和可持久性buffers都将存放在 ``destination`` 中。 默认值：None。
    - **include_sublayers** (bool, 可选) - 如果设置为True，则包括子层的参数和buffers。默认值：True。

返回：dict， 包含所有参数和可持久行buffers的dict

**代码示例**

.. code-block:: python

    import paddle

    emb = paddle.nn.Embedding(10, 10)

    state_dict = emb.state_dict()
    paddle.save( state_dict, "paddle_dy.pdparams")

.. py:method:: set_state_dict(state_dict, include_sublayers=True, use_structured_name=True)

根据传入的 ``state_dict`` 设置参数和可持久性buffers。 所有参数和buffers将由 ``state_dict`` 中的 ``Tensor`` 设置。

参数：
    - **state_dict** (dict) - 包含所有参数和可持久性buffers的dict。
    - **include_sublayers** (bool, 可选) - 如果设置为True，则还包括子层的参数和buffers。 默认值：True。
    - **use_structured_name** (bool, 可选) - 如果设置为True，将使用Layer的结构性变量名作为dict的key，否则将使用Parameter或者Buffer的变量名作为key。默认值：True。

返回：无

**代码示例**

.. code-block:: python

    import paddle

    emb = paddle.nn.Embedding(10, 10)
    
    state_dict = emb.state_dict()
    paddle.save(state_dict, "paddle_dy.pdparams")
    para_state_dict = paddle.load("paddle_dy.pdparams")
    emb.set_state_dict(para_state_dict)

.. py:method:: to(device=None, dtype=None, blocking=None)

根据给定的device、dtype和blocking 转换 Layer中的parameters 和 buffers。

参数：
    - **device** （str|paddle.CPUPlace()|paddle.CUDAPlace()|paddle.CUDAPinnedPlace()|None, 可选) - 希望存储Layer 的设备位置。如果为None， 设备位置和原始的Tensor 的设备位置一致。如果设备位置是string 类型，取值可为 ``cpu``, ``gpu:x`` and ``xpu:x`` ，这里的 ``x`` 是 GPUs 或者 XPUs的编号。默认值：None。
    - **dtype** （str|core.VarDesc.VarType|None, 可选) - 数据的类型。如果为None， 数据类型和原始的Tensor 一致。默认值：None。
    - **blocking** （bool|None, 可选）- 如果为False并且当前Tensor处于固定内存上，将会发生主机到设备端的异步拷贝。否则，会发生同步拷贝。如果为None，blocking 会被设置为True。默认为False。

**代码示例**

.. code-block:: python

    import paddle
    
    linear=paddle.nn.Linear(2, 2)
    linear.weight
    #Parameter containing:
    #Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=False,
    #       [[-0.32770029,  0.38653070],
    #        [ 0.46030545,  0.08158520]])
    
    linear.to(dtype='float64')
    linear.weight
    #Tenor(shape=[2, 2], dtype=float64, place=CUDAPlace(0), stop_gradient=False,
    #       [[-0.32770029,  0.38653070],
    #        [ 0.46030545,  0.08158520]])
    
    linear.to(device='cpu')
    linear.weight
    #Tensor(shape=[2, 2], dtype=float64, place=CPUPlace, stop_gradient=False,
    #       [[-0.32770029,  0.38653070],
    #        [ 0.46030545,  0.08158520]])
    linear.to(device=paddle.CUDAPinnedPlace(), blocking=False)
    linear.weight
    #Tensor(shape=[2, 2], dtype=float64, place=CUDAPinnedPlace, stop_gradient=False,
    #       [[-0.04989364, -0.56889004],
    #        [ 0.33960250,  0.96878713]])
    