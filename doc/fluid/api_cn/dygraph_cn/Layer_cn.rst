.. _cn_api_fluid_dygraph_Layer:

Layer
-------------------------------

.. py:class:: paddle.fluid.dygraph.Layer(name_scope=None, dtype=core.VarDesc.VarType.FP32)




基于OOD实现的动态图Layer，包含该Layer的参数、前序运行的结构等信息。

参数：
    - **name_scope** (str，可选) - 为Layer内部参数命名而采用的名称前缀。如果前缀为“mylayer”，在一个类名为MyLayer的Layer中，参数名为“mylayer_0.w_n”，其中w是参数的名称，n为自动生成的具有唯一性的后缀。如果为None，前缀名将为小写的类名。默认值为None。
    - **dtype** (str|core.VarDesc.VarType, 可选) - Layer中参数数据类型。如果设置为str，则可以是“bool”，“float16”，“float32”，“float64”，“int8”，“int16”，“int32”，“int64”，“uint8”或“uint16”。默认值为 ``core.VarDesc.VarType.FP32`` 。

返回：无

.. py:method:: train()

将此层及其所有子层设置为训练模式。这只会影响某些模块，如Dropout和BatchNorm。

返回：无

.. py:method:: eval()

将此层及其所有子层设置为预测模式。这只会影响某些模块，如Dropout和BatchNorm。

返回：无

.. py:method:: full_name()

Layer的全名。组成方式为： ``name_scope`` + “/” + MyLayer.__class__.__name__ 。

返回：Layer的全名

返回类型：str

.. py:method:: register_forward_pre_hook(hook)

为Layer注册一个 ``forward pre-hook`` 函数，该 ``hook`` 函数将会在 ``forward`` 函数调用之前被调用。

``hook`` 函数具有以下形式：它的 ``input`` 是 ``Layer`` 的 ``input`` ，并且可以返回一个元组或者单个修改值；如果返回单个修改值，则将值包装到一个元组中。用户可以使用该函数来查看或修改 ``Layer`` ``forward`` 函数的输入。

hook(Layer, input) -> None or modified input

参数：
    - **hook** (function) - 被注册为 ``forward pre-hook`` 的函数

返回：一个 ``HookRemoveHelper`` 类对象，可通过调用 ``hook_remove_helper.remove()`` 来删除注册的hook函数。

返回类型： ``HookRemoveHelper`` 类对象

**代码示例**

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np
    
    # forward_pre_hook函数修改了layer的输入：input = input * 2
    
    def forward_pre_hook(layer, input):
        # 改变输入值
        input_return = input[0] * 2
        return input_return
    
    
    with paddle.imperative.guard():
        linear = fluid.Linear(13, 5, dtype='float32')
    
        # 注册hook
        forward_pre_hook_handle = linear.register_forward_pre_hook(forward_pre_hook
    
            )
        value0 = np.arange(26).reshape(2, 13).astype('float32')
        in0 = paddle.imperative.to_variable(value0)
        out0 = linear(in0)
    
        # 移除hook
        forward_pre_hook_handle.remove()
    
        value1 = value0 * 2
        in1 = paddle.imperative.to_variable(value1)
        out1 = linear(in1)
    
        # hook改变了layer的输入（input = input * 2），所以out0等于out1
        assert (out0.numpy() == out1.numpy()).any()

.. py:method:: register_forward_post_hook(hook)

为Layer注册一个 ``forward post-hook`` 函数，该 ``hook`` 函数将会在 ``forward`` 函数调用之后被调用。

``hook`` 函数具有以下形式，它的 ``input`` 和 ``output`` 是 ``Layer`` 的 ``input`` 和 ``output`` 。用户可以用该函数来查看和修改 ``Layer`` ``forward`` 函数的输出。

hook(Layer, input, output) -> None or modified output

参数：
    - **hook** (function) - 被注册为 ``forward post-hook`` 的函数

返回：一个 ``HookRemoveHelper`` 类对象，可通过调用 ``hook_remove_helper.remove()`` 来删除注册的hook函数。

返回类型： ``HookRemoveHelper`` 类对象

**代码示例**

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np
    
    # forward_pre_hook函数修改了layer的输入：input = input * 2
    
    def forward_pre_hook(layer, input):
        # 改变输入值
        input_return = input[0] * 2
        return input_return
    
    
    with paddle.imperative.guard():
        linear = fluid.Linear(13, 5, dtype='float32')
    
        # 注册hook
        forward_pre_hook_handle = linear.register_forward_pre_hook(forward_pre_hook
    
            )
        value0 = np.arange(26).reshape(2, 13).astype('float32')
        in0 = paddle.imperative.to_variable(value0)
        out0 = linear(in0)
    
        # 移除hook
        forward_pre_hook_handle.remove()
    
        value1 = value0 * 2
        in1 = paddle.imperative.to_variable(value1)
        out1 = linear(in1)
    
        # hook改变了layer的输入（input = input * 2），所以out0等于out1
        assert (out0.numpy() == out1.numpy()).any()

.. py:method:: create_parameter(shape, attr=None, dtype="float32", is_bias=False, default_initializer=None)

为Layer创建参数。

参数：
    - **shape** (list) - 参数的形状。列表中的数据类型必须为int。
    - **attr** (ParamAttr，可选) - 指定权重参数属性的对象，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。默认值为None。
    - **dtype** (str|core.VarDesc.VarType, 可选) - Layer中参数数据类型。如果设置为str，则可以是“bool”，“float16”，“float32”，“float64”，“int8”，“int16”，“int32”，“int64”，“uint8”或“uint16”。默认值为“float32”。
    - **is_bias** (bool, 可选) - 是否是偏置参数。默认值：False。
    - **default_initializer** (Initializer, 可选) - 默认的参数初始化方法。如果设置为None，则设置非bias参数的初始化方式为 :ref:`cn_api_fluid_initializer_XavierInitializer` ，设置bias参数的初始化方式为 :ref:`cn_api_fluid_initializer_ConstantInitializer` 。默认值：None。

返回：创建的参数变量

返回类型： :ref:`cn_api_fluid_Variable`

.. py:method:: create_variable(name=None, persistable=None, dtype=None, type=VarType.LOD_TENSOR)

为Layer创建变量。

参数：
    - **name** (str, 可选) - 变量名。默认值：None。
    - **persistable** (bool, 可选) - 是否为持久性变量，后续会被移出。默认值：None。
    - **dtype** (str|core.VarDesc.VarType, 可选) - Layer中参数数据类型。如果设置为str，则可以是“bool”，“float16”，“float32”，“float64”，“int8”，“int16”，“int32”，“int64”，“uint8”或“uint16”。默认值为 ``core.VarDesc.VarType.FP32`` 。
    - **type** (core.VarDesc.VarType, 可选) - 变量类型，该参数不需要用户设置。默认值：core.VarDesc.VarType.LOD_TENSOR。

返回：创建的 ``Tensor`` 

返回类型： :ref:`cn_api_fluid_Variable`

.. py:method:: parameters(include_sublayers=True)

返回一个由当前层及其子层的所有参数组成的列表。

参数：
    - **include_sublayers** (bool, 可选) - 是否返回子层的参数。如果为True，返回的列表中包含子层的参数。默认值：True。

返回：一个由当前层及其子层的所有参数组成的列表，列表中的元素类型为Parameter(Variable)。

返回类型：list

.. py:method:: sublayers(include_sublayers=True)

返回一个由所有子层组成的列表。

参数：
    - **include_sublayers** (bool, 可选) - 是否返回子层中各个子层。如果为True，则包括子层中的各个子层。默认值：True。

返回： 一个由所有子层组成的列表，列表中的元素类型为Layer。

返回类型：list

.. py:method:: clear_gradients()

清除该层所有参数的梯度。

**代码示例**

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np
    
    # forward_pre_hook函数修改了layer的输入：input = input * 2
    
    def forward_pre_hook(layer, input):
        # 改变输入值
        input_return = input[0] * 2
        return input_return
    
    
    with paddle.imperative.guard():
        linear = fluid.Linear(13, 5, dtype='float32')
    
        # 注册hook
        forward_pre_hook_handle = linear.register_forward_pre_hook(forward_pre_hook
    
            )
        value0 = np.arange(26).reshape(2, 13).astype('float32')
        in0 = paddle.imperative.to_variable(value0)
        out0 = linear(in0)
    
        # 移除hook
        forward_pre_hook_handle.remove()
    
        value1 = value0 * 2
        in1 = paddle.imperative.to_variable(value1)
        out1 = linear(in1)
    
        # hook改变了layer的输入（input = input * 2），所以out0等于out1
        assert (out0.numpy() == out1.numpy()).any()

.. py:method:: named_parameters(prefix='', include_sublayers=True)

返回层中所有参数的迭代器，生成名称和参数的元组。

参数：
    - **prefix** (str, 可选) - 在所有参数名称前加的前缀。默认值：''。
    - **include_sublayers** (bool, 可选) - 是否返回子层的参数。如果为True，返回的列表中包含子层的参数。默认值：True。

返回：产出名称和参数的元组的迭代器。

返回类型：iterator

**代码示例**

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np
    
    # forward_pre_hook函数修改了layer的输入：input = input * 2
    
    def forward_pre_hook(layer, input):
        # 改变输入值
        input_return = input[0] * 2
        return input_return
    
    
    with paddle.imperative.guard():
        linear = fluid.Linear(13, 5, dtype='float32')
    
        # 注册hook
        forward_pre_hook_handle = linear.register_forward_pre_hook(forward_pre_hook
    
            )
        value0 = np.arange(26).reshape(2, 13).astype('float32')
        in0 = paddle.imperative.to_variable(value0)
        out0 = linear(in0)
    
        # 移除hook
        forward_pre_hook_handle.remove()
    
        value1 = value0 * 2
        in1 = paddle.imperative.to_variable(value1)
        out1 = linear(in1)
    
        # hook改变了layer的输入（input = input * 2），所以out0等于out1
        assert (out0.numpy() == out1.numpy()).any()

.. py:method:: named_sublayers(prefix='', include_sublayers=True, include_self=False, layers_set=None)

返回层中所有子层上的迭代器，生成名称和子层的元组。重复的子层只产生一次。

参数：
    - **prefix** (str, 可选) - 在所有参数名称前加的前缀。默认值：''。
    - **include_sublayers** (bool, 可选) - 是否返回子层中各个子层。如果为True，则包括子层中的各个子层。默认值：True。
    - **include_self** (bool, 可选) - 是否包含该层自身。默认值：False。
    - **layers_set** (set, 可选): 记录重复子层的集合。默认值：None。

返回：产出名称和子层的元组的迭代器。

返回类型：iterator

**代码示例**

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np
    
    # forward_pre_hook函数修改了layer的输入：input = input * 2
    
    def forward_pre_hook(layer, input):
        # 改变输入值
        input_return = input[0] * 2
        return input_return
    
    
    with paddle.imperative.guard():
        linear = fluid.Linear(13, 5, dtype='float32')
    
        # 注册hook
        forward_pre_hook_handle = linear.register_forward_pre_hook(forward_pre_hook
    
            )
        value0 = np.arange(26).reshape(2, 13).astype('float32')
        in0 = paddle.imperative.to_variable(value0)
        out0 = linear(in0)
    
        # 移除hook
        forward_pre_hook_handle.remove()
    
        value1 = value0 * 2
        in1 = paddle.imperative.to_variable(value1)
        out1 = linear(in1)
    
        # hook改变了layer的输入（input = input * 2），所以out0等于out1
        assert (out0.numpy() == out1.numpy()).any()

.. py:method:: forward(*inputs, **kwargs)

定义每次调用时执行的计算。应该被所有子类覆盖。

参数：
    - **\*inputs** (tuple) - 解包后的tuple参数。
    - **\*\*kwargs** (dict) - 解包后的dict参数。

.. py:method:: add_sublayer(name, sublayer)

添加子层实例。可以通过self.name访问该sublayer。

参数：
    - **name** (str) - 子层名。
    - **sublayer** (Layer) - Layer实例。

返回：添加的子层

返回类型：Layer

.. py:method:: add_parameter(name, parameter)

添加参数实例。可以通过self.name访问该parameter。

参数：
    - **name** (str) - 参数名。
    - **parameter** (Parameter) - Parameter实例。

返回：传入的参数实例

返回类型：Parameter( :ref:`cn_api_fluid_Variable` )

.. py:method:: state_dict(destination=None, include_sublayers=True)

获取当前层及其子层的所有参数。并将所有参数存放在dict结构中。

参数：
    - **destination** (dict, 可选) - 如果提供 ``destination`` ，则所有参数都将存放在 ``destination`` 中。 默认值：None。
    - **include_sublayers** (bool, 可选) - 如果设置为True，则包括子层的参数。默认值：True。

返回：包含所有参数的dict

返回类型：dict

**代码示例**

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np
    
    # forward_pre_hook函数修改了layer的输入：input = input * 2
    
    def forward_pre_hook(layer, input):
        # 改变输入值
        input_return = input[0] * 2
        return input_return
    
    
    with paddle.imperative.guard():
        linear = fluid.Linear(13, 5, dtype='float32')
    
        # 注册hook
        forward_pre_hook_handle = linear.register_forward_pre_hook(forward_pre_hook
    
            )
        value0 = np.arange(26).reshape(2, 13).astype('float32')
        in0 = paddle.imperative.to_variable(value0)
        out0 = linear(in0)
    
        # 移除hook
        forward_pre_hook_handle.remove()
    
        value1 = value0 * 2
        in1 = paddle.imperative.to_variable(value1)
        out1 = linear(in1)
    
        # hook改变了layer的输入（input = input * 2），所以out0等于out1
        assert (out0.numpy() == out1.numpy()).any()

.. py:method:: set_dict(stat_dict, include_sublayers=True)

根据传入的 ``stat_dict`` 设置参数。 所有参数将由 ``stat_dict`` 中的 ``Tensor`` 设置。

参数：
    - **state_dict** (dict) - 包含所有参数的dict。
    - **include_sublayers** (bool, 可选) - 如果设置为True，则还包括子层的参数。 默认值：True。

返回：None

**代码示例**

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np
    
    # forward_pre_hook函数修改了layer的输入：input = input * 2
    
    def forward_pre_hook(layer, input):
        # 改变输入值
        input_return = input[0] * 2
        return input_return
    
    
    with paddle.imperative.guard():
        linear = fluid.Linear(13, 5, dtype='float32')
    
        # 注册hook
        forward_pre_hook_handle = linear.register_forward_pre_hook(forward_pre_hook
    
            )
        value0 = np.arange(26).reshape(2, 13).astype('float32')
        in0 = paddle.imperative.to_variable(value0)
        out0 = linear(in0)
    
        # 移除hook
        forward_pre_hook_handle.remove()
    
        value1 = value0 * 2
        in1 = paddle.imperative.to_variable(value1)
        out1 = linear(in1)
    
        # hook改变了layer的输入（input = input * 2），所以out0等于out1
        assert (out0.numpy() == out1.numpy()).any()

.. py:method:: load_dict(stat_dict, include_sublayers=True)

.. warning::
    该函数将被弃用。请使用set_dict函数。

根据传入的 ``stat_dict`` 设置参数。 所有参数将由 ``stat_dict`` 中的 ``Tensor`` 设置。

参数：
    - **state_dict** (dict) - 包含所有参数的dict。
    - **include_sublayers** (bool, 可选) - 如果设置为True，则还包括子层的参数。 默认值：True。

返回：None

**代码示例**

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np
    
    # forward_pre_hook函数修改了layer的输入：input = input * 2
    
    def forward_pre_hook(layer, input):
        # 改变输入值
        input_return = input[0] * 2
        return input_return
    
    
    with paddle.imperative.guard():
        linear = fluid.Linear(13, 5, dtype='float32')
    
        # 注册hook
        forward_pre_hook_handle = linear.register_forward_pre_hook(forward_pre_hook
    
            )
        value0 = np.arange(26).reshape(2, 13).astype('float32')
        in0 = paddle.imperative.to_variable(value0)
        out0 = linear(in0)
    
        # 移除hook
        forward_pre_hook_handle.remove()
    
        value1 = value0 * 2
        in1 = paddle.imperative.to_variable(value1)
        out1 = linear(in1)
    
        # hook改变了layer的输入（input = input * 2），所以out0等于out1
        assert (out0.numpy() == out1.numpy()).any()

