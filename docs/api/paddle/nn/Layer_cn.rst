.. _cn_api_paddle_nn_Layer:

Layer
-------------------------------

.. py:class:: paddle.nn.Layer(name_scope=None, dtype="float32")




基于 OOD 实现的动态图 Layer，包含该 Layer 的参数、前序运行的结构等信息。

参数
::::::::::::

    - **name_scope** (str，可选) - 为 Layer 内部参数命名而采用的名称前缀。如果前缀为“mylayer”，在一个类名为 MyLayer 的 Layer 中，参数名为“mylayer_0.w_n”，其中 w 是参数的名称，n 为自动生成的具有唯一性的后缀。如果为 None，前缀名将为小写的类名。默认值为 None。
    - **dtype** (str 可选) - Layer 中参数数据类型。如果设置为 str，则可以是“bool”，“float16”，“float32”，“float64”，“int8”，“int16”，“int32”，“int64”，“uint8”或“uint16”。默认值为 "float32"。

方法
::::::::::::
train()
'''''''''

将此层及其所有子层设置为训练模式。这只会影响某些模块，如 Dropout 和 BatchNorm。

**返回**
无

**代码示例**

COPY-FROM: paddle.nn.Layer

eval()
'''''''''

将此层及其所有子层设置为预测模式。这只会影响某些模块，如 Dropout 和 BatchNorm。

**返回**
无

**代码示例**

COPY-FROM: paddle.nn.Layer.eval

full_name()
'''''''''

Layer 的全名。组成方式为：``name_scope`` + “/” + MyLayer.__class__.__name__ 。

**返回**
str， Layer 的全名

**代码示例**

COPY-FROM: paddle.nn.Layer.full_name

register_forward_pre_hook(hook)
'''''''''

为 Layer 注册一个 ``forward pre-hook`` 函数，该 ``hook`` 函数将会在 ``forward`` 函数调用之前被调用。

``hook`` 函数具有以下形式：它的 ``input`` 是 ``Layer`` 的 ``input``，并且可以返回一个元组或者单个修改值；如果返回单个修改值，则将值包装到一个元组中。用户可以使用该函数来查看或修改 ``Layer`` ``forward`` 函数的输入。

hook(Layer, input) -> None or modified input

**参数**

    - **hook** (function) - 被注册为 ``forward pre-hook`` 的函数

**返回**
HookRemoveHelper，可通过调用 ``hook_remove_helper.remove()`` 来删除注册的 hook 函数。

**代码示例**

COPY-FROM: paddle.nn.Layer.register_forward_pre_hook

register_forward_post_hook(hook)
'''''''''

为 Layer 注册一个 ``forward post-hook`` 函数，该 ``hook`` 函数将会在 ``forward`` 函数调用之后被调用。

``hook`` 函数具有以下形式，它的 ``input`` 和 ``output`` 是 ``Layer`` 的 ``input`` 和 ``output``。用户可以用该函数来查看和修改 ``Layer`` ``forward`` 函数的输出。

hook(Layer, input, output) -> None or modified output

**参数**

    - **hook** (function) - 被注册为 ``forward post-hook`` 的函数

**返回**
HookRemoveHelper，可通过调用 ``hook_remove_helper.remove()`` 来删除注册的 hook 函数。

**代码示例**

COPY-FROM: paddle.nn.Layer.register_forward_post_hook

create_parameter(shape, attr=None, dtype="float32", is_bias=False, default_initializer=None)
'''''''''

为 Layer 创建参数。

**参数**

    - **shape** (list) - 参数的形状。列表中的数据类型必须为 int。
    - **attr** (ParamAttr，可选) - 指定权重参数属性的对象，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_paddle_ParamAttr`。默认值为 None。
    - **dtype** (str|core.VarDesc.VarType，可选) - Layer 中参数数据类型。如果设置为 str，则可以是“bool”，“float16”，“float32”，“float64”，“int8”，“int16”，“int32”，“int64”，“uint8”或“uint16”。默认值为“float32”。
    - **is_bias** (bool，可选) - 是否是偏置参数。默认值：False。
    - **default_initializer** (Initializer，可选) - 默认的参数初始化方法。如果设置为 None，则设置非 bias 参数的初始化方式为 paddle.nn.initializer.Xavier，设置 bias 参数的初始化方式为 paddle.nn.initializer.Constant。默认值：None。

**返回**
Tensor，创建的参数变量

**代码示例**

COPY-FROM: paddle.nn.Layer.create_parameter

create_variable(name=None, persistable=None, dtype=None)
'''''''''

为 Layer 创建变量。

**参数**

    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
    - **persistable** (bool，可选) - 是否为持久性变量，后续会被移出。默认值：None。
    - **dtype** (str，可选) - Layer 中参数数据类型。如果设置为 str，则可以是“bool”，“float16”，“float32”，“float64”，“int8”，“int16”，“int32”，“int64”，“uint8”或“uint16”。默认值为 "float32" 。

**返回**
Tensor，返回创建的 ``Tensor``

**代码示例**

COPY-FROM: paddle.nn.Layer.create_variable

create_tensor(name=None, persistable=None, dtype=None)
'''''''''

为 Layer 创建变量。

**参数**

    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
    - **persistable** (bool，可选) - 是否为持久性变量，后续会被移出。默认值：None。
    - **dtype** (str，可选) - Layer 中参数数据类型。如果设置为 str，则可以是“bool”，“float16”，“float32”，“float64”，“int8”，“int16”，“int32”，“int64”，“uint8”或“uint16”。默认值为 "float32" 。

**返回**
Tensor，返回创建的 ``Tensor``

**代码示例**

COPY-FROM: paddle.nn.Layer.create_tensor

parameters(include_sublayers=True)
'''''''''

返回一个由当前层及其子层的所有参数组成的列表。

**参数**

    - **include_sublayers** (bool，可选) - 是否返回子层的参数。如果为 True，返回的列表中包含子层的参数。默认值：True。

**返回**
list，一个由当前层及其子层的所有参数组成的列表，列表中的元素类型为 Parameter(Tensor)。

**代码示例**

COPY-FROM: paddle.nn.Layer.parameters

children()
'''''''''

返回所有子层的迭代器。

**返回**
iterator，子层的迭代器。

**代码示例**

COPY-FROM: paddle.nn.Layer.children

named_children()
'''''''''

返回所有子层的迭代器，生成子层名称和子层的元组。

**返回**
iterator，产出子层名称和子层的元组的迭代器。

**代码示例**

COPY-FROM: paddle.nn.Layer.named_children

sublayers(include_self=False)
'''''''''

返回一个由所有子层组成的列表。

**参数**

    - **include_self** (bool，可选) - 是否包含本层。如果为 True，则包括本层。默认值：False

**返回**
 list，一个由所有子层组成的列表，列表中的元素类型为 Layer。

**代码示例**

COPY-FROM: paddle.nn.Layer.sublayers

clear_gradients()
'''''''''

清除该层所有参数的梯度。

**返回**
无

**代码示例**

COPY-FROM: paddle.nn.Layer.clear_gradients

named_parameters(prefix='', include_sublayers=True)
'''''''''

返回层中所有参数的迭代器，生成名称和参数的元组。

**参数**

    - **prefix** (str，可选) - 在所有参数名称前加的前缀。默认值：''。
    - **include_sublayers** (bool，可选) - 是否返回子层的参数。如果为 True，返回的列表中包含子层的参数。默认值：True。

**返回**
iterator，产出名称和参数的元组的迭代器。

**代码示例**

COPY-FROM: paddle.nn.Layer.named_parameters

named_sublayers(prefix='', include_self=False, layers_set=None)
'''''''''

返回层中所有子层上的迭代器，生成名称和子层的元组。重复的子层只产生一次。

**参数**

    - **prefix** (str，可选) - 在所有参数名称前加的前缀。默认值：''。
    - **include_self** (bool，可选) - 是否包含该层自身。默认值：False。
    - **layers_set** (set，可选)：记录重复子层的集合。默认值：None。

**返回**
iterator，产出名称和子层的元组的迭代器。

**代码示例**

COPY-FROM: paddle.nn.Layer.named_sublayers

register_buffer(name, tensor, persistable=True)
'''''''''

将一个 Tensor 注册为 buffer。

buffer 是一个不可训练的变量，不会被优化器更新，但在评估或预测阶段可能是必要的状态变量。比如 ``BatchNorm`` 中的均值和方差。

注册的 buffer 默认是可持久性的，会被保存到 ``state_dict`` 中。如果指定 ``persistable`` 参数为 False，则会注册一个非持久性的 buffer，即不会同步和保存到 ``state_dict`` 中。

**参数**

    - **name** (str) - 注册 buffer 的名字。可以通过此名字来访问已注册的 buffer。
    - **tensor** (Tensor) - 将被注册为 buffer 的变量。
    - **persistable** (bool，可选) - 注册的 buffer 是否需要可持久性地保存到 ``state_dict`` 中。

**返回**
None

**代码示例**

COPY-FROM: paddle.nn.Layer.register_buffer

buffers(include_sublayers=True)
'''''''''

返回一个由当前层及其子层的所有 buffers 组成的列表。

**参数**

    - **include_sublayers** (bool，可选) - 是否返回子层的 buffers。如果为 True，返回的列表中包含子层的 buffers。默认值：True。

**返回**
list，一个由当前层及其子层的所有 buffers 组成的列表，列表中的元素类型为 Tensor。

**代码示例**

COPY-FROM: paddle.nn.Layer.buffers

named_buffers(prefix='', include_sublayers=True)
'''''''''

返回层中所有 buffers 的迭代器，生成名称和 buffer 的元组。

**参数**

    - **prefix** (str，可选) - 在所有 buffer 名称前加的前缀。默认值：''。
    - **include_sublayers** (bool，可选) - 是否返回子层的 buffers。如果为 True，返回的列表中包含子层的 buffers。默认值：True。

**返回**
iterator，产出名称和 buffer 的元组的迭代器。

**代码示例**

COPY-FROM: paddle.nn.Layer.named_buffers

forward(*inputs, **kwargs)
'''''''''

定义每次调用时执行的计算。应该被所有子类覆盖。

**参数**

    - **\*inputs** (tuple) - 解包后的 tuple 参数。
    - **\*\*kwargs** (dict) - 解包后的 dict 参数。

**返回**
 无

add_sublayer(name, sublayer)
'''''''''

添加子层实例。可以通过 self.name 访问该 sublayer。

**参数**

    - **name** (str) - 子层名。
    - **sublayer** (Layer) - Layer 实例。

**返回**
Layer，添加的子层

**代码示例**

COPY-FROM: paddle.nn.Layer.add_sublayer

add_parameter(name, parameter)
'''''''''

添加参数实例。可以通过 self.name 访问该 parameter。

**参数**

    - **name** (str) - 参数名。
    - **parameter** (Parameter) - Parameter 实例。

**返回**
Parameter，传入的参数实例

**代码示例**

COPY-FROM: paddle.nn.Layer.add_parameter

state_dict(destination=None, include_sublayers=True, use_hook=True)
'''''''''

获取当前层及其子层的所有参数和可持久性 buffers。并将所有参数和 buffers 存放在 dict 结构中。

**参数**

    - **destination** (dict，可选) - 如果提供 ``destination``，则所有参数和可持久性 buffers 都将存放在 ``destination`` 中。默认值：None。
    - **include_sublayers** (bool，可选) - 如果设置为 True，则包括子层的参数和 buffers。默认值：True。
    - **use_hook** (bool，可选) - 如果设置为 True，将_state_dict_hooks 中注册的函数应用于 destination。默认值：True。

**返回**
dict，包含所有参数和可持久行 buffers 的 dict

**代码示例**

COPY-FROM: paddle.nn.Layer.state_dict

set_state_dict(state_dict, use_structured_name=True)
'''''''''

根据传入的 ``state_dict`` 设置参数和可持久性 buffers。所有参数和 buffers 将由 ``state_dict`` 中的 ``Tensor`` 设置。

**参数**

    - **state_dict** (dict) - 包含所有参数和可持久性 buffers 的 dict。
    - **use_structured_name** (bool，可选) - 如果设置为 True，将使用 Layer 的结构性变量名作为 dict 的 key，否则将使用 Parameter 或者 Buffer 的变量名作为 key。默认值：True。

**返回**
    - **missing_keys** (list) - 没有匹配到的参数名列表
    - **unexpected_keys** (list) - state_dict 传入的无效的参数名列表


**代码示例**

COPY-FROM: paddle.nn.Layer.set_state_dict

to(device=None, dtype=None, blocking=None)
'''''''''

根据给定的 device、dtype 和 blocking 转换 Layer 中的 parameters 和 buffers。

**参数**

    - **device** （str|paddle.CPUPlace()|paddle.CUDAPlace()|paddle.CUDAPinnedPlace()|paddle.XPUPlace()|None，可选) - 希望存储 Layer 的设备位置。如果为 None，设备位置和原始的 Tensor 的设备位置一致。如果设备位置是 string 类型，取值可为 ``cpu``, ``gpu:x`` and ``xpu:x``，这里的 ``x`` 是 GPUs 或者 XPUs 的编号。默认值：None。
    - **dtype** （str|numpy.dtype|paddle.dtype|None，可选) - 数据的类型。如果为 None，数据类型和原始的 Tensor 一致。默认值：None。
    - **blocking** （bool|None，可选）- 如果为 False 并且当前 Tensor 处于固定内存上，将会发生主机到设备端的异步拷贝。否则，会发生同步拷贝。如果为 None，blocking 会被设置为 True。默认为 False。

**代码示例**

COPY-FROM: paddle.nn.Layer.to

astype(dtype=None)
:::::::::

将 Layer 的所有 ``parameters`` 和 ``buffers`` 的数据类型转换为 ``dtype``，并返回这个 Layer。

**参数**
    - **dtype** (str | paddle.dtype | numpy.dtype) - 转换后的 dtype，str 类型支持"bool", "bfloat16", "float16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "complex64", "complex128"。

返回：类型转换后的 Layer

返回类型：Layer

**代码示例**
COPY-FROM: paddle.nn.Layer.astype

float(excluded_layers=None)
'''''''''

将所有浮点型的参数和通过 ``register_buffers()`` 注册的 Buffer 变量转换为 float 数据类型。

**参数**

    - **excluded_layers** （list|tuple|nn.Layer|None，可选） - 不需要转换数据类型的层。如果 ``excluded_layers`` 为 None，则转换所有浮点参数和缓冲区，默认值：None。

**代码示例**

COPY-FROM: paddle.nn.Layer.float

float16(excluded_layers=None)
'''''''''

将所有浮点型的参数和通过 ``register_buffers()`` 注册的 Buffer 变量转换为 float16 数据类型。

.. note::
   nn.BatchNorm 不支持 float16 类型的权重，默认不对其权重进行类型转换。

**参数**

    - **excluded_layers** （list|tuple|nn.Layer|None，可选） - 不需要转换数据类型的层。如果 ``excluded_layers`` 为 None，则转换除 ``nn.BatchNorm`` 之外的所有浮点参数和缓冲区，默认值：None。

**代码示例**

COPY-FROM: paddle.nn.Layer.float16

bfloat16(excluded_layers=None)
'''''''''

将所有浮点型的参数和通过 ``register_buffers()`` 注册的 Buffer 变量转换为 bfloat16 数据类型。

.. note::
   nn.BatchNorm 不支持 bfloat16 类型的权重，默认不对其权重进行类型转换。

**参数**

    - **excluded_layers** （list|tuple|nn.Layer|None，可选） - 不需要转换数据类型的层。如果 ``excluded_layers`` 为 None，则转换除 ``nn.BatchNorm`` 之外的所有浮点参数和缓冲区，默认值：None。

**代码示例**

COPY-FROM: paddle.nn.Layer.bfloat16
