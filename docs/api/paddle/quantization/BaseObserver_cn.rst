.. _cn_api_paddle_quantization_BaseObserver:

BaseObserver
---------------------

.. py:class:: paddle.quantization.BaseObserver
内置的观察器和定制的观察器应当扩展这个基类观察器，并且实现抽象方法。

方法
::::::::::::
add_parameter(name, parameter)
'''''''''
添加一个参数实例且添加的参数可以通过 self.name 来访问。

**参数**

 - **name** (str) - 是一个参数，这个参数用于指定或标识某个子层的名称。
 - **parameter** (Parameter) - 一个参数实例。

**返回**

返回值为 parameter 的传入的参数 Parameter 。

**代码示例**

COPY-FROM: paddle.quantization.BaseObserver.add_parameter

add_sublayer(name, sublayer)
'''''''''
添加一个子层实例并且添加的子层可以通过 self.name 来访问。

**参数**

 - **name** (str) - 这个子层的名称。
 - **sublayer** (Layer) - 一个 Layer 的实例，它将被作为子层添加。

**返回**

返回值为 Layer,函数返回传入的子层实例

**代码示例**

COPY-FROM: paddle.quantization.BaseObserver.add_sublayer

apply(fn)
'''''''''
将函数 fn 递归地应用到每个子层（如.sublayers()返回的）以及自身。典型的用途包括初始化模型的参数。

**参数**

 - **fn** (function) -  应用于每个子层的函数。

**返回**

返回值为 Layer，即对象本身。

**代码示例**

COPY-FROM: paddle.quantization.BaseObserver.apply

astype(dtype=None)
'''''''''
将所有参数和缓冲区转换为 dtype 类型，然后返回 Layer（层）。

**参数**

 - **dtype** (str|paddle.dtype|numpy.dtype) -  层的目标数据类型。如果设置为字符串，它可以是 “bool”、“bfloat16”、“float16”、“float32”、“float64”、“int8”、“int16”、“int32”、“int64”、“uint8”、“complex64” 或 “complex128”。默认值为 None。

**返回**

返回值为 Layer，即对象本身。

**代码示例**

COPY-FROM: paddle.quantization.BaseObserver.astype

bfloat16(excluded_layers=None)
'''''''''
将所有浮点数参数和缓冲区转换为 bfloat16 数据类型。

..note:
    nn.BatchNorm 不支持 bfloat16 权重，因此默认情况下不会进行转换。

**参数**

 - **excluded_layers**（nn.Layer|list|tuple|None，可选）- 指定需要保持原始数据类型的层。如果 excluded_layers 为 None，则转换所有浮点数参数和缓冲区，除了 nn.BatchNorm。默认值：None。

**返回**

返回值为 Layer，即对象本身。

**代码示例**

COPY-FROM: paddle.quantization.BaseObserver.bfloat16

abstract bit_length()
'''''''''
获取量化的比特长度。

buffers(include_sublayers=True)
'''''''''
返回当前层及其子层中所有缓冲区的列表。

**参数**

 - **include_sublayers**（布尔值，可选）- 是否包含子层的缓冲区。如果为 True，也将包括来自子层的缓冲区。默认值为 True,即包含子层的缓冲区。

**返回**

返回一个缓冲区的 Tensor 列表。

**代码示例**

COPY-FROM: paddle.quantization.BaseObserver.buffers

children()
'''''''''
返回一个遍历直接子层的迭代器。

**生成**

产生一个子层

**代码示例**

COPY-FROM: paddle.quantization.BaseObserver.children

clear_gradients()
'''''''''
清除此层所有参数的梯度。

**返回**

无。

**代码示例**

COPY-FROM: paddle.quantization.BaseObserver.clear_gradients

create_parameter(shape, attr=None, dtype=None, is_bias=False, default_initializer=None)
'''''''''
为这一层创建参数。

**参数**

 - **shape** (list) - 参数的形状。列表中的数据类型必须是 int。
 - **attr** (ParamAttr，可选) - 权重的参数属性。请参考 :ref:`cn_api_paddle_ParamAttr`。默认值：None。
 - **dtype** (str，可选) - 该参数的数据类型。如果设置为 str，可以是"bool"、“float16”、“float32”、“float64”、“int8”、“int16”、“int32”、“int64”、“uint8"或"uint16”。默认值：“float32”。
 - **is_bias** (bool，可选) - 是否为偏置参数。默认值：False。
 - **default_initializer** (Initializer，可选) - 该参数的默认初始化器。如果设置为 None，默认初始化器将为非偏置参数设置为 paddle.nn.initializer.Xavier，为偏置参数设置为 paddle.nn.initializer.Constant。默认值：None。

**返回**

Tensor,创建的参数张量。

**代码示例**

COPY-FROM: paddle.quantization.BaseObserver.create_parameter

create_tensor(name=None, persistable=None, dtype=None)
'''''''''
为这一层创建 Tensor。

**参数**

 - **name** (str，可选) - Tensor 的名称。请参考:ref:`cn_user_guide_broadcasting`。默认值：None。
 - **persistable** (bool，可选) -  如果设置这个张量为可持久化。默认值：False。
 - **dtype** (str，可选) - 该参数的数据类型。可以是 “bool”、“float16”、“float32”、“float64”、“int8”、“int16”、“int32”、“int64”、“uint8” 或 “uint16”。如果设为 None，则为 “float32”。默认值：None。

**返回**

Tensor,创建的张量。

**代码示例**

COPY-FROM: paddle.quantization.BaseObserver.create_tensor

create_variable(name=None, persistable=None, dtype=None)
'''''''''
为这一层创建 Tensor。

**参数**

 - **name** (str，可选) - Tensor 的名称。请参考:ref:`cn_user_guide_broadcasting`。默认值：None。
 - **persistable** (bool，可选) -  如果设置这个张量为可持久化。默认值：False。
 - **dtype** (str，可选) - 该参数的数据类型。可以是 “bool”、“float16”、“float32”、“float64”、“int8”、“int16”、“int32”、“int64”、“uint8” 或 “uint16”。如果设为 None，则为 “float32”。默认值：None。

**返回**

Tensor,创建的张量。

**代码示例**

COPY-FROM: paddle.quantization.BaseObserver.create_variable

eval()
'''''''''
将这个层及其所有子层设置为评估模式。这只会影响到一些模块，比如 Dropout 和 BatchNorm。

**返回**

无。

**代码示例**

COPY-FROM: paddle.quantization.BaseObserver.eval

extra_repr()
'''''''''
自定义层的额外表示，您可以对自己的层进行自定义实现。

float(excluded_layers=None)
'''''''''
将所有浮点型参数和缓冲区转换为浮点数据类型。

**参数**

 - **excluded_layers** (nn.Layer|list|tuple|None，可选) - 指定需要保留原始数据类型的层。如果 excluded_layers 参数为 None，则将所有浮点型参数和缓冲区转换为浮点数据类型。默认值：None。

**返回**

返回值为 Layer，即对象本身。

**代码示例**

COPY-FROM: paddle.quantization.BaseObserver.float

float16(excluded_layers=None)
'''''''''
将所有浮点型参数和缓冲区转换为 float16 数据类型。

..note:
    nn.BatchNorm 不支持 bfloat16 权重，因此默认情况下不会进行转换。

**参数**

- **excluded_layers** (nn.Layer|list|tuple|None，可选) - 指定需要保留原始数据类型的层。如果 excluded_layers 参数为 None，则将除了 nn.BatchNorm 之外的所有浮点型参数和缓冲区转换为 float16 数据类型。默认值：None。

**返回**

返回值为 Layer，即对象本身。

**代码示例**

COPY-FROM: paddle.quantization.BaseObserver.float16

abstract forward(input)
'''''''''
定义在每次调用时执行的计算。所有子类都应该重写这个方法。

**参数**
- **inputs** (tuple) - 是一个解包后的 tuple(元组) 参数。
- **kwargs** (dict) - 是一个解包后的 dict(字典) 参数。

full_name()
'''''''''
这个层的完整名称由 name_scope + "/" + MyLayer.__class__.__name__ 组成。

**返回**

返回值是字符串，代表这个层的完整名称。

**代码示例**

COPY-FROM: paddle.quantization.BaseObserver.full_name

load_dict(state_dict, use_structured_name=True)
'''''''''
从 state_dict 中设置参数和可持久化缓冲区。所有的参数和缓冲区将会被 state_dict 中的张量重置。

**参数**

- **state_dict** (dict) - 包含所有参数和可持久化缓冲区的 dict(字典)。
- **use_structured_name** (bool，可选) - 如果为真，使用结构化名称作为键，否则，使用参数或缓冲区的名称作为键。默认值：True，使用结构化名称作为键值。

**返回**

missing_keys(list)：一个包含缺失键的字符串列表
unexpected_keys(list)：一个包含意外键的字符串列表。

**返回**

返回一个包含缺失键的字符串列表。

**代码示例**

COPY-FROM: paddle.quantization.BaseObserver.load_dict

named_buffers(prefix='', include_sublayers=True)
'''''''''
返回一个在 Layer 中遍历所有缓冲区的迭代器，生成包含名称和 Tensor 的元组。

**参数**

- **prefix** (str，可选) - 需要添加到所有缓冲区名称前的前缀。默认值为 ''。
- **include_sublayers** (bool，可选) - 是否包括子层中的缓冲区。如果设为 True，则也会包含来自子层的已命名缓冲区。默认值为 True。

**生成**

(string, Tensor) - 包含名称和张量的元组。

**代码示例**

COPY-FROM: paddle.quantization.BaseObserver.named_buffers

named_children()
'''''''''
返回一个对直接子层的迭代器，它会生成层的名称以及层本身。

**生成**

(string, Layer) - 包含名称和子层的 Tuple(元组)。

**代码示例**

COPY-FROM: paddle.quantization.BaseObserver.named_children

named_parameters(prefix='', include_sublayers=True)
'''''''''
返回一个在 Layer 中所有参数上的迭代器，产出名称和参数的元组。

**参数**

- **prefix** (str，可选) - 需要添加到所有缓冲区名称前的前缀。默认值为 ''。
- **include_sublayers** (bool，可选) - 是否包括子层中的缓冲区。如果设为 True，则也会包含来自子层的已命名缓冲区。默认值为 True。

**生成**

(string, Parameter) - 名称和参数的 Tuple(元组)。

**代码示例**

COPY-FROM: paddle.quantization.BaseObserver.named_parameters

named_sublayers(prefix='', include_self=False, layers_set=None)
'''''''''
返回一个在 Layer 中所有子层上的迭代器，产出名称和子层的元组。重复的子层只会被产出一次。

**参数**

- **prefix** (str，可选) -  要添加到所有参数名称前的前缀。默认值为 ''。
- **include_self** (bool，可选) - 是否包含 Layer 本身。默认值为 False。
- **layers_set** (set，可选) - 用来记录重复子层的集合。默认值为 None。

**生成**

(string, Layer) - 名称和 Layer(层)的 Tuple(元组)。

**代码示例**

COPY-FROM: paddle.quantization.BaseObserver.named_sublayers

parameters(include_sublayers=True)
'''''''''
返回当前层及其子层的所有参数的列表。

**参数**

- **include_sublayers** (bool，可选) -  是否返回子层的参数。如果为 True，则返回的列表包含子层的参数。默认值：True。

**返回**

Tensor 列表，一个参数列表。

**代码示例**

COPY-FROM: paddle.quantization.BaseObserver.parameters

abstract quant_axis()→ Union[int, collections.abc.Iterable]
'''''''''
获取量化的轴。None 意味着对整个 Tensor 进行量化。

register_buffer(name, tensor, persistable=True)
'''''''''
在层中注册一个张量作为缓冲区。

缓冲区是一个非可训练的张量，不会被优化器更新，但对评估和推理是必要的。例如，BatchNorm 层中的平均值和方差。默认情况下，注册的缓冲区是可持久化的，并且会和参数一起被保存到状态字典（state_dict）中。如果设置 persistable=False，则它注册的是一个非持久化的缓冲区，因此它不会成为状态字典的一部分。

可以使用给定的名称作为属性来访问缓冲区。

**参数**

- **name** (string) -  缓冲区的名称。可以使用给定的名称从这个层访问该缓冲区。
- **tensor** (Tensor) -  要注册为缓冲区的 Tensor。
- **persistable** (bool) -  缓冲区是否是这个层状态字典的一部分。

**返回**

无。

**代码示例**

COPY-FROM: paddle.quantization.BaseObserver.register_buffer

register_forward_post_hook(hook)
'''''''''
为层注册一个前向传播后钩子（forward post-hook）。该钩子将在前向传播函数计算完成后被调用。

它应该具有以下形式，钩子的输入和输出分别是层的输入和输出。用户可以使用前向传播后钩子来改变层的输出或对层进行信息统计任务。

hook(Layer, input, output) -> 无 或 修改后的输出

**参数**

- **hook** (function) - 注册为前向传播后钩子的函数。

**返回**

HookRemoveHelper，一个 HookRemoveHelper 对象，可以通过调用 hook_remove_helper.remove() 来移除添加的钩子。

**代码示例**

COPY-FROM: paddle.quantization.BaseObserver.register_forward_post_hook

register_forward_pre_hook(hook)
'''''''''
为层注册一个前向传播前钩子（forward pre-hook）。该钩子将在前向传播函数计算之前被调用。

它应该具有以下形式：钩子的输入是层的输入，钩子可以返回一个元组或者在钩子中修改后的单个值。如果返回的是单个值（除非该值本身就是元组），我们会将该值包装成一个元组。用户可以使用前向传播前钩子来改变层的输入或者对层执行信息统计任务。

hook(Layer, input) -> 无 或 修改后的输出

**参数**

- **hook** (function) - 注册为前向传播后钩子的函数。

**返回**

HookRemoveHelper，一个 HookRemoveHelper 对象，可以通过调用 hook_remove_helper.remove() 来移除添加的钩子。

**代码示例**

COPY-FROM: paddle.quantization.BaseObserver.register_forward_pre_hook

abstract scales()→ Union[paddle.Tensor, numpy.ndarray]
'''''''''
获取用于量化的比例尺。它可以是空值，这意味着量化器没有保留用于量化的比例尺。

set_dict(state_dict, use_structured_name=True)
'''''''''
从 state_dict 设置参数和持久化缓冲区。所有的参数和缓冲区都将通过 state_dict 中的张量来重置。

**参数**

- **state_dict** (dict) - 包含所有参数和持久化缓冲区的 dict（字典)。
- **use_structured_name** (bool，可选) - 如果为真，使用结构化名称作为键，否则，使用参数或缓冲区名称作为键。默认值：True。

**返回**

missing_keys (list)：一个包含缺失键的字符串列表；
unexpected_keys (list)：一个包含意外键的字符串列表。

**代码示例**

COPY-FROM: paddle.quantization.BaseObserver.set_dict

set_state_dict(state_dict, use_structured_name=True)
'''''''''
从 state_dict 设置参数和持久性缓冲区。所有的参数和缓冲区将会被 state_dict 中的张量重置。

**参数**

- **state_dict** (bool，可选) - 包含所有参数和持久化缓冲区的 dict（字典)。
- **use_structured_name** (bool，可选) - 如果为真，使用结构化名称作为键，否则，使用参数或缓冲区名称作为键。默认值：True。

**返回**

missing_keys (list)：一个包含缺失键的字符串列表；
unexpected_keys (list)：一个包含意外键的字符串列表。

**代码示例**

COPY-FROM: paddle.quantization.BaseObserver.set_state_dict

state_dict(destination=None, include_sublayers=True, structured_name_prefix='', use_hook=True)
'''''''''
获取当前层及其子层的所有参数和可持久化缓冲区，并将它们设置到一个字典中。

**参数**

- **destination** (dict，可选) - 如果提供，所有的参数和可持久化缓冲区将被设置到这个字典中。默认值：None。
- **include_sublayers** (bool，可选) - 如果为真，也会包括子层中的参数和可持久化缓冲区。默认值：True。
- **use_hook** (bool，可选) - 如果为真，包含在 _state_dict_hooks 中的操作将会被追加到 destination 中。默认值：True。

**返回**

dict，一个包含所有参数和可持久化缓冲区的字典。

**代码示例**

COPY-FROM: paddle.quantization.BaseObserver.state_dict

sublayers(include_self=False)
'''''''''
返回子层的列表。

**参数**

- **include_self** (bool，可选) - 是否将自身作为子层返回。默认值：False。

**返回**

Layer 的列表，一个包含子层的列表。

**代码示例**

COPY-FROM: paddle.quantization.BaseObserver.sublayers

to(device=None, dtype=None, blocking=None)
'''''''''
将 Layer 的参数和缓冲区按照给定的设备、数据类型和暂停模式转换。

**参数**

- **device** (str|paddle.CPUPlace()|paddle.CUDAPlace()|paddle.CUDAPinnedPlace()|paddle.XPUPlace()|None，可选) - 想要存储的 Layer 的设备。
- **string** (如果设备与原始 Tensor 相同) -
- **cpu** (可选) -
- **xpu:x** (gpu:x) -
- **the** ( x 是) -
- **Default** (index of the GPUs or XPUs) -
- **dtype** (str|numpy.dtype|paddle.dtype|None，可选) - 数据的类型。如果为 None，dtype 与原始 Tensor 相同。默认值：None。
- **blocking** (bool|None，可选) - 如果为 False 并且源数据在固定内存中，复制将异步于主机。否则，该参数无效。如果为 None，blocking 被设置为 True。默认值：None。

**返回**

返回值为自身。

**代码示例**

COPY-FROM: paddle.quantization.BaseObserver.to

to_static_state_dict(destination=None, include_sublayers=True, structured_name_prefix='', use_hook=True)
'''''''''
获取当前层及其子层的所有参数和缓冲区，并将它们设定到一个字典中。

**参数**

- **destination** (dict，可选) - 如果提供，所有的参数和持久化缓冲区将会被设定到这个字典中。默认值：None。
- **include_sublayers** (bool，可选) - 如果为真，还会包含子层中的参数和持久化缓冲区。默认值：True。
- **use_hook** (bool，可选) - 如果为真，包含在 _state_dict_hooks 中的操作将会被追加到目的地字典中。默认值：True。

**返回**

dict（字典），一个包含所有参数和持久化缓冲区的字典。

**代码示例**

COPY-FROM: paddle.quantization.BaseObserver.to_static_state_dict

train()
'''''''''
将这个层及其所有子层设置为训练模式。这仅影响某些模块，如 Dropout 和 BatchNorm。

**返回**

无。

**代码示例**

COPY-FROM: paddle.quantization.BaseObserver.train

abstract zero_points()→ Union[paddle.Tensor, numpy.ndarray]
'''''''''
获取用于量化的零点。它可以是无，这意味着量化器没有为量化保留零点。
