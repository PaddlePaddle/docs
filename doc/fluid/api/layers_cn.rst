.. _cn_api_fluid_layers_equal:

equal
>>>>>>>>>>

.. py:class:: paddle.fluid.layers.equal(x,y,cond=None,**ignored)

**equal**
该层返回 :math:'x==y' 按逐元素运算而得的真值。

参数：
    - **x** (Variable)-equal的第一个操作数
    - **y** (Variable)-equal的第二个操作数
    - **cond** (Variable|None)-输出变量（可选），用来存储equal的结果

返回：张量类型的变量，存储equal的输出结果 

返回类型：变量（Variable） 

**代码示例**: 

.. code-block:: python

    less = fluid.layers.equal(x=label,y=limit)

.. _cn_api_fluid_layers_array_read:

array_read
>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.array_read(array,i)

此函数用于读取数据，数据以LOD_TENSOR_ARRAY数组的形式读入

::
    Given:
    array = [0.6,0.1,0.3,0.1]
    And:
    I=2
    Then:
    output = 0.3

参数：
    - **array** (Variable|list)-输入张量，存储要读的数据
    - **i** (Variable|list)-输入数组中数据的索引

返回：张量类型的变量，已有数据写入

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    tmp = fluid.layers.zeros(shape=[10],dtype='int32')
    i = fluid.layers.fill_constant(shape=[1],dtype='int64',value=10)
    arr = layers.array_read(tmp,i=i)

.. _cn_api_fluid_layers_array_length:

array_length
>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.array_length(array)

**得到输入LoDTensorArray的长度**

此功能用于查找输入数组LOD_TENSOR_ARRAY的长度。  

相关API:
    - ref:'api_fluid_layers_array_read',
    - ref:'api_fluid_layers_array_write',
    - ref:'api_fluid_layers_While'. 

参数：**array** (LOD_TENSOR_ARRAY)-输入数组，用来计算数组长度

返回：输入数组LoDTensorArray的长度

返回类型：变量（Variable）

**代码示例**:

.. code-block:: python

    tmp = fluid.layers.zeros(shape=[10], dtype='int32')
    i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
    arr = fluid.layers.array_write(tmp, i=i)
    arr_len = fluid.layers.array_length(arr)

.. _cn_api_fluid_layers_IfElse:

IfElse
>>>>>>>

.. py:class:: paddle.fluid.layers.IfElse(cond, name=None)

if-else控制流。  

参数：
    - **cond** (Variable)-用于比较的条件
    - **Name** (str,默认为空（None）)-该层名称

**代码示例**：

.. code-block:: python

    limit = fluid.layers.fill_constant_batch_size_like(
        input=label, dtype='int64', shape=[1], value=5.0)
    cond = fluid.layers.less_than(x=label, y=limit)
    ie = fluid.layers.IfElse(cond)
    with ie.true_block():
        true_image = ie.input(image)
        hidden = fluid.layers.fc(input=true_image, size=100, act='tanh')
        prob = fluid.layers.fc(input=hidden, size=10, act='softmax')
        ie.output(prob)

    with ie.false_block():
        false_image = ie.input(image)
        hidden = fluid.layers.fc(
            input=false_image, size=200, act='tanh')
        prob = fluid.layers.fc(input=hidden, size=10, act='softmax')
        ie.output(prob)
    prob = ie()

.. _cn_api_fluid_layers_Print:

Print
>>>>>>>

.. py:class:: paddle.fluid.layers.Print(input, first_n=-1, message=None, summarize=-1, print_tensor_name=True, print_tensor_type=True, print_tensor_shape=True, print_tensor_lod=True, print_phase='both')

**Print操作命令**

该操作命令创建一个打印操作，打印正在访问的张量。

封装传入的张量，以便无论何时访问张量，都会打印信息message和张量的当前值。

参数：

    - **input** (Variable)-将要打印的张量
    - **summarize** (int)-打印张量中的元素数目，如果值为-1则打印所有元素
    - **message** (str)-字符串类型消息，作为前缀打印
    - **first_n** (int)-只记录first_n次数
    - **print_tensor_name** (bool)-打印张量名称
    - **print_tensor_type** (bool)-打印张量类型
    - **print_tensor_shape** (bool)-打印张量维度
    - **print_tensor_lod** (bool)-打印张量lod
    - **print_phase** (str)-打印的阶段，包括"forward","backward"和"both".若设置为"backward"或者"both",则打印输入张量的梯度。

返回：输出张量，和输入张量同样的数据

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    value = some_layer(...)
    Print(value, summarize=10,
    message="The content of some_layer: ")

.. _cn_api_fluid_layers_is_empty:

is_empty
>>>>>>>>>

.. py:class:: paddle.fluid.layers.is_empty(x, cond=None, **ignored)

测试变量是否为空

参数：
    - **x** (Variable)-测试的变量
    - **cond** (Variable|None)-输出参数。返回给定x的测试结果，默认为空（None）

返回：布尔类型的标量。如果变量x为空则值为真

返回类型：变量（Variable）

抛出异常：''TypeError''-如果input不是变量或cond类型不是变量

**代码示例**：

.. code-block:: python

    res = fluid.layers.is_empty(x=input)
    # or:
    fluid.layers.is_empty(x=input, cond=res)

.. _cn_api_fluid_layers_data:

data
>>>>>

.. py:class:: paddle.fluid.layers.data(name, shape, append_batch_size=True, dtype='float32', lod_level=0, type=VarType.LOD_TENSOR, stop_gradient=True)

数据层(Data Layer)

该功能接受输入数据，根据是否返回minibatch用辅助函数创建全局变量。可通过图中所有操作命令访问全局变量。

该函数输入的所有变量作为局部变量传到LayerHelper构造器

参数：
    - **name** (str)-函数名或函数别名
    - **shape** (list)-声明维度的元组
    - **append_batch_size** (bool)-

        1.如果为真，则在维度shape的开头插入-1
        ''比如如果shape=[1],结果shape为[-1,1].'' 

        2.如果维度shape包含-1，比如shape=[-1,1],
        ''append_batch_size则为False（表示无效）''

    - **dtype** (int|float)-数据类型：float32,float_16,int等
    - **type** (VarType)-输出类型。默认为LOD_TENSOR.
    - **lod_level** (int)-LoD层。0表示输入数据不是一个序列
    - **stop_gradient** (bool)-布尔数，提示是否应该停止计算梯度

返回：全局变量，可进行数据访问

返回类型：变量(Variable)

**代码示例**：

.. code-block:: python

    data = fluid.layers.data(name='x', shape=[784], dtype='float32')


.. _cn_api_fluid_layers_open_files:

open_files
>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.open_files(filenames, shapes, lod_levels, dtypes, thread_num=None, buffer_size=None, pass_num=1, is_test=None)

打开文件(Open files)

该层读一列文件并返回Reader变量。通过Reader变量，可以从给定的文件中获取数据。所有的文件必须有后缀名，显示文件格式，比如”*.recordio”。

参数：
    - **filenames** (list)-文件名列表
    - **shape** (list)-元组类型值列表，声明数据维度
    - **lod_levels** (list)-整形值列表，声明数据的lod层级
    - **dtypes** (list)-字符串类型值列表，声明数据类型
    - **thread_num** (None)-用于读文件的线程数。默认：min(len(filenames),cpu_number)
    - **buffer_size** (None)-reader的缓冲区大小。默认：3*thread_num
    - **pass_num** (int)-用于运行的传递数量
    - **is_test** (bool|None)-open_files是否用于测试。如果用于测试，生成的数据顺序和文件顺序一致。反之，无法保证每一epoch之间的数据顺序是一致的

返回：一个Reader变量，通过该变量获取文件数据

返回类型：变量(Variable)

**代码示例**：

.. code-block:: python

    reader = fluid.layers.io.open_files(filenames=['./data1.recordio',
                                            './data2.recordio'],
                                    shapes=[(3,224,224), (1)],
                                    lod_levels=[0, 0],
                                    dtypes=['float32', 'int64'])

    # 通过reader, 可使用''read_file''层获取数据:
    image, label = fluid.layers.io.read_file(reader)

.. _cn_api_fluid_layers_read_file:

read_file
>>>>>>>>>>

.. py:class:: paddle.fluid.layers.read_file(reader)

执行给定的reader变量并从中获取数据

reader也是变量。可以为由fluid.layers.open_files()生成的原始reader或者由fluid.layers.double_buffer()生成的装饰变量，等等。

参数：
    **reader** (Variable)-将要执行的reader

返回：从给定的reader中读取数据

**代码示例**：

.. code-block:: python

    data_file = fluid.layers.open_files(
        filenames=['mnist.recordio'],
        shapes=[(-1, 748), (-1, 1)],
        lod_levels=[0, 0],
        dtypes=["float32", "int64"])
    data_file = fluid.layers.double_buffer(
        fluid.layers.batch(data_file, batch_size=64))
    input, label = fluid.layers.read_file(data_file)

.. _cn_api_fluid_layers_batch:

batch
>>>>>>>

.. py:class:: paddle.fluid.layers.batch(reader, batch_size)

该层是一个reader装饰器。接受一个reader变量并添加''batching''装饰。读取装饰的reader，输出数据自动组织成batch的形式。

参数：
    - **reader** (Variable)-装饰有“batching”的reader变量
    - **batch_size** (int)-批尺寸

返回：装饰有''batching''的reader变量

返回类型：变量(Variable)

**代码示例**：

.. code-block:: python

    raw_reader = fluid.layers.io.open_files(filenames=['./data1.recordio',
                                               './data2.recordio'],
                                        shapes=[(3,224,224), (1)],
                                        lod_levels=[0, 0],
                                        dtypes=['float32', 'int64'],
                                        thread_num=2,
                                        buffer_size=2)
    batch_reader = fluid.layers.batch(reader=raw_reader, batch_size=5)

    # 如果用raw_reader读取数据：
    #     data = fluid.layers.read_file(raw_reader)
    # 只能得到数据实例。
    #
    # 但如果用batch_reader读取数据：
    #     data = fluid.layers.read_file(batch_reader)
    # 每5个相邻的实例自动连接成一个batch。因此get('data')得到的是一个batch数据而不是一个实例。

.. _cn_api_fluid_layers_random_data_generator:

random_data_generator
>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.random_data_generator(low, high, shapes, lod_levels, for_parallel=True)

创建一个均匀分布随机数据生成器.

该层返回一个Reader变量。该Reader变量不是用于打开文件读取数据，而是自生成float类型的均匀分布随机数。该变量可作为一个虚拟reader，无需打开真实文件便可测试网络。

参数：
    - **low** (float)--数据均匀分布的下界
    - **high** (float)-数据均匀分布的上界
    - **shapes** (list)-元组数列表，声明数据维度
    - **lod_levels** (list)-整形数列表，声明数据
    - **for_parallel** (Bool)-若要运行一系列操作命令则将其设置为True

返回：Reader变量，可从中获取随机数据

返回类型：变量(Variable)

**代码示例**：

.. code-block:: python

    reader = fluid.layers.random_data_generator(
                                 low=0.0,
                                 high=1.0,
                                 shapes=[[3,224,224], [1]],
                                 lod_levels=[0, 0])
    # 通过reader, 可以用'read_file'层获取数据:
    image, label = fluid.layers.read_file(reader)

.. _cn_api_fluid_layers_Preprocessor:

Preprocessor
>>>>>>>>>>>>>

.. py:class:: class paddle.fluid.layers.Preprocessor(reader, name=None)

reader变量中数据预处理块。

参数：
    - **reader** (Variable)-reader变量
    - **name** (str,默认None)-reader的名称

**代码示例**:

.. code-block:: python

    preprocessor = fluid.layers.io.Preprocessor(reader=reader)
    with preprocessor.block():
        img, lbl = preprocessor.inputs()
        img_out = img / 2
        lbl_out = lbl + 1
        preprocessor.outputs(img_out, lbl_out)
    data_file = fluid.layers.io.double_buffer(preprocessor())

.. _cn_api_fluid_layers_load:

load
>>>>>

.. py:class:: paddle.fluid.layers.load(out, file_path, load_as_fp16=None)

Load操作命令将从磁盘文件中加载LoDTensor/SelectedRows变量。

.. code-block:: python

    import paddle.fluid as fluid
    tmp_tensor = fluid.layers.create_tensor(dtype='float32')
    fluid.layers.load(tmp_tensor, "./tmp_tensor.bin")

参数：
    - **out** (Variable)-需要加载的LoDTensor或SelectedRows
    - **file_path** (STRING)-预从”file_path”中加载的变量Variable
    - **load_as_fp16** (BOOLEAN)-如果为真，张量首先进行加载然后类型转换成float16。如果为假，张量无数据类型转换直接进行加载。默认为false。

返回：None

.. _cn_api_fluid_layers_embedding:

embedding
>>>>>>>>>>

.. py:class:: paddle.fluid.layers.embedding(input, size, is_sparse=False, is_distributed=False, padding_idx=None, param_attr=None, dtype='float32')

嵌入层(Embedding Layer)

该层用来在供查找的表中查找IDS的嵌入矩阵，IDS由input提供。查找的结果是input里每个ID对应的嵌入矩阵。
所有的输入变量都作为局部变量传入LayerHelper构造器

参数：
    - **input** (Variable)-包含IDs的张量
    - **size** (tuple|list)-查找表参数的维度。应当有两个参数，一个代表嵌入矩阵字典的大小，一个代表每个嵌入向量的大小。
    - **is_sparse** (bool)-代表是否用稀疏更新的标志
    - **is_distributed** (bool)-是否从远程参数服务端运行查找表、
    - **padding_idx** (int|long|None)-如果为''**None**''，对查找结果无影响。如果padding_idx不为空，表示一旦查找表中找到input中对应的''**padding_idz**''，则用0填充输出结果。如果 :math:'padding_{i}dx<0' ,在查找表中使用的''**padding_idx**''值为 :math:'*size[0]+dim*' 。
    - **param_attr** (ParamAttr)-该层参数
    - **dtype** (np.dtype|core.VarDesc.VarType|str)-数据类型：float32,float_16,int等。

返回：张量，存储已有输入的嵌入矩阵。

返回类型：变量(Variable)

**代码示例**:

.. code-block:: python

    dict_size = len(dataset.ids)
    data = fluid.layers.data(name='ids', shape=[32, 32], dtype='float32')
    fc = fluid.layers.embedding(input=data, size=[dict_size, 16])

.. _cn_api_fluid_cos_sim:

cos_sim 
>>>>>>>>

.. py:class:: paddle.fluid.layers.cos_sim(X, Y)

余弦相似度运算符（Cosine Similarity Operator）

.. math::

Out = \frac{X^{T}*Y}{\sqrt{X^{T}*X}*\sqrt{Y^{T}*Y}}

输入X和Y必须有相同维，除非输入Y的第一维只能为1（不同于输入X），传播到匹配输入X的维，然后计算X和Y的余弦相似度。
输入X和Y都可以携带LoD(Level of Detail)信息，或者都不。但输出仅和X共享LoD信息

参数：
    - **X** (Variable) - cos_sim操作函数的一个输入
    - **Y** (Variable) - cos_sim操作函数的第二个输入

返回：cosine(X,Y)的输出

返回类型：变量（Variable)

.. _cn_api_fluid_square_error_cost:

square_error_cost 
>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.square_error_cost(input,label)

方差估计层（Square error cost layer）

该层接受输入预测值和目标值，并返回方差估计

对于预测值X和目标值Y，公式为：

.. math::

    Out = (X-Y)^{2}

在以上等式中：
::
    - **X** : 输入预测值，张量（Tensor)
    - **Y** : 输入目标值，张量（Tensor）
    - **Out** : 输出值，维度和X的相同

参数：
    - **input** (Variable) - 输入张量（Tensor），带有预测值
    - **label** (Variable) - 标签张量（Tensor），带有目标值

返回：张量变量，存储输入张量和标签张量的方差

返回类型：变量（Variable）

**代码示例**：

 .. code_block:: python:

    y = layers.data(name='y', shape=[1], dtype='float32')
    y_predict = layers.data(name='y_predict', shape=[1], dtype='float32')
    cost = layers.square_error_cost(input=y_predict, label=y)

.. _cn_api_fluid_layers_sequence_conv:

sequence_conv 
>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.sequence_conv(input, num_filters, filter_size=3, filter_stride=1, padding=None, bias_attr=None, param_attr=None, act=None, name=None)

该函数的输入参数中给出了筛选器和步长，通过利用输入以及筛选器和步长的常规配置来为sequence_conv创建操作符。

参数：
    - **input** (Variable) - (LoD张量）输入X是LoD张量，支持可变的时间量的长度输入序列。该LoDTensor的标记张量是一个维度为（T,N)
    的矩阵，其中T是mini-batch的总时间步数，N是input_hidden_size
    - **num_filters** (int) - 筛选器的数量
    - **filter_size** (int) - 筛选器大小（H和W)
    - **filter_stride** (int) - 筛选器的步长
    - **padding** (bool) - 若为真，添加填充
    - **bias_attr** (ParamAttr|bool|None) - sequence_conv偏离率参数属性。若设为False,
    输出单元则不加入偏离率。若设为None或ParamAttr的一个属性，sequence_conv将创建一个ParamAttr作为bias_attr。
    如果未设置bias_attr的初始化函数，则将bias初始化为0.默认:None
    - **param_attr** (ParamAttr|None) - 可学习参数/sequence_conv的权重参数属性。若设置为None或ParamAttr的一个属性，sequence_conv将创建ParamAttr作为param_attr。
    若未设置param_attr的初始化函数，则用Xavier初始化参数。默认:None

返回：sequence_conv的输出

返回类型：变量（Variable）

.. _cn_api_fluid_layers_sequence_pool:

sequence_pool 
>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.sequence_pool(input, pool_type)

该函数为序列池添加操作符。将每个实例的所有时间步数特征加入池子，并用参数中提到的pool_type将特征运用到输入到首部。

支持四种pool_type:

- **average**: :math:'Out[i] = \frac{\sum_{i}X_{i}}{N}'
- **sum**: :math:'Out[i] = \sum _{j}X_{ij}'
- **sqrt**: :math:'Out[i] = \frac{ \sum _{j}X_{ij}}{\sqrt{len(\sqrt{X_{i}})}}'
- **max**: :math:'Out[i] = max(X_{i})'

::
    x是一级LoDTensor:
        x.lod = [[2, 3, 2]]
        x.data = [1, 3, 2, 4, 6, 5, 1]
        x.dims = [7, 1]
    输出为张量（Tensor）：
        out.dim = [3, 1]
        with condition len(x.lod[-1]) == out.dims[0]
    对于不同的pool_type：
        average: out.data = [2, 4, 3], where 2=(1+3)/2, 4=(2+4+6)/3, 3=(5+1)/2
        sum    : out.data = [4, 12, 6], where 4=1+3, 12=2+4+6, 6=5+1
        sqrt   : out.data = [2.82, 6.93, 4.24], where 2.82=(1+3)/sqrt(2),
             6.93=(2+4+6)/sqrt(3), 4.24=(5+1)/sqrt(2)
        max    : out.data = [3, 6, 5], where 3=max(1,3), 6=max(2,4,6), 5=max(5,1)
        last   : out.data = [3, 6, 1], where 3=last(1,3), 6=last(2,4,6), 1=last(5,1)
        first  : out.data = [1, 2, 5], where 1=first(1,3), 2=first(2,4,6), 5=first(5,1)

参数：
    - **input** (variable) - 输入变量，为LoDTensor
    - **pool_type** (string) - sequence_pool的池变量。支持average,sum,sqrt和max

返回：序列池变量，为张量（Tensor)

**代码示例**：

.. code_block:: python:

    x = fluid.layers.data(name='x', shape=[7, 1],
                 dtype='float32', lod_level=1)
    avg_x = fluid.layers.sequence_pool(input=x, pool_type='average')
    sum_x = fluid.layers.sequence_pool(input=x, pool_type='sum')
    sqrt_x = fluid.layers.sequence_pool(input=x, pool_type='sqrt')
    max_x = fluid.layers.sequence_pool(input=x, pool_type='max')
    last_x = fluid.layers.sequence_pool(input=x, pool_type='last')
    first_x = fluid.layers.sequence_pool(input=x, pool_type='first')

.. _cn_api_fluid_layers_softmax:

softmax
>>>>>>>>

.. py:class:: paddle.fluid.layers.softmax(input, use_cudnn=True, name=None)

softmax操作符的输入是任意阶的张量，输出张量和输入张量的维度相同。

首先逻辑上将输入张量压平至二维矩阵。矩阵的第二维（行数）和输入张量的最后一维相同。第一维（列数）
是输入张量除最后一维之外的所有维的产物。对矩阵的每一行,softmax操作符将任意实值k维向量压平至实值的k维向量，范围为[0,1]，总和为1（k是矩阵的宽度，也是输入张量最后一维的大小）

softmax操作符计算k维向量输入中所有其他维的指数和指数值的累加和。维的指数比例和所有其他维的指数值之和作为softmax操作符的输出。

对矩阵中的每行i和每列j有：
    Out[i,j] = \frac{exp(X[i,j])}{\sum{j}_exp(X[i,j])}

参数：
    - **input** (Variable) - 输入变量
    - **use_cudnn** (bool) - 不论是否用cudnn核，只有在cudnn库安装时有效
    - **name** (str|None) - 该层名称（可选）。若为空，则自动为该层命名。默认：None

返回： softmax输出

返回类型：变量（Variable）

**代码示例**：

.. code_block:: python:
    fc = fluid.layers.fc(input=x, size=10)
    softmax = fluid.layers.softmax(input=fc)

.. _cn_api_fluid_layers_pool3d:

pool3d
>>>>>>

.. py:class:: paddle.fluid.layers.pool3d(input, pool_size=-1, pool_type='max', pool_stride=1, pool_padding=0, global_pooling=False, use_cudnn=True, ceil_mode=False, name=None)

该函数用输入参数中提到的池配置项为三维池添加操作符。

参数：
    - **input** (Vairable) - ${input_comment}
    - **pool_size** (int) - ${ksize_comment}
    - **pool_type** (str) - ${pooling_type_comment}
    - **pool_stride** (int) - 池层的步长
    - **pool_padding** (int) - ${global_pooling_comment}
    - **global_pooling** (bool) - ${global_pooling_comment}
    - **use_cudnn** (bool) - ${use_cudnn_comment}
    - **ceil_mode** (bool) - ${ceil_mode_comment}
    - **name** (str) - 该层名称（可选）。若为空，则自动为该层命名。

返回：pool3d层的输出

返回类型：变量（Variable）

.. _cn_api_fluid_layers_beam_search_decode:

beam_search_decode
>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.beam_search_decode(ids, scores, beam_size, end_id, name=None)

Beam Search Decode层。沿着LoDTensorArray ''ids''往回走，为每个源句构造全假设。‘’ids''的lods可以用来存储beam search树的路径。下面是完整的beam search用例，请看如下demo：

:: fluid/tests/book/test_machine_translation.py

参数：
    - **ids** (Variable) - LodTensorArray变量，包含所有步中选中的ids
    - **scores** (Variable) - LodTensorArray变量，包含所有步选中的分数
    - **beam_size** (int) - beam search中使用的beam宽度
    - **end_id** (int) - 末尾token的id
    - **name** (str|None) - 该层名称（可选）。如果设为空，则自动为该层命名

返回：LodTensor对包含生成的id序列和相应的分数。两个LodTensor的维度和详细层相同。lod层为2，两层分别表示每个源句有多少假设，每个假设有多少ids

返回类型：变量（Variable）

**代码示例**：

.. _cn_api_fluid_layers_sequence_expand:

sequence_expand 
>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.sequence_expand(x, y, ref_level=-1, name=None)

序列扩张层（Sequence Expand Layer）。根据y的具体层lod扩展输入变量x。x的lod层至多为1，x的阶至少为2。x的阶大于2，将作为二维张量。以下示例解释sequence_expand是如何工作的：
::
* 例1
    x is a LoDTensor:
        x.lod  = [[2,        2]]
        x.data = [[a], [b], [c], [d]]
        x.dims = [4, 1]

    y is a LoDTensor:
        y.lod = [[2,    2],
                 [3, 3, 1, 1]]

    ref_level: 0

    then output is a 1-level LoDTensor:
        out.lod =  [[2,        2,        2,        2]]
        out.data = [[a], [b], [a], [b], [c], [d], [c], [d]]
        out.dims = [8, 1]

* 例2
    x is a Tensor:
        x.data = [[a], [b], [c]]
        x.dims = [3, 1]

    y is a LoDTensor:
        y.lod = [[2, 0, 3]]

    ref_level: -1

    then output is a Tensor:
        out.data = [[a], [a], [c], [c], [c]]
        out.dims = [5, 1]

参数：
    - **x** (Variable) - 输入变量，张量或LoDTensor
    - **y** (Variable) - 输入变量，为LoDTensor
    - **ref_level** (int) - x表示的y的Lod层。若设为-1，表示lod的最后一层
    - **name** (str|None) - 该层名称（可选）。如果设为空，则自动为该层命名

返回：扩展变量，LoDTensor

返回类型：变量（Variable）

**代码示例**：

.. code_block:: python

    x = fluid.layers.data(name='x', shape=[10], dtype='float32')
    y = fluid.layers.data(name='y', shape=[10, 20],
                 dtype='float32', lod_level=1)
    out = layers.sequence_expand(x=x, y=y, ref_level=0)

.. _cn_api_fluid_layers_sequence_expand_as:

sequence_expand_as 
>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.sequence_expand_as(x, y, name=None)

序列扩张为层（Sequence Expand As Layer）。该层格局y的第0层lod扩展输入变量x。
当前实现要求输入（Y）的lod层数必须为1，输入（X）的第一维应当和输入（Y）的第0层lod的大小相同，
并且不考虑输入（X）的lod。

以下示例解释sequence_expand如何工作：
* Case 1:
    给定一维LoDTensor input(X)
        X.data = [[a], [b], [c], [d]]
        X.dims = [4, 1]
    和 input(Y)
        Y.lod = [[0, 3, 6, 7, 8]]
    ref_level: 0
    得到1级 LoDTensor
        Out.lod =  [[0,            3,              6,  7,  8]]
        Out.data = [[a], [a], [a], [b], [b], [b], [c], [d]]
        Out.dims = [8, 1]

参数：
    - **x** (Variable) - 输入变量，类型为Tensor或LoDTensor
    - **y** (Variable) - 输入变量，为LoDTensor
    - **name** (str|None) - 该层名称（可选）。如果设为空，则自动为该层命名

返回：扩展变量，LoDTensor

返回类型：变量（Variable）

**代码示例**：

.. code_block:: python

    x = fluid.layers.data(name='x', shape=[10], dtype='float32')
    y = fluid.layers.data(name='y', shape=[10, 20],
                 dtype='float32', lod_level=1)
    out = layers.sequence_expand_as(x=x, y=y)

.. _cn_api_fluid_layers_sequence_pad:

sequence_pad
>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.sequence_pad(x,pad_value,maxlen=None)

序列填充操作符（Sequence Pad Operator）

该操作符填充同一个batch（批）里的序列，使这些序列的长度保持一致。长度具体‘paddle_length’属性指示。填充的新元素的值具体由输入‘PadValue'指示，并会添加到每一个序列的末尾，使得他们最终的长度保持一致。

以下的例子更清晰地解释此操作符的工作原理：

::
    例1:
    给定一级LoDTensor
    input(X):
    X.lod = [[0,2,5]]
    X.data = [a,b,c,d,e]
    input(PadValue):
    PadValue.data = [0]
    属性'padded_length'=4
    于是得到LoDTensor:Out.data = [[a,b,0,0],[c,d,e,0]]
    Length.data = [[2],[3]]

    例2:
    给定一级LoDTensor
    input(X):
    X.lod = [[0,2,5]]
    X.data = [[a1,a2],[b1,b2],[c1,c2],[d1,d2],[e1,e2]]
    input(PadValue):
    PadValue.data = [0]
    属性'padded_length' = -1,表示用最长输入序列的长度(此例为3)
    于是得到LoDTensor:
    Out.data = [[[a1,a2],[b1,b2],[0,0]],[[c1,c2],[d1,d2],[e1,e2]]]
    Length.data = [[2],[3]]

    例3:
    给定一级LoDTensor
    input(X):
    X.lod = [[0,2,5]]
    X.data = [[a1,a2],[b1,b2],[c1,c2],[d1,d2],[e1,e2]]
    input(PadValue):
    PadValue.data = [p1,p2]
    属性'padded_length' = -1,表示用最长输入序列的长度（此例为3）
    于是得到LoDTensor:
    Out.data = [[[a1,a2],[b1,b2],[0,0]],[[c1,c2],[d1,d2],[e1,e2]]]
    Length.data = [[2],[3]]

参数：
    - **x**(Vairable) - 输入变量，应包含lod信息
    - **pad_value**(Variable) - 变量，存有放入填充步的值。可以是scalar或tensor,维度和序列的时间步长相等。如果是scalar,则自动广播到时间步长的维度
    - **maxlen**(int,默认None) - 填充序列的长度。可以为空或者任意正整数。当为空时，以序列中最长序列的长度为准，其他所有序列填充至该长度。当是某个特定的正整数，最大长度必须大于最长初始序列的长度

返回：填充序列批（batch）和填充前的初始长度。所有序列的长度相等

返回类型：变量（Variable）

**代码示例**：

.. code_block:: python

    import numpy

    x = fluid.layers.data(name='y', shape=[10, 5],
                 dtype='float32', lod_level=1)
    pad_value = fluid.layers.assign(input=numpy.array([0]))
    out = fluid.layers.sequence_pad(x=x, pad_value=pad_value)





.. _cn_api_fluid_layers_sequence_first_step:

sequence_first_step
>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.sequence_first_step(input)

该功能获取序列的第一步

x是一级LoDTensor:

  x.lod = [[2, 3, 2]]

  x.data = [1, 3, 2, 4, 6, 5, 1]

  x.dims = [7, 1]

输出为张量:

.. code-block:: python

  out.dim = [3, 1]
  with condition len(x.lod[-1]) == out.dims[0]
  out.data = [1, 2, 5], where 1=first(1,3), 2=first(2,4,6), 5=first(5,1)

参数：**input** (variable)-输入变量，为LoDTensor

返回：序列第一步，为张量

**代码示例**：

.. code-block:: python

    x = fluid.layers.data(name='x', shape=[7, 1],
                 dtype='float32', lod_level=1)
    x_first_step = fluid.layers.sequence_first_step(input=x)

.. _cn_api_fluid_layers_sequence_last_step:

sequence_last_step
>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers. sequence_last_step(input)

该API可以获取序列的最后一步

x是level-1的LoDTensor:

    x.lod = [[2, 3, 2]]

    x.data = [1, 3, 2, 4, 6, 5, 1]

    x.dims = [7, 1]

输出为Tensor:

.. code-block:: python

    out.dim = [3, 1]
    with condition len(x.lod[-1]) == out.dims[0]
    out.data = [3, 6, 1], where 3=last(1,3), 6=last(2,4,6), 1=last(5,1)

参数：**input** (variable)-输入变量，为LoDTensor

返回：序列的最后一步，为张量

**代码示例**：

.. code-block:: python

    x = fluid.layers.data(name='x', shape=[7, 1],
                 dtype='float32', lod_level=1)
    x_last_step = fluid.layers.sequence_last_step(input=x)

.. _cn_api_fluid_layers_dropout:

dropout
>>>>>>>

.. py:class:: Paddle.fluid.layers. dropout(x,dropout_prob,is_test=False,seed=None,name=None,dropout_implementation=‘downgrade_in_infer’)

计算dropout。

丢弃x的每个元素或者保持x的每个元素独立。Dropout是一种正则化技术，通过在训练过程中阻止神经元节点间的联合适应性来减少过拟合。根据给定的丢弃概率dropout操作符随机将一些神经元输出设置为0，其他的仍保持不变。

参数：
    - **x**（Variable）-输入张量
    - **dropout_prob** (float)-设置为0的单元的概率
    - **is_test** (bool)-显示是否进行测试用语的标记
    - **seed** (int)-Python整型，用于创建随机种子。如果该参数设为None，则使用随机种子。注：如果给定一个整型种子，始终丢弃相同的输出单元。训练过程中勿用固定不变的种子。
    - **name** (str|None)-该层名称（可选）。如果设置为None,则自动为该层命名
    - **dropout_implementation** (string)-
        [‘downgrade_in_infer’(defauld)|’upscale_in_train’] 1.downgrade_in_infer(default), 降级在线推断的结果

            train: out = input * mask inference: out = input * dropout_prob 
            (make是一个张量，维度和输入维度相同，值为0或1，值为0的比例即为dropout_prob)
        
        2.upscale_in_train, 扩张训练时的结果(make是一个张量，维度和输入维度相同，值为0或1，值为0的比例即为dropout_prob)

            dropout操作符可以从程序中移除，程序变得高效。

返回：带有x维的张量

返回类型：变量

**代码示例**：

.. code-block:: python

    x = fluid.layers.data(name="data", shape=[32, 32], dtype="float32")
    droped = fluid.layers.dropout(x, dropout_prob=0.5)

.. _cn_api_fluid_layers_split:

split
>>>>>>

.. py:class:: paddle.fluid.layers. split(input,num_or_sections,dim=-1,name=None)

将输入张量分解成多个子张量

参数：
    - **input** (Variable)-输入变量，类型为Tensor或者LoDTensor
    - **num_or_sections** (int|list)-如果num_or_sections是整数，则表示张量平均划分为的相同大小子张量的数量。如果num_or_sections是一列整数，列表的长度代表子张量的数量，整数依次代表子张量的dim维度的大小
    - **dim** (int)-将要划分的维。如果dim<0,划分的维为rank(input)+dim
    - **name** (str|None)-该层名称（可选）。如果设置为空，则自动为该层命名

返回：一列分割张量

返回类型：列表(Variable)

**代码示例**：

.. code-block:: python

    # x是维为[3,9,5]的张量：
    x0, x1, x2 = fluid.layers.split(x, num_or_sections=3, dim=1)
    x0.shape  # [3, 3, 5]
    x1.shape  # [3, 3, 5]
    x2.shape  # [3, 3, 5]
    x0, x1, x2 = fluid.layers.split(
        x, num_or_sections=[2, 3, 4], dim=1)
    x0.shape  # [3, 2, 5]
    x1.shape  # [3, 3, 5]
    x2.shape  # [3, 4, 5]

.. _cn_api_fluid_layers_edit_distance:

edit_distance
>>>>>>>>>>>>>>

.. py:class:: Paddle.fluid.layers. edit_distance(input,label,normalized=True,ignored_tokens=None)

编辑距离运算符计算一批给定字符串及其参照字符串间的编辑距离。编辑距离也称Levenshtein距离，通过计算从一个字符串变成另一个字符串所需的最少操作步骤来衡量两个字符串的相异度。这里的操作包括插入、删除和替换。

比如给定假设字符串A=“kitten”和参照字符串B=“sitting”，从A变换成B编辑距离为3，至少需要两次替换和一次插入：

“kitten”->“sitten”->“sittn”->“sitting”

输入为LoDTensor,包含假设字符串（带有表示批尺寸的总数）和分离信息（具体为LoD信息）。并且批尺寸大小的参照字符串和输入LoDTensor的顺序保持一致。

输出包含批尺寸大小的结果，代表一对字符串中每个字符串的编辑距离。如果Attr(normalized)为真，编辑距离则处以参照字符串的长度。

参数：
    - **input** (Variable)-假设字符串的索引
    - **label** (Variable)-参照字符串的索引
    - **normalized** (bool,默认为True)-表示是否用参照字符串的长度进行归一化
    - **ignored_tokens** (list<int>,默认为None)-计算编辑距离前需要移除的token
    - **name** (str)-该层名称，可选

返回：[batch_size,1]中序列到序列到编辑距离

返回类型：变量

**代码示例**：

.. code-block:: python

    x = fluid.layers.data(name='x', shape=[8], dtype='float32')
    y = fluid.layers.data(name='y', shape=[7], dtype='float32')
    cost = fluid.layers.edit_distance(input=x,label=y)

.. _cn_api_fluid_layers_l2_normalize:

l2_normalize
>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers. l2_normalize(x, axis, epsilon=1e-12, name=None)

L2正则层（L2 normalize Layer）

该层用欧几里得距离之和对维轴的x归一化。对于1-D张量（系数矩阵的维度固定为0），该层计算公式如下：
公式

对于x多维的情况，该层分别对维轴的每个1-D切片单独归一化

参数：
    - **x** (Variable|list)-传给欧几里得距离之和正则层（l2_normalize layer）
    - **axis** (int)-运用归一化的轴。如果轴小于0，归一化的维是rank(X)+axis。-1是最后维
    - **epsilon** (float)-epsilon用于避免分母为0，默认值为1e-10
    - **name** (str|None)-该层名称（可选）。如果设为空，则自动为该层命名
    
    返回：输出张量，同x的维度一致
    
    返回类型：变量
    
**代码示例**：

.. code-block:: python

    data = fluid.layers.data(name="data",
                         shape=(3, 17, 13),
                         dtype="float32")
    normed = fluid.layers.l2_normalize(x=data, axis=1)

.. _cn_api_fluid_layers_matmul:

matmul
>>>>>>>

.. py:class:: paddle.fluid.layers. matmul(x, y, transpose_x=False, transpose_y=False, alpha=1.0, name=None)

对两个张量进行矩阵相乘

当前输入张量的阶可以任意，但当任何输入的阶大于3，则两个输入的阶必须相等。

实际的操作取决于x,y的维度和transpose_x,transpose_y的标记值。具体如下：

如果张量是维[D]中的一阶，那么x在非转置形式中作为[1,D]，在转置形式中作为[D,1],而y则相反，在非转置形式中作为[D,1]，在转置形式中作为[1,D]。
转置后，两个张量是2-D或者n-D，以如下方式执行矩阵相乘。

如果两个都是2-D，则同普通矩阵一样进行矩阵相乘

**如果有一个为n-D，则作为一堆矩阵存储在最后两维中，一批矩阵相乘支持两个张量broadcast**

**需注意如果原始张量x或y是一阶并未转置，矩阵相乘后需移除前置或后置维1.**


参数：
    - **x** (Variable)-输入变量，类型为Tensor或LoDTensor
    - **y** (Variable)-输入变量，类型为Tensor或LoDTensor
    - **transpose_x** (bool)-相乘前是否转置x
    - **transeptse_y** (bool)-相乘前是否转置y
    - **alpha** (float)-输出比例。默认为1.0
    - **name** (str|None)-该层名称（可选）。如果设置为空，则自动为该层命名

返回：张量积变量

返回类型：变量

**代码示例**：

.. code-block:: python

    # 以下是解释输入和输出维度的示例
    # x: [B, ..., M, K], y: [B, ..., K, N]
    fluid.layers.matmul(x, y)  # out: [B, ..., M, N]

    # x: [B, M, K], y: [B, K, N]
    fluid.layers.matmul(x, y)  # out: [B, M, N]

    # x: [B, M, K], y: [K, N]
    fluid.layers.matmul(x, y)  # out: [B, M, N]

    # x: [M, K], y: [K, N]
    fluid.layers.matmul(x, y)  # out: [M, N]

    # x: [B, M, K], y: [K]
    fluid.layers.matmul(x, y)  # out: [B, M]

    # x: [K], y: [K]
    fluid.layers.matmul(x, y)  # out: [1]

    # x: [M], y: [N]
    fluid.layers.matmul(x, y, True, True)  # out: [M, N]

.. _cn_api_fluid_layers_topk:

topk
>>>>>
.. py:class:: paddle.fluid.layers. topk(input, k, name=None)

该操作符用于寻找最后维前k最大项的值和索引。

如果输入是（1-D Tensor），则找到向量的前k最大项，并以向量的形式输出前k最大项的值和索引。values[j]是输入中第j最大项，其索引为indices[j]。
如果输入是更高阶的张量，则该operator会基于最后一维计算前k项

例如：

.. code-block:: python

    如果:
        input = [[5, 4, 2, 3],
                [9, 7, 10, 25],
                [6, 2, 10, 1]]
        k = 2

    则:
        第一个输入:
        values = [[5, 4],
                [10, 25],
                [6, 10]]

        第二个输入:
        indices = [[0, 1],
                [2, 3],
                [0, 2]]

参数：
    - **input**(Variable)-输入变量可以是一个向量或者更高阶的张量
    - **k** (int)-在输入最后一纬中寻找的前项数目
    - **name** (str|None)-该层名称（可选）。如果设为空，则自动为该层命名。默认为空

返回：含有两个元素的元组。元素都是变量。第一个元素是最后维切片的前k项。第二个元素是输入最后维里值索引

返回类型：元组[变量]

提示：抛出异常-如果k<1或者k不小于输入的最后维

**代码示例**：

.. code-block:: python 

    top5_values, top5_indices = layers.topk(input, k=5)

.. _cn_api_fluid_layers_sequence_reshape:

sequence_reshape
>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers. sequence_reshape(input, new_dim) 

Sequence Reshape Layer
该层重排输入序列。用户设置新维度。每一个序列的的长度通过原始长度、原始维度和新的维度计算得出。以下实例帮助解释该层的功能

.. code-block:: python

    x是一个LoDTensor:
        x.lod  = [[0, 2, 6]]
        x.data = [[1,  2], [3,  4],
                [5,  6], [7,  8],
                [9, 10], [11, 12]]
        x.dims = [6, 2]
    设置 new_dim = 4
    输出为LoDTensor:
        out.lod  = [[0, 1, 3]]

        out.data = [[1,  2,  3,  4],
                    [5,  6,  7,  8],
                    [9, 10, 11, 12]]
        out.dims = [3, 4]

目前仅提供1级LoDTensor，请确认初始长度与初始维度的乘积可被新维度整除，并且每一列没有多余。

参数：
    - **input** (Variable)-一个2-D LoDTensor,模型为[N,M]，维度为M
    - **new_dim** (int)-新维度，输入LoDTensor重新塑造后的新维度

返回：根据新维度重新塑造的LoDTensor

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    x = fluid.layers.data(shape=[5, 20], dtype='float32', lod_level=1)
    x_reshaped = fluid.layers.sequence_reshape(input=x, new_dim=10)

.. _cn_api_fluid_layers_transpose:

transpose
>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.transpose(x,perm,name=None)

按照置换perm置换输入的维度矩阵。

返回张量（tensor）的第i维对应输入维度矩阵的perm[i]。

参数：
    - **x**(Variable) - 输入张量（Tensor)
    - **perm**(list) - 输入维度矩阵的转置
    - **name**(str) - 该层名称（可选）

返回： 转置后的张量（Tensor）

返回类型：变量（Variable）

**代码示例**:

.. code_block:: python

    x = fluid.layers.data(name='x', shape=[5, 10, 15], dtype='float32')
    x_transposed = layers.transpose(x, perm=[1, 0, 2])

.. _cn_api_fluid_layers_one_hot:

one_hot 
>>>>>>>>

.. py:class:: paddle.fluid.layers.one_hot(input, depth)

该层创建输入指数的one-hot表示

参数：
    - **input** (Variable)-输入指数，最后维度必须为1
    - **depth** (scalar)-整数，定义one-hot维度的深度

返回：输入的one-hot表示

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python 

    label = layers.data(name="label", shape=[1], dtype="float32")
    one_hot_label = layers.one_hot(input=label, depth=10)

.. _cn_api_fluid_layers_autoincreased_step_counter:

autoincreased_step_counter
>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.autoincreased_step_counter(counter_name=None, begin=1, step=1)

创建一个自增变量，每个mini-batch返回主函数运行次数，变量自动加1，默认初始值为1.

参数：
    - **counter_name** (str)-计数名称，默认为'@STEP_COUNTER@'
    - **begin** (int)-开始计数
    - **step** (int)-执行之间增加的步数

返回：全局运行步数

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    global_step = fluid.layers.autoincreased_step_counter(
        counter_name='@LR_DECAY_COUNTER@', begin=begin, step=1)

.. _cn_api_fluid_layers_squeeze:

squeeze 
>>>>>>>>

.. py:class:: paddle.fluid.layers. squeeze(input, axes, name=None)

** 向张量维度中移除单维输入。传入用于压缩的轴。如果未提供轴，所有的单一维度将从维中移除。如果带有维入口的轴与其他轴不等，则报错。**
例如：
情况1：

.. code-block:: python

    Given
        X.shape = (1,3,1,5)
    and
        axes = [0]
    we get
        Out.shape = (3,1,5)
    Case 2：
        Given
            X.shape = (1,3,1,5)
        and
            axes = []
        we get
            Out.shape = (3,5)

参数：
        - **input** (Variable)-将要压缩的输入变量
        - **axes** (list)-一列整数，代表压缩的维
        - **name** (str|None)-该层名称

返回：输出压缩的变量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    x = layers.data(name='x', shape=[5, 1, 10])
    y = layers.sequeeze(input=x, axes=[1])      

.. _cn_api_fluid_layers_unsqueeze:

unsqueeze
>>>>>>>>>>

.. py:class:: paddle.fluid.layers. unsqueeze(input, axes, name=None)

向张量维度中插入单维入口。传入一个必须的参数轴，将插入一列维。输出张量中显示轴上划分的维。

比如：
给定一个张量，例如维度为[3,4,5]的张量，轴为[0,4]的未压缩张量，维度为[1,3,4,5,1]

参数：
    - **input** (Variable)-未压缩的输入变量
    - **axes** (list)-一列整数，代表要插入的维数
    - **name** (str|None)-该层名称

返回：输出未压缩变量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    x = layers.data(name='x', shape=[5, 10])
    y = layers.unsequeeze(input=x, axes=[1])


.. _cn_api_fluid_layers_lod_reset:

lod_reset
>>>>>>>>>>

.. py:class:: paddle.fluid.layers. lod_reset(x, y=None, target_lod=None)

设定x的LoD为y或者target_lod。如果提供y，首先将y.lod指定为目标LoD,否则y.data将指定为目标LoD。如果未提供y，
目标LoD则指定为target_lod。如果目标LoD指定为Y.data或target_lod，只提供一层LoD。

- 例1:

.. code-block:: python

    Given a 1-level LoDTensor x:
        x.lod =  [[ 2,           3,                   1 ]]
        x.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
        x.dims = [6, 1]

    target_lod: [4, 2]

    then we get a 1-level LoDTensor:
        out.lod =  [[4,                          2]]
        out.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
        out.dims = [6, 1]

- 例2:

.. code-block:: python

    Given a 1-level LoDTensor x:
        x.lod =  [[2,            3,                   1]]
        x.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
        x.dims = [6, 1]

    y is a Tensor:
        y.data = [[2, 4]]
        y.dims = [1, 3]

    then we get a 1-level LoDTensor:
        out.lod =  [[2,            4]]
        out.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
        out.dims = [6, 1]

- 例3:
.. code-block:: python

    Given a 1-level LoDTensor x:
        x.lod =  [[2,            3,                   1]]
        x.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
        x.dims = [6, 1]

    y is a 2-level LoDTensor:
        y.lod =  [[2, 2], [2, 2, 1, 1]]
        y.data = [[1.1], [2.1], [3.1], [4.1], [5.1], [6.1]]
        y.dims = [6, 1]

    then we get a 2-level LoDTensor:
        out.lod =  [[2, 2], [2, 2, 1, 1]]
        out.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
        out.dims = [6, 1]

参数：
    - **x** (Variable)-输入变量，可以为Tensor或者LodTensor
    - **y** (Variable|None)-若提供，输出的LoD则衍生自y
    - **target_lod** (list|tuple|None)-一层LoD，y未提供时作为目标LoD

返回：输出变量，该层指定为LoD

返回类型：变量

提示：抛出异常 - 如果y和target_lod都为空

**代码示例**：
.. code-block:: python

    x = layers.data(name='x', shape=[10])
    y = layers.data(name='y', shape=[10, 20], lod_level=2)
    out = layers.lod_reset(x=x, y=y)

.. _cn_api_fluid_layers_square:

square
>>>>>>>

.. py:class:: paddle.fluid.layers. square(x,name=None)

SquareDoc :参数x: 平方操作符的输入 :参数use_mkldnn: (bool, 默认false) 仅在mkldnn核中使用:类型use_mkldnn: BOOLEAN

返回：平方后的结果

.. _cn_api_fluid_layers_softplus:

softplus
>>>>>>>>>

.. py:class:: paddle.fluid.layers. softplus(x,name=None)

SoftplusDoc :参数x: Softplus操作符的输入 :参数use_mkldnn: (bool, 默认false) 仅在mkldnn核中使用:类型 use_mkldnn: BOOLEAN

返回：Softplus操作后的结果

.. _cn_api_fluid_layers_softsign:

softsign
>>>>>>>>>

.. py:class:: Paddle.fluid.layers. softsign(x,name=None)

SoftplusDoc :参数x: Softsign操作符的输入 :参数use_mkldnn: (bool, 默认false) 仅在mkldnn核中使用:类型 use_mkldnn: BOOLEAN

返回：Softsign操作后的结果

.. _cn_api_fluid_layers_uniform_random:

uniform_random
>>>>>>>>>>>>>>

.. py:class:: Paddle.fluid.layers. uniform_random(shape,dtype=None,min=None,max=None,seed=None)
该操作符初始化一个张量，该张量的值是从正太分布中抽样的随机值

参数：
    - **shape** (LONGS)-输出张量的维
    - **min** (FLOAT)-均匀随机分布的最小值。[默认 -1.0]
    - **max** (FLOAT)-均匀随机分布的最大值。[默认 1.0]
    - **seed** (INT)-随机种子，用于生成样本。0表示使用系统生成的种子。注意如果种子不为0，该操作符每次都生成同样的随机数。[默认 0]
    - **dtype** (INT)-输出张量数据类型。[默认5(FP32)]

返回：正态随机操作符的输出张量

**代码示例**：

.. code-block:: python

    result = fluid.layers.uniform_random(shape=[32, 784])

.. _cn_api_fluid_layers_hard_shrink:

hard_shrink
>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers. hard_shrink(x,threshold=None)

HardShrink激活函数(HardShrink activation operator)

公式

参数：
    - **x** -HardShrink激活函数的输入
    - **threshold** (FLOAT)-HardShrink激活函数的threshold值。[默认：0.5]

返回：HardShrink激活函数的输出

**代码示例**：

    .. code-block:: python

        data = fluid.layers.data(name="input", shape=[784])
        result = fluid.layers.hard_shrink(x=data, threshold=0.3)    

.. _cn_api_fluid_layers_cumsum:

cumsum
>>>>>>>

.. py:class:: paddle.fluid.layers. cumsum(x,axis=None,exclusive=None,reverse=None

给定轴上元素的累加。默认结果的第一个元素和输入的第一个元素一致。如果exlusive为真，结果的第一个元素则为0。

参数：
    - **x** -累加操作符的输入
    - **axis** (INT)-需要累加的维。-1代表最后一维。[默认 -1]。
    - **exclusive** (BOOLEAN)-是否执行exclusive累加。[默认false]。
    - **reverse** (BOOLEAN)-若为true,则以相反顺序执行累加。[默认 false]。

返回：累加器的输出

**代码示例**：

.. code-block:: python

    data = fluid.layers.data(name="input", shape=[32, 784])
    result = fluid.layers.cumsum(data, axis=0)

.. _cn_api_fluid_layers_thresholded_relu:

thresholded_relu
>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers thresholded_relu(x,threshold=None)

    ThresholdedRelu激活函数
        公式

    参数：
        - **x** -ThresholdedRelu激活函数的输入
        - **threshold** (FLOAT)-激活函数threshold的位置。[默认1.0]。
    
    返回：ThresholdedRelu激活函数的输出

    **代码示例**：

    .. code-block:: python

        data = fluid.layers.data(name="input", shape=[1])
        result = fluid.layers.thresholded_relu(data, threshold=0.4)

.. _cn_api_fluid_layers_create_tensor:

create_tensor
>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers. create_tensor(dtype,name=None,persistable=False)

创建一个变量，存储数据类型为dtype的LoDTensor。

参数：
    - **dtype** (string)-‘float32’|’int32’|..., 创建张量的数据类型。
    - **name** (string)-创建张量的名称。如果未设置，则随机取一个唯一的名称。
    - **persistable** (bool)-为创建张量设置的永久标记

返回：存储在创建张量中的张量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    tensor = fluid.layers.create_tensor(dtype='float32')

create_tensor
>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers. create_tensor(dtype,name=None,persistable=False)

创建一个变量，存储数据类型为dtype的LoDTensor。

参数：
    - **dtype** (string)-‘float32’|’int32’|..., 创建张量的数据类型。
    - **name** (string)-创建张量的名称。如果未设置，则随机取一个唯一的名称。
    - **persistable** (bool)-为创建张量设置的永久标记

返回：存储在创建张量中的张量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    tensor = fluid.layers.create_tensor(dtype='float32')

.. _cn_api_fluid_layers_create_parameter:

create_parameter
>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers. create_parameter(shape,dtype,name=None,attr=None,is_bias=False,default_initializer=None)

创建一个参数。该参数是一个可学习的变量，拥有梯度并且可优化。

注：这是一个非常低级的API。自创操作符时该API较为有用，而无需使用层。

参数：
    - **shape** (list[int])-参数的维度
    - **dtype** (string)-参数的元素类型
    - **attr** (ParamAttr)-参数的属性
    - **is_bias** (bool)-当default_initializer为空，该值会对选择哪个默认初始化程序产生影响。如果is_bias为真，则使用initializer.Constant(0.0)。
    否则使用Xavier()
    - **default_initializer** (Initializer)-参数的初始化程序

返回：创建的参数

**代码示例**：

.. code-block:: python

    W = fluid.layers.create_parameter(shape=[784, 200], dtype='float32')
    data = fluid.layers.data(name="img", shape=[64, 784], append_batch_size=False)
    hidden = fluid.layers.matmul(x=data, y=W)

.. _cn_api_fluid_layers_create_global_var:

create_global_var
>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers create_global_var(shape,value,dtype,persistable=False,force_cpu=False,name=None)

在全局块中创建一个新的带有值的张量。

参数：
    - **shape** (list[int])-变量的维度
    - **value** (float)-变量的值。填充新创建的变量
    - **dtype** (string)-变量的数据类型
    - **persistable** (bool)-如果是永久变量。默认：False
    - **force_cpu** (bool)-将该变量压入CPU。默认：False
    - **name** (str|None)-变量名。如果设为空，则自动创建变量名。默认：None.

返回：创建的变量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    var = fluid.create_global_var(shape=[2,3], value=1.0, dtype='float32',
                     persistable=True, force_cpu=True, name='new_var')

.. _cn_api_fluid_layers_cast:

cast 
>>>>>>

.. py:class:: paddle.fluid.layers. cast(x,dtype)

该层传入变量x,并用x.dtype将x转换成dtype类型，作为输出。

参数：
    - **x** (Variable)-转换函数的输入变量
    - **dtype** (np.dtype|core.VarDesc.VarType|str)-输出变量的数据类型

返回：转换后的输出变量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    data = fluid.layers.data(name='x', shape=[13], dtype='float32')
    result = fluid.layers.cast(x=data, dtype='float64')

.. _cn_api_fluid_layers_concat:

concat
>>>>>>>

.. py:class:: paddle.fluid.layers concat(input,axis=0,name=None)

**Concat** 

该函数将提到的轴上的输入连接起来，并作为输出返回。

参数：
    - **input** (list)-将要联结的张量列表
    - **axis** (int)-数据类型为整型的轴，其上的张量将被联结
    - **name** (str|None)-该层名称（可选）。如果设为空，则自动为该层命名。

返回：输出的联结变量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    out = fluid.layers.concat(input=[Efirst, Esecond, Ethird, Efourth])

.. _cn_api_fluid_layers_sums:

sums
>>>>>

.. py:class:: paddle.fluid.layers. sums(input,out=None)

该函数对输入进行求和，并返回求和结果作为输出。

参数：
    - **input** (Variable|list)-输入张量，有需要求和的元素
    - **out** (Variable|None)-输出参数。求和结果。默认：None

返回：输入的求和。和参数'out'等同

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    tmp = fluid.layers.zeros(shape=[10], dtype='int32')
    i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
    a0 = layers.array_read(array=tmp, i=i)
    i = layers.increment(x=i)
    a1 = layers.array_read(array=tmp, i=i)
    mean_a0 = layers.mean(a0)
    mean_a1 = layers.mean(a1)
    a_sum = layers.sums(input=[mean_a0, mean_a1])

.. _cn_api_fluid_layers_assign:

assign
>>>>>>>

.. py:class:: paddle.fluid.layers. assign(input,output=None)

**Assign**

该功能将输入变量复制到输出变量

参数：
    - **input** (Variable|numpy.ndarray)-源变量
    - **output** (Variable|None)-目标变量

返回：作为输出的目标变量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    out = fluid.layers.create_tensor(dtype='float32')
    hidden = fluid.layers.fc(input=data, size=10)
    fluid.layers.assign(hidden, out)

.. _cn_api_fluid_layers_fill_constant_batch_size_like:

fill_constant_batch_size_like
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers. fill_constant_batch_size_like(input,shape,dtype,value,input_dim_idx=0,output_dim_idx=0)

该功能创建一个张量，具体含有shape,dtype和batch尺寸。并用值中提供的常量初始化该张量。该批尺寸从输入张量中获取。

也将stop_gradient设置为True.

    data = fluid.layers.fill_constant_batch_size_like(
                input=like, shape=[1], value=0, dtype='int64')

参数：
    - **input** (Variable)-张量，其input_dim_idx个维具体指示batch_size
    - **shape** (INTS)-输出的维
    - **dtype** (INT)-可以为numpy.dtype。输出数据类型。默认为float32
    - **value** (FLOAT)-默认为0.将要被填充的值
    - **input_dim_idx** (INT)-默认为0.输入批尺寸维的索引
    - **output_dim_idx** (INT)-默认为0.输出批尺寸维的索引

返回：具体维的张量填充有具体值

.. _cn_api_fluid_layers_fill_constant:

fill_constant
>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers fill_constant(shape,dtype,value,force_cpu=False,out=None)

**fill_constant**

该功能创建一个张量，具体含有shape,dtype和batch尺寸。并用值中提供的常量初始化该张量。

创建张量的属性stop_gradient设为True。

参数：
    - **shape** (tuple|list|None)-输出张量的维
    - **dtype** (np.dtype|core.VarDesc.VarType|str)-输出张量的数据类型
    - **value** (float)-用于初始化输出张量的常量值
    - **out** (Variable)-输出张量
    - **force_cpu** (True|False)-若设为true,数据必须在CPU上

返回：存储在输出中的张量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    data = fluid.layers.fill_constant(shape=[1], value=0, dtype='int64')

.. _cn_api_fluid_layers_argmin:

argmin
>>>>>>>

.. py:class:: paddle.fluid.layers argmin(x,axis=0)
    
    **argmin**
    
    该功能计算输入张量元素中最小元素的索引，张量元素在提供的轴上。

    参数：
        - **x** (Variable)-计算最小元素索引的输入
        - **axis** (int)-计算索引的轴
    
    返回：存储在输出中的张量

    返回类型：变量（Variable）

    **代码示例**：

    .. code-block:: python

        out = fluid.layers.argmin(x=in, axis=0)
        out = fluid.layers.argmin(x=in, axis=-1)
    
.. _cn_api_fluid_layers_argmax:

argmax
>>>>>>

.. py:class:: paddle.fluid.layers argmin(x,axis=0)
    
    **argmax**
    
    该功能计算输入张量元素中最大元素的索引，张量的元素在提供的轴上。

    参数：
        - **x** (Variable)-用于计算最大元素索引的输入
        - **axis** (int)-用于计算索引的轴
    
    返回：存储在输出中的张量

    返回类型：变量（Variable）

    **代码示例**：

    .. code-block:: python

        out = fluid.layers.argmax(x=in, axis=0)
        out = fluid.layers.argmax(x=in, axis=-1)

.. _cn_api_fluid_layers_argsort:

argsort
>>>>>>>

.. py:class:: paddle.fluid.layers argsort(input,axis=-1,name=None)

对给定轴上的输入变量进行排序，输出排序好的数据和相应的索引，其维度和输入相同

.. code-block:: python

    For example, the given axis is -1 and the input Variable

        input = [[0.15849551, 0.45865775, 0.8563702 ],
                [0.12070083, 0.28766365, 0.18776911]],

    after argsort, the sorted Vairable becomes

        out = [[0.15849551, 0.45865775, 0.8563702 ],
            [0.12070083, 0.18776911, 0.28766365]],

    and the sorted indices along the given axis turn outs to be

        indices = [[0, 1, 2],
                [0, 2, 1]]

参数：
    - **input** (Variable)-用于排序的输入变量
    - **axis** (int)-含有用于排序输入变量的轴。当axis<0,实际的轴为axis+rank(input)。默认为-1，即最后一维。
    - **name** (str|None)-（可选）该层名称。如果设为空，则自动为该层命名。

返回：含有已排序的数据和索引

返回类型：元组

**代码示例**：

.. code-block:: python

    input = fluid.layers.data(data=[2, 3])
    out, indices = fluid.layers.argsort(input, axis=0)

.. _cn_api_fluid_layers_ones:

ones 
>>>>>

.. py:class:: paddle.fluid.layers. ones(shape,dtype,force_cpu=False)

**ones**

该功能创建一个张量，有具体的维度和dtype，初始值为1。

也将stop_gradient设置为True。

参数：
    - **shape** (tuple|list|None)-输出张量的维
    - **dtype** (np.dtype|core.VarDesc.VarType|str)-输出张量的数据类型

返回：存储在输出中的张量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    data = fluid.layers.ones(shape=[1], dtype='int64')

.. _cn_api_fluid_layers_zeros:

zeros
>>>>>>

.. py:class:: paddle.fluid.layers. zeros(shape,dtype,force_cpu=False)

**zeros**

该功能创建一个张量，含有具体的维度和dtype，初始值为1.

也将stop_gradient设置为True。

参数：
    - **shape** (tuple|list|None)-输出张量的维
    - **dtype** (np.dtype|core.VarDesc.VarType|str)-输出张量的数据类型
    - **force_cpu** (bool,default False)-是否将输出保留在CPU上

返回：存储在输出中的张量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python
    data = fluid.layers.zeros(shape=[1], dtype='int64')

.. _cn_api_fluid_layers_reverse:

reverse
>>>>>>>>

.. py:class:: paddle.fluid.layers. reverse(x,axis)
    
    **reverse**
    
    该功能将给定轴上的输入'x'逆序

    参数：
        - **x** (Variable)-预逆序到输入
        - **axis** (int|tuple|list)-其上元素逆序排列的轴。
    
    返回：逆序的张量

    返回类型：变量（Variable）

    **代码示例**：

    .. code-block:: python

        out = fluid.layers.reverse(x=in, axis=0)
        # or:
        out = fluid.layers.reverse(x=in, axis=[0,1])

.. _cn_api_fluid_layers_exponential_decay:

exponential_decay 
>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers exponential_decay(learning_rate,decay_steps,decay_rate,staircase=False)

在学习率上运用指数衰减。
训练模型时，在训练过程中通常推荐降低学习率。每次‘decay_steps’步骤中用'decay_rate'衰减学习率。

.. code-block:: python

    if staircase == True:
        decayed_learning_rate = learning_rate * decay_rate ^ floor(global_step / decay_steps)
    else:
        decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)    

参数：
    - **learning_rate** (Variable|float)-初始学习率
    - **decay_steps** (int)-见以上衰减运算
    - **decay_rate** (float)-衰减率。见以上衰减运算
    - **staircase** (Boolean)-若为True,按离散区间衰减学习率。默认：False

返回：衰减的学习率

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    base_lr = 0.1
    sgd_optimizer = fluid.optimizer.SGD(
        learning_rate=fluid.layers.exponential_decay(
            learning_rate=base_lr,
            decay_steps=10000,
            decay_rate=0.5,
            staircase=True))
    sgd_optimizer.minimize(avg_cost)

.. _cn_api_fluid_layers_natural_exp_decay:

natural_exp_decay
>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers. natural_exp_decay(learning_rate, decay_steps, decay_rate, staircase=False)

将自然指数衰减运用到初始学习率上。

.. code-block:: python

    if not staircase:
        decayed_learning_rate = learning_rate * exp(- decay_rate * (global_step / decay_steps))
    else:
        decayed_learning_rate = learning_rate * exp(- decay_rate * (global_step / decay_steps))

参数：
    - **learning_rate** -标量float32值或变量。是训练过程中的初始学习率。
    - **decay_steps** -Python int32数
    - **decay_rate** -Python float数
    - **staircase** -Boolean.若设为true，每个decay_steps衰减学习率

返回：衰减的学习率

.. _cn_api_fluid_layers_inverse_time_decay:

inverse_time_decay
>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers. inverse_time_decay(learning_rate, decay_steps, decay_rate, staircase=False)

在初始学习率上运用逆时衰减。

训练模型时，在训练过程中通常推荐降低学习率。通过执行该函数，将对初始学习率运用逆向衰减函数。

.. code-block:: python

    if staircase == True:
         decayed_learning_rate = learning_rate / (1 + decay_rate * floor(global_step / decay_step))
     else:
         decayed_learning_rate = learning_rate / (1 + decay_rate * global_step / decay_step)

参数：
    - **learning_rate** (Variable|float)-初始学习率
    - **decay_steps** (int)-见以上衰减运算
    - **decay_rate** (float)-衰减率。见以上衰减运算
    - **staircase** (Boolean)-若为True，按间隔区间衰减学习率。默认：False

    返回：衰减的学习率

    返回类型：变量（Variable）

    示例代码：

    .. code-block:: python

        base_lr = 0.1
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.layers.inverse_time_decay(
                learning_rate=base_lr,
                decay_steps=10000,
                decay_rate=0.5,
                staircase=True))
        sgd_optimizer.minimize(avg_cost)

.. _cn_api_fluid_layers_polynomial_decay:

polynomial_decay 
>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers. polynomial_decay(learning_rate,decay_steps,end_learning_rate=0.0001,power=1.0,cycle=False)

对初始学习率使用多项式衰减

.. code-block:: python

    if cycle:
        decay_steps = decay_steps * ceil(global_step / decay_steps)
    else:
        global_step = min(global_step, decay_steps)
        decayed_learning_rate = (learning_rate - end_learning_rate) *
            (1 - global_step / decay_steps) ^ power + end_learning_rate

参数：
    - **learning_rate** (Variable|float32)-标量float32值或变量。是训练过程中的初始学习率。
    - **decay_steps** (int32)-Python int32数
    - **end_learning_rate** (float)-Python float数
    - **power** (float)-Python float数
    - **cycle** (bool)-若设为true，每decay_steps衰减学习率

返回：衰减的学习率

返回类型：变量（Variable）

.. _cn_api_fluid_layers_piecewise_decay:

piecewise_decay
>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers. piecewise_decay(boundaries,values)

对初始学习率进行分段衰减。

该算法可用如下代码描述。

.. code-block:: python

    boundaries = [10000, 20000]
    values = [1.0, 0.5, 0.1]
    if step < 10000:
        learning_rate = 1.0
    elif 10000 <= step < 20000:
        learning_rate = 0.5
    else:
        learning_rate = 0.1

参数：
    - **boundaries** -一列代表步数的数字
    - **values** -一列学习率的值，从不同的步边界中挑选

返回：衰减的学习率

.. _cn_api_fluid_layers_noam_decay:

noam_decay
>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers noam_decay(d_model,warmup_steps)

Noam衰减方法。noam衰减的numpy实现如下。

.. code-block:: python

    import numpy as np
    lr_value = np.power(d_model, -0.5) * np.min([
                           np.power(current_steps, -0.5),
                           np.power(warmup_steps, -1.5) * current_steps])

请参照 attention is all you need。

参数：
    - **d_model** (Variable)-模型的输入和输出维度
    - **warmup_steps** (Variable)-超参数

返回：衰减的学习率

.. _cn_api_fluid_layers_append_LARS:

append_LARS 
>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers. append_LARS(params_grads,learning_rate,weight_decay)

对每一层的学习率运用LARS(LAYER-WISE ADAPTIVE RATE SCALING)

'''python
        learning_rate*=local_gw_ratio * sqrt(sumsq(param))
            / (sqrt(sumsq(gradient))+ weight_decay * sqrt(sumsq(param)))
'''
参数：
    - **learning_rate** -变量学习率。LARS的全局学习率。
    - **weight_decay** -Python float类型数

返回： 衰减的学习率

.. _cn_api_fluid_layers_prior_box:

prior_box 
>>>>>>>>>
.. py:class:: paddle.fluid.layers. prior_box(input,image,min_sizes=None,aspect_ratios=[1.0],variance=[0.1,0.1,0.2,0.2],flip=False,clip=False,steps=[0.0,0.0],offset=0.5,name=None,min_max_aspect_ratios_order=False)

**Prior Box Operator**

为SSD(Single Shot MultiBox Detector)算法生成先验盒。输入的每个位产生N个先验盒，N由min_sizes,max_sizes和aspect_ratios的数目决定，先验盒的尺寸在(min_size,max_size)之间，该尺寸根据aspect_ratios在序列中生成。

参数：
    - **input**(Variable)-输入变量，格式为NCHW
    - ** image** (Variable)-PriorBoxOp的输入图像数据，布局为NCHW
    - ** min_sizes** (list|tuple|float值)-生成的先验框的最小尺寸
    - ** max_sizes** (list|tuple|None)-生成的先验框的最大尺寸。默认：None
    - ** aspect_ratios** (list|tuple|float值)-生成的先验框的纵横比。默认：[1.]
    - ** variance** (list|tuple)-先验框中的变量，会被解码。默认：[0.1,0.1,0.2,0.2]
    - ** flip** (bool)-是否忽略纵横比。默认：False。
    - ** clip** (bool)-是否修建溢界框。默认：False。
    - ** step** (list|tuple)-先验框在
    - ** offset** (float)-先验框中心位移。默认：0.5
    - ** name** (str)-先验框操作符名称。默认：None
    - ** min_max_aspect_ratios_order** (bool)-若设为True,先验框的输出以[min,max,aspect_ratios]的顺序，和Caffe保持一致。请注意，该顺序会影响后面卷基层的权重顺序，但不影响最后的检测结果。默认：False。

返回：
    含有两个变量的元组(boxes,variances)
    boxes:PriorBox的输出先验框。布局是[H,W,num_priors,4]。H是输入的高度，W是输入的宽度，num_priors是输入每位的总框数
    variances:PriorBox的扩展变量。布局上[H,W,num_priors,4]。H是输入的高度，W是输入的宽度，num_priors是输入每位的总框数

返回类型：元组

**代码示例**：

.. code-block:: python

    box, var = fluid.layers.prior_box(
        input=conv1,
        image=images,
        min_sizes=[100.],
        flip=True,
        clip=True)

.. _cn_api_fluid_layers_multi_box_head:

multi_box_head
>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.multi_box_head(inputs, image, base_size, num_classes, aspect_ratios, min_ratio=None, max_ratio=None, min_sizes=None, max_sizes=None, steps=None, step_w=None, step_h=None, offset=0.5, variance=[0.1, 0.1, 0.2, 0.2], flip=True, clip=False, kernel_size=1, pad=0, stride=1, name=None, min_max_aspect_ratios_order=False)

为SSD(Single Shot MultiBox Detector)算法生成先验框。算法详情请参见SSD论文的2.2节SSD:Single Shot MultiBox Detector。

参数：
    - **inputs** (list|tuple)-一列输入变量，所有变量的格式为NCHW
    - **image** (Variable)-PriorBoxOp的输入图片，布局上NCHW
    - **base_size** (int)-根据min_ratio和max_ratio获取min_size和max_size
    - **num_classes** (int)0类的数目
    - **aspect_ratios** (list|tuple)-生成先验框的纵横比。输入长度和纵横比的长度项值必须相等
    - **min_ratio** (int)-生成先验框的最小比例
    - **max_ratio** (int)-生成先验框的最大比例
    - **min_sizes** (list|tuple|None)-如果len(inputs)<=2,必须设定min_sizes，并且min_sizes的长度应当和输入的长度相等。默认：None
    - **max_sizes** (list|tuple|None)-如果len(inputs)<2,必须设定max_sizes，并且max_sizes的长度应当和输入的长度相等。默认：None
    - **steps** (list|tuple)-如果step_w和step_h相同，step_w和step_h可替换成steps
    - **step_w** (list|tuple)-先验框在宽度方向上的步长。如果step_w[i]==0.0,则自动计算inputs[i]先验框在宽度方向上的步长。默认：None
    - **step_h** (list|tuple)-先验框在高度方向上的步长。如果step_h[i]==0.0,则自动计算inputs[i]先验框中高度方向上的步长。默认：None
    - **offset** (float)-先验框的中心偏移。默认：0.5
    - **variance** (list|tuple)-先验框中将被解码的变量。默认：[0.1,0.1,0.2,0.2]
    - **flip** (bool)-是否略过纵横比。默认：False
    - **clip** (bool)-是否剪裁出界框。默认：False
    - **kernel_size** (int)-conv2d的核尺寸。默认：1
    - **pad** (int|list|tuple)-conv2d的填充。默认：0
    - **stride** (int|list|tuple)-conv2d的步长。默认：1
    - **name** (str)-先验框层的名称。默认：None
    - **min_max_aspect_ratios_order** (bool)-如果设为True，输出先验框的顺序为[min,max,aspect_ratios]，和Caffe保持一致。请注意，该顺序影响后面卷积层的权重顺序，但不影响最终检测结果。默认：False

返回：
    含有四个变量的元组。(mbox_loc,mbox_conf,boxes,variances)

    **mbox_loc** :输入中预测框的位置。输出结果为[N,H*W*Priors,4]，Priors是每个位置的预测框数量。

    **mbox_conf** :输入中预测框的置信值。输出结果为[N,H*W*Priors,C]，Priors是每个位置的预测框数量，C是类的数量。

    **boxes** :PriorBox的输出先验框。布局为[num_priors,4]。num_priors是输入每个位上的总框数

    **variances** :PriorBox的扩展变量。布局为[num_priors,4]。num_priors是输入每个位上的总框数

返回类型：元组（tuple）

**代码示例**：

.. code-block:: python

    mbox_locs, mbox_confs, box, var = fluid.layers.multi_box_head(
        inputs=[conv1, conv2, conv3, conv4, conv5, conv5],
        image=images,
        num_classes=21,
        min_ratio=20,
        max_ratio=90,
        aspect_ratios=[[2.], [2., 3.], [2., 3.], [2., 3.], [2.], [2.]],
        base_size=300,
        offset=0.5,
        flip=True,
        clip=True)

.. _cn_api_fluid_layers_bipartite_match:

biparite_match
>>>>>>>>>>>>>>>>



.. py:class:: paddle.fluid.layers.bipartite_match(dist_matrix, match_type=None, dist_threshold=None, name=None)

该操作符实现一个贪婪二分图匹配算法，根据输入的距离矩阵获得最大距离匹配。对于输入的二维矩阵，二分图匹配算法找到每一行匹配的列（匹配即最大距离），
也能找到每一列匹配的行。并且该操作符只计算从列到行的匹配索引。对一个实例，匹配索引数是输入距离矩阵的列数。

输出包含匹配索引和距离。简要描述即该算法匹配距离最大的行到距离最大的列，在ColToRowMatchIndices的每一行不会复制匹配索引。如果行项没有匹配的列项，则在ColRowMatchIndices中置为-1。

注：输入DistMat可以是LoDTensor（含LoD)或者张量（Tensor）。如果LoDTensor带有LoD，ColToRowMatchIndices的高度为批尺寸。如果为张量，ColToRowMatchIndices的高度为1。

注：这是一个非常低级的API。用''ssd_loss''层。请考虑用''ssd_loss''。

参数：
    - **dist_matrix**(Vairable) - 输入是维度为[K,M]的二维LoDTensor，是行项和列项之间距离的矩阵。假设矩阵A,维度为K，矩阵B，维度为M。dist_matrix[i][j]即A[i]和B[j]的距离。最大距离即为行列项的最好匹配。
    
    注：该张量包含LoD信息，代表输入的批。该批的一个实例含有不同的项数。

    - **match_type**(string|None) - 匹配算法的类型，应为二分图或。默认为二分图

    - **dist_threshold**(float|None) - 如果match_type为，该临界值决定在最大距离基础上的额外matching bboxes。

返回：
    返回带有两个元素的元组。第一个元素是match_indices,第二个是matched_distance。

    matched_indices是一个二维张量，维度为[N,M]，类型为整型。N是批尺寸。如果match_indices[i][j]为-1，则表示在第i个实例中B[j]不匹配任何项。如果match_indeice不为-1，则表示在第i个实例中B[j]匹配行match_indices[i][j]。第i个实例的行数存在match_indices[i][j]中。

    matched_distance是一个二维张量，维度为[N,M]，类型为浮点型。N是批尺寸。如果match_indices[i][j]为-1，match_distance[i][j]也为-1.如果match_indices[i][j]不为-1，将设match_distance[i][j]=d，每个示例的行偏移两称为LoD。match_distance[i][j] = dist_matrix[d+LoD[i][j]]。

返回类型：元组（tuple）

**代码示例**：

.. code_block:: python

    x = fluid.layers.data(name='x', shape=[4], dtype='float32')
    y = fluid.layers.data(name='y', shape=[4], dtype='float32')
    iou = fluid.layers.iou_similarity(x=x, y=y)
    matched_indices, matched_dist = fluid.layers.bipartite_match(iou)

.. _cn_api_fluid_layers_target_assign:

target_assign
>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.target_assign(input, matched_indices, negative_indices=None, mismatch_value=None, name=None)

对于给定的目标边界框（bounding box）和标签（label），该操作符对每个预测赋予分类和逻辑回归目标函数以及预测权重。权重具体表示哪个预测无需贡献训练误差。

对于每个实例，根据match_indices和negative_indices赋予输入''out''和''out_weight''。将定输入中每个实例的行偏移称为lod，该操作符执行分类或回归目标函数，执行步骤如下：

1.根据match_indices分配所有输入

::
    If id = match_indices[i][j] > 0,

        out[i][j][0 : K] = X[lod[i] + id][j % P][0 : K]
        out_weight[i][j] = 1.

    Otherwise,

        out[j][j][0 : K] = {mismatch_value, mismatch_value, ...}
        out_weight[i][j] = 0.

2.如果提供neg_indices，根据neg_indices分配out_weight：

假设neg_indices中每个实例的行偏移称为neg_lod，该实例中第i个实例和neg_indices的每个id如下：

::
    out[i][id][0 : K] = {mismatch_value, mismatch_value, ...}
    out_weight[i][id] = 1.0

参数：
    - **inputs**(Variable) - 输入为三维LoDTensor，维度为[M,P,K]
    - **matched_indices**(Variable) - 张量（Tensor），整型，输入匹配索引为二维张量（Tensor），类型为整型32位，维度为[N,P]，如果MatchIndices[i][j]为-1，在第i个实例中第j列项不匹配任何行项。
    - **negative_indices**(Variable) - 输入负例索引，可选输入，维度为[Neg,1]，类型为整型32，Neg为负例索引的总数
    - **mismatch_value**(float32) - 为未匹配的位置填充值

返回：返回一个元组（out,out_weight）。out是三维张量，维度为[N,P,K],N和P与neg_indices中的N和P一致，K和输入X中的K一致。如果match_indices[i][j]存在，out_weight是输出权重，维度为[N,P,1]。

返回类型：元组（tuple）

**代码示例**：

.. code_block:: python

    matched_indices, matched_dist = fluid.layers.bipartite_match(iou)
    gt = layers.data(
            name='gt', shape=[1, 1], dtype='int32', lod_level=1)
    trg, trg_weight = layers.target_assign(
                gt, matched_indices, mismatch_value=0)

.. _cn_api_fluid_layers_detection_output:

detection_output
>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.detection_output(loc, scores, prior_box, prior_box_var, background_label=0, nms_threshold=0.3, nms_top_k=400, keep_top_k=200, score_threshold=0.01, nms_eta=1.0)

单点多盒检测的检测输出层（Detection Output Layer for Single Shot Multibox Detector(SSD))

该操作符用于获得检测结果，执行步骤如下：

    1.根据优先盒解码输入边界框（bounding box）预测

    2.通过运用多类非最大压缩(NMS)获得最终检测结果

请注意，该操作符不将最终输出边界框剪切至图像窗口。

参数：
    - **loc**(Variable) - 一个三维张量（Tensor），维度为[N,M,4]，代表M个bounding bboxes的预测位置。N是批尺寸，每个边界框（boungding box）有四个坐标值，布局为[xmin,ymin,xmax,ymax]
    - **scores**(Variable) - 一个三维张量（Tensor），维度为[N,M,C]，代表预测置信预测。N是批尺寸，C是类别数，M是边界框数。对每类一共M个分数，对应M个边界框
    - **prior_box**(Variable) - 一个二维张量（Tensor),维度为[M,4]，存储M个框，每个框代表[xmin,ymin,xmax,ymax]，[xmin,ymin]是anchor box的左上坐标，如果输入是图像特征图，靠近坐标系统的原点。[xmax,ymax]是anchor box的右下坐标
    - **prior_box_var**(Variable) - 一个二维张量（Tensor），维度为[M,4]，存有M变量群
    - **background_label**(float) - 背景标签索引，背景标签将会忽略。若设为-1，将考虑所有类别
    - **nms_threshold**(int) - 用于NMS的临界值（threshold）
    - **nms_top_k**(int) - 基于score_threshold过滤检测后，根据置信数维持的最大检测数
    - **keep_top_k**(int) - NMS步后，每一图像要维持的总bbox数
    - **score_threshold**(float) - 临界函数（Threshold），用来过滤带有低置信分数的边界框（bounding box）。若未提供，则考虑所有框
    - **nms_eta**(float) - 适应NMS的参数

返回：检测输出数一个LoDTensor，维度为[No,6]。每行有6个值：[label,confidence,xmin,ymin,xmax,ymax]。No是该mini-batch的总检测数。对每个实例，第一维偏移称为LoD，偏移数为N+1，N是批尺寸。第i个图像有LoD[i+1]-LoD[i]检测结果。如果为0，第i个图像无检测结果。如果所有图像都没有检测结果，LoD所有元素都为0，并且输出张量只包含一个值-1。

返回类型：变量（Variable）

**代码示例**：

.. code_block:: python

    pb = layers.data(name='prior_box', shape=[10, 4],
             append_batch_size=False, dtype='float32')
    pbv = layers.data(name='prior_box_var', shape=[10, 4],
              append_batch_size=False, dtype='float32')
    loc = layers.data(name='target_box', shape=[2, 21, 4],
              append_batch_size=False, dtype='float32')
    scores = layers.data(name='scores', shape=[2, 21, 10],
              append_batch_size=False, dtype='float32')
    nmsed_outs = fluid.layers.detection_output(scores=scores,
                           loc=loc,
                           prior_box=pb,
                           prior_box_var=pbv)

