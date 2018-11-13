.. _cn_api_fluid_layers_equal:
# equal
## Paddle.fluid.layers. equal(x,y,cond=None,**ignored)
### equal
该层返回元素x,y逐元素相等的值

参数：
- x(Variable)-equal的第一个操作数
- y(Variable)-equal的第二个操作数
- cond(Variable|None)-输出变量（可选），用来存储equal的结果

返回：张量类型的变量，存储equal的输出结果 

返回类型：变量（variable） 

代码示例:  

    less = fluid.layers.equal(x=label,y=limit)

.. _cn_api_fluid_layers_array_read: 
# array_read
## paddle.fluid.layers. array_read(array,i)  
此函数用于读取数据，数据以LOD_TENSOR_ARRAY数组的形式读入
~~~
Given:
array = [0.6,0.1,0.3,0.1]
And:
I=2
Then:
output = 0.3
~~~
参数：
- array(Variable|list)-输入张量，存储要读的数据
- i(Variable|list)-输入数组中数据的索引

返回：张量类型的变量，已有数据写入

返回类型：变量（variable）

代码示例：
~~~python
tmp = fluid.layers.zeros(shape=[10],dtype='int32')
i = fluid.layers.fill_constant(shape=[1],dtype='int64',value=10)
arr = layers.array_read(tmp,i=i)
~~~

.. _cn_api_fluid_layers_array_length:
# array_length
## paddle.fluid.layers.array_length(array)
### 得到输入的LoDTensorArray数组的长度

此功能用于找出输入数组LOD_TENSOR_ARRAY的长度。  
相关API:array_read,array_write,While. 

参数：array(LOD_TENSOR_ARRAY)-输入数组，用来计算数组长度

返回：输入数组LoDTensorArray的长度

返回类型：变量（Variable）

代码示例:
~~~python
tmp = fluid.layers.zeros(shape=[10], dtype='int32')
i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
arr = fluid.layers.array_write(tmp, i=i)
arr_len = fluid.layers.array_length(arr)
~~~

.. _cn_api_fluid_layers_IfElse:
# IfElse
## class paddle.fluid.layers.IfElse(cond, name=None)
if-else控制流。  

参数：
- cond(Variable)-用于比较的条件
- Name(str,默认为空（None）)-该层名称

代码示例：
~~~python
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
~~~

.. _cn_api_fluid_layers_Print:
# Print
## paddle.fluid.layers.Print(input, first_n=-1, message=None, summarize=-1, print_tensor_name=True, print_tensor_type=True, print_tensor_shape=True, print_tensor_lod=True, print_phase='both')

Print操作命令
该操作命令创建一个打印操作，打印正在访问的张量。
包裹传入的张量，以便无论何时访问张量，都会打印信息message和张量的当前值。

参数：
- input(Variable)-将要打印的张量
- summarize(int)-打印张量中的元素数目，如果值为-1则打印所有元素
- message(str)-字符串类型消息，作为前缀打印
- first_n(int)-只记录first_n次数
- print_tensor_name(bool)-打印张量名称
- print_tensor_type(bool)-打印张量类型
- print_tensor_shape(bool)-打印张量维度
- print_tensor_lod(bool)-打印张量lod
- print_phase(str)-所要放置的阶段，包括"forward","backward"和"both".若设置为"backward"或者"both",则打印输入张量的梯度。

返回：输出张量，和输入张量同样的数据

返回类型：变量（Variable）

代码示例：
~~~python
value = some_layer(...)
Print(value, summarize=10,
message="The content of some_layer: ")
~~~

.. _cn_api_fluid_layers_is_empty:
# is_empty
## paddle.fluid.layers.is_empty(x, cond=None, **ignored)

测试变量是否为空

参数：
- x(Variable)-测试的变量
- cond(Variable|None)-输出参数。返回给定x的测试结果，默认为空（None）

返回：布尔类型的标量。如果变量x为空则值为真

返回类型：变量（Variable）

提示：类型错误-如果输入条件不是变量或变量类型不是布尔类型

代码示例：
~~~python
res = fluid.layers.is_empty(x=input)
# or:
fluid.layers.is_empty(x=input, cond=res)
~~~

.. _cn_api_fluid_layers_data:
# data
## paddle.fluid.layers.data(name, shape, append_batch_size=True, dtype='float32', lod_level=0, type=VarType.LOD_TENSOR, stop_gradient=True)

数据层(Data Layer)

该功能接受输入数据，根据是否返回迷你批次minbatch用辅助函数创建全局变量。可通过以下所有操作命令访问全局变量。

该函数输入的所有变量作为局部变量传到LayerHelper构造器

参数：
- name(str)-函数的别名
- shape(list)-声明维度的元组
- append_batch_size(bool)-

        1.如果为真，则在维度shape的开头插入-1
        比如如果shape=[1],结果shape为[-1,1].
        2.如果维度shape包含-1，比如shape=[-1,1],
        append_batch_size则为False（表示无效）
- dtype(int|float)-数据类型：float32,float_16,int等
- type(VarType)-输出类型。默认为LOD_TENSOR.
- lod_level(int)-LoD层。0表示输入数据不是一个序列
- stop_gradient(bool)-布尔数，提示是否应该停止计算梯度

返回：全局变量，可进行数据访问

返回类型：变量(Variable)

代码示例：
~~~python
data = fluid.layers.data(name='x', shape=[784], dtype='float32')
~~~

.. _cn_api_fluid_layers_open_files:
# open_files
## paddle.fluid.layers.open_files(filenames, shapes, lod_levels, dtypes, thread_num=None, buffer_size=None, pass_num=1, is_test=None)

打开文件(Open files)

该层读一列文件并返回Reader变量。通过Reader变量，可以从给定的文件中获取数据。所有的文件必须有后缀名，显示文件格式，比如”*.recordio”。

参数：
- filenames(list)-文件名列表
- shape(list)-元组类型值列表，声明数据维度
- lod_levels(list)-整形值列表，声明数据的lod层级
- dtypes(list)-字符串类型值列表，声明数据类型
- thread_num(None)-用于读文件的线程数。默认：min(len(filenames),cpu_number)
- buffer_size(None)-reader的缓冲区大小。默认：3*thread_num
- pass_num(int)-用于运行的传递数量
- is_test(bool|None)-open_files是否用于测试。如果用于测试，生成的数据顺序和文件顺序一致。反之，无法保证时期间的数据顺序是一致的

返回：一个Reader变量，通过该变量获取文件数据

返回类型：变量(Variable)

代码示例：
~~~python
reader = fluid.layers.io.open_files(filenames=['./data1.recordio',
                                            './data2.recordio'],
                                    shapes=[(3,224,224), (1)],
                                    lod_levels=[0, 0],
                                    dtypes=['float32', 'int64'])

# Via the reader, we can use 'read_file' layer to get data:
image, label = fluid.layers.io.read_file(reader)
~~~
.. _cn_api_fluid_layers_read_file:
# read_file
## paddle.fluid.layers.read_file(reader)
执行给定的reader变量并从中获取数据
reader也是变量。可以为由fluid.layers.open_files()生成的原始reader或者由fluid.layers.double_buffer()生成的装饰变量，等等。

参数：
reader(Variable)-将要执行的reader

返回：从给定的reader中读取udall数据

代码示例：
~~~python
data_file = fluid.layers.open_files(
     filenames=['mnist.recordio'],
     shapes=[(-1, 748), (-1, 1)],
     lod_levels=[0, 0],
     dtypes=["float32", "int64"])
 data_file = fluid.layers.double_buffer(
     fluid.layers.batch(data_file, batch_size=64))
 input, label = fluid.layers.read_file(data_file)
~~~
.. _cn_api_fluid_layers_batch:
# batch
## paddle.fluid.layers.batch(reader, batch_size)

该层是一个reader装饰器。接受一个reader变量并添加“batching”装饰。读取装饰的reader，输出数据自动组织成batch的形式。

参数：
- reader(Variable)-装饰有“batching”的reader变量
- batch_size(int)-批尺寸

返回：装饰有“batching”的reader变量

返回类型：变量(Variable)

代码示例：
~~~python
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
# random_data_generator
## paddle.fluid.layers.random_data_generator(low, high, shapes, lod_levels, for_parallel=True)

创建一个均匀分布随机数据生成器
该层返回一个Reader变量。该Reader变量不是用于打开文件读取数据，而是自生成float类型的均匀分布随机数。该变量可作为一个虚拟reader，无需打开真实文件便可测试网络。

参数：
- low(float)--数据均匀分布的下界
- high(float)-数据均匀分布的上界
- shapes(list)-元组数列表，声明数据维度
- lod_levels(list)-整形数列表，声明数据
- lod_level
- for_parallel(Bool)-若要运行一系列操作命令则将其设置为True

返回：Reader变量，可从中获取随机数据

返回类型：变量(Variable)

代码示例：
~~~python
reader = fluid.layers.random_data_generator(
                                 low=0.0,
                                 high=1.0,
                                 shapes=[[3,224,224], [1]],
                                 lod_levels=[0, 0])
# Via the reader, we can use 'read_file' layer to get data:
image, label = fluid.layers.read_file(reader)
~~~

.. _cn_api_fluid_layers_Preprocessor:
# Preprocessor
## class paddle.fluid.layers.Preprocessor(reader, name=None)

reader变量中数据预处理块。

参数：
- reader(Variable)-reader变量
- name(str,默认None)-reader的名称

代码示例
~~~python
preprocessor = fluid.layers.io.Preprocessor(reader=reader)
with preprocessor.block():
    img, lbl = preprocessor.inputs()
    img_out = img / 2
    lbl_out = lbl + 1
    preprocessor.outputs(img_out, lbl_out)
data_file = fluid.layers.io.double_buffer(preprocessor())
~~~

.. _cn_api_fluid_layers_load:
# load
## paddle.fluid.layers.load(out, file_path, load_as_fp16=None)
Load操作命令将从磁盘文件中加载LoDTensor/SelectedRows变量。
~~~python
 import paddle.fluid as fluid
 tmp_tensor = fluid.layers.create_tensor(dtype='float32')
 fluid.layers.load(tmp_tensor, "./tmp_tensor.bin")
~~~
参数：
- out(Variable)-需要加载的LoDTensor/SelectedRows
- file_path(STRING)-预从”file_path”中加载的变量Variable
- load_as_fp16(BOOLEAN)-如果为真，张量首先进行加载然后转换成float16数据类型。如果为假，张量无数据类型转换直接进行加载。默认为false。

返回：None

.. _cn_api_fluid_layers_embedding:
# embedding
## paddle.fluid.layers.embedding(input, size, is_sparse=False, is_distributed=False, padding_idx=None, param_attr=None, dtype='float32')
嵌入层(Embedding Layer)

该层用来在供查找的表中查找IDS的嵌入矩阵，由input提供。查找的结果是input里每个ID的嵌入。
所有的输入变量都作为局部变量传入LayerHelper构造器

参数：
- input(Variable)-包含IDs的张量
- size(tuple|list)-查找表参数的维度。应当有两个参数，一个代表嵌入矩阵字典的大小，一个代表每个嵌入向量的大小。
- is_sparse(bool)-代表是否用稀疏更新的标志
- is_distributed(bool)-是否从远程参数服务端运行查找表、
- padding_idx(int|long|None)-如果为空，对查找结果无影响。如果padding_idx不为空，表示只要在input查找过程中遇到padding_idz则用0填充输出结果。如果paddingidx<0,在查找表中使用的padding_idx值为size[0]+dim。
param_attr(ParamAttr)-该层参数
dtype(np.dtype|core.VarDesc.VarType|str)-数据类型：float32,float_16,int etc。

返回：张量存储已有输入的嵌入矩阵。

返回类型：变量(Variable)

代码示例:
~~~python
dict_size = len(dataset.ids)
data = fluid.layers.data(name='ids', shape=[32, 32], dtype='float32')
fc = fluid.layers.embedding(input=data, size=[dict_size, 16])
~~~

.. _cn_api_fluid_layers_dynamic_lstmp:
# dynamic_lstmp
## paddle.fluid.layers.dynamic_lstmp(input, size, proj_size, param_attr=None, bias_attr=None, use_peepholes=True, is_reverse=False, gate_activation='sigmoid', cell_activation='tanh', candidate_activation='tanh', proj_activation='tanh', dtype='float32', name=None)

动态LSTMP层(Dynamic LSTMP Layer)
LSTMP层(具有循环映射的LSTM)在LSTM层后有一个分离的映射层，从原始隐藏状态映射到较低维的状态，用来减少参数总数，减少LSTM计算复杂度，特别是输出单元相对较大的情况下。(https://research.google.com/pubs/archive/43905.pdf)

公式如下：
    i<sub>t</sub> = 
在以上公式中：
W:代表权重矩阵（例如 是输入门道输入的权重矩阵）
：窥视孔链接的对角矩阵。
b:

返回：含有两个输出变量的元组：隐藏状态的映射和LSTMP的

返回类型：元组(tuple)

代码示例：
~~~python
dict_dim, emb_dim = 128, 64
data = fluid.layers.data(name='sequence', shape=[1],
                         dtype='int32', lod_level=1)
emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim])
hidden_dim, proj_dim = 512, 256
fc_out = fluid.layers.fc(input=emb, size=hidden_dim * 4,
                         act=None, bias_attr=None)
proj_out, _ = fluid.layers.dynamic_lstmp(input=fc_out,
                                         size=hidden_dim * 4,
                                         proj_size=proj_dim,
                                         use_peepholes=False,
                                         is_reverse=True,
                                         cell_activation="tanh",
                                         proj_activation="tanh")
~~~
.. _cn_api_fluid_layers_warpctc:
# warpctc 
## paddle.fluid.layers.warpctc(input, label, blank=0, norm_by_times=False)


.. _cn_api_fluid_layers_sequence_reshape:
# sequence_reshape
## paddle.fluid.layers.sequence_reshape(input, new_dim)
Sequence Reshape Layer
该层重排输入序列。用户设置新维度。每一个序列的的长度通过原始长度、原始维度和新的维度计算得出。以下实例帮助解释该层的功能
~~~
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
- input(Variable)-一个2-D LoDTensor,模型为[N,M]，维度为M
- new_dim(int)-新维度，输入LoDTensor重新塑造后的新维度
返回：根据新维度重新塑造的LoDTensor
返回类型：变量（Variable）
代码示例：
~~~
x = fluid.layers.data(shape=[5, 20], dtype='float32', lod_level=1)
x_reshaped = fluid.layers.sequence_reshape(input=x, new_dim=10)
~~~

~~~


.. _cn_api_fluid_layers_one_hot:
# one_hot 
## paddle.fluid.layers.one_hot(input, depth)
该层创建输入指数的one-hot表示
参数：
- input(Variable)-输入指数，最后维度必须为1
- depth(scalar)-整数，定义one-hot维度的深度
返回：输入的one-hot表示
返回类型：变量（Variable）
代码示例：
~~~
label = layers.data(name="label", shape=[1], dtype="float32")
one_hot_label = layers.one_hot(input=label, depth=10)
~~~
.. _cn_api_fluid_layers_autoincreased_step_counter:
# autoincreased_step_counter
## paddle.fluid.layers.autoincreased_step_counter(counter_name=None, begin=1, step=1)
创建一个自增变量，每个mini-batch返回主函数运行次数，变量自动加1，默认初始值为1.
参数：
- counter_name(str)-计数名称，默认为'@STEP_COUNTER@'
- begin(int)-技术的第一个值
- step(int)-执行之间增加的步数
返回：全局运行步数
返回类型：变量（Variable）
代码示例：
~~~
global_step = fluid.layers.autoincreased_step_counter(
    counter_name='@LR_DECAY_COUNTER@', begin=begin, step=1)
~~~

.. _cn_api_fluid_layers_squeeze:
# squeeze 
## paddle.fluid.layers.squeeze(input, axes, name=None)
向张量维度中移除单维输入。传入用于压缩的轴。如果未提供轴，所有的单一维度将从维中移除。如果带有维入口的轴与其他轴不等，则报错。
例如：情况1：
~~~给定
    X.shape = (1,3,1,5)
   并且
    axes = [0]
   得到
    Out.shape = (3,1,5)
   情况2：
        给定
            X.shape = (1,3,1,5)
        并且
            axes = []
        得到
            Out.shape = (3,5)
 参数：
 - input(Variable)-将要压缩的输入变量
 - axes(list)-一列整数，代表压缩的维
 - name(str|None)-该层名称
 返回：输出压缩的变量
 返回类型：变量（Variable）
 代码示例：
 ~~~
 x = layers.data(name='x', shape=[5, 1, 10])
 y = layers.sequeeze(input=x, axes=[1])
 ~~~           

.. _cn_api_fluid_layers_unsqueeze:
# unsqueeze
## paddle.fluid.layers.unsqueeze(input, axes, name=None)
向张量维度中插入单维入口。传入一个必须的参数轴，将插入一列维。输出张量中显示轴上划分的维。
比如：
给定一个张量，例如维度为[3,4,5]的张量，轴为[0,4]的未压缩张量，维度为[1,3,4,5,1]
参数：
- input(Variable)-未压缩的输入变量
- axes(list)-一列整数，代表要插入的维数
- name(str|None)-该层名称
返回：输出未压缩变量
返回类型：变量（Variable）
代码示例：
~~~
x = layers.data(name='x', shape=[5, 10])
y = layers.unsequeeze(input=x, axes=[1])
~~~

.. _cn_api_fluid_layers_lod_reset:
# lod_reset
## paddle.fluid.layers.lod_reset(x, y=None, target_lod=None)
设定x的LoD为y或者target_lod。如果提供y，首先将y.lod指定为目标LoD,否则y.data将指定为目标LoD。如果未提供y，
目标LoD则指定为target_lod。如果目标LoD指定为Y.data或target_lod，只提供一层LoD。
~~~
* 例1:

    Given a 1-level LoDTensor x:
        x.lod =  [[ 2,           3,                   1 ]]
        x.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
        x.dims = [6, 1]

    target_lod: [4, 2]

    then we get a 1-level LoDTensor:
        out.lod =  [[4,                          2]]
        out.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
        out.dims = [6, 1]

* 例2:

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

* 例3:

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
~~~
参数：
- x(Variable)-输入变量，可以为Tensor或者LodTensor
- y(Variable|None)-若提供，输出的LoD则衍生自y
- target_lod(list|tuple|None)-一层LoD，y未提供时作为目标LoD
返回：输出变量，该层指定为LoD
返回类型：变量
提示：ValueError - 如果y和target_lod都为空
代码示例：
~~~
x = layers.data(name='x', shape=[10])
y = layers.data(name='y', shape=[10, 20], lod_level=2)
out = layers.lod_reset(x=x, y=y)
~~~
.. _cn_api_fluid_layers_lrn:
# lrn
## paddle.fluid.layers. lrn(input,n=5,k=1.0,alpha=0.0001,beta=0.75,name=None)
Local Response Normalization Layer.
该层通过对本地输入域归一化实现侧向抑制。
公式如下：

在以上等式中：
- n: 累加的渠道数
- k: 位移（避免除数为0）
- alpha: 参数，代表缩放比例
- beta: 参数，代表指数

参考ImageNet Classification with Deep Convolutional Neural Networks

参数:
- input(Variable)-该层输入张量，输入张量的维度必须为4
- n(int,默认为5)-累加的渠道数
- k(float,默认为1.0)-位移（通常避免除数为0）
- alpha(float,默认为1e-4)-缩放比例
- beta(float,默认为0.75)-指数
- name(str,默认None)-操作名称

提示：ValueError-如果输入张量级别不为4
返回：张量变量，存储转换结果

代码示例：
~~~
data = fluid.layers.data(
    name="data", shape=[3, 112, 112], dtype="float32")
lrn = fluid.layers.lrn(input=data)
~~~

