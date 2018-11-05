
.. _cn_api_fluid_Program:

Program
>>>>>>>>>>>>

class paddle.fluid.Program
""""""""""""""""""""""""""""""""""""""""""

创建python program， 在paddleFluid内部会被转换为ProgramDesc描述语言，是被用来创建c++ Program。Program像容器一样也是一种独立的程序语言。Program包括至少一个块（Block），控制流比如conditional_block包括while_op，该Program将会含有嵌套快（nested block）。详情请参阅framework.proto。

注意：默认情况下，paddleFluid内部默认含有default_startup_program和default_main_program，它们将共享参数。default_startup_program只运行一次来初始化参数，default_main_program在每个mini batch中运行并调整权重。

返回： empty program

**代码示例**

..  code-block:: python

  main_program = fluid.Program()
  startup_program = fluid.Program()
  with fluid.program_guard(main_program=main_program, startup_program=startup_program):
        fluid.layers.data(name="x", shape=[-1, 784], dtype='float32')
        fluid.layers.data(name="y", shape=[-1, 1], dtype='int32')
        fluid.layers.fc(name="fc", shape=[10], dtype='float32', act="relu")

op_role
""""""""""""""""""""""""""""""""""""""""""
operator的角色，值只能是枚举变量{Forward, Backward, Optimize}。

注意：这是一个底层API。它仅用于ParallelExecutor复制或调度operator到设备。

例如，Forward operator应该在每个设备上执行。Backward operator在每个设备上执行，并将后向传播的参数梯度(使用op_role_var获得该变量)合并到一个设备上。Optimize operator只在一个设备上执行，并向其他设备广播新的参数，

set_op_role
""""""""""""""""""""""""""""""""""""""""""
operator的角色，值只能是枚举变量{Forward, Backward, Optimize}。

注意：这是一个底层API。它仅用于ParallelExecutor复制或调度operator到设备上执行。

例如，Forward operator应该在每个设备上执行。Backward operato应该在每个设备上执行，并将后向传播的参数梯度(使用op_role_var获得该变量)合并到一个设备上。Optimize operator只在一个设备上执行，并向其他设备广播新的参数

op_role_var
""""""""""""""""""""""""""""""""""""""""""
op_role的辅助变量。

参考:Program.op_role 文档。

注意:这是一个底层API，用户不应该直接使用它。

set_op_role_var
""""""""""""""""""""""""""""""""""""""""""
op_role的辅助变量。

参考:Program.op_role 文档。

注意:这是一个底层API。用户不应该直接使用它。

to_string(throw_on_error, with_details=False)
""""""""""""""""""""""""""""""""""""""""""

用于debug

参数：  
		- throw_on_error(bool): 有设置任何必需的字段时，抛出值错误。
		- with_details(bool): 值为true时，打印更多关于变量和参数的信息，如trainable, optimize_attr等

返回：

(str): debug 字符串

抛出异常：

ValueError：当throw_on_error = true时，但没有设置任何必需的字段时，抛出ValueError。

clone(for_test=False)
""""""""""""""""""""""""""""""""""""""""""
创建一个新的、相同的Program。

有些operator，在训练和测试之间的行为是不同的，比如batch_norm。它们有一个属性is_test来控制行为。当for_test=True时，此方法将把它们的is_test属性更改为True。

- 克隆Program，该Program用于训练时，将for_test设置为False。
- 克隆Program，该Program用于测试时，将for_test设置为True。

注意:此API不会删除任何操作符。请在backward和optimization之前使用clone(for_test=True)。

**代码示例**

..  code-block:: python

  test_program = fluid.default_main_program().clone(for_test=True)
  optimizer = fluid.optimizer.Momentum(learning_rate=0.01, momentum=0.9)
  optimizer.minimize()

参数：for_test (bool) – 取值为True时，clone方法内部会把operator的属性is_test设置为true.

返回：一个新的、相同的Program.

返回类型:Program

**代码示例**

1. 克隆一个Program，示例代码如下：

..  code-block:: python

  train_program = fluid.Program()
  startup_program = fluid.Program()
  with fluid.program_guard(train_program, startup_program):
        img = fluid.layers.data(name='image', shape=[784])
        hidden = fluid.layers.fc(input=img, size=200, act='relu')
        hidden = fluid.layers.dropout(hidden, dropout_prob=0.5)
        loss = fluid.layers.cross_entropy(
                     input=fluid.layers.fc(hidden, size=10, act='softmax'),
                     label=fluid.layers.data(name='label', shape=[1], dtype='int64'))
  test_program = train_program.clone(for_test=True)
  sgd = fluid.optimizer.SGD(learning_rate=1e-3)
  with fluid.program_guard(train_program, startup_program):
        sgd.minimize(loss)    
	
2.如果分别运行train Program 和 test Program，则可以不使用clone。

..  code-block:: python

>>> import paddle.fluid as fluid
>>>
>>> def network(is_test):
>>>     img = fluid.layers.data(name='image', shape=[784])
>>>     hidden = fluid.layers.fc(input=img, size=200, act='relu')
>>>     hidden = fluid.layers.dropout(hidden, dropout_prob=0.5, is_test=is_test)
>>>     loss = fluid.layers.cross_entropy(
>>>                 input=fluid.layers.fc(hidden, size=10, act='softmax'),
>>>                 label=fluid.layers.data(name='label', shape=[1], dtype='int64'))
>>>     return loss
>>>
>>> train_program = fluid.Program()
>>> startup_program = fluid.Program()
>>> test_program = fluid.Program()
>>>
>>> with fluid.program_guard(train_program, startup_program):
>>>     with fluid.unique_name.guard():
>>>         loss = network(is_test=False)
>>>         sgd = fluid.optimizer.SGD(learning_rate=1e-3)
>>>         sgd.minimize(loss)
>>>
>>> # the test startup program is not used.
>>> with fluid.program_guard(test_program, fluid.Program()):
>>>     with fluid.unique_name.guard():
>>>         loss = network(is_test=True)

上边两个代码片段生成的Program是一样的。

static parse_from_string(binary_str)
""""""""""""""""""""""""""""""""""""""""""
反序列化protobuf，转换成program

注意:在序列化和反序列化之后，所有关于参数的信息都会丢失。

参数:	binary_str_type (str) – prootbuf二进制字符串

返回:	反序列化后的ProgramDesc

返回类型：Program

num_blocks
""""""""""""""""""""""""""""""""""""""""""
该program中的block的个数

random_seed
""""""""""""""""""""""""""""""""""""""""""

程序中随机运算符的默认随机种子。0意味着从随机设备中获取随机种子。

注意：必须在operator被添加之前设置。

global_block()
""""""""""""""""""""""""""""""""""""""""""
获取该program的第一个block。

block(index)
""""""""""""""""""""""""""""""""""""""""""
返回该program中 ，index指定的block。index类型为int

返回：index对应的block

返回类型：Block

current_block()
""""""""""""""""""""""""""""""""""""""""""
获取当前block。当前block是用来添加operators。

list_vars()
""""""""""""""""""""""""""""""""""""""""""
获取当前program中所有变量。返回值是一个可迭代对象（iterable object)。

返回：generator 会yield每个Program中的变量

返回类型：iterable
	

.. _cn_api_fluid_name_scope:

name_scope
>>>>>>>>>>>>

paddle.fluid.name_scope(*args, **kwds)
""""""""""""""""""""""""""""""""""""""""""

为operators生成层次名称前缀

注意： 这个函数只能用于调试和可视化。不要将其用于分析，比如graph/program转换。

.. _cn_api_fluid_global_scope:

global_scope
>>>>>>>>>>>>

paddle.fluid.global_scope()
""""""""""""""""""""""""""""""""""""""""""

获取全局/默认作用域实例。很多api使用默认global_scope，例如Executor.run

返回：全局/默认作用域实例

返回类型：Scope

.. _cn_api_fluid_scope_guard:

scope_guard
>>>>>>>>>>>>

paddle.fluid.scope_guard(*args, **kwds)()
""""""""""""""""""""""""""""""""""""""""""

修改全局/默认作用scope,  运行时中的所有变量都将分配给新的scope。

参数：scope -新的全局/默认 scope。

**代码示例**

..  code-block:: python

>>> import paddle.fluid as fluid
>>> new_scope = fluid.Scope()
>>> with fluid.scope_guard(new_scope):
>>>     ...


.. _cn_api_fluid_memory_optimize:

memory_optimize
>>>>>>>>>>>>

paddle.fluid.memory_optimize(input_program, skip_opt_set=None, print_log=False, level=0, skip_grads=False)
""""""""""""""""""""""""""""""""""""""""""

通过重用var内存来优化内存。

注意:它不支持block中嵌套子block。

参数:
	- input_program (str) – 输入Program。
	- skip_opt_set (set) – set中的vars将不被内存优化。
	- print_log (bool) – 是否打印debug日志。
	- level (int)  如果 level=0 并且shape是完全相等，则重用。
	
返回: None


.. _cn_api_fluid_DistributeTranspilerConfig:

DistributeTranspilerConfig
>>>>>>>>>>>>

class paddle.fluid.DistributeTranspilerConfig
""""""""""""""""""""""""""""""""""""""""""

slice_var_up (bool): 使用Tensor切片保存, 默认为True

split_method (PSDispatcher): 可使用 RoundRobin 或者 HashName 

注意: 尝试选择最佳方法来达到负载均衡。

min_block_size (int): 最小数据块的大小

注意: 根据：https：//github.com/PaddlePaddle/Paddle/issues/8638#issuecomment-369912156, 当数据块大小超过2MB时，我们可以有效地使用带宽。如果你想更改它，请详细查看slice_variable函数。

.. _cn_api_fluid_LoDTensor:

LoDTensor
>>>>>>>>>>>>

class paddle.fluid.LoDTensor
""""""""""""""""""""""""""""""""""""""""""

LoDTensor是一个具有LoD信息的张量(Tensor)

np.array(lod_tensor)可以将LoDTensor转换为numpy array。lod_tensor.lod()可以获得LoD信息。
LoD是多层序列（Level of Details）的缩写，通常用于不同长度的序列。如果您不需要了解LoD信息，可以跳过下面的注解。

举例:

X 为 LoDTensor，它包含两个序列。第一个长度是2，第二个长度是3。

从Lod中可以计算出X的第一维度为5， 因为5=2+3， 说明X中有5个序列。在X中的每个序列中的每个元素有2列，因此X的shape为[5,2]。

::

	x.lod = [[2, 3]] x.data = [[1, 2], [3, 4], // seq 1

	[5, 6], [7, 8], [9, 10]] // seq 2

	x.shape = [5, 2]


LoD可以有多个level(例如，一个段落可以有多个句子，一个句子可以有多个单词)。下面的例子中，Y为LoDTensor ，lod_level为2。表示有2个序列，第一个序列的长度是2(有2个子序列)，其中第二个序列的长度是1。第一序列的两个子序列长度分别为2和2。第二个序列的子序列的长度是3。


::

	y.lod = [[2 1], [2 2 3]] y.shape = [2+2+3, ...]


.. note::

	在上面的描述中，LoD是基于长度的。在paddle内部实现中，lod是基于偏移的。因此,在内部,y.lod表示为[[0,2,3]，[0,2,4,7]](基于长度的Lod表示为为[[2-0,3-2]，[2-0,4-2,7-4]])。

	可以将LoD理解为recursive_sequence_length（递归序列长度）。此时，LoD必须是基于长度的。由于历史原因。当LoD在API中被称为lod时，它可能是基于偏移的。用户应该注意。


::

	has_valid_recursive_sequence_lengths(self: paddle.fluid.core.LoDTensor) → bool


::

	lod(self: paddle.fluid.core.LoDTensor) → List[List[int]]


::

	recursive_sequence_lengths(self: paddle.fluid.core.LoDTensor) → List[List[int]]


::

	set_lod(self: paddle.fluid.core.LoDTensor, arg0: List[List[int]]) → None


::

	set_recursive_sequence_lengths(self: paddle.fluid.core.LoDTensor, arg0: List[List[int]]) → None





.. _cn_api_fluid_WeightNormParamAttr:

WeightNormParamAttr
>>>>>>>>>>>>

class paddle.fluid.WeightNormParamAttr(dim=None, name=None, initializer=None, learning_rate=1.0, regularizer=None, trainable=True, gradient_clip=None, do_model_average=False)
""""""""""""""""""""""""""""""""""""""""""

用于取得权重范数。权重范数将权重向量的长度与其方向解耦。`Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks <https://arxiv.org/pdf/1602.07868.pdf>`_ 这篇paper中讨论了权重范数的实现

参数:

	- dim(list)——参数的名称。默认None。（ 原文错误）
	- name (str) -参数的名称。默认None。
	- initializer（initializer)——初始化参数的方法。默认None。
	- learning_rate (float)——学习率。优化时学习速率global_lr∗parameter_lr∗scheduler_factor。默认1.0。
	- regularizer (WeightDecayRegularizer) 。正则化因子。默认None。
	- trainable(bool) -参数是否可训练。默认True。
	- gradient_clip (BaseGradientClipAttr)——梯度下降裁剪（Gradient Clipping）的方法。默认None。
	- do_model_average (bool) -参数是否应该model average。默认False。

返回： empty program

**代码示例**

..  code-block:: python

	data = fluid.layers.data(name="data", shape=[3, 32, 32], dtype="float32")
	fc = fluid.layers.fc(input=data,
			     size=1000,
			     param_attr=WeightNormParamAttr(
				  dim=None,
				  name='weight_norm_param'))

