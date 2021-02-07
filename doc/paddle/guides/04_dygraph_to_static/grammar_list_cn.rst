支持语法列表
==============

ProgramTranslator本质是把Python运行语法转写为PaddlePaddle静态图代码，但是Python语法的表达能力和PaddlePaddle静态图表达能力存在不同，这使得一些代码无法被转换。

本章节我们将详细讲述在动转静过程中支持转化哪些语法，不支持哪些语法，并且讲述如何改写代码能够解决语法不支持的场景。

动转静支持的语法分为以下几个大类：

控制流相关关键词
------------------

控制流指if-elif-else，while等能够控制程序语句执行顺序的关键字。PaddlePaddle静态图通过cond，while_loop API来实现条件判断和循环，如果动态图Python控制流的判断条件或循环条件依赖 PaddlePaddle Tensor，动转静后会被转化为等价的PaddlePaddle控制流接口，否则仍然使用Python控制流逻辑运行。在动转静过程中这些关键字的转化情况为：

1. if-elif-else 条件

当 ``if <条件>`` 中的条件是Tensor时，ProgramTranslator会把该if-elif-else语句转化为等价的cond API语句。否则会按普通Python if-elif-else的逻辑运行。需注意cond支持的Tensor只能是numel为1的bool Tensor，所以请使用这种Tensor进行条件判断，其他Tensor会报错。

2. while 循环

当while循环中的条件是Tensor时，ProgramTranslator会把该while语句转化为等价的while_loop API语句，否则会按普通Python while运行。需注意while循环条件中的Tensor只能是numel为1的bool Tensor，所以请使用这种Tensor进行条件判断，其他Tensor会报错。


3. for 循环

3.1 ``for _ in range(__)`` 循环

ProgramTranslator先将其转化为等价的Python while循环，然后按while循环的逻辑进行动静转换。

3.2 ``for _ in x`` 循环

当x是Python容器或迭代器，则会用普通Python逻辑运行。当x是Tensor时，会转化为循环中每次对应拿出x[0], x[1], ... 。

3.3 ``for idx, val in enumerate(x)`` 循环

当x是Python容器或迭代器，则会用普通Python逻辑运行。当x是Tensor时，idx会转化为依次0，1，...的1-D Tensor。val会转化为循环中每次对应拿出x[0], x[1], ... 。

4. break，continue

ProgramTranslator 可以支持在循环中添加break，continue语句，其底层实现原理是对于要break，continue的部分在相应时候使用cond在一定条件下跳过执行。

5. return

ProgramTranslator 支持在循环，条件判断中return结果而不需要一定在函数末尾return。也能够支持return不同长度tuple和不同类型的Tensor。其底层实现原理是对return后的部分相应使用cond在一定条件下跳过执行。


一些需要转化的运算类型
------------------------

1. +，-，*，/，**, >, <, >= , <=, == 等Python内置运算

由于静态图有重载这些基本运算符，所以这些被ProgramTranslator转化后都适用相应重载的运算符，动转静支持此类运算。

2. and，or，not 逻辑运算

Python内置and，or，not逻辑运算关键词，ProgramTranslator在语句的运算时会判断逻辑运算关键词运行的对象是否是Tensor，如果都是Tensor，我们将其转化为静态图对应的逻辑运算接口并运行。

3. 类型转化

动态图中可以直接用Python的类型转化语法来转化Tensor类型。例如x是Tensor时，float(x)可以将x的类型转化为float。ProgramTranslator在运行时判断x是否是Tensor，如果是，则在动转静时使用静态图cast接口转化相应的Tensor类型。

Python 函数相关
---------------------

1. print

如果x是Tensor，在动态图模式中print(x)可以打印x的值。在动转静过程中我们把此转化为静态图的Print接口实现，使得在静态图中也能打印。如果print的参数不是Tensor，那么我们没有把相应print语句进行转写。

2. len

如果x是Tensor，在动态图模式中len(x)可以获得x第0维度的长度。在动转静中我们把此转化为静态图shape接口，并返回shape的第0维。另外如果x是个TensorArray，那么len(x)将会使用静态图接口control_flow.array_length返回TensorArray的长度。对于其他情况，动转静时会按照普通Python len函数运行。

3. lambda 表达式

动转静允许写带有Python lambda表达式的语句，并且我们会适当改写使得返回对应结果。

4. 函数内再调用函数

对于函数内调用其他函数的情况，ProgramTranslator也会对内部的函数递归地进行动转静，这样做的好处是可以在最外层函数只需加一次装饰器即可，而不需要每个函数都加装饰器。但需要注意，动转静还不支持函数递归调用自己，详细原因请查看下文动转静无法正确运行的情况。

报错异常相关
--------------

1. assert

如果x是Tensor，在动态图中可以通过assert x来强制x为True或者非0值，在动转静中我们把此转化为静态图Assert接口支持此功能。


Python基本容器
---------------

1. list：对于一个list如果里面元素都是Tensor，那么动转静会转化其为TensorArray，静态图TensorArray可以支持append，pop，修改操作。因此ProgramTranslator在元素皆为Tensor的list中支持上面三种操作。换言之，其他list操作，比如sort无法支持。对于list中并非所有元素是Tensor的情况，ProgramTranslator会将其作为普通Python list运行。

2. dict：ProgramTranslator会将相应的dict中的Tensor添加进静态图Program，因此使用dict是动转静支持的语法。

动转静无法正确运行的情况
--------------------------

1. 改变变量的shape后调用其shape作为PaddlePaddle API参数。

具体表现比如 ``x = reshape(x, shape=shape_tensor)`` ，再使用 ``x.shape[0]`` 的值进行其他操作。这种情况会由于动态图和静态图的本质不同而使得动态图能够运行，但静态图运行失败。其原因是动态图情况下，API是直接返回运行结果，因此 ``x.shape`` 在经过reshape运算后是确定的。但是在转化为静态图后，因为静态图API只是组网，``shape_tensor`` 的值在组网时是不知道的，所以 ``reshape`` 接口组网完，静态图并不知道 ``x.shape`` 的值。PaddlePaddle静态图用-1表示未知的shape值，此时 ``x`` 的shape每个维度会被设为-1，而不是期望的值。同理，类似expand等更改shape的API，其输出Tensor再调用shape也难以进行动转静。

遇到这类情况我们建议尽量固定shape值，减少变化shape操作。

2. 多重list嵌套读写Tensor

具体表现如 ``l = [[tensor1, tensor2], [tensor3, tensor4]]`` ，因为现在动转静将元素全是Tensor的list转化为TensorArray，而PaddlePaddle的TensorArray还不支持多维数组，因此这种情况下，动转静无法正确运行。

遇到这类情况我们建议尽量用一维list，或者自己使用PaddlePaddle的create_array，array_read，array_write接口编写为TensorArray。

3. Tensor值在被装饰函数中转成numpy array进行运算

具体表现为在被装饰函数中没有返回Tensor时就使用 ``numpy.array(tensor)`` 将Tensor转化为numpy array并使用numpy接口进行运算。这种情况在动态图下因为Tensor有值是可以正常运行的，但是在静态图时由于Tensor只是组网变量，在没有运行时没有数值，因此无法进行numpy运算。

遇到这种情况我们建议在动转静的函数中尽量使用PaddlePaddle接口替代numpy接口进行运算。

4. 一个函数递归调用本身

ProgramTranslator还无法支持一个函数递归调用本身，原因是递归常常会用 ``if-else`` 构造停止递归的条件。然而这样的停止条件在静态图下只是一个 ``cond`` 组网，组网并不能在编译阶段得到递归条件决定本身组多少次，会导致函数运行时一直组网递归直至栈溢出，因此ProgramTranslator还无法支持一个函数递归调用自己本身。

遇到这种情况我们建议将代码改为非递归写法。

