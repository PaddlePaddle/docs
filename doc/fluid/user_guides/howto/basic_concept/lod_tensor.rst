.. _cn_user_guide_lod_tensor:
##################
LoD-Tensor使用说明
##################

LoD(Level-of-Detail) Tensor是Fluid中特有的概念，它在Tensor基础上附加了序列信息。Fluid中可传输的数据包括：输入、输出、网络中的可学习参数，全部统一使用LoD-Tensor表示。

阅读本文档将帮助您了解 Fluid 中的 LoD-Tensor 设计思想，以便您更灵活的使用这一数据类型。

变长序列的挑战
================

大多数的深度学习框架使用Tensor表示一个mini-batch。

例如一个mini-batch中有10张图片，每幅图片大小为32x32，则这个mini-batch是一个10x32x32的 Tensor。

或者在处理NLP任务中，一个mini-batch包含N个句子，每个字都用一个D维的one-hot向量表示，假设所有句子都用相同的长度L，那这个mini-batch可以被表示为NxLxD的Tensor。

上述两个例子中序列元素都具有相同大小，但是在许多情况下，训练数据是变长序列。基于这一场景，大部分框架采取的方法是确定一个固定长度，对小于这一长度的序列数据以0填充。

在Fluid中，由于LoD-Tensor的存在，我们不要求每个mini-batch中的序列数据必须保持长度一致，因此您不需要执行填充操作，也可以满足处理NLP等具有序列要求的任务需求。

Fluid引入了一个索引数据结构（LoD）来将张量分割成序列。


LoD 索引
===========

为了更好的理解LoD的概念，本节提供了几个例子供您参考：

**句子组成的 mini-batch**

假设一个mini-batch中有3个句子，每个句子中分别包含3个、1个和2个单词。我们可以用(3+1+2)xD维Tensor 加上一些索引信息来表示这个mini-batch:

.. code-block :: text

  3       1   2
  | | |   |   | |

上述表示中，每一个 :code:`|` 代表一个D维的词向量，数字3，1，2构成了 1-level LoD。

**递归序列**

让我们来看另一个2-level LoD-Tensor的例子：假设存在一个mini-batch中包含3个句子、1个句子和2个句子的文章，每个句子都由不同数量的单词组成，则这个mini-batch的样式可以看作：

.. code-block:: text


  3            1 2
  3   2  4     1 2  3
  ||| || ||||  | || |||


表示的LoD信息为：

.. code-block:: text

  [[3，1，2]/*level=0*/，[3，2，4，1，2，3]/*level=1*/]


**视频的mini-batch**

在视觉任务中，时常需要处理视频和图像这些元素是高维的对象，假设现存的一个mini-batch包含3个视频，分别有3个，1个和2个帧，每个帧都具有相同大小：640x480，则这个mini-batch可以被表示为：

.. code-block:: text

  3     1  2
  口口口 口 口口


最底层tensor大小为（3+1+2）x640x480，每一个 :code:`口` 表示一个640x480的图像

**图像的mini-batch**

在传统的情况下，比如有N个固定大小的图像的mini-batch，LoD-Tensor表示为:

.. code-block:: text

  1 1 1 1     1
  口口口口 ... 口

在这种情况下，我们不会因为索引值都为1而忽略信息，仅仅把LoD-Tensor看作是一个普通的张量:

.. code-block:: text

  口口口口 ... 口

**模型参数**

模型参数只是一个普通的张量，在Fluid中它们被表示为一个0-level LoD-Tensor。

LoDTensor的偏移表示
=====================

为了快速访问基本序列，Fluid提供了一种偏移表示的方法——保存序列的开始和结束元素，而不是保存长度。

在上述例子中，您可以计算基本元素的长度：

.. code-block:: text

  3 2 4 1 2 3

将其转换为偏移表示：

.. code-block:: text

  0  3  5   9   10  12   15
     =  =   =   =   =    =
     3  2+3 4+5 1+9 2+10 3+12

所以我们知道第一个句子是从单词0到单词3，第二个句子是从单词3到单词5。

类似的，LoD的顶层长度

.. code-block:: text

  3 1 2

可以被转化成偏移形式：

.. code-block:: text

  0 3 4   6
    = =   =
    3 3+1 4+2

因此该LoD-Tensor的偏移表示为：

.. code-block:: text

  0       3    4      6
    3 5 9   10   12 15


LoD-Tensor
=============
一个LoD-Tensor可以被看作是一个树的结构，树叶是基本的序列元素，树枝作为基本元素的标识。

在 Fluid 中 LoD-Tensor 的序列信息有两种表述形式：原始长度和偏移量。在 Paddle 内部采用偏移量的形式表述 LoD-Tensor，以获得更快的序列访问速度；在 python API中采用原始长度的形式表述 LoD-Tensor 方便用户理解和计算，并将原始长度称为： :code:`recursive_sequence_lengths` 。

以上文提到的一个2-level LoD-Tensor为例：

.. code-block:: text

  3           1  2
  3   2  4    1  2  3
  ||| || |||| |  || |||

- 以偏移量表示此 LoD-Tensor:[ [0,3,4,6] , [0,3,5,9,10,12,15] ]，
- 以原始长度表达此 Lod-Tensor：recursive_sequence_lengths=[ [3-0 , 4-3 , 6-4] , [3-0 , 5-3 , 9-5 , 10-9 , 12-10 , 15-12] ]。


以文字序列为例： [3,1,2] 可以表示这个mini-batch中有3篇文章，每篇文章分别有3、1、2个句子，[3,2,4,1,2,3] 表示每个句子中分别含有3、2、4、1、2、3个字。

recursive_seq_lens 是一个双层嵌套列表，也就是列表的列表，最外层列表的size表示嵌套的层数，也就是lod-level的大小；内部的每个列表，对应表示每个lod-level下，每个元素的大小。

下面三段代码分别介绍如何创建一个LoD-Tensor，如何将LoD-Tensor转换成Tensor，如何将Tensor转换成LoD-Tensor：

* 创建 LoD-Tensor

.. code-block:: python

  #创建lod-tensor
  import paddle.fluid as fluid
  import numpy as np
  
  a = fluid.create_lod_tensor(np.array([[1],[1],[1],
                                    [1],[1],
                                    [1],[1],[1],[1],
                                    [1],
                                    [1],[1],
                                    [1],[1],[1]]).astype('int64') ,
                            [[3,1,2] , [3,2,4,1,2,3]],
                            fluid.CPUPlace())
  
  #查看lod-tensor嵌套层数
  print (len(a.recursive_sequence_lengths()))
  # output：2

  #查看最基础元素个数
  print (sum(a.recursive_sequence_lengths()[-1]))
  # output:15 (3+2+4+1+2+3=15)

* LoD-Tensor 转 Tensor

.. code-block:: python

  import paddle.fluid as fluid
  import numpy as np

  # 创建一个 LoD-Tensor
  a = fluid.create_lod_tensor(np.array([[1.1], [2.2],[3.3],[4.4]]).astype('float32'), [[1,3]], fluid.CPUPlace())

  def LodTensor_to_Tensor(lod_tensor):
    # 获取 LoD-Tensor 的 lod 信息
    lod = lod_tensor.lod()
    # 转换成 array
    array = np.array(lod_tensor)
    new_array = []
    # 依照原LoD-Tensor的层级信息，转换成Tensor
    for i in range(len(lod[0]) - 1):
        new_array.append(array[lod[0][i]:lod[0][i + 1]])
    return new_array

  new_array = LodTensor_to_Tensor(a)

  # 输出结果
  print(new_array)

* Tensor 转 LoD-Tensor

.. code-block:: python

  import paddle.fluid as fluid
  import numpy as np

  def to_lodtensor(data, place):
    # 存储Tensor的长度作为LoD信息
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    # 对待转换的 Tensor 降维
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    # 为 Tensor 数据添加lod信息
    res = fluid.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res

  # new_array 为上段代码中转换的Tensor
  lod_tensor = to_lodtensor(new_array,fluid.CPUPlace())

  # 输出 LoD 信息
  print("The LoD of the result: {}.".format(lod_tensor.lod()))

  # 检验与原Tensor数据是否一致
  print("The array : {}.".format(np.array(lod_tensor)))




代码示例
===========

本节代码将根据指定的级别y-lod，扩充输入变量x。本例综合了LoD-Tensor的多个重要概念，跟随代码实现，您将：

-  直观理解Fluid中 :code:`fluid.layers.sequence_expand` 的实现过程
-  掌握如何在Fluid中创建LoD-Tensor
-  学习如何打印LoDTensor内容


  
**定义计算过程**

layers.sequence_expand通过获取 y 的 lod 值对 x 的数据进行扩充，关于 :code:`fluid.layers.sequence_expand` 的功能说明，请先阅读 :ref:`cn_api_fluid_layers_sequence_expand` 。

序列扩充代码实现：

.. code-block:: python

  x = fluid.layers.data(name='x', shape=[1], dtype='float32', lod_level=1)
  y = fluid.layers.data(name='y', shape=[1], dtype='float32', lod_level=2)
  out = fluid.layers.sequence_expand(x=x, y=y, ref_level=0)

*说明*：输出LoD-Tensor的维度仅与传入的真实数据维度有关，在定义网络结构阶段为x、y设置的shape值，仅作为占位，并不影响结果。

**创建Executor**

.. code-block:: python

  place = fluid.CPUPlace()
  exe = fluid.Executor(place)
  exe.run(fluid.default_startup_program())

**准备数据**

这里我们调用 :code:`fluid.create_lod_tensor` 创建 :code:`sequence_expand` 的输入数据，通过定义 y_d 的 LoD 值，对 x_d 进行扩充。其中，输出值只与 y_d 的 LoD 值有关，y_d 的 data 值在这里并不参与计算，维度上与LoD[-1]一致即可。

:code:`fluid.create_lod_tensor()` 的使用说明请参考 :ref:`cn_api_fluid_create_lod_tensor` 。

实现代码如下：

.. code-block:: python

  x_d = fluid.create_lod_tensor(np.array([[1.1],[2.2],[3.3],[4.4]]).astype('float32'), [[1,3]], place)
  y_d = fluid.create_lod_tensor(np.array([[1.1],[1.1],[1.1],[1.1],[1.1],[1.1]]).astype('float32'), [[1,3], [2,1,2,1]],place)


**执行运算**

在Fluid中，LoD>1的Tensor与其他类型的数据一样，使用 :code:`feed` 定义数据传入顺序。此外，由于输出results是带有LoD信息的Tensor，需在exe.run( )中添加 :code:`return_numpy=False` 参数，获得LoD-Tensor的输出结果。

.. code-block:: python

  results = exe.run(fluid.default_main_program(),
                    feed={'x':x_d, 'y': y_d },
                    fetch_list=[out],return_numpy=False)

**查看LodTensor结果**

由于LoDTensor的特殊属性，无法直接print查看内容，常用操作时将LoD-Tensor作为网络的输出fetch出来，然后执行 numpy.array(lod_tensor), 就能转成numpy array：

.. code-block:: python

  np.array(results[0])

输出结果为：

.. code-block:: text

  array([[1.1],[2.2],[3.3],[4.4],[2.2],[3.3],[4.4],[2.2],[3.3],[4.4]])

**查看序列长度**

可以通过查看序列长度得到 LoDTensor 的递归序列长度：

.. code-block:: python

    results[0].recursive_sequence_lengths()
    
输出结果为：

.. code-block:: text
    
    [[1L, 3L, 3L, 3L]]

**完整代码**

您可以运行下列完整代码，观察输出结果：

.. code-block:: python
    
    #加载库
    import paddle
    import paddle.fluid as fluid
    import numpy as np
    #定义前向计算
    x = fluid.layers.data(name='x', shape=[1], dtype='float32', lod_level=1)
    y = fluid.layers.data(name='y', shape=[1], dtype='float32', lod_level=2)
    out = fluid.layers.sequence_expand(x=x, y=y, ref_level=0)
    #定义运算场所
    place = fluid.CPUPlace()
    #创建执行器
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    #创建LoDTensor
    x_d = fluid.create_lod_tensor(np.array([[1.1], [2.2],[3.3],[4.4]]).astype('float32'), [[1,3]], place)
    y_d = fluid.create_lod_tensor(np.array([[1.1],[1.1],[1.1],[1.1],[1.1],[1.1]]).astype('float32'), [[1,3], [1,2,1,2]], place)
    #开始计算
    results = exe.run(fluid.default_main_program(),
                      feed={'x':x_d, 'y': y_d },
                      fetch_list=[out],return_numpy=False)
    #输出执行结果
    print("The data of the result: {}.".format(np.array(results[0])))
    #输出 result 的序列长度
    print("The recursive sequence lengths of the result: {}.".format(results[0].recursive_sequence_lengths()))
    #输出 result 的 LoD
    print("The LoD of the result: {}.".format(results[0].lod()))


总结
========

至此，相信您已经基本掌握了LoD-Tensor的概念，尝试修改上述代码中的 x_d 与 y_d，观察输出结果，有助于您更好的理解这一灵活的结构。

更多LoDTensor的模型应用，可以参考新手入门中的 `词向量 <../../../beginners_guide/basics/word2vec/index.html>`_ 、`个性化推荐 <../../../beginners_guide/basics/recommender_system/index.html>`_、`情感分析 <../../../beginners_guide/basics/understand_sentiment/index.html>`_ 等指导教程。

更高阶的应用案例，请参考 `模型库 <../../../user_guides/models/index_cn.html>`_ 中的相关内容。
