.. _user_guide_use_numpy_array_as_train_data:

##############
同步数据读取
##############

PaddlePaddle Fluid支持使用 :code:`fluid.data()` 配置数据层；
再使用 Numpy Array 或者直接使用Python创建C++的
:code:`fluid.LoDTensor` , 通过 :code:`Executor.run(feed=...)` 传给
:code:`fluid.Executor` 或 :code:`fluid.ParallelExecutor` 。

数据层配置
##########

通过 :code:`fluid.data()` 可以配置神经网络中需要的数据层。具体方法为:

.. code-block:: python

   import paddle.fluid as fluid

   image = fluid.data(name="image", shape=[None, 3, 224, 224])
   label = fluid.data(name="label", shape=[None, 1], dtype="int64")

   # use image/label as layer input
   prediction = fluid.layers.fc(input=image, size=1000, act="softmax")
   loss = fluid.layers.cross_entropy(input=prediction, label=label)
   ...

上段代码中，:code:`image` 和 :code:`label` 是通过 :code:`fluid.data`
创建的两个输入数据层。其中 :code:`image` 是 :code:`[None, 3, 224, 224]` 维度的浮点数据;
:code:`label` 是 :code:`[None, 1]` 维度的整数数据。这里需要注意的是:

1. Executor在执行的时候，会检查定义的数据层数据和feed的数据的 :code:`shape` 和 :code:`dtype` 是否一致，如果不一致，程序会报错退出。对于一些任务，在不同的轮数，数据的某些维度会变化，可以将维度的值设置为None，例如第0维会变化，可以将 :code:`shape` 设置为 :code:`[None, 3, 224, 224]` 。


2. Fluid中用来做类别标签的数据类型是 :code:`int64`，并且标签从0开始。可用数据类型请参考 :ref:`user_guide_paddle_support_data_types`。

.. _user_guide_feed_data_to_executor:

传递训练数据给执行器
####################

:code:`Executor.run` 和 :code:`ParallelExecutor.run` 都接受一个 :code:`feed` 参数。
这个参数是一个Python的字典。它的键是数据层的名字，例如上文代码中的 :code:`image`。
它的值是对应的numpy array。

例如:

.. code-block:: python

   exe = fluid.Executor(fluid.CPUPlace())
   # init Program
   exe.run(fluid.default_startup_program())
   exe.run(feed={
      "image": numpy.random.random(size=(32, 3, 224, 224)).astype('float32'),
      "label": numpy.random.random(size=(32, 1)).astype('int64')
   })

进阶使用
########

如何传入序列数据
----------------

序列数据是PaddlePaddle Fluid支持的特殊数据类型，可以使用 :code:`LoDTensor` 作为
输入数据类型。它需要用户: 1. 传入一个mini-batch需要被训练的所有数据;
2.每个序列的长度信息。
用户可以使用 :code:`fluid.create_lod_tensor` 来创建 :code:`LoDTensor` 。

传入序列信息的时候，需要设置序列嵌套深度，:code:`lod_level` 。
例如训练数据是词汇组成的句子，:code:`lod_level=1` ；训练数据是 词汇先组成了句子，
句子再组成了段落，那么 :code:`lod_level=2` 。

例如:

.. code-block:: python

   sentence = fluid.data(name="sentence", dtype="int64", shape=[None, 1], lod_level=1)

   ...

   exe.run(feed={
     "sentence": create_lod_tensor(
       data=numpy.array([1, 3, 4, 5, 3, 6, 8], dtype='int64').reshape(-1, 1),
       recursive_seq_lens=[[4, 1, 2]],
       place=fluid.CPUPlace()
     )
   })

训练数据 :code:`sentence` 包含三个样本，他们的长度分别是 :code:`4, 1, 2` 。
他们分别是 :code:`data[0:4]`， :code:`data[4:5]` 和 :code:`data[5:7]` 。

如何分别设置ParallelExecutor中每个设备的训练数据
------------------------------------------------

用户将数据传递给使用 :code:`ParallelExecutor.run(feed=...)` 时，
可以显示指定每一个训练设备(例如GPU)上的数据。
用户需要将一个列表传递给 :code:`feed` 参数，列表中的每一个元素都是一个字典。
这个字典的键是数据层的名字，值是数据层的值。

例如:

.. code-block:: python

   parallel_executor = fluid.ParallelExecutor()
   parallel_executor.run(
     feed=[
        {
          "image": numpy.random.random(size=(32, 3, 224, 224)).astype('float32'),
          "label": numpy.random.random(size=(32, 1)).astype('int64')
        },
        {
          "image": numpy.random.random(size=(16, 3, 224, 224)).astype('float32'),
          "label": numpy.random.random(size=(16, 1)).astype('int64')
        },
     ]
   )

上述代码中，GPU0会训练 32 个样本，而 GPU1训练 16 个样本。


.. _user_guide_paddle_support_data_types:

Fluid目前支持的数据类型
-----------------------

PaddlePaddle Fluid目前支持的数据类型包括:

   * float16： 部分操作支持
   * float32:  主要实数类型
   * float64:  次要实数类型，支持大部分操作
   * int32:  次要标签类型
   * int64: 主要标签类型
   * uint64: 次要标签类型
   * bool: 控制流数据类型
   * int16: 次要标签类型
   * uint8: 输入数据类型，可用于图像像素
