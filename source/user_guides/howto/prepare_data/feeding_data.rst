
.. _user_guide_use_numpy_array_as_train_data:

###########################
使用Numpy Array作为训练数据
###########################

PaddlePaddle Fluid支持使用 :ref:`api_fluid_layers_data` 配置数据层；
再使用 Numpy Array 或者直接使用Python创建C++的
:ref:`api_guide_lod_tensor` , 通过 :code:`Executor.run(feed=...)` 传给
:ref:`api_guide_executor` 或 :ref:`api_guide_parallel_executor` 。

数据层配置
##########

通过 :ref:`api_fluid_layers_data` 可以配置神经网络中需要的数据层。具体方法为:

.. code-block:: python

   import paddle.fluid as fluid

   image = fluid.layers.data(name="image", shape=[3, 224, 224])
   label = fluid.layers.data(name="label", shape=[1], dtype="int64")

   # use image/label as layer input
   prediction = fluid.layers.fc(input=image, size=1000, act="softmax")
   loss = fluid.layers.cross_entropy(input=prediction, label=label)
   ...

上段代码中，:code:`image` 和 :code:`label` 是通过 :code:`fluid.layers.data`
创建的两个输入数据层。其中 :code:`image` 是 :code:`[3, 224, 224]` 维度的浮点数据;
:code:`data` 是 :code:`[1]` 维度的整数数据。这里需要注意的是:

1. Fluid中默认使用 :code:`-1` 表示 batch size 维度，默认情况下会在 :code:`shape`
的第一个维度添加 :code:`-1` 。 所以 上段代码中， 我们可以接受将一个
:code:`[32, 3, 224, 224]`的numpy array传给 :code:`image`。 如果想自定义batch size
维度的位置的话，请设置 :code:`fluid.layers.data(append_batch_size=False)` 。

2. Fluid中目前使用 :code:`int64` 表示类别标签。

传递训练数据给执行器
####################

:code:`Executor.run` 和 :code:`ParallelExecutor.run` 都接受一个 :code:`feed` 参数。
这个参数是一个Python的字典。他的键是数据层的名字，例如上文代码中的:code:`image`。
他的值是对应的numpy array。

例如:

.. code-block:: python

   exe = fluid.Executor(fluid.CPUPlace())
   exe.run(feed={
      "image": numpy.random.random(size=(32, 3, 224, 224)).astype('float32'),
      "label": numpy.random.random(size=(32, 1)).astype('int64')
   })

进阶使用
########

如何传入序列数据
----------------

序列数据是PaddlePaddle Fluid支持的特殊数据类型，可以使用 :code:`LoDTensor` 作为
输入数据类型。它需要用户传入一个mini-batch需要被训练的所有数据和每个序列的长度信息。
具体可以使用 :code:`fluid.create_lod_tensor` 来创建 :code:`LoDTensor`。

传入序列信息的时候，需要设置序列嵌套深度，:code:`lod_level`。
例如训练数据是词汇组成的句子，:code:`lod_level=1`；训练数据是 词汇先组成了句子，
句子再组成了段落，那么 :code:`lod_level=2`。

例如:

.. code-block:: python

   sentence = fluid.layers.data(name="sentence", dtype="int64", shape=[1], lod_level=1)

   ...

   exe.run(feed={
     "sentence": create_lod_tensor(
       data=numpy.array([1, 3, 4, 5, 3, 6, 8], dtype='int64').reshape(-1, 1),
       lod=[4, 1, 2],
       place=fluid.CPUPlace()
     )
   })

训练数据 :code:`sentence` 包含三个样本，他们的长度分别是 :code:`4, 1, 2`。
他们分别是 :code:`data[0:4]`， :code:`data[4:5]` 和 :code:`data[5:7]`。

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