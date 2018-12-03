.. _api_guide_sparse_update:

#####
稀疏更新
#####

Fluid的 :ref:`api_fluid_layers_embedding`  层在单机训练和分布式训练时，均可以支持“稀疏更新”，即梯度以sparse tensor 结构存储，只保存梯度不为0的行。
在分布式训练中，对于较大的embedding层，开启稀疏更新有助于减少通信数据量，提升训练速度

<<<<<<< HEAD
在paddle内部，我们用lookup_table来实现embedding。下边这张图说明了embedding在正向和反向计算的过程：
=======
embedding输入参数
---------------------

embedding需要输入(input)，形状(size)，是否需要稀疏更新(is_sparse)，是否使用分布式table(is_distributed)，是否padding输出(padding_idx)，参数属性(param_attr)，数据类型(dtype)来决定如何计算。

- input:

  input是一个Fluid的Variable, 其内容为需要查询的id向量。
- size:

  size为lookup table的shape，必须为两维。以NLP应用为例，第0维一般为词典的大小，第1维一般为每个词对应向量的大小。
- is_sparse:

  反向计算的时候梯度是否为 `sparse tensor <https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/design/modules/selected_rows.md>`_  。如果不设置，梯度是一个 `LodTensor <https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/design/concepts/lod_tensor.md>`_  。默认为False。
- is_distributed:

  标志是否用在分布式的场景下。一般大规模稀疏更新（embedding的第0维维度很大，比如几百万以上）才需要设置。具体可以参考大规模稀疏的API guide  :ref:`api_guide_async_training`  。默认为False。
- padding_idx:

  标志需要set为0的id的值。不设置时对结果没有影响。默认为None。
- param_attr:

  设置参数的属性。可以有两种设置方式：

  #. 如果设置为一个string，即为参数的名字
  #. 可以使用 :ref:`api_fluid_paramattr` 设置更多的属性。

  默认为None。不设置时，框架将设置创建具有唯一名字的默认属性参数。
- dtype:

  标志数据的具体类型，如float或者double等。默认为float32。
>>>>>>> 2f03e19ebd6e9ee44927a9455191bb7e58b3b7c1

.. image:: ../../../../images/lookup_table_training.png
   :scale: 50 %

embedding使用例子:
---------------------

API详细使用方法参考 :ref:`api_fluid_layers_embedding` ，以下是一个简单的例子：

.. code-block:: python

   DICT_SIZE = 10000 * 10
   EMBED_SIZE = 64
   IS_SPARSE = False
   def word_emb(word, dict_size=DICT_SIZE, embed_size=EMBED_SIZE):
       embed = fluid.layers.embedding(
           input=word,
           size=[dict_size, embed_size],
           dtype='float32',
           param_attr=fluid.ParamAttr(
               initializer=fluid.initializer.Normal(scale=1/math.sqrt(dict_size))),
           is_sparse=IS_SPARSE,
           is_distributed=False)
       return embed

以上参数中：

- :code:`is_sparse` ： 反向计算的时候梯度是否为sparse tensor。如果不设置，梯度是一个 `LodTensor <https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/user_guides/howto/prepare_data/lod_tensor.md>`_  。默认为False。

- :code:`is_distributed` ： 标志是否是用在分布式的场景下。一般大规模稀疏更新（embedding的第0维维度很大，比如几百万以上）才需要设置。具体可以参考大规模稀疏的API guide  :ref:`api_guide_async_training`  。默认为False。

- API汇总:
 - :ref:`api_fluid_layers_embedding`
