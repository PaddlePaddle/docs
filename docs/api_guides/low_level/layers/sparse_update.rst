.. _api_guide_sparse_update:

#####
稀疏更新
#####

Fluid 的 :ref:`cn_api_fluid_layers_embedding`  层在单机训练和分布式训练时，均可以支持“稀疏更新”，即梯度以 sparse tensor 结构存储，只保存梯度不为 0 的行。
在分布式训练中，对于较大的 embedding 层，开启稀疏更新有助于减少通信数据量，提升训练速度。

在 paddle 内部，我们用 lookup_table 来实现 embedding。下边这张图说明了 embedding 在正向和反向计算的过程：

如图所示：一个 Tensor 中有两行不为 0，正向计算的过程中，我们使用 ids 存储不为 0 的行，并使用对应的两行数据来进行计算；反向更新的过程也只更新这两行。

.. image:: ../../../images/lookup_table_training.png
   :scale: 50 %

embedding 使用例子:
---------------------

API 详细使用方法参考 :ref:`cn_api_fluid_layers_embedding` ，以下是一个简单的例子：

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

- :code:`is_sparse` ： 反向计算的时候梯度是否为 sparse tensor。如果不设置，梯度是一个 :ref:`Lod_Tensor <cn_user_guide_lod_tensor>` 。默认为 False。

- :code:`is_distributed` ： 标志是否是用在分布式的场景下。一般大规模稀疏更新（embedding 的第 0 维维度很大，比如几百万以上）才需要设置。具体可以参考大规模稀疏的 API guide  :ref:`cn_api_guide_async_training`  。默认为 False。

- API 汇总:
 - :ref:`cn_api_fluid_layers_embedding`
