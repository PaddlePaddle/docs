.. _api_guide_sparse_update_en:

###############
Sparse update
###############

Fluid's :ref:`api_fluid_layers_embedding` layer supports both "sparse updates" in both stand-alone and distributed training, which means gradients are stored in a sparse tensor structure and only rows with gradients other than 0 are saved.
In distributed training, for larger embedding layers, turning on sparse updates helps reduce the amount of communication data and speed up training.

Inside paddle, we use lookup_table to implement embedding. The figure below illustrates the process of embedding in the forward and reverse calculations:

As shown in the figure: two rows in a Tensor are not 0. In the process of forward calculation, we use ids to store rows that are not 0, and use the corresponding two rows of data for calculation; the process of reverse update is only Update these two lines.

.. image:: ../../../../images/lookup_table_training.png
   :scale: 50 %

Embedding use example:
--------------------------

API detailed usage reference :ref:`api_fluid_layers_embedding` , the following is a simple example:

.. code-block:: python

   DICT_SIZE = 10000 * 10
   EMBED_SIZE = 64
   IS_SPARSE = False
   Def word_emb(word, dict_size=DICT_SIZE, embed_size=EMBED_SIZE):
       Embed = fluid.layers.embedding(
           Input=word,
           Size=[dict_size, embed_size],
           Dtype='float32',
           Param_attr=fluid.ParamAttr(
               Initializer=fluid.initializer.Normal(scale=1/math.sqrt(dict_size))),
           Is_sparse=IS_SPARSE,
           Is_distributed=False)
       Return embed

Among the above parameters:

- :code:`is_sparse` : Whether the gradient is a sparse tensor in the reverse calculation. If not set, the gradient is a `LodTensor <https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/user_guides/howto/prepare_data/lod_tensor.md>`_ . The default is False.

- :code:`is_distributed` : Whether the flag is used in a distributed scenario. Generally, large-scale sparse updates (the 0th dimension of embedding is very large, such as several million or more) need to be set. For details, please refer to the massive sparse API guide :ref:`api_guide_async_training`. The default is False.

- API summary:
 - :ref:`api_fluid_layers_embedding`