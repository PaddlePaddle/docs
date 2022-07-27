.. _api_guide_large_scale_sparse_feature_training_en:

###################################################
Training of Models with Large Scale Sparse Features
###################################################


Model configuration and training
==================================

Embedding is widely used in various network structures, especially in the text processing related models. In some scenarios, such as recommendation systems or search engines, the number of feature ids of embedding may be very large. When the feature id reaches a certain amount, the embedding parameter will become very large, which will bring two problems:

1. The single-machine memory cannot be trained because it cannot store such huge embedding parameters.
2. The normal training mode needs to synchronize the complete parameters for each iteration. If the parameters are too large, the communication will become very slow, which will affect the training speed.

Fluid supports the training of hundreds of millions of large-scale sparse features embedding. The embedding parameter is only saved on the parameter server. Through the parameter prefetch and gradient sparse update method, the communication volume is greatly reduced and the communication speed is improved.

This function is only valid for distributed training and cannot be used in single node training. It need to work with `sparse update <../layers/sparse_update_en.html>`_.

How to use: When configuring embedding, add the parameters :code:`is_distributed=True` and :code:`is_sparse=True`.
Parameter :code:`dict_size` defines the total number of ids in the data. The id can be any value in the int64 range. As long as the total number of ids is less than or equal to dict_size, it can be supported. So before you configure, you need to estimate the total number of feature ids in the data.

.. code-block:: python

  emb = fluid.layers.embedding(
      is_distributed=True,
      input=input,
      size=[dict_size, embedding_width],
      is_sparse=True,
      is_distributed=True)


Model storage and inference
===============================

When the number of features reaches 100 billion, the parameters are very large, and the single machine can't hold such huge amount of parameters. As the result, the model is stored and loaded differently from the normal mode:

1. In normal mode, parameters are saved and loaded on the trainer side;
2. In distributed mode, all the parameters are saved and loaded on the pserver side. Each pserver only saves and loads the parameters corresponding to the pserver itself.
