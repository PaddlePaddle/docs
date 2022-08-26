.. _cn_overview_paddle:

paddle.incubate.sparse
---------------------

paddle.incubate.sparse 目录包含飞桨框架支持稀疏数据存储和计算相关的API。具体如下：

-  :ref:`稀疏数据结构相关 <about_sparse_tensor>`
-  :ref:`数学操作API <about_sparse_math>`
-  :ref:`NN相关API <about_sparse_nn>`

.. _about_sparse_tensor:

稀疏数据结构相关
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    
    " :ref:`paddle.incubate.sparse.sparse_coo_tensor <cn_api_paddle_incubate_sparse_coo_tensor>` ", "构造COO格式的SparseTensor"
    " :ref:`paddle.incubate.sparse.sparse_csr_tensor <cn_api_paddle_incubate_sparse_csr_tensor>` ", "构造CSR格式的SparseTensor"


.. _about_sparse_math:

数学操作相关
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    
    " :ref:`paddle.incubate.sparse.abs` ", "绝对值函数"
    " :ref:`paddle.incubate.sparse.add` ", "Sparse Tensor逐元素相加"
    " :ref:`paddle.incubate.sparse.asin` ", "arcsine函数"
    " :ref:`paddle.incubate.sparse.asinh` ", "反双曲正弦函数"
    " :ref:`paddle.incubate.sparse.atan` ", "反双曲正切函数"


.. _about_sparse_nn:

NN相关
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    
    " :ref:`paddle.incubate.sparse.nn.Conv3D` ", "三维卷积"
    " :ref:`paddle.incubate.sparse.nn.SubmConv3D` ", "三维的submanifold卷积"
    " :ref:`paddle.incubate.sparse.nn.Relu` ", "激活函数"
