.. _cn_overview_paddle:

paddle.incubate.sparse
---------------------

paddle.incubate.sparse 目录包含飞桨框架支持稀疏数据存储和计算相关的 API。具体如下：

-  :ref:`稀疏数据结构相关 <about_sparse_tensor>`
-  :ref:`数学操作 API <about_sparse_math>`
-  :ref:`NN 相关 API <about_sparse_nn>`

.. _about_sparse_tensor:

稀疏数据结构相关
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"

    " :ref:`paddle.incubate.sparse.sparse_coo_tensor <cn_api_paddle_incubate_sparse_coo_tensor>` ", "创建一个 COO 格式的 SparseTensor"
    " :ref:`paddle.incubate.sparse.sparse_csr_tensor <cn_api_paddle_incubate_sparse_csr_tensor>` ", "创建一个 CSR 格式的 SparseTensor"
    " :ref:`paddle.incubate.sparse.is_same_shape <cn_api_paddle_incubate_sparse_is_same_shape>` ", "判断两个 Tensor 的形状是否相同, 支持 DenseTensor 与 SparseTensor 相互比较"
    " :ref:`paddle.incubate.sparse.transpose <cn_api_paddle_incubate_sparse_transpose>` ", "在不改变数据的情况下改变 ``x`` 的维度顺序, 支持 COO 格式的多维 SparseTensor 以及 COO 格式的 2 维和 3 维 SparseTensor"
    " :ref:`paddle.incubate.sparse.reshape <cn_api_paddle_incubate_sparse_reshape>` ", "改变一个 SparseTensor 的形状"


.. _about_sparse_math:

数学操作相关
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"

    " :ref:`paddle.incubate.sparse.abs` ", "绝对值函数"
    " :ref:`paddle.incubate.sparse.add` ", "Sparse Tensor 逐元素相加"
    " :ref:`paddle.incubate.sparse.asin` ", "arcsine 函数"
    " :ref:`paddle.incubate.sparse.asinh` ", "反双曲正弦函数"
    " :ref:`paddle.incubate.sparse.atan` ", "反双曲正切函数"
    " :ref:`paddle.incubate.sparse.add <cn_api_paddle_incubate_sparse_add>` ", "逐元素加法"
    " :ref:`paddle.incubate.sparse.subtract <cn_api_paddle_incubate_sparse_subtract>` ", "逐元素减法"
    " :ref:`paddle.incubate.sparse.multiply <cn_api_paddle_incubate_sparse_multiply>` ", "逐元素乘法"
    " :ref:`paddle.incubate.sparse.divide <cn_api_paddle_incubate_sparse_divide>` ", "逐元素除法"


.. _about_sparse_nn:

NN 相关
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"

    " :ref:`paddle.incubate.sparse.nn.Conv3D` ", "三维卷积"
    " :ref:`paddle.incubate.sparse.nn.SubmConv3D` ", "三维的 submanifold 卷积"
    " :ref:`paddle.incubate.sparse.nn.Relu` ", "激活函数"
