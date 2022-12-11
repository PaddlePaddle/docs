.. _cn_overview_paddle_sparse:

paddle.sparse
---------------------

paddle.sparse 目录包含飞桨框架支持稀疏数据存储和计算相关的 API。具体如下：

-  :ref:`稀疏 Tensor 创建 <about_sparse_tensor>`
-  :ref:`稀疏 Tensor 运算 <about_sparse_math>`
-  :ref:`稀疏组网类 <about_sparse_nn>`
-  :ref:`稀疏组网类的函数式 API <about_sparse_nn_functional>`

.. _about_sparse_tensor:

稀疏 Tensor 创建
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"

    " :ref:`paddle.sparse.sparse_coo_tensor <cn_api_paddle_sparse_coo_tensor>` ", "创建一个 COO 格式的 SparseTensor"
    " :ref:`paddle.sparse.sparse_csr_tensor <cn_api_paddle_sparse_csr_tensor>` ", "创建一个 CSR 格式的 SparseTensor"

.. _about_sparse_math:

稀疏 Tensor 运算
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"

    " :ref:`paddle.sparse.sin <cn_api_paddle_sparse_sin>` ", "对稀疏 Tensor 逐元素求正弦"
    " :ref:`paddle.sparse.tan <cn_api_paddle_sparse_tan>` ", "对稀疏 Tensor 逐元素求正切"
    " :ref:`paddle.sparse.asin <cn_api_paddle_sparse_asin>` ", "对稀疏 Tensor 逐元素求反正弦"
    " :ref:`paddle.sparse.atan <cn_api_paddle_sparse_atan>` ", "对稀疏 Tensor 逐元素求反正切"
    " :ref:`paddle.sparse.sinh <cn_api_paddle_sparse_sinh>` ", "对稀疏 Tensor 逐元素求双曲正弦"
    " :ref:`paddle.sparse.tanh <cn_api_paddle_sparse_tanh>` ", "对稀疏 Tensor 逐元素求双曲正切"
    " :ref:`paddle.sparse.asinh <cn_api_paddle_sparse_asinh>` ", "对稀疏 Tensor 逐元素求反双曲正弦"
    " :ref:`paddle.sparse.atanh <cn_api_paddle_sparse_atanh>` ", "对稀疏 Tensor 逐元素求反双曲正切"
    " :ref:`paddle.sparse.sqrt <cn_api_paddle_sparse_sqrt>` ", "对稀疏 Tensor 逐元素求算数平方根"
    " :ref:`paddle.sparse.square <cn_api_paddle_sparse_square>` ", "对稀疏 Tensor 逐元素求平方"
    " :ref:`paddle.sparse.log1p <cn_api_paddle_sparse_log1p>` ", "对稀疏 Tensor 逐元素计算 ln(x+1)"
    " :ref:`paddle.sparse.abs <cn_api_paddle_sparse_abs>` ", "对稀疏 Tensor 逐元素求绝对值"
    " :ref:`paddle.sparse.pow <cn_api_paddle_sparse_pow>` ", "对稀疏 Tensor 逐元素计算 x 的 y 次幂"
    " :ref:`paddle.sparse.cast <cn_api_paddle_sparse_cast>` ", "对稀疏 Tensor 逐元素转换类型"
    " :ref:`paddle.sparse.neg <cn_api_paddle_sparse_neg>` ", "对稀疏 Tensor 逐元素计算相反数"
    " :ref:`paddle.sparse.deg2rad <cn_api_paddle_sparse_deg2rad>` ", "对稀疏 Tensor 逐元素从度转换为弧度"
    " :ref:`paddle.sparse.rad2deg <cn_api_paddle_sparse_rad2deg>` ", "对稀疏 Tensor 逐元素从弧度转换为度"
    " :ref:`paddle.sparse.expm1 <cn_api_paddle_sparse_expm1>` ", "对稀疏 Tensor 逐元素进行以自然数 e 为底的指数运算并减 1"
    " :ref:`paddle.sparse.mv <cn_api_paddle_sparse_mv>` ", "稀疏矩阵乘向量，第一个参数为稀疏矩阵，第二个参数为稠密向量"
    " :ref:`paddle.sparse.matmul <cn_api_paddle_sparse_matmul>` ", "稀疏矩阵乘，第一个参数为稀疏矩阵，第二个参数为稠密矩阵或者稀疏矩阵"
    " :ref:`paddle.sparse.addmm <cn_api_paddle_sparse_addmm>` ", "稀疏矩阵乘与加法的组合运算"
    " :ref:`paddle.sparse.masked_matmul <cn_api_paddle_sparse_masked_matmul>` ", "稀疏矩阵乘，第一、二个参数均为稠密矩阵，返回值为稀疏矩阵"
    " :ref:`paddle.sparse.add <cn_api_paddle_sparse_add>` ", "对稀疏 Tensor 逐元素相加"
    " :ref:`paddle.sparse.subtract <cn_api_paddle_sparse_subtract>` ", "对稀疏 Tensor 逐元素相减"
    " :ref:`paddle.sparse.multiply <cn_api_paddle_sparse_multiply>` ", "对稀疏 Tensor 逐元素相乘"
    " :ref:`paddle.sparse.divide <cn_api_paddle_sparse_divide>` ", "对稀疏 Tensor 逐元素相除"
    " :ref:`paddle.sparse.is_same_shape <cn_api_paddle_sparse_is_same_shape>` ", "判断两个稀疏 Tensor 或稠密 Tensor 的 shape 是否一致"
    " :ref:`paddle.sparse.reshape <cn_api_paddle_sparse_reshape>` ", "改变一个 SparseTensor 的形状"
    " :ref:`paddle.sparse.coalesce<cn_api_paddle_sparse_coalesce>` ", "对 SparseCooTensor 进行排序并合并"
    " :ref:`paddle.sparse.transpose <cn_api_paddle_sparse_transpose>` ", "在不改变数据的情况下改变 ``x`` 的维度顺序, 支持 COO 格式的多维 SparseTensor 以及 COO 格式的 2 维和 3 维 SparseTensor"

.. _about_sparse_nn:

稀疏组网类
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"

    " :ref:`paddle.sparse.nn.ReLU <cn_api_paddle_sparse_nn_ReLU>` ", "激活层"
    " :ref:`paddle.sparse.nn.ReLU6 <cn_api_paddle_sparse_nn_ReLU6>` ", "激活层"
    " :ref:`paddle.sparse.nn.LeakyReLU <cn_api_paddle_sparse_nn_LeakyReLU>` ", "激活层"
    " :ref:`paddle.sparse.nn.Softmax <cn_api_paddle_sparse_nn_Softmax>` ", "激活层"
    " :ref:`paddle.sparse.nn.Conv3D <cn_api_paddle_sparse_nn_Conv3D>` ", "三维卷积层"
    " :ref:`paddle.sparse.nn.SubmConv3D <cn_api_paddle_sparse_nn_SubmConv3D>` ", "子流形三维卷积层"
    " :ref:`paddle.sparse.nn.BatchNorm<cn_api_paddle_sparse_nn_BatchNorm>` ", " Batch Normalization 层"
    " :ref:`paddle.sparse.nn.SyncBatchNorm<cn_api_paddle_sparse_nn_SyncBatchNorm>` ", " Synchronized Batch Normalization 层"
    " :ref:`paddle.sparse.nn.MaxPool3D<cn_api_paddle_sparse_nn_MaxPool3D>` ", "三维最大池化层"

.. _about_sparse_nn_functional:

稀疏组网类函数式 API
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"

    " :ref:`paddle.sparse.nn.functional.relu <cn_api_paddle_sparse_nn_functional_relu>` ", "激活函数"
    " :ref:`paddle.sparse.nn.functional.relu6 <cn_api_paddle_sparse_nn_functional_relu6>` ", "激活函数"
    " :ref:`paddle.sparse.nn.functional.leaky_relu <cn_api_paddle_sparse_nn_functional_leaky_relu>` ", "激活函数"
    " :ref:`paddle.sparse.nn.functional.softmax <cn_api_paddle_sparse_nn_functional_softmax>` ", "激活函数"
    " :ref:`paddle.sparse.nn.functional.attention <cn_api_paddle_sparse_nn_functional_attention>` ", "稀疏 attention 函数"
    " :ref:`paddle.sparse.nn.functional.conv3d <cn_api_paddle_sparse_nn_functional_conv3d>` ", "三维卷积函数"
    " :ref:`paddle.sparse.nn.functional.subm_conv3d <cn_api_paddle_sparse_nn_functional_subm_conv3d>` ", "子流形三维卷积函数"
    " :ref:`paddle.sparse.nn.functional.max_pool3d <cn_api_paddle_sparse_nn_functional_max_pool3d>` ", "三维最大池化函数"
