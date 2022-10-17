.. _cn_overview_paddle:

paddle.incubate.sparse
---------------------

paddle.incubate.sparse 目录包含飞桨框架支持稀疏数据存储和计算相关的 API。具体如下：

-  :ref:`稀疏 Tensor 创建 <about_sparse_tensor>`
-  :ref:`稀疏 Tensor 运算 <about_sparse_math>`
-  :ref:`稀疏组网类 <about_sparse_nn>`
-  :ref:`稀疏组网类的函数式 API <about_sparse_nn_functional>`

.. _about_sparse_tensor:

稀疏 Tensor 创建
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"

    " :ref:`paddle.incubate.sparse.sparse_coo_tensor <cn_api_paddle_incubate_sparse_coo_tensor>` ", "创建一个 COO 格式的稀疏 Tensor"
    " :ref:`paddle.incubate.sparse.sparse_csr_tensor <cn_api_paddle_incubate_sparse_csr_tensor>` ", "创建一个 CSR 格式的稀疏 Tensor"

.. _about_sparse_math:

稀疏 Tensor 运算
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"

    " :ref:`paddle.incubate.sparse.sin <cn_api_paddle_incubate_sparse_sin>` ", "对稀疏 Tensor 逐元素求正弦"
    " :ref:`paddle.incubate.sparse.tan <cn_api_paddle_incubate_sparse_tan>` ", "对稀疏 Tensor 逐元素求正切"
    " :ref:`paddle.incubate.sparse.asin <cn_api_paddle_incubate_sparse_asin>` ", "对稀疏 Tensor 逐元素求反正弦"
    " :ref:`paddle.incubate.sparse.atan <cn_api_paddle_incubate_sparse_atan>` ", "对稀疏 Tensor 逐元素求反正切"
    " :ref:`paddle.incubate.sparse.sinh <cn_api_paddle_incubate_sparse_sinh>` ", "对稀疏 Tensor 逐元素求双曲正弦"
    " :ref:`paddle.incubate.sparse.tanh <cn_api_paddle_incubate_sparse_tanh>` ", "对稀疏 Tensor 逐元素求双曲正切"
    " :ref:`paddle.incubate.sparse.asinh <cn_api_paddle_incubate_sparse_asinh>` ", "对稀疏 Tensor 逐元素求反双曲正弦"
    " :ref:`paddle.incubate.sparse.atanh <cn_api_paddle_incubate_sparse_atanh>` ", "对稀疏 Tensor 逐元素求反双曲正切"
    " :ref:`paddle.incubate.sparse.sqrt <cn_api_paddle_incubate_sparse_sqrt>` ", "对稀疏 Tensor 逐元素求算数平方根"
    " :ref:`paddle.incubate.sparse.square <cn_api_paddle_incubate_sparse_square>` ", "对稀疏 Tensor 逐元素求平方"
    " :ref:`paddle.incubate.sparse.log1p <cn_api_paddle_incubate_sparse_log1p>` ", "对稀疏 Tensor 逐元素计算 ln(x+1)"
    " :ref:`paddle.incubate.sparse.abs <cn_api_paddle_incubate_sparse_abs>` ", "对稀疏 Tensor 逐元素求绝对值"
    " :ref:`paddle.incubate.sparse.pow <cn_api_paddle_incubate_sparse_pow>` ", "对稀疏 Tensor 逐元素计算 x 的 y 次幂"
    " :ref:`paddle.incubate.sparse.cast <cn_api_paddle_incubate_sparse_cast>` ", "对稀疏 Tensor 逐元素转换类型"
    " :ref:`paddle.incubate.sparse.neg <cn_api_paddle_incubate_sparse_neg>` ", "对稀疏 Tensor 逐元素计算相反数"
    " :ref:`paddle.incubate.sparse.deg2rad <cn_api_paddle_incubate_sparse_deg2rad>` ", "对稀疏 Tensor 逐元素从度转换为弧度"
    " :ref:`paddle.incubate.sparse.rad2deg <cn_api_paddle_incubate_sparse_rad2deg>` ", "对稀疏 Tensor 逐元素从弧度转换为度"
    " :ref:`paddle.incubate.sparse.expm1 <cn_api_paddle_incubate_sparse_expm1>` ", "对稀疏 Tensor 逐元素进行以自然数 e 为底的指数运算并减 1"
    " :ref:`paddle.incubate.sparse.mv <cn_api_paddle_incubate_sparse_mv>` ", "稀疏矩阵乘向量，第一个参数为稀疏矩阵，第二个参数为稠密向量"
    " :ref:`paddle.incubate.sparse.matmul <cn_api_paddle_incubate_sparse_matmul>` ", "稀疏矩阵乘，第一个参数为稀疏矩阵，第二个参数为稠密矩阵或者稀疏矩阵"
    " :ref:`paddle.incubate.sparse.addmm <cn_api_paddle_incubate_sparse_addmm>` ", "稀疏矩阵乘与加法的组合运算"
    " :ref:`paddle.incubate.sparse.masked_matmul <cn_api_paddle_incubate_sparse_masked_matmul>` ", "稀疏矩阵乘，第一、二个参数均为稠密矩阵，返回值为稀疏矩阵"
    " :ref:`paddle.incubate.sparse.add <cn_api_paddle_incubate_sparse_add>` ", "对稀疏 Tensor 逐元素相加"
    " :ref:`paddle.incubate.sparse.subtract <cn_api_paddle_incubate_sparse_subtract>` ", "对稀疏 Tensor 逐元素相减"
    " :ref:`paddle.incubate.sparse.multiply <cn_api_paddle_incubate_sparse_multiply>` ", "对稀疏 Tensor 逐元素相乘"
    " :ref:`paddle.incubate.sparse.divide <cn_api_paddle_incubate_sparse_divide>` ", "对稀疏 Tensor 逐元素相除"
    " :ref:`paddle.incubate.sparse.is_same_shape <cn_api_paddle_incubate_sparse_is_same_shape>` ", "判断两个稀疏 Tensor 或稠密 Tensor 的 shape 是否一致"

.. _about_sparse_nn:

稀疏组网类
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"

    " :ref:`paddle.incubate.sparse.nn.ReLU <cn_api_paddle_incubate_sparse_nn_ReLU>` ", "激活层"
    " :ref:`paddle.incubate.sparse.nn.ReLU6 <cn_api_paddle_incubate_sparse_nn_ReLU6>` ", "激活层"
    " :ref:`paddle.incubate.sparse.nn.LeakyReLU <cn_api_paddle_incubate_sparse_nn_LeakyReLU>` ", "激活层"
    " :ref:`paddle.incubate.sparse.nn.Softmax <cn_api_paddle_incubate_sparse_nn_Softmax>` ", "激活层"
    " :ref:`paddle.incubate.sparse.nn.Conv3D <cn_api_paddle_incubate_sparse_nn_Conv3D>` ", "三维卷积层"

.. _about_sparse_nn_functional:

稀疏组网类函数式 API
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"

    " :ref:`paddle.incubate.sparse.nn.functional.relu <cn_api_paddle_incubate_sparse_nn_functional_relu>` ", "激活函数"
    " :ref:`paddle.incubate.sparse.nn.functional.relu6 <cn_api_paddle_incubate_sparse_nn_functional_relu6>` ", "激活函数"
    " :ref:`paddle.incubate.sparse.nn.functional.leaky_relu <cn_api_paddle_incubate_sparse_nn_functional_leaky_relu>` ", "激活函数"
    " :ref:`paddle.incubate.sparse.nn.functional.softmax <cn_api_paddle_incubate_sparse_nn_functional_softmax>` ", "激活函数"
    " :ref:`paddle.incubate.sparse.nn.functional.attention <cn_api_paddle_incubate_sparse_nn_functional_attention>` ", "稀疏 attention 函数"
