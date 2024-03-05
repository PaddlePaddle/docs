.. _cn_overview_linalg:

paddle.linalg
---------------------

paddle.linalg 目录下包含飞桨框架支持的线性代数相关 API。具体如下：

-  :ref:`矩阵属性相关 API <about_matrix_property>`
-  :ref:`矩阵计算相关 API <about_matrix_functions>`
-  :ref:`矩阵分解相关 API <about_matrix_decompositions>`
-  :ref:`线性方程求解相关 API <about_solvers>`


.. _about_matrix_property:

矩阵属性相关 API
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`paddle.linalg.det <cn_api_paddle_linalg_det>` ", "计算方阵的行列式"
    " :ref:`paddle.linalg.slogdet <cn_api_paddle_linalg_slogdet>` ", "计算方阵行列式的符号、绝对值的自然对数"
    " :ref:`paddle.linalg.cond <cn_api_paddle_linalg_cond>` ", "根据矩阵的范数，来计算矩阵的条件数"
    " :ref:`paddle.linalg.norm <cn_api_paddle_linalg_norm>` ", "计算矩阵范数或向量范数"
    " :ref:`paddle.linalg.matrix_rank <cn_api_paddle_linalg_matrix_rank>` ", "计算矩阵的秩"


.. _about_matrix_functions:

矩阵计算相关 API
:::::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`paddle.linalg.multi_dot <cn_api_paddle_linalg_multi_dot>` ", "2 个或更多矩阵的乘法，会自动选择计算量最少的乘法顺序"
    " :ref:`paddle.linalg.matrix_power <cn_api_paddle_linalg_matrix_power>` ", "计算方阵的 n 次幂"
    " :ref:`paddle.linalg.inv <cn_api_paddle_linalg_inv>` ", "计算方阵的逆矩阵"
    " :ref:`paddle.linalg.pinv <cn_api_paddle_linalg_pinv>` ", "计算矩阵的广义逆"
    " :ref:`paddle.linalg.cov <cn_api_paddle_linalg_cov>` ", "计算矩阵的协方差矩阵"
    " :ref:`paddle.linalg.matrix_exp <cn_api_paddle_linalg_matrix_exp>` ", "计算方阵的矩阵指数"


.. _about_matrix_decompositions:

矩阵分解相关 API
:::::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`paddle.linalg.eig <cn_api_paddle_linalg_eig>` ", "计算一般方阵的特征值与特征向量"
    " :ref:`paddle.linalg.eigvals <cn_api_paddle_linalg_eigvals>` ", "计算一般方阵的特征值"
    " :ref:`paddle.linalg.eigh <cn_api_paddle_linalg_eigh>` ", "计算厄米特矩阵或者实数对称矩阵的特征值和特征向量"
    " :ref:`paddle.linalg.eigvalsh <cn_api_paddle_linalg_eigvalsh>` ", "计算厄米特矩阵或者实数对称矩阵的特征值"
    " :ref:`paddle.linalg.cholesky <cn_api_paddle_linalg_cholesky>` ", "计算一个实数对称正定矩阵的 Cholesky 分解"
    " :ref:`paddle.linalg.svd <cn_api_paddle_linalg_svd>` ", "计算矩阵的奇异值分解"
    " :ref:`paddle.linalg.pca_lowrank <cn_api_paddle_linalg_pca_lowrank>` ", "对矩阵进行线性主成分分析"
    " :ref:`paddle.linalg.qr <cn_api_paddle_linalg_qr>` ", "计算矩阵的正交三角分解（也称 QR 分解）"
    " :ref:`paddle.linalg.lu <cn_api_paddle_linalg_lu>` ", "计算矩阵的 LU 分解"
    " :ref:`paddle.linalg.lu_unpack <cn_api_paddle_linalg_lu_unpack>` ", "对矩阵的 LU 分解结果进行展开得到各单独矩阵"
    " :ref:`paddle.linalg.householder_product <cn_api_paddle_linalg_householder_product>` ", "计算 Householder 矩阵乘积的前 n 列(输入矩阵为 `[*,m,n]` )"

.. _about_solvers:

线性方程求解相关 API
:::::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`paddle.linalg.lstsq <cn_api_paddle_linalg_lstsq>` ", "求解线性方程组的最小二乘问题"
    " :ref:`paddle.linalg.solve <cn_api_paddle_linalg_solve>` ", "计算具有唯一解的线性方程组，方程左边为方阵，右边为矩阵"
    " :ref:`paddle.linalg.triangular_solve <cn_api_paddle_linalg_triangular_solve>` ", "计算具有唯一解的线性方程组，方程左边为上(下)三角方阵，右边为矩阵"
    " :ref:`paddle.linalg.cholesky_solve <cn_api_paddle_linalg_cholesky_solve>` ", "通过 Cholesky 分解矩阵，计算具有唯一解的线性方程组"
