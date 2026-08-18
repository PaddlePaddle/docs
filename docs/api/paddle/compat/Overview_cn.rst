.. _cn_overview_compat:

paddle.compat
---------------------

paddle.compat 目录下包含飞桨框架支持的 PyTorch 兼容函数与模块接口

.. _about_compat_funcs:

PyTorch 兼容函数
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`allclose <cn_api_paddle_compat_allclose>` ", "逐元素近似比较并返回 Python bool"
    " :ref:`equal <cn_api_paddle_compat_equal>` ", "比较两个 Tensor 是否完全相等并返回 Python bool"
    " :ref:`max <cn_api_paddle_compat_max>` ", "包含 ``amax``、同时返回 values 及 indices 的轴向最大值、``maximum`` 三种功能"
    " :ref:`median <cn_api_paddle_compat_median>` ", "兼容版中位数，支持 dim/keepdim/out 签名"
    " :ref:`min <cn_api_paddle_compat_min>` ", "包含 ``amin``、同时返回 values 及 indices 的轴向最小值、``minimum`` 三种功能"
    " :ref:`nanmedian <cn_api_paddle_compat_nanmedian>` ", "忽略 NaN 的兼容版中位数，支持 dim/keepdim/out 签名"
    " :ref:`slogdet <cn_api_paddle_compat_slogdet>` ", "slogdet 函数"
    " :ref:`sort <cn_api_paddle_compat_sort>` ", "同时返回 values 及 indices 的排序"
    " :ref:`split <cn_api_paddle_compat_split>` ", "允许非整除块大小输入的 Tensor 轴向切分"
    " :ref:`unique <cn_api_paddle_compat_unique>` ", "PyTorch 对齐的唯一元素统计接口"


.. _about_compat_nn:

PyTorch 兼容的 nn 模块
::::::::::::::::::::::::::::::

.. csv-table::
    :header: "类名称", "类功能"
    :widths: 10, 30

    " :ref:`AvgPool1d <cn_api_paddle_compat_nn_AvgPool1d>` ", "PyTorch 对齐的一维平均池化层"
    " :ref:`AvgPool2d <cn_api_paddle_compat_nn_AvgPool2d>` ", "PyTorch 对齐的二维平均池化层"
    " :ref:`BatchNorm1d <cn_api_paddle_compat_nn_BatchNorm1d>` ", "PyTorch 对齐的一维批归一化层"
    " :ref:`BatchNorm2d <cn_api_paddle_compat_nn_BatchNorm2d>` ", "PyTorch 对齐的二维批归一化层"
    " :ref:`BatchNorm3d <cn_api_paddle_compat_nn_BatchNorm3d>` ", "PyTorch 对齐的三维批归一化层"
    " :ref:`Linear <cn_api_paddle_compat_nn_Linear>` ", "PyTorch 对齐的 ``paddle.nn.Linear`` 兼容版本"
    " :ref:`MultiheadAttention <cn_api_paddle_compat_nn_MultiheadAttention>` ", "PyTorch 对齐的多头注意力层"
    " :ref:`Unfold <cn_api_paddle_compat_nn_Unfold>` ", "允许 Tensor 输入的 ``paddle.nn.Unfold`` 兼容版本"


.. _about_compat_nn_functional:

PyTorch 兼容的 nn.functional 模块
::::::::::::::::::::::::::::::::::::::::

.. csv-table::
    :header: "函数名称", "函数功能"
    :widths: 10, 30

    " :ref:`linear <cn_api_paddle_compat_nn_functional_linear>` ", "PyTorch 对齐的 ``paddle.nn.functional.linear`` 兼容版本"
    " :ref:`log_softmax <cn_api_paddle_compat_nn_functional_log_softmax>` ", "PyTorch 对齐的 log-softmax 函数"
    " :ref:`pad <cn_api_paddle_compat_nn_functional_pad>` ", "从最后一维度开始进行 padding、padding 正确兼容转换的填充函数"
    " :ref:`scaled_dot_product_attention <cn_api_paddle_compat_nn_functional_scaled_dot_product_attention>` ", "PyTorch 布局的缩放点积注意力"
    " :ref:`softmax <cn_api_paddle_compat_nn_functional_softmax>` ", "softmax 函数"
    " :ref:`unfold <cn_api_paddle_compat_nn_functional_unfold>` ", "PyTorch 参数名的 im2col 操作"

.. _about_compat_distribution:

PyTorch 兼容的概率分布
::::::::::::::::::::::::::::::

.. csv-table::
    :header: "类名称", "类功能"
    :widths: 10, 30

    " :ref:`Categorical <cn_api_paddle_compat_distributions_Categorical>` ", "支持 probs 或 logits 的类别分布"
