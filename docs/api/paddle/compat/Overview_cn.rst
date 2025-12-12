.. _cn_overview_compat:

paddle.compat
---------------------

paddle.compat 目录下包含飞桨框架支持的 PyTorch 兼容函数与模块接口

.. _about_compat_proxy:

PyTorch 兼容代理
::::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`disable_torch_proxy <cn_api_paddle_compat_disable_torch_proxy>` ", "禁用 PyTorch 代理"
    " :ref:`enable_torch_proxy <cn_api_paddle_compat_enable_torch_proxy>` ", "启用 PyTorch 代理"
    " :ref:`extend_torch_proxy_blocked_modules <cn_api_paddle_compat_extend_torch_proxy_blocked_modules>` ", "将模块添加到 PyTorch 代理阻止列表中"
    " :ref:`use_torch_proxy_guard <cn_api_paddle_compat_use_torch_proxy_guard>` ", "用于临时启用或禁用 PyTorch 代理的上下文管理器"


.. _about_compat_funcs:

PyTorch 兼容函数
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`max <cn_api_paddle_compat_max>` ", "包含 `amax`、同时返回 values 及 indices 的轴向最大值、`maximum` 三种功能"
    " :ref:`median <cn_api_paddle_compat_median>` ", "兼容版中位数，支持 dim/keepdim/out 签名"
    " :ref:`min <cn_api_paddle_compat_min>` ", "包含 `amin`、同时返回 values 及 indices 的轴向最小值、`minimum` 三种功能"
    " :ref:`nanmedian <cn_api_paddle_compat_nanmedian>` ", "忽略 NaN 的兼容版中位数，支持 dim/keepdim/out 签名"
    " :ref:`slogdet <cn_api_paddle_compat_slogdet>` ", "slogdet 函数"
    " :ref:`sort <cn_api_paddle_compat_sort>` ", "同时返回 values 及 indices 的排序"
    " :ref:`split <cn_api_paddle_compat_split>` ", "允许非整除块大小输入的 Tensor 轴向切分"


.. _about_compat_nn:

PyTorch 兼容的 nn 模块
::::::::::::::::::::::::::::::

.. csv-table::
    :header: "类名称", "类功能"
    :widths: 10, 30

    " :ref:`Linear <cn_api_paddle_compat_nn_Linear>` ", "PyTorch 对齐的 ``paddle.nn.Linear`` 兼容版本"
    " :ref:`Unfold <cn_api_paddle_compat_nn_Unfold>` ", "允许 Tensor 输入的 ``paddle.nn.Unfold`` 兼容版本"


.. _about_compat_nn_functional:

PyTorch 兼容的 nn.functional 模块
::::::::::::::::::::::::::::::::::::::::

.. csv-table::
    :header: "函数名称", "函数功能"
    :widths: 10, 30

    " :ref:`linear <cn_api_paddle_compat_nn_functional_linear>` ", "PyTorch 对齐的 ``paddle.nn.functional.linear`` 兼容版本"
    " :ref:`pad <cn_api_paddle_compat_nn_functional_pad>` ", "从最后一维度开始进行 padding、padding 正确兼容转换的填充函数"
    " :ref:`softmax <cn_api_paddle_compat_nn_functional_softmax>` ", "softmax 函数"
