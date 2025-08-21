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

    " :ref:`split <cn_api_paddle_compat_split>` ", "允许非整除块大小输入的 Tensor 轴向切分"


.. _about_compat_class:

PyTorch 兼容模块
::::::::::::::::::::

.. csv-table::
    :header: "类名称", "类功能"
    :widths: 10, 30

    " :ref:`Unfold <cn_api_paddle_compat_Unfold>` ", "允许 Tensor 输入的 ``paddle.nn.Unfold`` 兼容版本"
