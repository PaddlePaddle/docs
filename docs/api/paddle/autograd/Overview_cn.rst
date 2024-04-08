.. _cn_overview_autograd:

paddle.autograd
---------------------

paddle.autograd 目录下包含飞桨框架支持的自动微分相关的 API 和类。具体如下：

-  :ref:`自动微分相关 API <about_autograd>`
-  :ref:`自动微分相关类 <about_autograd_class>`

.. _about_autograd:

自动微分相关 API
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`backward <cn_api_paddle_autograd_backward>` ", "计算给定的 Tensors 的反向梯度"
    " :ref:`hessian <cn_api_paddle_autograd_hessian>` ", "计算因变量 ``ys`` 对 自变量 ``xs`` 的海森矩阵"
    " :ref:`jacobian <cn_api_paddle_autograd_jacobian>` ", "计算因变量 ``ys`` 对 自变量 ``xs`` 的雅可比矩阵"
    " :ref:`saved_tensors_hooks <cn_api_paddle_autograd_saved_tensors_hooks>` ", "用于动态图中为保存的 Tensor 注册一对 pack / unpack hook"


.. _about_autograd_class:
自动微分相关类
::::::::::::::::::::

.. csv-table::
    :header: "类名称", "类功能"
    :widths: 10, 30

    " :ref:`PyLayer <cn_api_paddle_autograd_PyLayer>` ", "通过创建 ``PyLayer`` 子类的方式实现 Python 端自定义算子"
    " :ref:`PyLayerContext <cn_api_paddle_autograd_PyLayerContext>` ", "``PyLayerContext`` 对象能够辅助 :ref:`cn_api_paddle_autograd_PyLayer` 实现某些功能"
