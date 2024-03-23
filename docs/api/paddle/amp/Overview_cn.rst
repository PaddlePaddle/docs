.. _cn_overview_amp:

paddle.amp
---------------------

paddle.amp 目录下包含飞桨框架支持的动态图自动混合精度(AMP)相关的 API。具体如下：

-  :ref:`AMP 相关 API <about_amp>`
-  :ref:`开启 AMP 后默认转化为 float16 计算的相关 OP <about_amp_white_list_ops>`
-  :ref:`开启 AMP 后默认使用 float32 计算的相关 OP <about_amp_black_list_ops>`

paddle.amp 目录下包含 debugging 目录， debugging 目录中存放用于算子模型精度问题定位的 API。具体如下：

-  :ref:`Debug 相关辅助类 <about_debugging>`
-  :ref:`算子调用统计相关的 API <about_amp_debugging_op_list>`
-  :ref:`模块级别精度问题定位的 API <about_amp_debugging_check_api>`

.. _about_amp:

AMP 相关 API
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`auto_cast <cn_api_paddle_amp_auto_cast>` ", "创建 AMP 上下文环境"
    " :ref:`decorate <cn_api_paddle_amp_decorate>` ", "根据选定混合精度训练模式，改写神经网络参数数据类型"
    " :ref:`GradScaler <cn_api_paddle_amp_GradScaler>` ", "控制 loss 的缩放比例"

.. _about_amp_white_list_ops:

开启 AMP 后默认转化为 float16 计算的相关 OP
:::::::::::::::::::::::

.. csv-table::
    :header: "OP 名称", "OP 功能"
    :widths: 10, 30

    "conv2d", "卷积计算"
    "matmul", "矩阵乘法"
    "matmul_v2", "矩阵乘法"
    "mul", "矩阵乘法"

.. _about_amp_black_list_ops:

开启 AMP 后默认使用 float32 计算的相关 OP
:::::::::::::::::::::::

.. csv-table::
    :header: "OP 名称", "OP 功能"
    :widths: 10, 30

    "exp", "指数运算"
    "square", "平方运算"
    "log", "对数运算"
    "mean", "取平均值"
    "sum", "求和运算"
    "cos_sim", "余弦相似度"
    "softmax", "softmax 操作"
    "softmax_with_cross_entropy", "softmax 交叉熵损失函数"
    "sigmoid_cross_entropy_with_logits", "按元素的概率误差"
    "cross_entropy", "交叉熵"
    "cross_entropy2", "交叉熵"

.. _about_debugging:

Debug 相关辅助类
::::::::::::::::::::

.. csv-table::
    :header: "类名称", "辅助类功能"
    :widths: 10, 30

    " :ref:`DebugMode <cn_api_paddle_amp_debugging_DebugMode>` ", "精度调试模式"
    " :ref:`TensorCheckerConfig <cn_api_paddle_amp_debugging_TensorCheckerConfig>` ", "精度调试配置类"

.. _about_amp_debugging_op_list:

算子调用统计相关 API
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`collect_operator_stats <cn_api_paddle_amp_debugging_collect_operator_stats>` ", "收集不同数据类型的算子调用次数"
    " :ref:`enable_operator_stats_collection <cn_api_paddle_amp_debugging_enable_operator_stats_collection>` ", "启用以收集不同数据类型的算子调用次数"
    " :ref:`disable_operator_stats_collection <cn_api_paddle_amp_debugging_disable_operator_stats_collection>` ", "禁用收集不同数据类型的算子调用次数"

.. _about_amp_debugging_check_api:

模块级别精度定位 API
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`enable_tensor_checker <cn_api_paddle_amp_debugging_enable_tensor_checker>` ", "开启模块级别的精度检查"
    " :ref:`disable_tensor_checker <cn_api_paddle_amp_debugging_disable_tensor_checker>` ", "关闭模块级别的精度检查"
    " :ref:`compare_accuracy <cn_api_paddle_amp_debugging_compare_accuracy>` ", "精度比对接口"

数值检查相关 API
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`check_layer_numerics <cn_api_paddle_amp_debugging_check_layer_numerics>` ", "输入输出数据的数值检查"
    " :ref:`check_numerics <cn_api_paddle_amp_debugging_check_numerics>` ", "调试Tensor数值"