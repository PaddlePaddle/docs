.. _cn_overview_amp:

paddle.amp
---------------------

paddle.amp 目录下包含飞桨框架支持的动态图自动混合精度(AMP)相关的API。具体如下：

-  :ref:`AMP相关API <about_amp>`
-  :ref:`开启AMP后默认转化为float16计算的相关OP <about_amp_white_list_ops>`
-  :ref:`开启AMP后默认使用float32计算的相关OP <about_amp_black_list_ops>`



.. _about_amp:

AMP相关API
::::::::::::::::::::

.. csv-table::
    :header: "API名称", "API功能"
    :widths: 10, 30

    " :ref:`auto_cast <cn_api_amp_auto_cast>` ", "创建AMP上下文环境"
    " :ref:`GradScaler <cn_api_amp_GradScaler>` ", "控制loss的缩放比例"
    
.. _about_amp_white_list_ops:

开启AMP后默认转化为float16计算的相关OP
:::::::::::::::::::::::

.. csv-table::
    :header: "OP名称", "OP功能"
    :widths: 10, 30

    "conv2d", "卷积计算"
    "matmul", "矩阵乘法"
    "matmul_v2", "矩阵乘法"
    "mul", "矩阵乘法"

.. _about_amp_black_list_ops:

开启AMP后默认使用float32计算的相关OP
:::::::::::::::::::::::

.. csv-table::
    :header: "OP名称", "OP功能"
    :widths: 10, 30

    "exp", "指数运算"
    "square", "平方运算"
    "log", "对数运算"
    "mean", "取平均值"
    "sum", "求和运算"
    "cos_sim", "余弦相似度"
    "softmax", "softmax操作"
    "softmax_with_cross_entropy", "softmax交叉熵损失函数"
    "sigmoid_cross_entropy_with_logits", "按元素的概率误差"
    "cross_entropy", "交叉熵"
    "cross_entropy2", "交叉熵"


