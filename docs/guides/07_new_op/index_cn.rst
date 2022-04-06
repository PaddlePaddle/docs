#############
自定义算子
#############

本部分将指导您如何在飞桨中新增算子（Operator，简称Op），也包括一些必要的注意事项。此处的算子是一个广义的概念，包括以下几类：

1. 原生算子：遵循飞桨框架内部算子开发规范，源码合入到飞桨框架中，与框架一起编译后使用的算子
2. 外部算子：编写方法较为简洁，不涉及框架内部概念，无需重新编译飞桨框架，以外接模块的方式使用的算子
3. Python算子：使用Python编写实现前向（forward）和反向（backward）方法，在模型组网中使用的自定义API

- `原生算子开发注意事项 <./op_notes_cn.html>`_

- `自定义外部算子 <./new_custom_op_cn.html>`_

- `自定义Python算子 <./new_python_op_cn.html>`_

- `Kernel Primitives API <./kernel_primitive_api/index_cn.html>`_ : 介绍 PaddlePaddle 为加快算子开发提供的 Block 级 CUDA 函数。

.. toctree::
   :hidden:

   op_notes_cn.md
   new_custom_op_cn.md
   new_python_op_cn.md
   kernel_primitive_api/index_cn.rst
