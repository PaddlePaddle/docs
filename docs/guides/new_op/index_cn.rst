#############
自定义算子
#############

本部分将指导您如何使用飞桨的自定义算子（Operator，简称Op）机制，包括以下两类：

1. C++算子：编写方法较为简洁，不涉及框架内部概念，无需重新编译飞桨框架，以外接模块的方式使用的算子
2. Python算子：使用Python编写实现前向（forward）和反向（backward）方法，在模型组网中使用的自定义API

- `自定义C++算子 <./new_custom_op_cn.html>`_

- `自定义Python算子 <./new_python_op_cn.html>`_

- `Kernel Primitives API <./kernel_primitive_api/index_cn.html>`_ : 介绍 PaddlePaddle 为加快算子开发提供的 Block 级 CUDA 函数。

.. toctree::
   :hidden:

   op_notes_cn.md
   new_custom_op_cn.md
   new_python_op_cn.md
   kernel_primitive_api/index_cn.rst
