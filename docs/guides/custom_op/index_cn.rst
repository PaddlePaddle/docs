#############
自定义算子
#############

介绍如何使用飞桨的自定义算子（Operator，简称 Op）机制，包括以下三类：

1. C++算子：编写方法较为简洁，不涉及框架内部概念，无需重新编译飞桨框架，以外接模块的方式使用的算子
2. C++扩展：可将 C++ 类绑定至 Python，支持调用无 Tensor 参数的 C++ 函数。不涉及框架的调度开销
3. Python 算子：使用 Python 编写实现前向（forward）和反向（backward）方法，在模型组网中使用的自定义 API

- `自定义 C++算子 <./new_cpp_op_cn.html>`_

- `自定义 C++ 扩展 <./cpp_extension_cn.html>`_

- `自定义 Python 算子 <./new_python_op_cn.html>`_


.. toctree::
   :hidden:

   new_cpp_op_cn.md
   cpp_extension_cn.md
   new_python_op_cn.md
