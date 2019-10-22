############
基本概念
############

本文介绍 Fluid 中的基本概念：

- `Tensor <tensor.html>`_ : Tensor表示数据。
- `LoD-Tensor <lod_tensor.html>`_ : LoD-Tensor是Fluid中特有的概念，它在Tensor基础上附加了序列信息，支持处理变长数据。
- `Variable <variable.html>`_ : Variable表示变量，在Fluid中可以包含任何类型的值，在大多数情况下是一个Lod_Tensor。
- `Operator <operator.html>`_ : Operator表示对数据的操作。
- `Executor <executor.html>`_ : Executor表示执行引擎。

..  toctree::
    :hidden:

    tensor.rst
    lod_tensor.rst
    variable.rst
    operator.rst
    executor.rst

