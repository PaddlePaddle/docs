############
基本概念
############

本文介绍 Fluid 中的基本概念：

- `Variable <variable.html>`_ : Variable表示变量，在Fluid中可以包含任何类型的值，在大多数情况下是一个Lod-Tensor。
- `Tensor <tensor.html>`_ : Tensor表示数据。
- `LoD-Tensor <lod_tensor.html>`_ : LoD-Tensor是Fluid中特有的概念，它在Tensor基础上附加了序列信息，支持处理变长数据。
- `Operator <operator.html>`_ : Operator表示对数据的操作。
- `Program <program.html>`_ : Program表示对计算过程的描述。
- `Executor <executor.html>`_ : Executor表示执行引擎。

..  toctree::
    :hidden:

    variable.rst
    tensor.rst
    lod_tensor.rst
    operator.rst
    program.rst
    executor.rst

