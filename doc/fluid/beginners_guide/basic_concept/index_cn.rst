############
基本概念
############

本文介绍飞桨核心框架中的基本概念：

- `编程指南 <./programming_guide/programming_guide.html>`_ : 介绍飞桨的基本概念和使用方法。
- `Variable <variable.html>`_ : Variable表示变量，在飞桨中可以包含任何类型的值，在大多数情况下是一个Lod-Tensor。
- `Tensor <tensor.html>`_ : Tensor表示数据。
- `LoD-Tensor <lod_tensor.html>`_ : LoD-Tensor是飞桨的高级特性，它在Tensor基础上附加了序列信息，支持处理变长数据。
- `Operator <operator.html>`_ : Operator表示对数据的操作。
- `Program <program.html>`_ : Program表示对计算过程的描述。
- `Executor <executor.html>`_ : Executor表示执行引擎。
- `Broadcasting <broadcasting.html>`_ : Paddle对广播支持的说明。
..  toctree::
    :hidden:

    programming_guide/programming_guide.md
    variable.rst
    tensor.rst
    lod_tensor.rst
    operator.rst
    program.rst
    executor.rst
    broadcasting.rst
