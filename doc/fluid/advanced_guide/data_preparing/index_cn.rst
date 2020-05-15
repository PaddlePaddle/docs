########
准备数据
########

本章详细介绍了如何为神经网络提供数据，包括数据的前期处理与后期的同步、异步读取。

由于静态图与动态图在执行机制上存在差异，这两种模式在数据读取上也略有差别，此处将分别进行介绍：

    - `静态图数据准备 <../data_preparing/static_mode/index_cn.html>`_：介绍静态图模式下的同步异步数据读取方法

    - `动态图数据准备 <../data_preparing/imperative_mode/index_cn.html>`_ ：介绍动态图模式下的同步异步数据读取方法

..  toctree::
    :hidden:

    static_mode/index_cn.rst
    imperative_mode/index_cn.rst
