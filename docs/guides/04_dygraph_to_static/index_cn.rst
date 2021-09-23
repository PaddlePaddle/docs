###############
动态图转静态图
###############

动态图有诸多优点，包括易用的接口，Python风格的编程体验，友好的debug交互机制等。在动态图模式下，代码是按照我们编写的顺序依次执行。这种机制更符合Python程序员的习
惯，可以很方便地将大脑中的想法快速地转化为实际代码，也更容易调试。但在性能方面，
Python执行开销较大，与C++有一定差距。因此在工业界的许多部署场景中（如大型推荐系统、移动端）都倾向于直接使用C++来提速。

相比动态图，静态图在部署方面更具有性能的优势。静态图程序在编译执行时，先搭建模型
的神经网络结构，然后再对神经网络执行计算操作。预先搭建好的神经网络可以脱离Python依赖，在C++端被重新解析执行，而且拥有整体网络结构也能进行一些网络结构的优化。

动态图代码更易编写和debug，但在部署性能上，静态图更具优势。因此我们新增了动态图转静态图的功能，支持用户依然使用动态图编写组网代码。PaddlePaddle会对用户代码进行
分析，自动转换为静态图网络结构，兼顾了动态图易用性和静态图部署性能两方面优势。

我们在以下链接介绍PaddlePaddle动态图转静态图的各个部分：

- `基本用法 <basic_usage_cn.html>`_ : 介绍了动态图转静态图的基本使用方法

- `内部架构原理 <program_translator_cn.html>`_ ：介绍了动态图转静态图的架构原理

- `支持语法列表 <grammar_list_cn.html>`_ ：介绍了动态图转静态图支持的语法以及罗列不支持的语法写法

- `InputSpec功能介绍 <input_spec_cn.html>`_ ：介绍了动态图转静态图指定输入InputSpec的功能和用法

- `报错信息处理 <error_handling_cn.html>`_ ：介绍了动态图转静态图的报错信息处理方法

- `调试方法 <debugging_cn.html>`_ ：介绍了动态图转静态图支持的调试方法

- `预测模型导出教程 <./export_model/index_cn.html>`_ ：介绍了如何导出预测模型的详细教程


..  toctree::
    :hidden:

    basic_usage_cn.rst    
    program_translator_cn.rst
    grammar_list_cn.rst
    input_spec_cn.rst
    error_handling_cn.md
    debugging_cn.md
    export_model/index_cn.rst

