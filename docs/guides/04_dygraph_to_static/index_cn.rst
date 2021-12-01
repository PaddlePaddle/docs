###############
动态图转静态图
###############

动态图在接口易用性，交互式调试等方面具有诸多优势，但在工业界的许多部署场景中（如大型推荐系统、移动端）Python执行开销较大，与C++有一定的差距，静态图部署更具优势。

PaddlePaddle 在2.0版本之后，正式支持动态图转静态图（@to_static）的功能，对动态图代码进行智能化分析，自动转换为静态图网络结构，兼顾了动态图易用性和静态图部署性能两方面的优势。

如下将详细地介绍动静转换的各个模块内容：

- `使用样例 <basic_usage_cn.html>`_ : 介绍了动静转换 @to_static 的基本用法

- `转换原理 <principle_cn..html>`_ ：介绍了动静转换的内部原理

- `支持语法 <grammar_list_cn.html>`_ ：介绍了动静转换功能已支持的语法概况

- `案例解析 <./case_analysis_cn.html>`_ : 介绍使用 @to_static 时常见的问题和案例解析

- `报错调试 <debugging_cn.html>`_ ：介绍了动静转换 @to_static 的调试方法和经验



..  toctree::
    :hidden:

    basic_usage_cn.rst    
    principle_cn.md
    grammar_list_cn.md
    case_analysis_cn.md
    debugging_cn.md

