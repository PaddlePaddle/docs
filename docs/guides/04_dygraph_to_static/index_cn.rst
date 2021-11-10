###############
动态图转静态图
###############

动态图在接口易用性，交互式调试等方面具有诸多优势，但在工业界的许多部署场景中（如大型推荐系统、移动端）Python执行开销较大，与C++有一定的差距，静态图部署更具优势。

PaddlePaddle 在2.0版本之后，正式支持动态图转静态图（@to_static）的功能，对动态图代码进行智能化分析，自动转换为静态图网络结构，兼顾了动态图易用性和静态图部署性能两方面的优势。

如下将详细地介绍动静转换的各个模块内容：

- `基础接口用法 <basic_usage_cn.html>`_ : 介绍了动静转换 @to_static 的基本用法

- `语法支持列表 <grammar_list_cn.html>`_ ：介绍了动静转换功能已支持的语法概况

- `预测模型导出 <./export_model/index_cn.html>`_ ：介绍了导出动态图预测模型的详细教程

- `常见案例解析 <./case_analysis_cn.html>`_ : 介绍使用 @to_static 时常见的问题和案例解析

- `报错调试经验 <debugging_cn.html>`_ ：介绍了动静转换 @to_static 的调试方法和经验



..  toctree::
    :hidden:

    basic_usage_cn.rst    
    grammar_list_cn.md
    export_model_cn.md
    case_analysis_cn.md
    debugging_cn.md

