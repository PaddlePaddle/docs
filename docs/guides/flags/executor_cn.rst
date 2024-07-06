
执行器
==================


FLAGS_print_sub_graph_dir
*******************************************
(始于 1.2.0)

该 flag 用于调试。如果程序中转换图的某些子图失去连接，则结果可能会出错。我们可以将这些断开连接的子图打印到该 flag 指定的文件中。如果禁用则设为 empty。

取值范围
---------------
String 型，缺省值为 empty ("")。

示例
-------
FLAGS_print_sub_graph_dir="./sub_graphs.txt" - 将断开连接的子图打印到"./sub_graphs.txt"。


FLAGS_use_ngraph
*******************************************
(始于 1.4.0)

在预测或训练过程中，可以通过该选项选择使用英特尔 nGraph（https://github.com/NervanaSystems/ngraph）引擎。它将在英特尔 Xeon CPU 上获得很大的性能提升。

取值范围
---------------
Bool 型，缺省值为 False。

示例
-------
FLAGS_use_ngraph=True - 开启使用 nGraph 运行。

注意
-------
英特尔 nGraph 目前仅在少数模型中支持。我们只验证了[ResNet-50]（https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/README_ngraph.md）的训练和预测。
