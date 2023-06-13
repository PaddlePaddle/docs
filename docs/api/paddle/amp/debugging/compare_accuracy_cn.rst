.. _cn_api_amp_debugging_compare_accuracy:

compare_accuracy
-------------------------------
.. py:function:: paddle.amp.debugging.compare_accuracy(dump_path, another_dump_path, output_filename, loss_scale=1, dump_all_tensors=False)

`compare_accuracy` 是一个精度比对的工具，可以用来比较 float16 和 float32 的 log 数据。

参数
:::::::::
    - **dump_path** (str): 运行日志的路径，比如数据类型是 float32 的训练日志。
    - **another_dump_path** (str): 另一份运行日志的路径，比如数据类型是 float16 的训练日志。
    - **output_filename** (str): 输出 excel 的文件名，用于保存两份日志的比对结果。
    - **loss_scale** (float, 可选): 训练阶段的 loss_scale，默认是 1。
    - **dump_all_tensors** (bool, 可选): True 表示 dump 所有的 tensor 数据，False 表示不做处理，当前还不支持这个参数，默认是 False。


返回值
:::::::::
无

代码示例
:::::::::

COPY-FROM: paddle.amp.debugging.compare_accuracy
