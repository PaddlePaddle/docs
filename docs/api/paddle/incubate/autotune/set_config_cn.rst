.. _cn_api_paddle_incubate_autotune_set_config:

set_config
---------------------

.. py:function:: paddle.incubate.autotune.set_config(config=None)

配置 kernel、layout 和 dataloader 的自动调优功能。

1. kernel：当开启自动调优，将使用穷举搜索的方法在调优的迭代区间内为算子选择最佳算法，并将其缓存下来。调优参数如下：

    - **enable** (bool) - 是否开启 kernel 的调优功能。
    - **tuning_range** (list) - 自动调优开始和结束的迭代序号。默认值：[1, 10]。

2. layout：当开启自动调优，将根据设备和数据类型确定最优的数据布局如 NCHW 或者 NHWC。当原始的 layout 设置并非最优时，将会自动进行 layout 的转换以提升模型的性能。**该功能当前仅支持动态图模式**。调优参数如下：

    - **enable** (bool) - 是否开启 layout 的调优功能。

3. dataloader：当开启自动调优，将自动选择最佳的 num_workers 替换原始的配置。调优参数如下：

    - **enable** (bool) - 是否开启 dataloader 的调优功能。

参数
:::::::::

    - **config** (dict|str|None，可选) - 自动调整的配置。如果它是字典，则键是调优类型，值是调优参数构成的字典。如果它是字符串，则表示 json 文件路径，将根据 json 文件内容配置调优参数。默认值：None，kernel、layout 和 dataloader 的自动调优功能将全被开启。

代码示例
::::::::::

COPY-FROM: paddle.incubate.autotune.set_config
