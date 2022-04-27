.. _cn_api_incubate_autotune_set_config:

set_config
---------------------

.. py:function:: paddle.incubate.autotune.set_config(config=None)

配置kernel、layout和dataloader的自动调优功能。

1. kernel：当开启自动调优，将使用穷举搜索的方法在调优的迭代区间内为算子选择最佳算法，并将其缓存下来。调优参数如下：

    - **enable** (bool) - 是否开启kernel的调优功能。
    - **tuning_range** (list) - 自动调优开始和结束的迭代序号。默认值：[1, 10]。

2. layout：当开启自动调优，将根据设备和数据类型确定最优的数据布局如NCHW或者NHWC。当原始的layout设置并非最优时，将会自动进行layout的转换以提升模型的性能。调优参数如下：

    - **enable** (bool) - 是否开启layout的调优功能。

3. dataloader：当开启自动调优，将自动选择最佳的num_workers替换原始的配置。调优参数如下：

    - **enable** (bool) - 是否开启dataloader的调优功能。

参数
:::::::::

    - **config** (dict|str|None，可选) - 自动调整的配置。如果它是字典，则键是调优类型，值是调优参数构成的字典。如果它是字符串，则表示json文件路径，将根据json 文件内容配置调优参数。默认值：None，kernel、layout和dataloader的自动调优功能将全被开启。

代码示例
::::::::::

COPY-FROM: paddle.incubate.autotune.set_config:auto-tuning
