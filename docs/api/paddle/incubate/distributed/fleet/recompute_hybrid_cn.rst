.. _cn_api_paddle_incubate_distributed_fleet_recompute_hybrid:

recompute_hybrid
-------------------------------

.. py:function:: paddle.incubate.distributed.fleet.recompute_hybrid(ctx, function, *args, **kwargs)
在混合并行场景中，重新计算中间激活以节省内存。#NODTE（shenliang03）当前的混合并行重新计算具有局限性。
#它不能处理以下几种情况：
#1。重新计算的计算输出，有不需要梯度的张量。
# 2. 前向输出张量没有梯度。通过 detach（）可以暂时解决这个问题。
# 3. 这里，我们只使用 float dtype 来区分输出张量中是否需要梯度。

参数
::::::::::::

    - **ctx** (dict) – 包括“mp_group”、“offload”和“partition”键。键“mp_group”（Group），表示活动被拆分到哪个组。键“offload”（bool，可选，默认值为 False）表示是否卸载到 cpu。键'partition'（bool，可选，默认值为 False）表示是否拆分 mp_group 中的活动。
    - **function** (paddle.nn.Layer) - 层序列的层，描述模型的前向通道的一部分，其中间激活将在前向阶段被释放以节省内存，并将在后向阶段被重新计算以进行梯度计算。
    - ***args** (Tensor) - 函数的输入（元组）。
    - ****kwargs** (Dict) - 函数的输入（字典）。

返回
:::::::::

args 和 kwargs 上的函数输出。
