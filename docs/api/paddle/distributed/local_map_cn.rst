.. _cn_api_paddle_distributed_local_map:

local_map
-------------------------------

.. py:function:: paddle.distributed.local_map(func, out_placements, in_placements=None, process_mesh=None, reshard_inputs=False)

local_map 是一个函数装饰器，允许用户将分布式张量（DTensor）传递给为普通张量（Tensor）编写的函数。它通过提取分布式张量的本地分量，调用目标函数，并根据 out_placements 将输出包装为分布式张量来实现这一功能，通过自动处理张量转换，使得用户可以像编写单卡代码一样实现这些局部操作。


参数
:::::::::

    - **func** (Callable) - 要应用于分布式张量本地分片的函数。
    - **out_placements** (list[list[dist.Placement]]) - 指定输出张量的分布策略。外层列表长度必须与函数输出数量匹配，每个内层列表描述对应输出张量的分布方式。对于非张量输出必须设为 None。
    - **in_placements** (list[list[dist.Placement]]，可选) - 指定输入张量的要求分布。如果指定，每个内层列表描述对应输入张量的分布要求。外层列表长度必须与输入张量数量匹配。对于不具有分布式属性的输入应设为 None，默认为 None，表示输入张量不需要分布或从输入张量推断分布。
    - **process_mesh** (Optional[ProcessMesh]，可选) - 计算设备网格。所有分布式张量必须位于同一个 process_mesh 上。如未指定，默认为 None，表示从输入张量推断 process_mesh。
    - **reshard_inputs** (bool，可选) - 提示当输入分布式张量的分布方式与要求的 in_placements 不匹配时，是否自动 reshard。默认 False，表示不自动 reshard。

返回
:::::::::

    返回一个可调用对象（Callable），该对象将 func 应用于输入分布式张量的每个本地分片，并根据返回值构造新的分布式张量。

代码示例
:::::::::

COPY-FROM: paddle.distributed.local_map
