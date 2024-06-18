.. _cn_api_paddle_DataParallel:

DataParallel
------------

.. py:class:: paddle.DataParallel(layers, strategy=None, comm_buffer_size=25, last_comm_buffer_size=1, find_unused_parameters=False)


通过数据并行模式执行动态图模型。

目前，``DataParallel`` 仅支持以多进程的方式执行动态图模型。

支持两种使用方式：

1. 使用 ``paddle.distributed.spawn`` 方法启动，例如：

 ``python demo.py`` (spawn need to be called in ``__main__`` method)

2. 使用 ``paddle.distributed.launch`` 方法启动，例如：

``python -m paddle.distributed.launch –selected_gpus=0,1 demo.py``

其中 ``demo.py`` 脚本的代码可以是下面的示例代码。

参数
::::::::::::

    - **Layer** (Layer) - 需要通过数据并行方式执行的模型。
    - **strategy** (ParallelStrategy，可选) - (deprecated) 数据并行的策略，包括并行执行的环境配置。默认为 None。
    - **comm_buffer_size** (int，可选) - 它是通信调用（如 NCCLAllReduce）时，参数梯度聚合为一组的内存大小（MB）。默认值：25。
    - **last_comm_buffer_size** （float，可选）它限制通信调用中最后一个缓冲区的内存大小（MB）。减小最后一个通信缓冲区的大小有助于提高性能。默认值：1。默认值：1
    - **find_unused_parameters** (bool，可选) 是否在模型 forward 函数的返回值的所有 Tensor 中，遍历整个向后图。对于不包括在 loss 计算中的参数，其梯度将被预先标记为 ready 状态用于后续多卡间的规约操作。请注意，模型参数的所有正向输出必须参与 loss 的计算以及后续的梯度计算。否则，将发生严重错误。请注意，将 find_unused_parameters 设置为 True 会影响计算性能，因此，如果确定所有参数都参与了 loss 计算和自动反向图的构建，请将其设置为 False。默认值：False。

返回
::::::::::::
支持数据并行的 ``Layer``。

代码示例
::::::::::::
COPY-FROM: paddle.DataParallel:dp-example

.. note::
    目前数据并行不支持 PyLayer 自定义算子。如有此类需求，推荐先使用 no_sync 接口暂停多卡通信，然后在优化器前手动实现梯度同步；具体实现过程可参考下述示例。

代码示例
::::::::::::
COPY-FROM: paddle.DataParallel:dp-pylayer-example




代码示例
::::::::::::



方法
::::::::::::
no_sync()
'''''''''
用于暂停梯度同步的上下文管理器。在 no_sync()中参数梯度只会在模型上累加；直到 with 之外的第一个 forward-backward，梯度才会被同步。

**代码示例**

COPY-FROM: paddle.DataParallel.no_sync

state_dict(destination=None, include_sublayers=True)
'''''''''

获取当前层及其子层的所有 parameters 和持久的 buffers。并将所有 parameters 和 buffers 存放在 dict 结构中。

**参数**

    - **destination** (dict，可选) - 如果提供 ``destination``，则所有参数和持久的 buffers 都将存放在 ``destination`` 中。默认值：None。
    - **include_sublayers** (bool，可选) - 如果设置为 True，则包括子层的参数和 buffers。默认值：True。

**返回**
dict，包含所有 parameters 和持久的 buffers 的 dict。

**代码示例**

COPY-FROM: paddle.DataParallel.state_dict


set_state_dict(state_dict, use_structured_name=True)
'''''''''

根据传入的 ``state_dict`` 设置 parameters 和持久的 buffers。所有 parameters 和 buffers 将由 ``state_dict`` 中的 ``Tensor`` 设置。

**参数**

    - **state_dict** (dict) - 包含所有 parameters 和可持久性 buffers 的 dict。
    - **use_structured_name** (bool，可选) - 如果设置为 True，将使用 Layer 的结构性变量名作为 dict 的 key，否则将使用 Parameter 或者 Buffer 的变量名作为 key。默认值：True。


**返回**
无

**代码示例**

COPY-FROM: paddle.DataParallel.set_state_dict
