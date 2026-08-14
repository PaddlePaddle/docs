.. _cn_api_paddle_distributed_new_group:

new_group
-------------------------------


.. py:function:: paddle.distributed.new_group(ranks=None, backend=None, timeout=datetime.timedelta(seconds=1800), nccl_comm_init_option=0, nccl_config=None)

创建分布式通信组。


参数
:::::::::
    - **ranks** (list，可选) - 用于新建通信组的全局 rank 列表。默认值为 None。
    - **backend** (str，可选) - 用于新建通信组的后端支持，目前仅支持 nccl。默认值为 None。
    - **timeout** (datetime.timedelta，可选) - 等待 store 相关选项的超时时间。默认值为 30 分钟。
    - **nccl_comm_init_option** (int，可选) - NCCL 通信器初始化选项。默认值为 0。
    - **nccl_config** (NCCLConfig|None，可选) - NCCL 配置。默认值为 None。


返回
:::::::::
Group：新建的通信组对象

代码示例
::::::::::::
COPY-FROM: paddle.distributed.new_group
