.. _cn_api_paddle_incubate_distributed_utils_io_dist_save_save:

save
-------------------------------

.. py:function:: paddle.incubate.distributed.utils.io.dist_save.save(state_dict, path, **configs)

在分布式和单卡环境中将状态字典保存到指定路径。

.. note::
        现在支持保存 Layer/Optimizer 的 ``state_dict``, Tensor 和包含 Tensor、Program 的嵌套结构。

.. note::
        与 ``paddle.jit.save`` 不同的是, 由于 ``paddle.save`` 的保存结果为单个文件,因此无需通过添加后缀来区分多个保存的文件。
        ``paddle.save`` 的 ``path`` 参数将直接用作保存的文件名，而不是前缀。
        为了统一保存的文件名格式, 我们推荐使用 paddle 的标准后缀:
        1. 对于 ``Layer.state_dict`` , 推荐使用 ``.pdparams`` ;
        2. 对于 ``Optimizer.state_dict`` , 推荐使用 ``.pdopt`` .
        具体示例请参考 API 代码示例。

参数
:::::::::
    - **obj** (Object) : 要保存的对象。
    - **path** (str|BytesIO) : 保存的对象的路径/缓冲区。如果保存在当前目录中，则输入路径字符串将用作文件名。
    - **protocol** (int, 可选): pickle 模块的协议版本必须大于 1 且小于 5。默认值为 4。
    - ****configs** (dict, 可选): 可选的关键字参数。目前支持以下选项：
            
            1. use_binary_format(bool):
               在 paddle.save 中使用。当保存的对象是静态图形变量时, 可以指定 ``use_binary_for_var``。
               如果为 True , 则在保存单个静态图变量时, 以 c++ 二进制格式保存文件;否则, 请将其保存为 pickle 格式。
               默认值为 False.

            2. gather_to(int|list|tuple|None):
               指定要保存的全局进程。默认值为 None.
               None 表示分布式保存不收集到单个卡上。
            
            3. state_type(str):
               值可以是 'params' 或 'opt'，指定保存参数或优化器状态。
            
            4. max_grouped_size(str|int):
               限制对象组在一段时间内传输的最大大小（位数）。
               如果是字符串, 格式必须为 num+'G/M/K'，例如 3G、2K、10M 等。默认值为 3G。


返回
:::::::::
    None


代码示例
::::::::::

COPY-FROM: paddle.incubate.distributed.utils.io.dist_save.save
