.. _cn_api_paddle_incubate_asp_prune_model:

prune_model
-------------------------------

.. py:function:: paddle.incubate.asp.prune_model(model, n=2, m=4, mask_algo='mask_1d', with_mask=True)


使用 mask_algo 指定的掩码生成函数修剪 model 中支持 ASP 的子层参数。使用 with_mask 来控制模型训练和推理，如果 with_mask 是 True ，还有修剪参数相关的 ASP 掩码变量，如果是 False，仅仅裁剪参数。

.. note::
    - 在静态图模式下，使用 `with_mask` 调用函数时，需要先调用 OptimizerWithSparsityGuarantee.minimize 和 exe.run(startup_program) 来成功获取掩码变量。通常情况下训练时（已调用 OptimizerWithSparsityGuarantee.minimize）设置 `with_mask` 为 True。而仅进行推理时，设置 `with_mask` 为 False。 获取 OptimizerWithSparsityGuarantee 请参考 :ref:`paddle.incubate.asp.decoreate <cn_api_paddle_incubate_asp_decoreate>`。
    - 在动态图模式下，使用 with_mask 调用函数是，需要先调用 paddle.incubate.asp.decorate() 来获取掩码变量。


参数
:::::::::
- **model** (Program|nn.Layer) - 带有模型定义和参数的 Program 或者 paddle.nn.Layer 对象
- **n** (int，可选) - n:m 稀疏中的 n
- **m** (int，可选) - n:m 稀疏中的 m
- **mask_algo** (string，可选) - 生成掩码的函数名。默认值为 mask_1d。有效输入应为 mask_1d ， mask_2d_greedy 和 mask_2d_best 之一。
- **with_mask** (bool，可选) - 选择是否裁剪参数相关的 ASP 掩码变量，True 是要裁剪，False 就是不裁剪。默认是 True。

返回
:::::::::

**dictionary** - 一个字典，key 是参数名称，value 是 对应的掩码变量。

代码示例
:::::::::

1. 动态图模式

COPY-FROM: paddle.incubate.asp.prune_model:dynamic_graph

2. 静态图模式

COPY-FROM: paddle.incubate.asp.prune_model:static_graph
