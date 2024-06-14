..  _group_sharded_parallel:

分组切分并行
========================

当模型参数达到百亿或者千亿时， 传统的数据并行训练可能会遇到显存瓶颈。
在数据并行训练中，每个 gpu worker 都有一份完整模型参数和优化器状态副本。
`《ZeRO: Memory Optimizations Toward Training Trillion Parameter Models》 <https://arxiv.org/abs/1910.02054>`__
指出在每个 GPU 上都保存一份模型参数和优化器状态副本是冗余的。 我们可以通过将上述参数和副本划分到不同 GPU 中，
在每个 GPU 只保存部分副本，来减少每张 GPU 上显存的占用，从而可以支持更大模型的训练。


一、原理介绍
-------------------

1.1 GroupSharded
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GroupSharded 实现了类似 ZeRO-DP 的训练策略，将模型状态包括：模型参数（parameter）、参数梯度（gradient）、参数对应的优化器状态（以 Adam 为例 moment 和 varience）切分到每一张 GPU 上。让模型参数部分所占的显存随并行卡数的增加而减少。
通过 paddle.distributed.sharding.group_sharded_parallel 提供的简单易用接口, 用户只需要添加几行代码就可将策略加入到原有的训练中。

模型训练过程中的显存消耗主要由两大部分组成：模型参数及优化器状态、训练产生的中间变量（activations）。
GroupSharded 策略可以根据用户配置支持，分别切分模型参数、对应参数梯度和优化器状态，因此模型状态所消耗的显存可以随着并行 GPU 数量增加而线性减少；
但是每张 GPU 上仍然维护着模型完整的前向和反向，所以每张 GPU 依然需要存放模型的训练过程中的产生的全部的中间变量，这部分显存消耗
不会随着 GPU 数量的增加而减少。 用户可以通过结合 recompute 策略来减少 activation 这部分的显存消耗。

通过 GroupSharded 和增加并行 GPU 数量，用户可以在 A100-40G 设备下 8 卡训练 16.25B 参量的模型 （需要结合 recompute, amp 策略）。

1.2 GroupSharded-hybrid-dp
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GroupSharded hybrid 数据并行策略，在 GroupSharded 并行的基础上再增加一层数据并行逻辑。
该策略的目的是通过 ``限制 GroupSharded 通信的节点数`` 和 ``增加多路数据并行`` 来提高训练吞吐。 如果一个模型在普通 GroupSharded 训练时需要 M 张 GPU，则开启 hybrid-dp 至少需要 N*M GPU （N>= 2）。

GroupSharded-hybrid-dp 适用的场景如下：

  * 当前有 4 个 8 卡 A100 节点
  * 目标模型 A 在 GroupSharded 训练时至少需要 8 卡 A100 （一个完整的 8 卡 A100 节点）
  * 希望利用全部的 4 个节点来加速训练

上述情况如果直接使用全部的 4 个节点 进行普通的 GroupSharded 训练， 那么全部的 32 gpus 之间组成一个完整 GroupSharded parallelism。这样会因为通信瓶颈造成训练速度非常慢：

  * GroupSharded 中的 allgather 通信 会涉及全部的 32 张卡，且为跨节点通信。
  * GroupSharded 中的 allreduce 通信 会涉及全部的 32 张卡，且为跨节点通信。

开启 hybrid-dp 并设置 ``GroupSharded_group_size = 8`` 后， 每个节点内的 8 张卡组成一个完整的 GroupSharded parallelism，4 个节点构成 4 路 hybrid data parallelism：

  * GroupSharded 中的 allgather 通信被限制在每个节点内的 8 张 GPU 之间， 没有跨节点通信。
  * GroupSharded 中的 allreduce 可以先进行机内通信，再跨节点通信，且每张 GPU 每次仅需要 allreduce 通信 1/8 的模型参数。

GroupSharded-hybrid-dp 通过上述措施，可以较大程度 减少 GroupSharded 训练 从 1 节点扩展到 4 节点时的（跨节点）通信量。提高节点增加时的加速比，提高训练吞吐。

P.S. hybrid dp 是因为 GroupSharded parallelism 本身内含一层 data parallelism 逻辑， hybrid dp 是在 GroupSharded parallelism 之上再增加新的一层 data parallelism 逻辑。


二、功能效果
--------------------

下面表格将对比 GroupSharded 策略对显存的影响。

模型为 GPT(11.375B)，试验环境为 A100 （40GB）， recompute = ON, amp（O2) = ON, hybrid-dp = OFF。
模型不变，单卡 batch size 不变，当并行 GPU 数量增加时，显存的消耗将减小。 省下的显存可以用来增大模型。

+------------------------------+----------+----------------+
| setting                      | GPU Mem  | Speed          |
+==============================+==========+================+
| GroupSharded—off             | 30474 MB | 9344 tokens/s  |
+------------------------------+----------+----------------+
| GroupSharded—on N2C16        | 30160 MB | 12663 tokens/s |
+------------------------------+----------+----------------+
| GroupSharded—on N4C32        | 30190 MB | 22235 tokens/s |
+------------------------------+----------+----------------+

GroupSharded 结合 amp （O2) + recompute，可以在 8 张 40GB A100 并行的情况下支持百亿参数（16.25B）GPT 训练。


三、使用方法
----------------------

首先简单总结 GroupSharded stage1、stage2、stage3 分别实现减小参数规模的原理。stage1、stage2、stage3 分别在训练过程中对模型优化器状态、梯度+优化器状态、参数+梯度+优化器状态进行切分，通过减小训练的相关 Tensor（参数、梯度、优化器状态）达到同样计算资源下能够训练更大模型的效果。

以下是分别从 GroupSharded 的三种实现阶段的实现方式：
  * 使用 group_sharded_parallel 和 save_group_sharded_model 两个 API 可以进行训练和保存。使用 group_sharded_parallel 提供 stage1 的选项，内部使用 stage2 完成优化实现。参考 `group_sharded_parallel <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/sharding/group_sharded_parallel_cn.html>`__， `save_group_sharded_model <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/sharding/save_group_sharded_model_cn.html>`__。
  * 此处需要注意，使用 save_group_sharded_model 保存模型，再次 load 时需要在调用 group_sharded_parallel 前对 model 和 optimizer 进行 set_state_dict。
  * 目前 stage2、3 已经适配 GPT 模型，可以参考请参考 `示例代码 <https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/gpt-3/dygraph>`__。
  * 其次解决组网中共享参数训练问题，stage3 需要额外在组网中加入外置参数注册逻辑，在组网中需要注册 ``self.extra_parameters = [self.gpt.embeddings.word_embeddings.weight]``，这部分可以参考 PaddleNLP 中 GPT-3 的组网。`示例代码 <https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/language_model/gpt-3/dygraph/modeling.py>`__。

.. code-block::

    import paddle
    from paddle.vision.models import ResNet
    from paddle.vision.models.resnet import BasicBlock
    from paddle.distributed import fleet
    from paddle.distributed.sharding import group_sharded_parallel, save_group_sharded_model

    fleet.init(is_collective=True)
    group = paddle.distributed.new_group([0, 1])
    use_pure_fp16 = True

    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
    model = ResNet(BasicBlock, 18)
    optimizer = paddle.optimizer.AdamW(learning_rate=0.001, parameters=model.parameters(), weight_decay=0.00001, grad_clip=clip)

    scaler = None
    if use_pure_fp16:
        scaler = paddle.amp.GradScaler(init_loss_scaling=32768)
        # level O2 means converting the network to FP16
        model = paddle.amp.decorate(
            models=model,
            level='O2',
            save_dtype='float32')

    # wrap GroupSharded model, optimizer and scaler
    model, optimizer, scaler = group_sharded_parallel(model, optimizer, "os_g", scaler=scaler)

    for step_id in range(1, 100):
        x = paddle.rand([1, 3, 224, 224])
        with paddle.amp.auto_cast(use_pure_fp16):
            out = model(x)
        loss = out.mean()

        if use_pure_fp16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.clear_grad()

        print("=== step_id : {}    loss : {}".format(step_id, loss.numpy()))

    # save model and optimizer state_dict
    save_group_sharded_model(model, output_dir, optimizer)


运行方式（需要保证当前机器有两张 GPU）：

.. code-block:: bash

  export CUDA_VISIBLE_DEVICES=0,1
  python -m paddle.distributed.launch run_pretrain.py # run_pretrain.py 是用户运行动态图 GroupSharded 的 python 文件


控制台输出信息如下：

.. code-block:: bash

  launch train in GPU mode!
  INFO 2022-05-18 09:34:51,803 launch_utils.py:561] Local start 2 processes. First process distributed environment info (Only For Debug):
    +=======================================================================================+
    |                        Distributed Envs                      Value                    |
    +---------------------------------------------------------------------------------------+
    |                       PADDLE_TRAINER_ID                        0                      |
    |                 PADDLE_CURRENT_ENDPOINT                 127.0.0.1:12532               |
    |                     PADDLE_TRAINERS_NUM                        2                      |
    |                PADDLE_TRAINER_ENDPOINTS         127.0.0.1:12532,127.0.0.1:58759       |
    |                     PADDLE_RANK_IN_NODE                        0                      |
    |                 PADDLE_LOCAL_DEVICE_IDS                        6                      |
    |                 PADDLE_WORLD_DEVICE_IDS                       6,7                     |
    |                     FLAGS_selected_gpus                        6                      |
    |             FLAGS_selected_accelerators                        6                      |
    +=======================================================================================+

日志信息位于 log 目录下:

.. code-block:: bash

  [2022-05-18 09:35:15,062] [    INFO] - global step 1, epoch: 0, batch: 0, loss: 11.059432030, avg_reader_cost: 0.15902 sec, avg_batch_cost: 0.61838 sec, speed: 1.62 step/s, ips: 13247 tokens/s, learning rate: 9.37500e-08
  [2022-05-18 09:35:15,274] [    INFO] - global step 2, epoch: 0, batch: 1, loss: 11.050725937, avg_reader_cost: 0.00041 sec, avg_batch_cost: 0.21061 sec, speed: 4.75 step/s, ips: 38897 tokens/s, learning rate: 1.40625e-07
  [2022-05-18 09:35:15,432] [    INFO] - global step 3, epoch: 0, batch: 2, loss: 11.051848412, avg_reader_cost: 0.00022 sec, avg_batch_cost: 0.15722 sec, speed: 6.36 step/s, ips: 52105 tokens/s, learning rate: 1.87500e-07
  [2022-05-18 09:35:15,566] [    INFO] - global step 4, epoch: 0, batch: 3, loss: 11.052285194, avg_reader_cost: 0.00022 sec, avg_batch_cost: 0.13303 sec, speed: 7.52 step/s, ips: 61579 tokens/s, learning rate: 2.34375e-07
  [2022-05-18 09:35:15,722] [    INFO] - global step 5, epoch: 0, batch: 4, loss: 11.028432846, avg_reader_cost: 0.00036 sec, avg_batch_cost: 0.15526 sec, speed: 6.44 step/s, ips: 52764 tokens/s, learning rate: 2.81250e-07
  [2022-05-18 09:35:15,880] [    INFO] - global step 6, epoch: 0, batch: 5, loss: 11.032807350, avg_reader_cost: 0.00021 sec, avg_batch_cost: 0.15763 sec, speed: 6.34 step/s, ips: 51971 tokens/s, learning rate: 3.28125e-07
