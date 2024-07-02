# 分布式训练常见问题

## 综合问题

##### 问题：怎样了解飞桨分布式 Fleet API 用法?

+ 答复：可查看覆盖高低阶应用的[分布式用户文档](https://github.com/PaddlePaddle/PaddleFleetX)和[分布式 API 文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distributed/Overview_cn.html)

----------

##### 问题：机房训练模型的分布式环境用什么比较合适？

+ 答复： 推荐使用 K8S 部署，K8S 的环境搭建可[参考文档](https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/paddle_on_k8s.html)

----------

##### 问题：目前飞桨分布式对哪些模型套件/工具支持？

+ 答复：
1. 多机多卡支持 paddlerec，PGL，paddleHelix，paddleclas，paddlenlp，paddledetection。
2. 单机多卡支持全部飞桨的模型套件和高层 API 写法，无需修改单卡训练代码，默认启用全部可见的卡。

----------

##### 问题：怎样自定义单机多卡训练的卡数量？

+ 答复：如果直接使用飞桨模型套件（paddleclas，paddleseg 等）或高层 API 写的代码，可以直接用这条命令指定显卡启动程序，文档源代码不用改（文档内不要用 set_device 指定卡）：
  `python3 -m paddle.distributed.launch --gpus="1, 3" train.py`
  使用基础 API 的场景下，在程序中修改三处：
  * 第 1 处改动，import 库`import paddle.distributed as dist`
  * 第 2 处改动，初始化并行环境`dist.init_parallel_env()`
  * 第 3 处改动，对模型增加 paddle.DataParallel 封装 `net = paddle.DataParallel(paddle.vision.models.LeNet())`
修改完毕就可以使用 `python3 -m paddle.distributed.launch --gpus="1, 3" xxx `来启动了。可参考[AI Studio 项目示例](https://aistudio.baidu.com/aistudio/projectdetail/1222066)

----------

## Fleet API 的使用

##### 问题：飞桨 2.0 版本分布式 Fleet API 的目录在哪？

+ 答复：2.0 版本分布式 API 从 paddle.fluid.incubate.fleet 目录挪至 paddle.distributed.fleet 目录下，且对部分 API 接口做了兼容升级。import 方式如下：

  ```python
  import paddle.distributed.fleet as fleet
  fleet.init()
  ```

 不再支持老版本 paddle.fluid.incubate.fleet API，2.0 版本会在分布式计算图拆分的阶段报语法相关错误。未来的某个版本会直接移除废弃 paddle.fluid 目录下的 API。

----------

##### 问题：飞桨 2.0 版本的 fleet 配置初始化接口 init 和 init_server 用法有什么变化？

+ 答复：
1. `fleet.init`接口，2.0 版本支持`role_maker`，`is_collective`，`strategy`等参数，且均有缺省值，老版本仅支持`role_maker`，且无缺省配置。[点击这里](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distributed/fleet/Fleet_cn.html) 参考 2.0 Fleet API 的使用方式。
2. `fleet.init_server`接口，除支持传入`model_dir`之外，2.0 版本还支持传入`var_names`，加载指定的变量。

----------

##### 问题： 飞桨 2.0 版本的分布式 paddle.static.nn.sparse_embedding 和 paddle.nn.embedding 有什么差别？

+ 答复：`paddle.nn.embedding`和`paddle.static.nn.sparse_embedding`的稀疏参数将会在每个 PServer 段都用文本的一部分保存，最终整体拼接起来是完整的 embedding。推荐使用`paddle.static.nn.sparse_embedding`直接采用分布式预估的方案。虽然 `nn.embedding`目前依旧可以正常使用，但后续的某个版本会变成与使用`paddle.static.nn.sparse_embedding`一样的保存方案。老版本中使用的 0 号节点的本地预测功能在加载模型的时候会报模型加载错误。

----------

##### 问题：飞桨 2.0 分布式可以用哪些配置类？

+ 答复：2.0 之后统一为`paddle.distributed.fleet.DistributedStrategy()`，与下述老版本配置类不兼容。2.0 之前的版本参数服务器配置类：`paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler.distributed_strategy.DistributedStrategy`，2.0 之前的版本 collective 模式配置类：`paddle.fluid.incubate.fleet.collective.DistributedStrategy`

----------

##### 问题：飞桨 2.0 分布式配置项统一到 DistributedStrategy 后有哪些具体变化？

+ 答复：
2.0 版本之后，建议根据 [DistributedStrategy 文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distributed/fleet/DistributedStrategy_cn.html) 和 [BuildStrategy 文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/static/BuildStrategy_cn.html#buildstrategy) 修改配置选项。

2.0 版本将 3 个环境变量配置变为`DistributedStrategy`配置项，3 个环境变量将不生效，包括

  * `FLAGS_sync_nccl_allreduce`→ `strategy.sync_nccl_allreduce`
  * `FLAGS_fuse_parameter_memory_size` → `strategy.fuse_grad_size_in_MB`
  * `FLAGS_fuse_parameter_groups_size` → `strategy.fuse_grad_size_in_TFLOPS`

 DistributedStrategy 中`forward_recompute`配置项不兼容升级为`recompute`。

 DistributedStrategy 中`recompute_checkpoints`配置项不兼容升级为`recompute_configs`字典下的字段，如下：

  ```python
  import paddle.distributed.fleet a fleet
  strategy = fleet.Distributedstrategy()
  strategy.recompute = True
  strategy.recompute_configs = {
      "checkpoints": ["x","y"],
      "enable_offload": True,
      "checkpoint_shape": [100, 512, 1024]}
  ```

 DistributedStrategy 中`use_local_sgd`配置项变为不兼容升级为 localsgd。

----------

##### 问题：飞桨 2.0 分布式 Fleet 的 program 接口是否还能继续用？
+ 答复：2.0 版本后，fleet 接口下 main_program 和_origin_program 均已废弃，会报错没有这个变量，替换使用`paddle.static.default_main_program`即可。

----------

##### 问题：怎样在本地测试 Fleet API 实现的分布式训练代码是否正确？

+ 答复：首先写好分布式 train.py 文件

  * 在 PServer 模式下，命令行模拟启动分布式：`python -m paddle.distributed.launch_ps --worker_num 2 --server_num 2 train.py`
  * 在 Collective 模式下，命令改为`python -m paddle.distributed.launch --gpus=0,1 train.py`

----------

##### 问题：Paddle Fleet 怎样做增量训练，有没有文档支持？

+ 答复：增量训练可参考[文档示例](https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/parameter_server/ps_incremental_learning.html)

----------

##### 问题：飞桨 2.0 分布式 distributed_optimizer 如何使用自动混合精度 amp 的 optimizer？

+ 答复：`amp_init`接口支持 pure_fp16，可以直接调用`optimizer.amp_init`。

----------

##### 问题：Paddle Fleet 可以在 K8S GPU 集群上利用 CPU 资源跑 pserver 模式的 MPI 程序吗？

+ 答复：可以，GPU 可设置为 trainer。

----------

##### 问题：使用 Fleet Collective 模式进行开发时，已使用 fleet.distributed_optimizer 对 optimizer 和 fleet.DistributedStrategy 包了一层。想确认模型是否也需要使用 fleet.distributed_model 再包一层？

+ 答复：需要将`fleet.distributed_model`在封装一层。原因是动态图主要在`fleet.distributed_model`进行分布式设计，静态图是在`fleet.distributed_optimizer`进行分布式设计。所以，如果不区分动态图和静态图，两个接口都是需要的。

----------

## 环境配置和训练初始化

##### 问题：分布式环境变量 FLAGS 参数定义可以在哪查看，比如 communicator 相关的？

+ 答复：参考使用[DistributedStrategy](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distributed/fleet/DistributedStrategy_cn.html#distributedstrategy)配置分布式策略。

----------

##### 问题：2.0 分布式训练的启动命令有什么变化？

+ 答复：为了统一启动分布式 Collective/PS 模式任务方式以及易用性考虑，2.0 版本中 launch/fleetrun 启动分布式任务时参数产生不兼容升级，`--cluster_node_ips`改为`--ips`，`--selected_gpus`改为`--gpus`、`--node_ip`、`--use_paddlecloud`、`--started_port`、`--log_level`、`--print_config` 5 个参数已废弃，使用旧参数会直接报错没有此参数。代码迁移至 python/paddle/distributed/fleet/launch.py。

----------

##### 问题：分布式环境依赖为什么出现第三方 libssl 库的依赖？

+ 答复：分布式 RPC 从 GRPC 迁移至 BRPC， 会导致在运行时依赖 libssl 库。使用 docker 的情况下，基础镜像拉一下官方最新的 docker 镜像，或自行安装 libssl 相关的依赖也可以。未安装 libssl 的情况下，import paddle 的时候，出现找不到 libssl.so 的库文件相关报错。使用 MPI 的情况下，需要将编译包时用到的 libssl.so、libcrypto.so 等依赖手动通过`LD_LIBRARY_PATH`进行指定。

----------

## 分布式的动态图模式

##### 问题：飞桨 2.0 版本动态图 DataParallel 用法有哪些简化？

+答复：老版本用法依然兼容，建议使用以下新用法：`apply_collective_grads`、`scale_loss`可以删除不使用。loss 会根据环境除以相应的卡数，`scale_loss`不再进行任何处理。

----------

##### 问题：飞桨 2.0 版本调用 model.eval 之后不再自动关闭反向计算图的构建，引入显存的消耗增加，可能会引入 OOM，怎么解决？

+ 答复：动态图`no_grad`和 `model.eval` 解绑，应使用`with paddle.no_grad():` 命令，显示关闭反向计算图的构建。

----------

##### 问题：飞桨 2.0 版本动态图环境初始化新接口怎样用？

+ 答复：建议调用新接口`paddle.distributed.init_parallel_env`，不需要输入参数。1.8 的`fluid.dygraph.prepare_context`依然兼容。

----------

##### 问题：分布式支持哪些飞桨 2.0 版本的模型保存和加载接口？

+ 答复： 与单机相同，分布式动态图推荐使用`paddle.jit.save`保存，使用`paddle.jit.load`加载，无需切换静态图，存储格式与推理模型存储一致。对比 1.8 动态图使用不含控制流的模型保存接口`TracedLayer.save_inference_model`，含控制流的模型保存接口`ProgramTranslator.save_inference_model`，加载模型需要使用静态图接口`fluid.io.load_inference_model`。
`fluid.save_dygraph`和`fluid.load_dygraph`升级为`paddle.save`和`paddle.load`，推荐使用新接口。`paddle.save`不再默认添加后缀，建议用户指定使用标椎后缀（模型参数：.pdparams，优化器参数：.pdopt）。

##### 问题：飞桨 2.0 版本为什么不能使用 minimize 和 clear_gradient？

+ 答复：2.0 版本中重新实现 optimizer，放在`paddle.optimizer`，建议使用新接口和参数。老版本的`paddle.fluid.optimizer`仍然可用。

 新版增加接口`step`替换`minimize`。老版动态图需要调用`loss.backward()`，用 minimize 来表示梯度的更新行为，词语意思不太一致。

新版使用简化的`clear_grad`接口替换`clear_gradient`。



----------

## 报错查错

##### 问题：集合通信 Collective 模式报参数未初始化的错误是什么原因？

+ 答复：2.0 版本需要严格先`run(startup_program)`，然后再调用`fleet.init_worker()`启动 worker 端通信相关，并将 0 号 worker 的参数广播出去完成其他节点的初始化。先`init_worker`，再`run(startup_program)`，会报参数未初始化的错误

 2.0 之前的版本是在 server 端做初始化，无需 0 号节点广播，所以`init_worker()`可以在`run(startup_program)`执行。

----------

##### 问题：分布式任务跑很久 loss 突然变成 nan 的原因？

+ 答复：可设置环境变量`export FLAGS_check_nan_inf=1`定位出错的地方，可以从 checkpoint 开始训练，参数服务器和集合通信模式均可使用这种方式查错。

----------

##### 问题：任务卡在 role init 怎么解决？

+ 答复：通常可能是 gloo 的初始化问题，需要检查是否有节点任务挂了。建议调小`train_data`配置的数据量,由于启动 trainer 前要下载数据，大量数据会导致拖慢。
