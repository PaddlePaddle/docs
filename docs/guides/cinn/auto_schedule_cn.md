# CINN 自动调优框架
  CINN 在算子(融合算子)实现层采用的是 compute 与 schedule 分离的思想，compute 表示算子的朴素实现，schedule 表示具体的计算方式。auto-schedule 的作用是自动生成算子的 schedule 配置，降低新硬件接入编译器的人力成本和技术门槛，并且满足极致追求性能场景的优化需求, 下面简要介绍下自动调优框架在 CINN 整体架构中的位置和核心模块的功能。

  如下 CINN 整体架构流程图所示，auto-schedule 位于**低层优化**这一层次中，在 compute 获得 CINN IR 之后进行调用，调优结束后再经过低层 Pass 进行通用优化。
  从自动调优子系统自身来看，可以分为三个阶段，六个关键模块部分，代码实现在[cinn/auto_schedule](https://github.com/PaddlePaddle/CINN/tree/develop/cinn/auto_schedule)目录中, 下面结合 auto-schedule 架构图简述各模块的主要功能及对应的代码位置:
  - 抽取任务: Graph Fusion 后的每个子图初始化为一个任务。核心代码位于 cinn/auto_schedule/task, 该目录包含了 task 的定义、抽取、注册、以及执行器
  - 任务调度: 选取对整体性能提升最大的任务, 决定下一个时间片调优的任务。核心代码位于 cinn/auto_schedule/task_scheduler，该目录下定义和实现了若干个任务调度方式，分别适用对搜索 overhead 敏感程度不同的场景。
  - 搜索空间: 定义了一系列 sketch 生成规则，它们通过分析 IR AST 的计算语义，结合当前后端 target 的硬件架构特点，粗略生成若干种计算方式的 schedule 配置，作为后续搜索算法的候选集合。核心代码位于 cinn/auto_schedule/search_space, 该目录定义了搜索空间、搜索状态等结构体，生成规则的接口及实现存放在子目录 cinn/auto_schedule/search_space/auto_gen_rule 中。
  - 搜索策略: 在搜索空间生成的 schedule 骨架的基础上，采用进化算法随机突变的思想，进一步确定 schedule 中的参数数值，并结合 learning-based CostModel 进行最终决策，选取最优的 schedule 配置作为结果。核心代码位于 cinn/auto_schedule/search_strategy，该目录定义了进化算法的执行方式，并且定义和实现了若干随机突变规则位于子目录 cinn/auto_schdule/search_strategy/mutate_rule 中。
  - 运行评估: 在实际硬件上测试候选集性能，收集样本数据，提升 cost model 的预测效果。核心代码位于 cinn/auto_schedule/cost_model，该目录定义了 cost model 的驱动类、CINN IR 的特征抽取逻辑等。
  - 数据序列化与加载: 调优记录序列化及反序列化，存储与加载，支持断点热启动，离线配置在线应用等需求。核心代码位于 cinn/auto_schedule/databse, 该目录定义了数据库访问接口并且实现了文件数据库，schedule 配置的序列化功能则实现在[schedule_desc](https://github.com/PaddlePaddle/CINN/blob/develop/cinn/ir/schedule_desc.h)中。
