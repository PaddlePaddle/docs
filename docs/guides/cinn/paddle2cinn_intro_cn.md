# Paddle 训练框架应用 CINN 进行编译优化加速
  CINN 提供了图层优化、低层 kernel 调优等编译优化技术加速计算图的运行效率，我们通过对 Paddle 训练框架执行过程进行一定升级改造，使得它能够联合 CINN 提升整体训练性能。本文介绍 Paddle 侧接入 CINN 的流程方案，包括框架编译期和运行时两方面的逻辑。

## 编译期修改计算图
  这部分的工作是圈定计算图中可被 CINN 编译优化的子图, 并将待编译的子图替换为大 Op, 使得框架运行时通过执行器调度驱动后续的子图编译和执行，具体实现逻辑简述如下:
  模型网络 ProgramDesc 转换为 Graph 之后，我们在框架 Graph Pass 阶段插入一个[build_cinn_pass](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/paddle2cinn/build_cinn_pass.cc), 该 Pass 首先搜索识别原始计算图中哪些算子可以支持 CINN, 然后合并相邻选中算子聚合成若干个待编译子图，接下来在原始计算图中将每个子图替换为一个`cinn_launch_op`算子, 并将相应的子图保存到全局单例类 CinnCompiler 中, 该对象返回唯一的 cache key 子图识别码，此 key 作为 cinn_launch_op 的属性保存到其 OpDesc 中，Pass 运行结束后计算图形态表现为 cinn_launch_op 之间通过其它常规算子连通，但彼此互不邻接。

## 运行时编译(JIT)与执行编译结果
  子图的编译和计算都在运行期完成，逻辑设计实现在[cinn_launch_op](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/cinn/cinn_launch_op.h)算子中, 每个 Op 负责一个子图并分别独立运行,从框架执行器视角来看，它与常规算子 kernel 一致，同样都是在运行期调度执行该 kernel, 因此保证了与现有运行逻辑兼容。具体到 kernel 的逻辑上, 它先后执行子图编译和子图计算两阶段:
  - 子图编译: 首先将前述步骤保存在 CinnCompiler 中的子图对象通过 cache key 取出，通过 CinnCompiler::Compile 接口编译子图, 编译成功后获得编译结果结构体[CinnCompiledObject](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/paddle2cinn/cinn_compiler.h#L56), 它包括编译过程收集到的信息、必要的上下文环境、以及进行具体计算的可执行对象。
  - 子图计算: 这部分的工作是运行编译得到的可执行对象, 完成子图的计算过程, 它先后经历准备输入数据、运行计算 kernel、同步输出数据三个步骤。
    (1) 数据同步：子图的输入、输出数据来源于 Paddle 侧的变量, 而计算 kernel 则是由 CINN 生成，因此涉及到了 CINN 与 Paddle 两套体系的数据交互。为了避免不必要的数据拷贝开销, 我们建立了 Paddle 变量 <--> CINN 内存 buffer 之间的数据映射, 将 Paddle 变量内存指针包裹成 CINN cinn_buffer_t 结构体, 作为参数供生成 kernel 在执行时直接进行 Paddle 侧变量的读取/写入内存操作。
    (2) 运行计算 kernel：目前支持 CINN runtime 调度以及 Paddle 执行器调度(PE 或新执行器)两种方式，可以根据开关 FLAGS_enable_interpretercore_launch_cinn 或 FLAGS_enable_pe_launch_cinn 进行切换，其中 CINN runtime 可以看作是以子图的拓扑序依次执行，Paddle 执行器的方式则是通过`编译生成的指令序列` --> `ProgramDesc` --> `Graph`的转换过程，将可执行对象转换为 Paddle 计算图，再复用执行器调度执行。
    上述数据映射，计算图转换等编译结果上下文的逻辑主要实现在[cinn_launch_context](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/cinn/cinn_launch_context.h)类中，它由 CinnCompiler 产出编译结果时进行构造，并作为子图计算时的辅助结构体进行使用。

## 显存管理
