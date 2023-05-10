# Paddle 训练框架应用 CINN 进行编译优化加速
  CINN 提供了图层优化、低层 kernel 调优等编译优化技术加速计算图的运行效率，我们通过对 Paddle 训练框架执行过程进行一定升级改造，使得它能够联合 CINN 提升整体训练性能。本文介绍 Paddle 侧接入 CINN 的流程方案，包括框架编译期和运行时两方面的逻辑。

## 编译期修改计算图
  这部分的工作是圈定计算图中可被 CINN 编译优化的子图, 并将待编译的子图替换为大 Op, 使得框架运行时通过执行器调度驱动后续的子图编译和执行，它是在编译期模型网络 ProgramDesc 转换为 Graph 之后，在框架 Graph Pass 阶段插入一个[build_cinn_pass](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/paddle2cinn/build_cinn_pass.cc)来实现的，具体可分为如下两个步骤:

  - 可编译算子搜索与子图标记: 首先搜索识别原始计算图中哪些算子可以支持 CINN, 然后合并相邻选中算子聚合成若干个待编译子图，过程如下示例图:![算子搜索与子图标记](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/cinn/8f0f98b32f54445e4fc027a02.png)

  - 子图替换为指定算子: 在原始计算图中将每个被标记为可编译的子图替换为一个`cinn_launch_op`算子, 并将相应的子图保存到全局单例类 CinnCompiler 中, 该对象返回唯一的 cache key 子图识别码，此 key 作为 cinn_launch_op 的属性保存到其 OpDesc 中，过程如下示例图:![待编译子图替换为指定算子](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/cinn/e5bafac158974aad1e1f0ef05.png)

Pass 运行结束后计算图形态表现为 cinn_launch_op 之间通过其它常规算子连通，但彼此互不邻接。

## 运行时编译(JIT)与执行编译结果
  子图的编译和计算都在运行期完成，逻辑设计实现在[cinn_launch_op](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/cinn/cinn_launch_op.h)算子中, 每个 Op 负责一个子图并分别独立运行,从框架执行器视角来看，它与常规算子 kernel 一致，同样都是在运行期调度执行该 kernel, 因此保证了与现有运行逻辑兼容。具体到 kernel 的逻辑上, 它先后执行子图编译和子图计算两阶段:
  - 子图编译: 首先将前述步骤保存在 CinnCompiler 中的子图对象通过 cache key 取出，通过 CinnCompiler::Compile 接口编译子图, 编译成功后获得编译结果结构体[CinnCompiledObject](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/paddle2cinn/cinn_compiler.h#L56), 它包括编译过程收集到的信息、必要的上下文环境、以及进行具体计算的可执行对象。关键代码片段如下:
  ```
    std::map<std::string, const phi::DenseTensor*> inputs_name2tensor;  // 输入变量信息

    auto target = details::PlaceToCinnTarget(place);  // 将 Paddle 的 Place 转换为 CINN 的 Target 对象

    const auto& cinn_compiled_object = CinnCompiler::GetInstance()->Compile(compilation_key, inputs_name2tensor, target, stream);  // 调用 CinnCompiler 编译

  ```

  - 子图计算: 这部分的工作是运行编译得到的可执行对象, 完成子图的计算过程, 它先后经历准备输入数据、运行计算 kernel、同步输出数据三个步骤。

    (1) 数据同步：子图的输入、输出数据来源于 Paddle 侧的变量, 而计算 kernel 则是由 CINN 生成，因此涉及到了 CINN 与 Paddle 两套体系的数据交互。为了避免不必要的数据拷贝开销, 我们建立了 Paddle 变量 <--> CINN 内存 buffer 之间的数据映射, 将 Paddle 变量内存指针包裹成 CINN cinn_buffer_t 结构体, 作为参数供生成 kernel 在执行时直接进行 Paddle 侧变量的读取/写入内存操作。

    (2) 运行计算 kernel：目前支持 CINN runtime 调度以及 Paddle 执行器调度(PE 或新执行器)两种方式，可以根据开关 FLAGS_enable_interpretercore_launch_cinn 或 FLAGS_enable_pe_launch_cinn 进行切换，其中 CINN runtime 可以看作是以子图的拓扑序依次执行，Paddle 执行器的方式则是通过`编译生成的指令序列` --> `ProgramDesc` --> `Graph`的转换过程，将可执行对象转换为 Paddle 计算图，再复用执行器调度执行。关键代码片段如下:
```
    auto* launch_context = cinn_compiled_object.launch_context.get();

    if (FLAGS_enable_pe_launch_cinn) {  // PE 执行器调度运行
      if (FLAGS_enable_interpretercore_launch_cinn) {  // 新执行器调度运行
        platform::RecordEvent record_event_4(
            "Step 4. Execute the runtime program by InterpreterCore.");
        VLOG(4) << "Execute the runtime program by InterpreterCore";
        auto* interpreter_core = launch_context->InitializeInterpreterCore(
            place, const_cast<framework::Scope*>(&scope));
        interpreter_core->Run({}, false);
      } else {
        platform::RecordEvent record_event_4(
            "Step 4. Execute the runtime graph by PE.");
        VLOG(4) << "Execute the runtime graph by PE";
        framework::Scope& exec_scope = scope.NewScope();
        auto* pe = launch_context->InitializePE(place, &exec_scope);
        pe->RunWithoutFetch(launch_context->GetSkipEagerVars());
      }
    } else {  // CINN runtime 调度运行
      platform::RecordEvent record_event_4(
          "Step 4. Execute the compiled executable program.");
      VLOG(4) << "Execute the compiled executable program";
      launch_context->UpdateCapturedEnv(scope, place);
      LaunchCinnExecution(cinn_compiled_object, *launch_context, stream);
    }
```

上述数据映射，计算图转换等编译结果上下文的逻辑主要实现在[cinn_launch_context](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/cinn/cinn_launch_context.h)类中，它由 CinnCompiler 产出编译结果时进行构造，并作为子图计算时的辅助结构体进行使用，主要功能函数接口见该类接口和实现的说明注释。

## 显存管理

  对应于上述两种不同的执行调度方式，我们也分别实现了不同的策略来进行显存管理。

  首先是 Paddle 执行器调度的方式，它是将可执行序列回转成了 Paddle Graph，因此子图计算时可以直接应用框架侧已有的显存回收与复用策略，但是需要额外处理的问题是：每个子图包裹在一个 cinn_launch 算子中，算子外还有一个主图，计算图呈现嵌套结构，子图的输入/输出变量可能会被主图中其它算子使用，因此需要将子图外部变量(输入/输出)的使用信息透传至子图内，以避免外部被在子图计算时被提前释放。对于这个问题，在使用 PE 调度方式时我们新增了一个[share_varinfo_into_cinn_pass](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/ir/memory_optimize_pass/share_varinfo_into_cinn_pass.cc)来实现将外部变量的`MemOptVarInfo`信息同步到子图内中，而新执行器调度方式下则是通过设置`ExecutionConfig::skip_gc_vars`来子图外部变量的回收延迟至主图计算调度时进行。

  其次是 CINN runtime 调度的方式, 它是依次串行调度可执行序列(CINN instruction)，instruction 计算时要求输入/输出数据的内存空间均已分配完成，最粗暴的方式是整个子图计算前将所有数据集中分配，计算完成后再集中释放临时中间数据, 但是这种方式显然会导致显存使用量急剧攀升。为此，我们参考了 Paddle 框架显存回收的思想，通过分析变量的使用关系，在 CINN 生成的子图可执行序列中插入一类特殊的 BufferMalloc/BufferFree 指令来完成即时分配/释放显存的工作，具体地，在每个 instruction 计算前若有数据尚未分配则被插入一个 BufferMalloc 进行即时申请，在每个 instruction 计算后若有临时数据不再使用则插入一个 BufferFree 进行即时回收。另外，前面提到通过 Paddle 与 CINN 变量之间的映射机制来避免数据同步开销，因此实际显存申请/释放是由 Paddle 侧的内存管理器执行的，我们在建立数据映射时为每个变量构造一个 external_malloc/external_free 的 Callback，供 CINN BufferMalloc/BufferFree 指令调用，从而实现在 CINN runtime 调用 Paddle 进行显存管理。具体逻辑可见`CinnLaunchContext::AssignExternalVariable`与`CinnLaunchContext::AssignInternalVariable`的定义和调用。
