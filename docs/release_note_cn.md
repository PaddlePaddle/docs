# 3.2 Release Note

飞桨框架 3.2 版本在大模型训练推理性能、硬件适配、主流大模型及高性能加速库的支持上进一步提升。

- 大模型训练方面，飞桨框架在计算、并行策略、容错能力三方面进行了升级：
  - 从基础计算性能层面，提出了存算重叠的稀疏掩码注意力计算 FlashMask V3，极致优化 Attention 的计算效率，同时还实现了高效的 FP8 混合精度效果无损训练技术。
  - 在分布式并行策略层面，提出了动态自适应的显存卸载策略，实现存算最优均衡，再结合创新设计的显存友好的流水线并行调度，进一步降低显存开销。
  - 增强了框架原生的容错能力，实现了大规模集群训练容错系统，可在不影响训练效率的前提下在线监测静默数据损坏等难以察觉的故障，并实现了高可用的检查点容灾方法，降低中断恢复损失。
- 在硬件适配方面，面向类 CUDA 芯片，全面升级插件式适配方案。
  - 在设备资源的管理调度和高性能集合通讯库方面，针对类 CUDA 芯片做了管理接口升级和通信能力的增强，特别增强了分布式通信能力，使 XCCL 对齐 NCCL 的各结构体和功能。
  - 新增了类 CUDA 算子注册机制。以沐曦适配为例，在复用 GPU 算子内核的基础上，仅需一行代码即可完成算子内核注册。经过统计计算，算子内核的复用率最高可以达到 92%，可大幅降低硬件适配成本。
- 使用体验方面，重点提升了兼容能力，包括开发接口兼容业界用法、safetensors 模型格式兼容、和第三方高性能加速库的兼容。
  - 新增和修改开发接口兼容业界用法，新增系列 API 和别名，新增参数别名，新增专有和通用的参数。
  - 全面兼容 Safetensors 模型格式。新增 FlexCheckpoint 机制，支持跨分布式策略、跨模型结构间自动实现参数重切分，可显著降低权重转换成本，进而提升大模型端到端的训练与推理研发效率。
  - 系统性增强了接口兼容与算子注册能力，实现了高性能加速库一键导入，无需修改代码直接复用于飞桨的模型训练与推理加速过程中。

## 1. 用户体验

### 新特性
- 新增 API：`paddle.msort`、`paddle.ravel`、`paddle.nn.functional.dropout1d`、`paddle.Tensor.type_as`、`paddle.Tensor.requires_grad`、`paddle.view_as_complex`、`paddle.view_as_real`、`paddle.nn.Parameter`、`paddle.broadcast_shapes`、`paddle.range`、`paddle.as_tensor`、`paddle.scatter_reduce/scatter_reduce_`、`paddle.scatter_add`、`paddle.tensor`、`paddle.softmax`、`paddle.Tensor.softmax`、`paddle.rand_like`、`paddle.is_autocast_enabled`、`paddle.get_autocast_gpu_dtype`、`paddle.Tensor.repeat`、`paddle.permute`。[#74421](https://github.com/PaddlePaddle/Paddle/pull/74421),[#74439](https://github.com/PaddlePaddle/Paddle/pull/74439),[#74444](https://github.com/PaddlePaddle/Paddle/pull/74444),[#74454](https://github.com/PaddlePaddle/Paddle/pull/74454),[#74459](https://github.com/PaddlePaddle/Paddle/pull/74459),[#74491](https://github.com/PaddlePaddle/Paddle/pull/74491)、[#74466](https://github.com/PaddlePaddle/Paddle/pull/74466),[#74438](https://github.com/PaddlePaddle/Paddle/pull/74438),[#74594](https://github.com/PaddlePaddle/Paddle/pull/74594),[#74542](https://github.com/PaddlePaddle/Paddle/pull/74542),[#74694](https://github.com/PaddlePaddle/Paddle/pull/74694),[#74564](https://github.com/PaddlePaddle/Paddle/pull/74564),[#74540](https://github.com/PaddlePaddle/Paddle/pull/74540),[#74586](https://github.com/PaddlePaddle/Paddle/pull/74586),[#74651](https://github.com/PaddlePaddle/Paddle/pull/74651),[#74807](https://github.com/PaddlePaddle/Paddle/pull/74807),[#74632](https://github.com/PaddlePaddle/Paddle/pull/74632),[#74834](https://github.com/PaddlePaddle/Paddle/pull/74834),[#74952](https://github.com/PaddlePaddle/Paddle/pull/74952),[#74772](https://github.com/PaddlePaddle/Paddle/pull/74772),[#74441](https://github.com/PaddlePaddle/Paddle/pull/74441),[#74561](https://github.com/PaddlePaddle/Paddle/pull/74561),[#74525](https://github.com/PaddlePaddle/Paddle/pull/74525)
- 新增`paddle.compat.*`一系列 API，支持业界的通用用法，便于迁移代码，包括 `paddle.compat.median`、`paddle.compat.nanmedian`、`paddle.compat.softmax`、`paddle.compat.sort`、`paddle.compat.split`、`paddle.compat.min/max`、`paddle.compat.Unfold`。[#74865](https://github.com/PaddlePaddle/Paddle/pull/74865),[#74874](https://github.com/PaddlePaddle/Paddle/pull/74874)
- 新增初始化一系列 API，支持业界通用的参数初始化方式，包括`paddle.nn.init.kaiming_uniform_`、`paddle.nn.init.xavier_uniform_`、`paddle.nn.init.uniform_`、`paddle.nn.init.kaiming_normal_`、`paddle.nn.init.xavier_normal_`、`paddle.nn.init.normal_`、`paddle.nn.init.calculate_gain`、`paddle.nn.init.constant_`、`paddle.nn.init.dirac_`、`paddle.nn.init.eye_`、`paddle.nn.init.ones_`、`paddle.nn.init.orthogonal_`、`paddle.nn.init.trunc_normal_`、`paddle.nn.init.zeros_`。[#74478](https://github.com/PaddlePaddle/Paddle/pull/74478)
- API 新增参数别名用法，例如既可以输入`x`，也可以输入`input`，用法更为灵活。包括 `paddle.maximum`、`paddle.minimum`、`paddle.sqrt`、`paddle.topk`、`paddle.polar`、`paddle.stack`、`paddle.cos`、`paddle.floor`、`paddle.log`、`paddle.pow`、`paddle.rsqrt`、`paddle.sign`、`paddle.sin`、`paddle.multiply`、`paddle.where`等。[#74683](https://github.com/PaddlePaddle/Paddle/pull/74683),[#74795](https://github.com/PaddlePaddle/Paddle/pull/74795),[#74887](https://github.com/PaddlePaddle/Paddle/pull/74887),[#74592](https://github.com/PaddlePaddle/Paddle/pull/74592)
- `paddle.Tensor`新增支持多种初始化方式，支持灵活的创建 Tensor。[#74619](https://github.com/PaddlePaddle/Paddle/pull/74619),[#75022](https://github.com/PaddlePaddle/Paddle/pull/75022),[#75065](https://github.com/PaddlePaddle/Paddle/pull/75065)
- API 新增一些专有参数，增强原有功能。包括 `paddle.nn.functional.gelu`、`paddle.divide/div/div_`、`paddle.add`、`paddle.Tensor.copy_`、`paddle.norm`、`paddle.linalg.norm`、`paddle.nn.functional.silu`、`paddle.repeat_interleave`。[#74485](https://github.com/PaddlePaddle/Paddle/pull/74485),[#74562](https://github.com/PaddlePaddle/Paddle/pull/74562),[#74420](https://github.com/PaddlePaddle/Paddle/pull/74420),[#74768](https://github.com/PaddlePaddle/Paddle/pull/74768),[#74855](https://github.com/PaddlePaddle/Paddle/pull/74855),[#74903](https://github.com/PaddlePaddle/Paddle/pull/74903),[#74788](https://github.com/PaddlePaddle/Paddle/pull/74788),[#74631](https://github.com/PaddlePaddle/Paddle/pull/74631),[#74947](https://github.com/PaddlePaddle/Paddle/pull/74947)
- API 新增一些通用参数：`out`、`device`、`dtype`、`requires_grad`、`pin_memory`、`bias`，增强原有功能。包括 `paddle.zeros`、`paddle.zeros_like`、`paddle.ones`、`paddle.ones_like`、`paddle.arange`、`paddle.eye`、`paddle.empty`、`paddle.empty_like`、`paddle.full`、`paddle.full_like`、`paddle.randn`、`paddle.Tensor.new_full`、`paddle.Tensor.new_empty`、`paddle.Tensor.new_ones`、`paddle.Tensor.new_zeros`、`paddle.tril/triu`、`paddle.bmm`、`paddle.nn.Conv1D/Conv2D/Conv3D/Embedding`、`paddle.diff`、`paddle.cumsum`、`paddle.var`、`paddle.multinomial`、`paddle.mean`等。[#74477](https://github.com/PaddlePaddle/Paddle/pull/74477),[#74526](https://github.com/PaddlePaddle/Paddle/pull/74526),[#74711](https://github.com/PaddlePaddle/Paddle/pull/74711),[#74582](https://github.com/PaddlePaddle/Paddle/pull/74582),[#74624](https://github.com/PaddlePaddle/Paddle/pull/74624),[#74849](https://github.com/PaddlePaddle/Paddle/pull/74849),[#74612](https://github.com/PaddlePaddle/Paddle/pull/74612),[#74875](https://github.com/PaddlePaddle/Paddle/pull/74875),[#74641](https://github.com/PaddlePaddle/Paddle/pull/74641),[#74949](https://github.com/PaddlePaddle/Paddle/pull/74949),[#74918](https://github.com/PaddlePaddle/Paddle/pull/74918),[#74914](https://github.com/PaddlePaddle/Paddle/pull/74914),[#74934](https://github.com/PaddlePaddle/Paddle/pull/74934),[#74920](https://github.com/PaddlePaddle/Paddle/pull/74920),[#74955](https://github.com/PaddlePaddle/Paddle/pull/74955),[#74226](https://github.com/PaddlePaddle/Paddle/pull/74226),[#74946](https://github.com/PaddlePaddle/Paddle/pull/74946)
- API 新增别名，支持更多调用方式。包括 `paddle.Tensor.mul_/mul`、`paddle.autograd.Function`、`paddle.argwhere`、`paddle.cat`、`paddle.clamp`、`paddle.ger`、`paddle.take_along_dim`、`paddle.linalg.matmul`、`paddle.special.logsumexp`、`paddle.concatenate`、`paddle.eq/gt、`paddle.Tensor.take_along_dim`、`paddle.nn.Conv1d/Conv2d/Conv3d`等。[#74493](https://github.com/PaddlePaddle/Paddle/pull/74493),[#74569](https://github.com/PaddlePaddle/Paddle/pull/74569),[#74870](https://github.com/PaddlePaddle/Paddle/pull/74870)

### Bug 修复
- 修复 `paddle.nanmedian` 精度问题。[#74263](https://github.com/PaddlePaddle/Paddle/pull/74263)
- 修复 `paddle.distributed.fleet.utils.hybrid_parallel_util.fused_allreduce_gradients` 在 0-D 下的问题。[#74957](https://github.com/PaddlePaddle/Paddle/pull/74957)
- 修复 `paddle.matmul` 在分布式下的问题。[#74989](https://github.com/PaddlePaddle/Paddle/pull/74989)

### 功能增强
- 针对返回多个 Tensor 的情况，通过 paddle 数据结构来封装，优化体验。包括 `paddle.topk`。[#74931](https://github.com/PaddlePaddle/Paddle/pull/74931)
- 创建类 API 支持 size 为可变参数的用法。[#74494](https://github.com/PaddlePaddle/Paddle/pull/74494)

### 文档
- 新增或修复文档。[#74453](https://github.com/PaddlePaddle/Paddle/pull/74453),[#74846](https://github.com/PaddlePaddle/Paddle/pull/74846),[#74982](https://github.com/PaddlePaddle/Paddle/pull/74982)

### 其他
- 代码风格相关的优化。[#74654](https://github.com/PaddlePaddle/Paddle/pull/74654),[#74655](https://github.com/PaddlePaddle/Paddle/pull/74655),[#74665](https://github.com/PaddlePaddle/Paddle/pull/74665),[#74660](https://github.com/PaddlePaddle/Paddle/pull/74660),[#74667](https://github.com/PaddlePaddle/Paddle/pull/74667),[#74664](https://github.com/PaddlePaddle/Paddle/pull/74664),[#74662](https://github.com/PaddlePaddle/Paddle/pull/74662),[#74661](https://github.com/PaddlePaddle/Paddle/pull/74661),[#74658](https://github.com/PaddlePaddle/Paddle/pull/74658),[#74657](https://github.com/PaddlePaddle/Paddle/pull/74657),[#74666](https://github.com/PaddlePaddle/Paddle/pull/74666),[#74659](https://github.com/PaddlePaddle/Paddle/pull/74659),[#74663](https://github.com/PaddlePaddle/Paddle/pull/74663),[#74656](https://github.com/PaddlePaddle/Paddle/pull/74656),[#74673](https://github.com/PaddlePaddle/Paddle/pull/74673),[#74672](https://github.com/PaddlePaddle/Paddle/pull/74672),[#74671](https://github.com/PaddlePaddle/Paddle/pull/74671),[#74674](https://github.com/PaddlePaddle/Paddle/pull/74674),[#74675](https://github.com/PaddlePaddle/Paddle/pull/74675),[#74670](https://github.com/PaddlePaddle/Paddle/pull/74670),[#74669](https://github.com/PaddlePaddle/Paddle/pull/74669),[#74677](https://github.com/PaddlePaddle/Paddle/pull/74677),[#74709](https://github.com/PaddlePaddle/Paddle/pull/74709),[#74714](https://github.com/PaddlePaddle/Paddle/pull/74714),[#74712](https://github.com/PaddlePaddle/Paddle/pull/74712),[#74713](https://github.com/PaddlePaddle/Paddle/pull/74713),[#74704](https://github.com/PaddlePaddle/Paddle/pull/74704),[#74746](https://github.com/PaddlePaddle/Paddle/pull/74746),[#74748](https://github.com/PaddlePaddle/Paddle/pull/74748),[#74743](https://github.com/PaddlePaddle/Paddle/pull/74743),[#74742](https://github.com/PaddlePaddle/Paddle/pull/74742),[#74744](https://github.com/PaddlePaddle/Paddle/pull/74744),[#74745](https://github.com/PaddlePaddle/Paddle/pull/74745),[#74747](https://github.com/PaddlePaddle/Paddle/pull/74747),[#74794](https://github.com/PaddlePaddle/Paddle/pull/74794),[#74789](https://github.com/PaddlePaddle/Paddle/pull/74789),[#74793](https://github.com/PaddlePaddle/Paddle/pull/74793),[#74786](https://github.com/PaddlePaddle/Paddle/pull/74786),[#74791](https://github.com/PaddlePaddle/Paddle/pull/74791),[#74787](https://github.com/PaddlePaddle/Paddle/pull/74787),[#74827](https://github.com/PaddlePaddle/Paddle/pull/74827),[#74608](https://github.com/PaddlePaddle/Paddle/pull/74608),[#74288](https://github.com/PaddlePaddle/Paddle/pull/74288),[#74287](https://github.com/PaddlePaddle/Paddle/pull/74287),[#74385](https://github.com/PaddlePaddle/Paddle/pull/74385),[#74395](https://github.com/PaddlePaddle/Paddle/pull/74395),[#74475](https://github.com/PaddlePaddle/Paddle/pull/74475),[#74647](https://github.com/PaddlePaddle/Paddle/pull/74647)
- MKLDNN/ONEDNN 相关的优化。[#74299](https://github.com/PaddlePaddle/Paddle/pull/74299),[#74244](https://github.com/PaddlePaddle/Paddle/pull/74244),[#74230](https://github.com/PaddlePaddle/Paddle/pull/74230),[#74314](https://github.com/PaddlePaddle/Paddle/pull/74314),[#74327](https://github.com/PaddlePaddle/Paddle/pull/74327),[#74325](https://github.com/PaddlePaddle/Paddle/pull/74325),[#74326](https://github.com/PaddlePaddle/Paddle/pull/74326),[#74315](https://github.com/PaddlePaddle/Paddle/pull/74315),[#74399](https://github.com/PaddlePaddle/Paddle/pull/74399),[#74398](https://github.com/PaddlePaddle/Paddle/pull/74398),[#74393](https://github.com/PaddlePaddle/Paddle/pull/74393),[#74392](https://github.com/PaddlePaddle/Paddle/pull/74392),[#74367](https://github.com/PaddlePaddle/Paddle/pull/74367),[#74391](https://github.com/PaddlePaddle/Paddle/pull/74391),[#74423](https://github.com/PaddlePaddle/Paddle/pull/74423),[#74424](https://github.com/PaddlePaddle/Paddle/pull/74424),[#74436](https://github.com/PaddlePaddle/Paddle/pull/74436),[#74417](https://github.com/PaddlePaddle/Paddle/pull/74417),[#74410](https://github.com/PaddlePaddle/Paddle/pull/74410),[#74473](https://github.com/PaddlePaddle/Paddle/pull/74473),[#74458](https://github.com/PaddlePaddle/Paddle/pull/74458),[#74501](https://github.com/PaddlePaddle/Paddle/pull/74501),[#74487](https://github.com/PaddlePaddle/Paddle/pull/74487),[#74502](https://github.com/PaddlePaddle/Paddle/pull/74502),[#74513](https://github.com/PaddlePaddle/Paddle/pull/74513),[#74518](https://github.com/PaddlePaddle/Paddle/pull/74518),[#74516](https://github.com/PaddlePaddle/Paddle/pull/74516),[#74507](https://github.com/PaddlePaddle/Paddle/pull/74507),[#74504](https://github.com/PaddlePaddle/Paddle/pull/74504),[#74505](https://github.com/PaddlePaddle/Paddle/pull/74505),[#74509](https://github.com/PaddlePaddle/Paddle/pull/74509),[#74535](https://github.com/PaddlePaddle/Paddle/pull/74535),[#74536](https://github.com/PaddlePaddle/Paddle/pull/74536),[#74517](https://github.com/PaddlePaddle/Paddle/pull/74517),[#74503](https://github.com/PaddlePaddle/Paddle/pull/74503),[#74557](https://github.com/PaddlePaddle/Paddle/pull/74557),[#74550](https://github.com/PaddlePaddle/Paddle/pull/74550),[#74575](https://github.com/PaddlePaddle/Paddle/pull/74575),[#74587](https://github.com/PaddlePaddle/Paddle/pull/74587),[#74576](https://github.com/PaddlePaddle/Paddle/pull/74576),[#74588](https://github.com/PaddlePaddle/Paddle/pull/74588),[#74549](https://github.com/PaddlePaddle/Paddle/pull/74549),[#74581](https://github.com/PaddlePaddle/Paddle/pull/74581),[#74583](https://github.com/PaddlePaddle/Paddle/pull/74583),[#74628](https://github.com/PaddlePaddle/Paddle/pull/74628),[#74630](https://github.com/PaddlePaddle/Paddle/pull/74630),[#74635](https://github.com/PaddlePaddle/Paddle/pull/74635),[#74679](https://github.com/PaddlePaddle/Paddle/pull/74679),[#74648](https://github.com/PaddlePaddle/Paddle/pull/74648),[#74127](https://github.com/PaddlePaddle/Paddle/pull/74127),[#74636](https://github.com/PaddlePaddle/Paddle/pull/74636),[#74552](https://github.com/PaddlePaddle/Paddle/pull/74552),[#74551](https://github.com/PaddlePaddle/Paddle/pull/74551),[#74678](https://github.com/PaddlePaddle/Paddle/pull/74678),[#74680](https://github.com/PaddlePaddle/Paddle/pull/74680),[#74730](https://github.com/PaddlePaddle/Paddle/pull/74730),[#74751](https://github.com/PaddlePaddle/Paddle/pull/74751),[#74895](https://github.com/PaddlePaddle/Paddle/pull/74895),[#74821](https://github.com/PaddlePaddle/Paddle/pull/74821),[#74897](https://github.com/PaddlePaddle/Paddle/pull/74897),[#74734](https://github.com/PaddlePaddle/Paddle/pull/74734)
- 代码实现相关的优化，变量与文件重命名。[#74309](https://github.com/PaddlePaddle/Paddle/pull/74309),[#74597](https://github.com/PaddlePaddle/Paddle/pull/74597),[#74613](https://github.com/PaddlePaddle/Paddle/pull/74613),[#74376](https://github.com/PaddlePaddle/Paddle/pull/74376),[#74479](https://github.com/PaddlePaddle/Paddle/pull/74479),[#74960](https://github.com/PaddlePaddle/Paddle/pull/74960),[#74968](https://github.com/PaddlePaddle/Paddle/pull/74968),[#74977](https://github.com/PaddlePaddle/Paddle/pull/74977)
- 单测相关的优化，单测问题修复。[#74595](https://github.com/PaddlePaddle/Paddle/pull/74595)
- 编译相关的优化，CI 问题修复。[#74356](https://github.com/PaddlePaddle/Paddle/pull/74356),[#74936](https://github.com/PaddlePaddle/Paddle/pull/74936)
- 优化调试与打印信息，优化报错信息。[#74765](https://github.com/PaddlePaddle/Paddle/pull/74765),[#74381](https://github.com/PaddlePaddle/Paddle/pull/74381),[#74384](https://github.com/PaddlePaddle/Paddle/pull/74384),[#74386](https://github.com/PaddlePaddle/Paddle/pull/74386),[#74387](https://github.com/PaddlePaddle/Paddle/pull/74387),[#74383](https://github.com/PaddlePaddle/Paddle/pull/74383),[#74519](https://github.com/PaddlePaddle/Paddle/pull/74519),[#74520](https://github.com/PaddlePaddle/Paddle/pull/74520),[#74468](https://github.com/PaddlePaddle/Paddle/pull/74468)
- 自定义算子相关优化。[#74402](https://github.com/PaddlePaddle/Paddle/pull/74402)
- 分布式 FlexCheckpoint 支持。[#74966](https://github.com/PaddlePaddle/Paddle/pull/74966),[#74593](https://github.com/PaddlePaddle/Paddle/pull/74593),[#74785](https://github.com/PaddlePaddle/Paddle/pull/74785),[#74814](https://github.com/PaddlePaddle/Paddle/pull/74814)

## 2. 基础执行架构

### 新功能
- 动态图支持。[#74484](https://github.com/PaddlePaddle/Paddle/pull/74484)
- 支持 safetensors。[#74642](https://github.com/PaddlePaddle/Paddle/pull/74642), [#74609](https://github.com/PaddlePaddle/Paddle/pull/74609), [#75049](https://github.com/PaddlePaddle/Paddle/pull/75049)
- 添加 offloader 优化计算效率。 [#74837](https://github.com/PaddlePaddle/Paddle/pull/74837)
- 为 conv_transpose 前向计算添加 API 支持。 [#74431](https://github.com/PaddlePaddle/Paddle/pull/74431)
- 添加 offloader 优化计算效率。 [#74837](https://github.com/PaddlePaddle/Paddle/pull/74837)
- 推理部署增加了 w4afp8 量化推理，支持 w4afp8 量化权重纯排及 all2all 通信[#74270](https://github.com/PaddlePaddle/Paddle/pull/74270)

### Bug 修复
- 核心框架与基础设施优化。[#74336](https://github.com/PaddlePaddle/Paddle/pull/74336), [#74554](https://github.com/PaddlePaddle/Paddle/pull/74554), [#74634](https://github.com/PaddlePaddle/Paddle/pull/74634)
- 计算精度与类型处理。 [#74278](https://github.com/PaddlePaddle/Paddle/pull/74278), [#74222](https://github.com/PaddlePaddle/Paddle/pull/74222), [#74830](https://github.com/PaddlePaddle/Paddle/pull/74830)
- 动态维度检查逻辑优化。 [#74633](https://github.com/PaddlePaddle/Paddle/pull/74633), [#74650](https://github.com/PaddlePaddle/Paddle/pull/74650)
- 内存与非法访问修复。 [#74347](https://github.com/PaddlePaddle/Paddle/pull/74347), [#73443](https://github.com/PaddlePaddle/Paddle/pull/73443), [#74953](https://github.com/PaddlePaddle/Paddle/pull/74953)
- 修复报错/告警信息打印。 [#74474](https://github.com/PaddlePaddle/Paddle/pull/74474), [#74533](https://github.com/PaddlePaddle/Paddle/pull/74533), [#74685](https://github.com/PaddlePaddle/Paddle/pull/74685), [#74721](https://github.com/PaddlePaddle/Paddle/pull/74721), [#74754](https://github.com/PaddlePaddle/Paddle/pull/74754)
- 代码质量与文档修正。 [#74378](https://github.com/PaddlePaddle/Paddle/pull/74378), [#74828](https://github.com/PaddlePaddle/Paddle/pull/74828)
- 修复 flashmask API 处理逻辑。 [#74928](https://github.com/PaddlePaddle/Paddle/pull/74928)
- 修复动转静模式下切分 CudaGraph 子图未生效的问题。 ([#74749](https://github.com/PaddlePaddle/Paddle/pull/74749))

### 功能增强
- C++ 扩展开发。 [#74338](https://github.com/PaddlePaddle/Paddle/pull/74338)
- FlexCP 功能优化。 [#74752](https://github.com/PaddlePaddle/Paddle/pull/74752), [#74981](https://github.com/PaddlePaddle/Paddle/pull/74981)
- 优化内存分配。[#74463](https://github.com/PaddlePaddle/Paddle/pull/74463)

### 废弃
- 清理动转静旧 IR 相关单测。 [#74698](https://github.com/PaddlePaddle/Paddle/pull/74698), [#74715](https://github.com/PaddlePaddle/Paddle/pull/74715), [#74718](https://github.com/PaddlePaddle/Paddle/pull/74718), [#74782](https://github.com/PaddlePaddle/Paddle/pull/74782), [#74962](https://github.com/PaddlePaddle/Paddle/pull/74962)

### 其他
- 更改补丁版本。 [#74940](https://github.com/PaddlePaddle/Paddle/pull/74940)

## 3. 分布式&自动并行

### 并行策略
在 3.2 版本中，我们对流水线并行功能进行了多项增强，包括实现了字典参数传递的支持，并扩展了 Pipeline Layer 和 SharedLayerDesc 对非流水线并行的兼容性；同时修复了多个关键问题，包括大尺寸张量的 IPC API 异常、流水线并行中的评估批次和非计算损失问题、MoE 模型的梯度释放错误、PP 场景下 NCCL 通信重建导致的 hang 问题，以及双流水线并行的 event 管理错误；此外还进行了多项性能优化，改进了双流水线并行的计算重叠效率以提升训练性能，并升级了 clear_param_storage 方法使其支持 sharding 模式下多 color 集合的清除和重置操作。

#### 功能新增
- 实现流水线并行（Pipeline Parallel）中字典参数传递的支持。[#74574](https://github.com/PaddlePaddle/Paddle/pull/74574),[#74867](https://github.com/PaddlePaddle/Paddle/pull/74867)
- Pipeline Layer 和 SharedLayerDesc 支持非流水线并行（nonpp parallel）。[#74573](https://github.com/PaddlePaddle/Paddle/pull/74573)

#### Bug 修复
- 修复大尺寸张量的 IPC API 问题。[#74472](https://github.com/PaddlePaddle/Paddle/pull/74472)
- 修复流水线并行中的评估批次（eval batch）及非计算损失（non-compute_loss）问题。[#74170](https://github.com/PaddlePaddle/Paddle/pull/74170)
- 修复 MoE 模型上的梯度释放问题。[#74972](https://github.com/PaddlePaddle/Paddle/pull/74972)
- 修复在 pp 的场景下重建 NCCL comm 存在 hang 的问题。[#73625](https://github.com/PaddlePaddle/Paddle/pull/73625)
- 修复双流水线并行（dual pp）的 event 管理错误。[#74158](https://github.com/PaddlePaddle/Paddle/pull/74158)

#### 优化改进
- 优化双流水线并行的计算重叠（overlap）效率，提升训练性能。[#74527](https://github.com/PaddlePaddle/Paddle/pull/74527)
- 升级 clear_param_storage 方法，支持 sharding 下多个 color 集合清除和重置。[#74741](https://github.com/PaddlePaddle/Paddle/pull/74741)

### 自动并行
#### 功能改进
- 支持分布式张量的同一维度被多个 mesh 维度切分时的默认切分推导规则。[#74396](https://github.com/PaddlePaddle/Paddle/pull/74396)
- 改进 `reshape` 算子的切分推导规则，以支持分布式张量的同一维度被多个 mesh 维度切分的场景。[#74352](https://github.com/PaddlePaddle/Paddle/pull/74352),[#74579](https://github.com/PaddlePaddle/Paddle/pull/74579), [#74565](https://github.com/PaddlePaddle/Paddle/pull/74565)
- 支持在不改变分布式张量数据的情况下改变张量的 mesh。[#74248](https://github.com/PaddlePaddle/Paddle/pull/74248)

#### Bug 修复
- 修复调用 `ProcessMesh` 的 `get_group` 方法时重复创建通信组的 bug。[#73099](https://github.com/PaddlePaddle/Paddle/pull/73099)
- 修复 MoE 场景下`get_local_slices` 方法的 bug。[#74705](https://github.com/PaddlePaddle/Paddle/pull/74705)
- 修复 MoE 场景下梯度裁剪的 bug。[#74916](https://github.com/PaddlePaddle/Paddle/pull/74916)
- 修复流水线并行场景下不同 stage 间无法传递`stop_gradient`参数的 bug。[#73459](https://github.com/PaddlePaddle/Paddle/pull/73459)
- 修复流水线并行场景下梯度裁剪的精度 bug。[#74409](https://github.com/PaddlePaddle/Paddle/pull/74409)
- 修复动态图流水线并行场景下产生冗余输出的 bug。[#74913](https://github.com/PaddlePaddle/Paddle/pull/74913)
- 修复算子`moe_combine`和`moe_gate_dispatch`在 MoE 场景下跑不通的 bug。[#74645](https://github.com/PaddlePaddle/Paddle/pull/74645)

#### 其他
- 支持 dataloader 手动并行和自动并行的精度对齐。[#73941](https://github.com/PaddlePaddle/Paddle/pull/73941)
- 优化动态图流水并行调度逻辑。[#74720](https://github.com/PaddlePaddle/Paddle/pull/74720)

### 通信库
在 3.2 版本中，我们修复了 DeepEP 支持 sm90 编译的一个报错，同时对 DeepEP 申请的显存分配添加了预分配功能，并升级了其 intranode 和 internode 计算 kernel，进一步优化了性能和稳定性。

#### Bug 修复
- 修复 DeepEP 支持 sm90 编译的一个报错。[#74762](https://github.com/PaddlePaddle/Paddle/pull/74762)

#### 功能改进
- 对 DeepEP 申请的显存分配添加预分配功能。[#74465](https://github.com/PaddlePaddle/Paddle/pull/74465)
- 升级 DeepEP 的 intranode 和 internode 计算 kernel。[#74284](https://github.com/PaddlePaddle/Paddle/pull/74284)

## 4. 算子机制
### 新特性
- API 兼容性支持。 [#74506](https://github.com/PaddlePaddle/Paddle/pull/74506), [#74676](https://github.com/PaddlePaddle/Paddle/pull/74676), [#74558](https://github.com/PaddlePaddle/Paddle/pull/74558), [#74572](https://github.com/PaddlePaddle/Paddle/pull/74572), [#74691](https://github.com/PaddlePaddle/Paddle/pull/74691), [#74703](https://github.com/PaddlePaddle/Paddle/pull/74703), [#74750](https://github.com/PaddlePaddle/Paddle/pull/74750), [#74757](https://github.com/PaddlePaddle/Paddle/pull/74757), [#74802](https://github.com/PaddlePaddle/Paddle/pull/74802), [#74546](https://github.com/PaddlePaddle/Paddle/pull/74546), [#74547](https://github.com/PaddlePaddle/Paddle/pull/74547), [#74802](https://github.com/PaddlePaddle/Paddle/pull/74802), [#74859](https://github.com/PaddlePaddle/Paddle/pull/74859), [#74910](https://github.com/PaddlePaddle/Paddle/pull/74910), [#74873](https://github.com/PaddlePaddle/Paddle/pull/74873), [#74882](https://github.com/PaddlePaddle/Paddle/pull/74882), [#74901](https://github.com/PaddlePaddle/Paddle/pull/74901), [#74899](https://github.com/PaddlePaddle/Paddle/pull/74899), [#74449](https://github.com/PaddlePaddle/Paddle/pull/74449)
- 新增 fused_partial_rope 算子。 [#74577](https://github.com/PaddlePaddle/Paddle/pull/74577)

### Bug 修复
- 0-size Tensor 相关修复。 [#74295](https://github.com/PaddlePaddle/Paddle/pull/74295), [#74305](https://github.com/PaddlePaddle/Paddle/pull/74305), [#74323](https://github.com/PaddlePaddle/Paddle/pull/74323), [#74354](https://github.com/PaddlePaddle/Paddle/pull/74354)
- 大 Tensor 相关修复。 [#74242](https://github.com/PaddlePaddle/Paddle/pull/74242), [#74293](https://github.com/PaddlePaddle/Paddle/pull/74293), [#74289](https://github.com/PaddlePaddle/Paddle/pull/74289), [#74279](https://github.com/PaddlePaddle/Paddle/pull/74279), [#74330](https://github.com/PaddlePaddle/Paddle/pull/74330), [#74329](https://github.com/PaddlePaddle/Paddle/pull/74329), [#74342](https://github.com/PaddlePaddle/Paddle/pull/74342), [#74369](https://github.com/PaddlePaddle/Paddle/pull/74369), [#74370](https://github.com/PaddlePaddle/Paddle/pull/74370), [#74404](https://github.com/PaddlePaddle/Paddle/pull/74404), [#74537](https://github.com/PaddlePaddle/Paddle/pull/74537), [#74451](https://github.com/PaddlePaddle/Paddle/pull/74451), [#74172](https://github.com/PaddlePaddle/Paddle/pull/74172), [#74324](https://github.com/PaddlePaddle/Paddle/pull/74324), [#74964](https://github.com/PaddlePaddle/Paddle/pull/74964), [#74360](https://github.com/PaddlePaddle/Paddle/pull/74360), [#74379](https://github.com/PaddlePaddle/Paddle/pull/74379), [#74377](https://github.com/PaddlePaddle/Paddle/pull/74377), [#74380](https://github.com/PaddlePaddle/Paddle/pull/74380), [#74362](https://github.com/PaddlePaddle/Paddle/pull/74362), [#74197](https://github.com/PaddlePaddle/Paddle/pull/74197)
- API 兼容性相关修复。 [#74764](https://github.com/PaddlePaddle/Paddle/pull/74764), [#74869](https://github.com/PaddlePaddle/Paddle/pull/74869), [#74935](https://github.com/PaddlePaddle/Paddle/pull/74935)
- 【开源任务】Paddle CPU/GPU Kernel 精度问题推全。 [#74149](https://github.com/PaddlePaddle/Paddle/pull/74149), [#74598](https://github.com/PaddlePaddle/Paddle/pull/74598), [#74719](https://github.com/PaddlePaddle/Paddle/pull/74719), [#74625](https://github.com/PaddlePaddle/Paddle/pull/74625), [#74555](https://github.com/PaddlePaddle/Paddle/pull/74555)
- 其他重要修复。 [#74282](https://github.com/PaddlePaddle/Paddle/pull/74282), [#74313](https://github.com/PaddlePaddle/Paddle/pull/74313), [#74303](https://github.com/PaddlePaddle/Paddle/pull/74303), [#74306](https://github.com/PaddlePaddle/Paddle/pull/74306), [#74298](https://github.com/PaddlePaddle/Paddle/pull/74298), [#74044](https://github.com/PaddlePaddle/Paddle/pull/74044), [#74290](https://github.com/PaddlePaddle/Paddle/pull/74290), [#74348](https://github.com/PaddlePaddle/Paddle/pull/74348), [#74364](https://github.com/PaddlePaddle/Paddle/pull/74364), [#74332](https://github.com/PaddlePaddle/Paddle/pull/74332), [#74224](https://github.com/PaddlePaddle/Paddle/pull/74224), [#74382](https://github.com/PaddlePaddle/Paddle/pull/74382), [#74406](https://github.com/PaddlePaddle/Paddle/pull/74406), [#74434](https://github.com/PaddlePaddle/Paddle/pull/74434), [#74448](https://github.com/PaddlePaddle/Paddle/pull/74448), [#74457](https://github.com/PaddlePaddle/Paddle/pull/74457), [#74322](https://github.com/PaddlePaddle/Paddle/pull/74322), [#74530](https://github.com/PaddlePaddle/Paddle/pull/74530), [#74716](https://github.com/PaddlePaddle/Paddle/pull/74716), [#74839](https://github.com/PaddlePaddle/Paddle/pull/74839), [#74842](https://github.com/PaddlePaddle/Paddle/pull/74842), [#74854](https://github.com/PaddlePaddle/Paddle/pull/74854), [#74919](https://github.com/PaddlePaddle/Paddle/pull/74919), [#74767](https://github.com/PaddlePaddle/Paddle/pull/74767), [#75003](https://github.com/PaddlePaddle/Paddle/pull/75003)

### 功能增强
- API 兼容能力提升。 [#74456](https://github.com/PaddlePaddle/Paddle/pull/74456), [#74480](https://github.com/PaddlePaddle/Paddle/pull/74480), [#74523](https://github.com/PaddlePaddle/Paddle/pull/74523), [#74490](https://github.com/PaddlePaddle/Paddle/pull/74490), [#74548](https://github.com/PaddlePaddle/Paddle/pull/74548), [#74596](https://github.com/PaddlePaddle/Paddle/pull/74596), [#74568](https://github.com/PaddlePaddle/Paddle/pull/74568), [#74559](https://github.com/PaddlePaddle/Paddle/pull/74559), [#74629](https://github.com/PaddlePaddle/Paddle/pull/74629), [#74623](https://github.com/PaddlePaddle/Paddle/pull/74623), [#74700](https://github.com/PaddlePaddle/Paddle/pull/74700), [#74643](https://github.com/PaddlePaddle/Paddle/pull/74643), [#74602](https://github.com/PaddlePaddle/Paddle/pull/74602), [#74783](https://github.com/PaddlePaddle/Paddle/pull/74783), [#74781](https://github.com/PaddlePaddle/Paddle/pull/74781), [#74735](https://github.com/PaddlePaddle/Paddle/pull/74735), [#74725](https://github.com/PaddlePaddle/Paddle/pull/74725), [#74815](https://github.com/PaddlePaddle/Paddle/pull/74815), [#74856](https://github.com/PaddlePaddle/Paddle/pull/74856), [#74925](https://github.com/PaddlePaddle/Paddle/pull/74925), [#74545](https://github.com/PaddlePaddle/Paddle/pull/74545), [#74932](https://github.com/PaddlePaddle/Paddle/pull/74932), [#74784](https://github.com/PaddlePaddle/Paddle/pull/74784)
- slice/stride 相关优化。 [#74731](https://github.com/PaddlePaddle/Paddle/pull/74731), [#74740](https://github.com/PaddlePaddle/Paddle/pull/74740), [#74769](https://github.com/PaddlePaddle/Paddle/pull/74769), [#74810](https://github.com/PaddlePaddle/Paddle/pull/74810), [#74841](https://github.com/PaddlePaddle/Paddle/pull/74841), [#74954](https://github.com/PaddlePaddle/Paddle/pull/74954), [#74888](https://github.com/PaddlePaddle/Paddle/pull/74888), [#74944](https://github.com/PaddlePaddle/Paddle/pull/74944), [#74312](https://github.com/PaddlePaddle/Paddle/pull/74312), [#74291](https://github.com/PaddlePaddle/Paddle/pull/74291), [#74271](https://github.com/PaddlePaddle/Paddle/pull/74271), [#74320](https://github.com/PaddlePaddle/Paddle/pull/74320), [#74344](https://github.com/PaddlePaddle/Paddle/pull/74344), [#74727](https://github.com/PaddlePaddle/Paddle/pull/74727), [#74637](https://github.com/PaddlePaddle/Paddle/pull/74637)
- 算子优化与 CUDA 支持。 [#74693](https://github.com/PaddlePaddle/Paddle/pull/74693), [#74922](https://github.com/PaddlePaddle/Paddle/pull/74922), [#74967](https://github.com/PaddlePaddle/Paddle/pull/74967)
- 改进调试信息、兼容性增强。 [#74372](https://github.com/PaddlePaddle/Paddle/pull/74372), [#74622](https://github.com/PaddlePaddle/Paddle/pull/74622)
- 算子功能扩展与优化。 [#74790](https://github.com/PaddlePaddle/Paddle/pull/74790), [#74979](https://github.com/PaddlePaddle/Paddle/pull/74979)

### 性能优化
- FP8 计算优化。 [#74471](https://github.com/PaddlePaddle/Paddle/pull/74471), [#74684](https://github.com/PaddlePaddle/Paddle/pull/74684), [#74911](https://github.com/PaddlePaddle/Paddle/pull/74911)
- 基础算子性能优化。 [#74442](https://github.com/PaddlePaddle/Paddle/pull/74442), [#74638](https://github.com/PaddlePaddle/Paddle/pull/74638)
- 支持 fa3 变长序列反向计算并优化前向 API。 [#73831](https://github.com/PaddlePaddle/Paddle/pull/73831)
- 新增 FlashMask V2 功能。 [#74729](https://github.com/PaddlePaddle/Paddle/pull/74729)

### 文档
- 修复英文文档问题以及版权年份问题。 [#74737](https://github.com/PaddlePaddle/Paddle/pull/74737)

### 其他
- 在 XPU 硬件上默认开启 WITH_XPU_FFT 选项。 [#74699](https://github.com/PaddlePaddle/Paddle/pull/74699)

## 5. 硬件适配
### 类 CUDA 硬件接入方案完善
- 类 CUDA 硬件接入方案支持 cuBlas kernel 的复用 [#74591](https://github.com/PaddlePaddle/Paddle/pull/74591),
- 类 CUDA 硬件接入方案已知问题修复
  [#74397](https://github.com/PaddlePaddle/Paddle/pull/74397), [#74411](https://github.com/PaddlePaddle/Paddle/pull/74411), [#74428](https://github.com/PaddlePaddle/Paddle/pull/74428), [#74877](https://github.com/PaddlePaddle/Paddle/pull/74877), [#74939](https://github.com/PaddlePaddle/Paddle/pull/74939)

### 主仓单测支持多硬件
- 单测支持多硬件 [#74349](https://github.com/PaddlePaddle/Paddle/pull/74349), [#74363](https://github.com/PaddlePaddle/Paddle/pull/74363)，[#74806](https://github.com/PaddlePaddle/Paddle/pull/74806), [#74868](https://github.com/PaddlePaddle/Paddle/pull/74868), [#74820](https://github.com/PaddlePaddle/Paddle/pull/74820), [#74927](https://github.com/PaddlePaddle/Paddle/pull/74927)

### 新增 Custom Device API 支持
- 新增 Custom Device API 支持 [#74308](https://github.com/PaddlePaddle/Paddle/pull/74308), [#74371](https://github.com/PaddlePaddle/Paddle/pull/74371), [#74539](https://github.com/PaddlePaddle/Paddle/pull/74539)

## 6. 安装环境
### Bug 修复

- 修复 flashattent 编译缓存的 bug。[#74388](https://github.com/PaddlePaddle/Paddle/pull/74388)
- 修复 site.USER_SITE 为 None 的 bug。 [#74373](https://github.com/PaddlePaddle/Paddle/pull/74373)
- 修复多架构 Linux 系统下 gtest 的编译 bug。 [#74723](https://github.com/PaddlePaddle/Paddle/pull/74723)
- 修复在 WITH_GPU=ON 情况下 DEBUG 模式编译多个报错。 [#74401](https://github.com/PaddlePaddle/Paddle/pull/74401)
- 修复 Windows 下 CUDA12.6 编译 bug。 [#74990](https://github.com/PaddlePaddle/Paddle/pull/74990)
- 修复 api-benchmark 基线流水线 bug。 [#74770](https://github.com/PaddlePaddle/Paddle/pull/74770)
- 修复 api-benchmark 基线流水线 bug。 [#74778](https://github.com/PaddlePaddle/Paddle/pull/74778)
- 修复 api-benchmark 基线流水线 bug。 [#74779](https://github.com/PaddlePaddle/Paddle/pull/74779)
- 修复 api-benchmark 基线流水线 bug。 [#74780](https://github.com/PaddlePaddle/Paddle/pull/74780)
- 修复 api-benchmark 基线流水线 bug。 [#74800](https://github.com/PaddlePaddle/Paddle/pull/74800)
- 修复 api-benchmark 基线流水线 bug。 [#74803](https://github.com/PaddlePaddle/Paddle/pull/74803)

### 其他

- 禁用 test_custom_contiguous 单测。 [#74337](https://github.com/PaddlePaddle/Paddle/pull/74337)
- 支持录取 slice 流水线基线任务定时触发。 [#74419](https://github.com/PaddlePaddle/Paddle/pull/74419)
- 支持 slice 录基线添加手动指定 pr。 [#74445](https://github.com/PaddlePaddle/Paddle/pull/74445)
- 检查代码中是否带有中问题。 [#74460](https://github.com/PaddlePaddle/Paddle/pull/74460)
- 支持 CI PaddleX 在 XPU 上的任务。 [#74426](https://github.com/PaddlePaddle/Paddle/pull/74426)
- 支持 slice 流水线豁免机制。 [#74482](https://github.com/PaddlePaddle/Paddle/pull/74482)
- 更新 paddle 基础镜像。 [#73423](https://github.com/PaddlePaddle/Paddle/pull/73423)
- windows 固定 ninja 版本 1.11。 [#74590](https://github.com/PaddlePaddle/Paddle/pull/74590)
- 支持添加关闭 pr 取消 CI。 [#74604](https://github.com/PaddlePaddle/Paddle/pull/74604)
- 支持快速跳过所有 CI。 [#74696](https://github.com/PaddlePaddle/Paddle/pull/74696)
- 增加 api-benchmark 基线流水线。 [#74690](https://github.com/PaddlePaddle/Paddle/pull/74690)
- 更新 nccl 版本。 [#74809](https://github.com/PaddlePaddle/Paddle/pull/74809)
- 更新 approve 流水线 RD 名单。 [#74838](https://github.com/PaddlePaddle/Paddle/pull/74838)
- 更新 approve 流水线 RD 名单。 [#74902](https://github.com/PaddlePaddle/Paddle/pull/74902)
- 更新 safetensor 到镜像中。  [#74904](https://github.com/PaddlePaddle/Paddle/pull/74904)
- 添加 flashatten 的编译 flag。 [#74959](https://github.com/PaddlePaddle/Paddle/pull/74959)
- 临时禁用 win-inference 流水线。 [#74980](https://github.com/PaddlePaddle/Paddle/pull/74980)
- 支持 windows 编译 phi 动态库。 [#74950](https://github.com/PaddlePaddle/Paddle/pull/74950)

## 7. 贡献者名单
AIbin, Ayakouji, baiyue, baoqiwen, Chang Lu, Chen Zhiyang, co63oc, cyberslack_lee, cyy536, datutu-L, Deng Haodong, Difer, Eddie-Wang, enzodechine, fangfangssj, feri, fxyfxy777, ggggxm, GoldPancake, gouzil, Gu Shiwei, Haze188 灏喆, hohdiy, hong, HU Shenwei, huangjiyi, HydrogenSulfate, kjagsdq, LCStayingdullCircuit, Leo Guo, lightbrother, liufengwei0103, liuruyan, LiYuRio, LLSGYN, Lucas, Luckycheng222, lzy, Nana, Nyakku Shigure, ooo oo, Qianyue He, risemeup1, Ruibiao Chen, Ryan, Shuhao Liang, sneaxiy, Starrysea996, SUN Dong, Tao Luo, Tian, tianhaodongbd, tianshuo78520a, umiswing, waliwali777, wanghuancoder, Wenhao.Dai, wyw, XiaoguangHu, xiaoguoguo626807, xingmingyyj, Yichen Zhang, Yohanna, yongqiangma, Yuan Xiaolan, YUNSHEN XIE, Yuntao Nie, Yuqiang Ge, Yutian Rao, Zero Rains, Zhan Rongrui, Zhang Ting, zhanghonggeng, Zhaowu Pan, zhengshengning, ZhenxingLi, Zhou Xin, zhupengyang, zhwesky2010, Zichao, zty-king, Zx, zyfncg, zzm, 周周周, 正在学习, 苍天荒
