# 报错信息文案书写规范

规范概要：

1. 第 1 节，报错文案书写模板，属于推荐参考的形式，根据情景不同，如果有简洁并且更易于用户理解的写法，可以灵活使用
2. 第 2 节，必须遵循的规范条目，为必须遵守的报错信息书写规则，前四条已加入 CI 监控
3. 第 3 节，报错信息规范示例库，是从 Paddle 中抽取的一些已有的 PADDLE_ENFORCE，将其改写为合规的示例，便于参考
4. 第 4 节，后续补充的一些辅助性规范

执行说明：

1. 规范在执行过程中，可能会发现现有规范未考虑到的方面，需要在实施过程中不断补充与完善，也请大家积极反馈意见
2. 报错信息规范示例库，示例越丰富，越有参考价值，非常鼓励大家补充新的示例
3. 规范匹配情况较为复杂，可能出现符合规范的写法被匹配为不合规，届时请找 [chenwhql](https://github.com/chenwhql) approve

## 1. 报错文案书写模板

Paddle 中的报错提示文案均推荐按照以下结构书写，具体包括：
- Python 端抛出的各类错误的提示信息
- C++端 PADDLE_ENFORCE_*与 PADDLE_THROW 的提示信息

> 注：文案的关键是要将错误描述清楚，模板仅供参考

### 1.1 三段式报错文案书写（错误 - 预期 - 建议）

#### 1.1.1 第一段：指明错误（必写）

- 直接陈述错误：
  - 推荐的描述：
    - A 出错，B 没有初始化，C 不存在，D 的值不正确，E 不匹配等
      - 示例：`Mismatched label shape.`
  - 不推荐的描述：A 应该怎么样，B 不应该怎么样
    - 出错了，首先直接告诉用户出错即可
    - 除非必要，不建议以应该/不应该的语气指出错误
    - 应该或者不应该如何，属于第二段的阐述期望结果的内容

- 本段注意事项：
  1. 属性变量要指明错误的主体，例如 Op 输入输出要写明是哪个 Op 的输入输出错误，区分前反向 Op
  2. 指明错误是告诉用户一个事实，一般不允许出现 magic number（意义不明的数字），用英语句子陈述即可

#### 1.1.2 第二段：期望值与实际值对比（尽可能提供）

- 写明此处期望的输入是什么，而实际的输入是什么
  - 示例：`Expected labels dimension=1. Received 4.`

- 本段注意事项：
  1. 将必要信息提供完整，比如 Shape 出错，需要将具体的 Shape 输出进行对比，并指明出错的维度
  2. 如果第一段的错误是单值描述，本段可以省略。例如 A 为空指针，B 不存在，没必要在此处写明期望 A 不为空，B 应该存在等

#### 1.1.3 第三段：修改建议（尽可能提供）

- 写明此处的错误可能是由什么导致的，应该如何修改
  - 示例：`Suggested Fix: If your classifier expects one-hot encoding label, check your n_classes argument to the estimatorand/or the shape of your label.Otherwise, check the shape of your label.`

- 本段注意事项：
  - 可以写明修改建议的情况一般适用于一些共性问题，例如
    - startup_program 没有执行
    - 某个重要参数没有设置
    - 某个环境配置可能有问题

更多具体的示例见第 3 节。

## 2. 必须遵循的规范条目

目前框架内可以使用的报错检查宏包括：

```
PADDLE_ENFORCE：布尔检查宏
PADDLE_ENFORCE_NOT_NULL：空指针检查宏
PADDLE_ENFORCE_EQ：相等检查宏
PADDLE_ENFORCE_NE：不相等检查宏
PADDLE_ENFORCE_GT：大于检查宏
PADDLE_ENFORCE_GE：大于等于检查宏
PADDLE_ENFORCE_LT：小于检查宏
PADDLE_ENFORCE_LE：小于等于检查宏
PADDLE_THROW：抛出异常宏
PADDLE_ENFORCE_GPU_SUCCESS：GPU API 执行成功检查宏
```

详细见：[paddle/phi/core/enforce.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/enforce.h)

PADDLE_ENFORCE_*与 PADDLE_THROW 的提示信息书写必须遵循以下条目：

### 2.1 C++端原则上不允许使用`PADDLE_ENFORCE`表达式（CI 测试中有检查）

- 为什么不推荐 `PADDLE_ENFORCE` 表达式？
  - `PADDLE_ENFORCE(COND, ...)`表达式接收的是 bool 型表达式，只能判断 true 和 false，不能给出具体错误提示。
  - `PADDLE_ENFORCE_GT(__VAL0, __VAL1, ...)` 等表达式接收具体参数，并对应着具体比较条件，易于给出精准的错误信息。
  - `如 PADDLE_ENFORCE(A>B)`只能报`true`和`false`，但改成`PADDLE_ENFORCE_GT(A, B)`，会报出 A 和 B 的具体值是什么。

> 注意：
> - GPU 内核函数中需要检查报错的话，由于 cuda C 语法限制，仍然需要使用 `PADDLE_ENFORCE` ，且不需要带有报错类型。
> - GPU 内核函数在`.cu`文件中，但并不是所有`.cu`文件中的都是 GPU 内核函数。只有使用` __global__ `，`HOSTDEVICE`等关键字装饰的函数才是。

```c++
__global__ void ComputeDifferent(T *centers_diff, const T *X, const T *centers,
                                 const int64_t *ids, const int64_t N,
                                 const int64_t K, const int64_t D) {
  int idx = threadIdx.x;
  int idy = blockIdx.x + threadIdx.y * GridDimX;

  while (idy < K) {
    int64_t id = ids[idy];
    PADDLE_ENFORCE(id >= 0, "received id:", id);
    PADDLE_ENFORCE(id < N, "received id:", id);
    T *out = centers_diff + idy * D;
    const T *x = X + idy * D;
    const T *cent = centers + id * D;
    for (int i = idx; i < D; i += BlockDimX) {
      out[i] = x[i] - cent[i];
    }
    idy += BlockDimY * GridDimX;
  }
}
```

### 2.2 不允许省略或为空字符串（CI 测试中有检查）

- 错误示例：
```c++
PADDLE_ENFORCE(ctx->HasInput("X"));

PADDLE_ENFORCE(ctx->HasInput("X"), "");
```

### 2.3 不允许提示过短，至少长于 20 个字符（CI 测试中有检查）

- 错误示例：
```c++
PADDLE_ENFORCE(i != nullptr, "I must be set");
```

### 2.4 必须指明错误类型（CI 测试中有检查）

- Python 端直接选择合适的 Python 类型错误即可

- C++端当前声明了 12 种错误类型（具体见第三节中的详细示例）
  - InvalidArgument：参数错误
  - NotFound：目标未找到
  - OutOfRange：越界错误
  - AlreadyExists：目标已存在，目标重复
  - ResourceExhausted：资源耗尽
  - PreconditionNotMet：前提条件不满足
  - PermissionDenied：操作不允许
  - ExecutionTimeout：执行超时
  - Unimplemented：功能未实现
  - Unavailable：服务不可用
  - Fatal：严重错误
  - External：外部错误，第三方库错误

> 用法概要：
> 在整个错误提示字符串（包含变长参数列表）的外面包裹`phi::errors::ErrorType()`
>
> 简要示例（注意后面括号的位置）：
> - 未指明错误类型的: `PADDLE_ENFORCE(true, "example: %s", str);`
> - 指明错误类型的: `PADDLE_ENFORCE(true, phi::errors::InvalidArgument("example: %s", str));`

正确示例：
```c++
PADDLE_ENFORCE_GT(y_dims.size(), y_num_col_dims,
                      phi::errors::InvalidArgument("The input tensor Y's dimensions of MulOp "
                      "should be larger than y_num_col_dims. But received Y's "
                      "dimensions = %d, Y's shape = [%s], y_num_col_dims = %d.",
                      y_dims.size(), y_dims, y_num_col_dims));
```

错误示例：
```c++
PADDLE_ENFORCE_GT(y_dims.size(), y_num_col_dims,
                      "The input tensor Y's dimensions of MulOp "
                      "should be larger than y_num_col_dims. But received Y's "
                      "dimensions = %d, Y's shape = [%s], y_num_col_dims = %d.",
                      y_dims.size(), y_dims, y_num_col_dims);
```

> 注意：__CUDA_ARCH__下的 PADDLE_ENFORCE 尚不支持声明错误类型，如果遇到，找审核人员 approve 即可


### 2.5 不允许在提示中使用 C++端开发人员定义的变量缩写，应展开为完整英语单词

错误示例：
```c++
PADDLE_ENFORCE(forward_pd != nullptr,
               "Fail to find eltwise_fwd_pd in device context");
```

### 2.6 确保提示不存在语法错误

错误示例：
```c++
PADDLE_ENFORCE(context->HasInput("X"),
               "ArrayToLoDTensorOp must has input X."); //must has?
```

## 3. 报错信息规范示例库

考虑到开发者对于前述标准理解存在差异，对于错误的归类也可能存在疑惑，所以此处尽可能地提供了各类错误的示例，以及相关提示的参考写法，请开发者在优化报错信息的时候，主动参考此处的规范示例。

### 3.1 InvalidArgument - 参数有误

用户传入了非法的参数，包含各种参数类型错误，应该是最为普遍的错误类型

#### 3.1.1 ShapeError

```c++
PADDLE_ENFORCE_EQ(
    output_shape[unk_dim_idx] * capacity, -in_size,
    phi::errors::InvalidArgument(
        "The 'shape' attribute in ReshapeOp is invalid. "
        "The input tensor X'size must be divisible by known "
        "capacity of 'shape'. "
        "But received X's shape = [%s], X's size = %d, "
        "'shape' is [%s], known "
        "capacity of 'shape' is %d.",
        in_dims, in_size, framework::make_ddim(shape), capacity));
```

#### 3.1.2 参数为空（列表为空，空指针等）

```c++
PADDLE_ENFORCE_NE(vars.empty(), true, phi::errors::InvalidArgument(
                                          "Variable names are empty."));
```

#### 3.1.3 参数有误，与预期值不相等

```c++
PADDLE_ENFORCE_GT(batch_size, 0, phi::errors::InvalidArgument(
                                    "Batch size %d is illegal.", batch_size));

PADDLE_ENFORCE_NE(
    num, 0,
    phi::errors::InvalidArgument(
        "The number of ids can not be zero, you need padding "
        "it in data generator, or if there is something wrong with "
        "the data, please check if the data contains unresolvable "
        "characters.\nplease check this error line: %s.",
        str));
```

#### 3.1.4 参数格式错误

```c++
PADDLE_ENFORCE_NE(in.format(), MKLDNNMemoryFormat::format_undef,
          phi::errors::InvalidArgument(
              "Input tensor format is invalid. Input tensor should "
              "have specified memory format."));
```

#### 3.1.5 参数未初始化

```c++
PADDLE_ENFORCE_EQ(proto_->IsInitialized(), true,
                  phi::errors::InvalidArgument(
                      "Operator's Proto in op info is not initialized."));

PADDLE_ENFORCE_EQ(
    t->IsInitialized(), true,
    phi::errors::InvalidArgument(
        "The Tensor in the %s Op's Input Variable %s(%s) is "
        "not initialized.",
        Type(), name, ctx.Inputs(name).at(i)));
```

#### 3.1.6 参数类型错误

```c++
PADDLE_ENFORCE(
    tmp == *data_type || *data_type == dafault_data_type,
    phi::errors::InvalidArgument(
        "The DataType of %s Op's duplicable Variable %s must be "
        "consistent. The current variable type is (%s), but the "
        "previous variable type is (%s).",
        Type(), name, DataTypeToString(tmp),
        DataTypeToString(*data_type)));

PADDLE_ENFORCE_EQ(
    valid, true,
    phi::errors::InvalidArgument(
        "Tensor holds the wrong type, it holds %s, but desires to be %s.",
        DataTypeToString(type_),
        DataTypeToString(DataTypeTrait<T>::DataType())));
```

#### 3.1.7 参数解析错误

```c++
PADDLE_ENFORCE_EQ(success, true,
                  phi::errors::InvalidArgument(
                      "Fail to parse DataFeedDesc from string: %s.",
                      data_feed_desc_str.c_str()));
```

#### 3.1.8 LoD 错误

```c++
PADDLE_ENFORCE_GT(lod_level, 0, phi::errors::InvalidArgument(
                                    "Input(X) Tensor of SequencePoolOp "
                                    "does not contain LoD information."));
```

### 3.2 NotFound - 未找到目标

申请的实体找不到，要找的变量为空，输入输出不存在等

- 和空指针区分开，找不到变量和变量没有被正确赋值，是两个层面的概念

#### 3.2.1 Op 输入输出未找到

```c++
PADDLE_ENFORCE_EQ(
    ctx->HasInput("X"), true,
    phi::errors::NotFound("Input(X) of MulOp is not found."));
PADDLE_ENFORCE_EQ(
    ctx->HasInput("Y"), true,
    phi::errors::NotFound("Input(Y) of MulOp is not found."));
PADDLE_ENFORCE_EQ(
    ctx->HasOutput("Out"), true,
    phi::errors::NotFound("Output(Out) of MulOp is not found."));
```

#### 3.2.2 缺少节点

```c++
PADDLE_ENFORCE_NOT_NULL(
    p, phi::errors::NotFound("subgraph has no node %s.", name.c_str()));
```

#### 3.2.3 文件未找到

```c++
PADDLE_ENFORCE_GT(file_cnt, 0,
                  phi::errors::NotFound("Input file list is empty."));
```

#### 3.2.4 其他

```c++
PADDLE_ENFORCE_NOT_NULL(
    var_desc, phi::errors::NotFound("%s is not found.", var_name));

PADDLE_ENFORCE_NOT_NULL(
    proto_,
    phi::errors::NotFound("Operator's Proto has not been registered."));
```

### 3.3 OutOfRange - 越界错误

```c++
PADDLE_ENFORCE_LT(
    i, N, phi::errors::OutOfRange("Array index out of bounds."));

PADDLE_ENFORCE_GT(value, lower_bound_,
                  phi::errors::OutOfRange("Attribute GreaterThan check failed."));
```

### 3.4 AlreadyExists - 目标已存在 / 目标重复

查找的实体已存在，或者某些仅允许存在单个实例的个体，却找到了多个

```c++
PADDLE_ENFORCE_EQ(
    attrs_.count(attr_name), 0,
    phi::errors::AlreadyExists(
        "The attribute %s has been set in the graph.", attr_name));

PADDLE_ENFORCE_NE(Has(pass_type), true,
    phi::errors::AlreadyExists(
        "Pass %s has been registered.", pass_type));

PADDLE_ENFORCE_LE(ins.size(), 1UL,
    phi::errors::AlreadyExists(
        "Operator %s's input %s should contain only one variable.", type_, name));

PADDLE_ENFORCE_EQ(
    fused_var_set.count(fused_var_name), 0,
    phi::errors::AlreadyExists(
         "The fused variable already exists."));
```

### 3.5 PermissionDenied - 操作不允许

当前操作不允许被执行。

```c++
PADDLE_ENFORCE_NE(a, b, phi::errors::PermissionDenied(
                            "Cannot connect the same node in the graph."));
```

### 3.6 ResourceExhausted - 资源耗尽

```c++
PADDLE_THROW_BAD_ALLOC(phi::errors::ResourceExhausted(
    "\n\nOut of memory error on GPU %d. "
    "Cannot allocate %s memory on GPU %d, "
    "available memory is only %s.\n\n"
    "Please check whether there is any other process using GPU %d.\n"
    "1. If yes, please stop them, or start PaddlePaddle on another GPU.\n"
    "2. If no, please decrease the batch size of your model.\n",
    place_.device, string::HumanReadableSize(size), place_.device,
    string::HumanReadableSize(avail), place_.device));

PADDLE_THROW_BAD_ALLOC(phi::errors::ResourceExhausted(
     "\n\nOut of memory error on GPU %d. "
     "Cannot allocate %s memory on GPU %d, "
     "available memory is only %s.\n\n"
     "Please check whether there is any other process using GPU %d.\n"
     "1. If yes, please stop them, or start PaddlePaddle on another GPU.\n"
     "2. If no, please try one of the following suggestions:\n"
     "   1) Decrease the batch size of your model.\n"
     "   2) FLAGS_fraction_of_gpu_memory_to_use is %.2lf now, "
     "please set it to a higher value but less than 1.0.\n"
     "      The command is "
     "`export FLAGS_fraction_of_gpu_memory_to_use=xxx`.\n\n",
     gpu_id_, string::HumanReadableSize(size), gpu_id_,
     string::HumanReadableSize(avail), gpu_id_,
     FLAGS_fraction_of_gpu_memory_to_use));
```

### 3.7 PreconditionNotMet - 前提条件有误

当前执行的操作，需要一定的前提条件满足才能够执行

```c++
PADDLE_ENFORCE_NOT_NULL(
    mutex_for_pick_file_,
    phi::errors::PreconditionNotMet(
        "You should call SetFileListMutex before PickOneFile"));

PADDLE_ENFORCE_NOT_NULL(
    root_scope_,
    phi::errors::PreconditionNotMet(
        "root_scope should be set before creating thread scope."));

PADDLE_ENFORCE_NE(
    fetched_var_it, fetched_vars->end(),
    phi::errors::PreconditionNotMet(
        "Cannot find fetched variable(%s). Perhaps the main_program "
        "is not set to ParallelExecutor.",
        var_name));

PADDLE_ENFORCE_EQ(finish_start_, true,
                  phi::errors::PreconditionNotMet(
                      "Datafeed has not started running yet."));

PADDLE_ENFORCE_NE(framework::product(y_dims), 0,
                  phi::errors::PreconditionNotMet(
                      "The Input variable Y(%s) has not "
                      "been initialized. You may need to confirm "
                      "if you put exe.run(startup_program) "
                      "after optimizer.minimize function.",
                      ctx->Inputs("Y").front());

PADDLE_ENFORCE_NE(FLAGS_use_ngraph, true,
                  phi::errors::PreconditionNotMet(
                      "Please compile with NGRAPH first to use NGRAPH."));
```

### 3.8 ExecutionTimeout - 执行超时

执行响应时间过长，或者通信超时。

```c++
PADDLE_THROW(
    phi::errors::ExecutionTimeout(
        "Node communication timeout, self rank is %d pair rank is %d.",
        self_rank_,
        last_check_rank));
```

### 3.9. Unimplemented - 功能尚未实现

尚未实现或支持，但之后有可能会实现

```c++
PADDLE_ENFORCE_NE(iter, operations_.end(),
                  phi::errors::Unimplemented(
                      "Operation %s is not supported yet.", op_type));

PADDLE_ENFORCE_EQ(
    all_reduce_ops.size(), grads.size(),
    phi::errors::Unimplemented(
        "The number of all_reduce OpHandle is not equal to the "
        "number of grads. Maybe some gradients are sparse type, "
        "it is not supported currently."));
```

### 3.10 Unavailable - 服务不可用

当前服务不可用，或当前操作不能执行。

10.1 IO 错误

```c++
PADDLE_ENFORCE_NE(file_descriptor, -1, phi::errors::Unavailable(
                                            "Cannot open file %s.", filename));

PADDLE_ENFORCE_EQ(fin.good(), true, phi::errors::Unavailable(
                                        "Cannot open file %s.", filename));

PADDLE_ENFORCE_EQ(
    file.is_open(), true,
    phi::errors::Unavailable("Can not open %s to add benchmark.", path));
```

### 3.11 Fatal - 严重错误

未预料到的，严重的错误，例如段错误。

> 用于后期增加 try-catch 处理非预期的异常，开发者一般不会用到。

```c++
PADDLE_THROW(platform::errors::Fatal(
        "Custom operator raises an unknown exception in runtime."));
```

### 3.12 External - 外部 / 第三方库错误

使用 PADDLE_ENFORCE_GPU_SUCCESS 检查宏的时候，会自动封装这个错误类型，一般来说，不需要开发者专门指定这个类型

如果有除 CUDA 之外的类库使用可能会出错的话，也可以使用这个类型

## 4. 补充规范

### 4.1 PADDLE_ENFORCE_GPU_SUCCESS 为特殊宏，无需再报错类型

- PADDLE_ENFORCE_GPU_SUCCESS 用于处理 Cuda/Hip 及其相关 lib 的错误，但该类错误环境原因居多，即使要求开发人员写报错信息，也较难起到关键的提示作用，使用 CUDA 官方提供的错误代码及描述是更好的选择，因此我们对该宏进行了升级

- 使用该宏时，直接在其中写 CUDA 的相关调用即可，不再需要写报错类型与报错信息，例如

```c++
PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllReduce(
      src_ptr, dst_ptr, src.numel(), nccl_dtype, ncclSum, comm, stream));
```

### 4.2 访问 variant 推荐使用 PADDLE_GET 系列宏，并禁止直接使用`paddle::get`

- `paddle::get`是一个不安全的第三方库函数，get 失败会直接抛出`bad_variant_access`，可能不会有栈信息，也不会有出错文件与行号的提示，导致此类问题很难调试，所以新定义了 PADDLE_GET 系列宏

- 共新增了三个 PADDLE_GET 宏，具体如下：

```
- PADDLE_GET (e.g. return int& or int*)
- PADDLE_GET_CONST(e.g. return const int& or const int*)
- PADDLE_GET_MUTABLE (e.g. return int or int*)
```

这三个宏可以满足的 paddle 中对于`paddle::get`从 variant 获取值的需求

- CI 中对此有规则检查，不允许直接使用`paddle::get`，需要使用新增的 PADDLE_GET 系列宏，如果不想使用 PADDLE_GET 系列宏，可以为`paddle::get`加 try-catch，然后找指定人员 approve

### 4.3 禁止使用 LOG(FATAL)

- LOG(FATAL)也可以在程序运行中抛出错误，但是抛出的错误难以阅读，无效信息多，且没有出错文件行号等关键信息，paddle 中有 100 余处使用了 LOG(FATAL)，这些均替换为 PADDLE_THROW，且在 CI 中禁止 LOG(FATAL)的使用
