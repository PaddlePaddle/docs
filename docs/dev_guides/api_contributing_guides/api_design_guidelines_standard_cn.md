# API 设计和命名规范

## API 设计规范

### 总体原则

1. 单一职责，每个 API 应只完成单一的任务
2. 接口设计应考虑通用性，避免只适用于某些单一场景
3. 符合行业标准，综合参考开源深度学习框架的接口设计，借鉴各框架的优点；除非飞桨 API 设计有明确优势的情况下可保留自有特色，否则需要符合行业标准
4. 功能类似的接口，参数名和行为需要保持一致，比如，lstm 和 gru
5. 优先保证清晰，然后考虑简洁，避免使用不容易理解的缩写
6. 历史一致，如无必要原因，应避免接口修改
7. 动静统一，如无特殊原因，动态图和静态图下的 API 输入、输出要求一致。开发者使用相同的代码，均可以在动态图和静态图模式下执行

### 动态图与静态图模式

关于飞桨框架支持的开发模式，为了便于用户理解，代码和 API 均采用“动态图”和“静态图”的说法；文档优先使用“动态图”和“静态图”的说法，不推荐使用“命令式编程”和“声明式编程”的说法。

### API 目录结构规范

- 公开 API 代码应该放置到以下列出的对应位置目录/文件中，并添加到目录下\_\_init\_\_.py 的 all 列表中。非公开 API 不能添加到 all 列表中


| paddle                         | paddle 基础 API，Tensor 操作相关 API                             |
| ------------------------------ | ------------------------------------------------------------ |
| paddle.tensor                  | 跟 tensor 操作相关的 API，比如：创建 zeros, 矩阵运算 matmul, 变换 concat, 计算 add, 查找 argmax 等 |
| paddle.nn                      | 跟组网相关的 API，比如：Linear, Conv2D，损失函数，卷积，LSTM 等，激活函数等 |
| paddle.nn.functional           | 跟组网相关的函数类 API，比如：conv2d、avg_pool2d 等            |
| paddle.nn.initializer          | 网络初始化相关 API，比如 Normal、Uniform 等                     |
| paddle.nn.utils                | 网络相关工具类 API，比如 weight_norm、spectral_norm 等          |
| paddle.static.nn               | 静态图下组网专用 API，比如：输入占位符 data/Input，控制流 while_loop/cond |
| paddle.static                  | 静态图下基础框架相关 API，比如：Variable, Program, Executor 等 |
| paddle.optimizer               | 优化算法相关 API，比如：SGD、Adagrad、Adam 等                  |
| paddle.optimizer.lr（文件）    | 学习率策略相关 API，比如 LinearWarmup、LRScheduler 等           |
| paddle.metric                  | 评估指标计算相关的 API，比如：accuracy, auc 等                 |
| paddle.io                      | 数据输入输出相关 API，比如：save, load, Dataset, DataLoader 等 |
| paddle.device（文件）          | 设备管理相关 API，通用类，比如：CPUPlace 等                    |
| paddle.device.cuda             | 设备管理相关 API，CUDA 相关，比如：CUDAPlace 等                 |
| paddle.distributed             | 分布式相关基础 API                                            |
| paddle.distributed.fleet       | 分布式相关高层 API                                            |
| paddle.distributed.fleet.utils | 分布式高层 API 文件系统相关 API，比如 LocalFS 等                  |
| paddle.distributed.utils       | 分布式工具类 API，比如 get_host_name_ip 等                      |
| paddle.vision                  | 视觉领域 API，基础操作且不属于子目录所属大类                  |
| paddle.vision.datasets         | 视觉领域 API，公开数据集相关，比如 Cifar10、MNIST 等            |
| paddle.vision.models           | 视觉领域 API，模型相关，比如 ResNet、LeNet 等                   |
| paddle.vision.transforms       | 视觉领域 API，数据预处理相关，比如 CenterCrop、hflip 等         |
| paddle.vision.ops              | 视觉领域 API，基础 op 相关，比如 DeformConv2D、yolo_box 等        |
| paddle.text                    | NLP 领域 API, 比如，数据集，数据处理，常用网络结构，比如 transformer |
| paddle.utils                   | paddle.utils 目录下包含飞桨框架工具类的 API，且不属于子目录所属大类 |
| paddle.utils.download          | 工具类自动下载相关 API，比如 get_weights_path_from_url         |
| paddle.utils.profiler          | 工具类通用性能分析器相关 API，比如 profiler 等                  |
| paddle.utils.cpp_extension     | 工具类 C++扩展相关的 API，比如 CppExtension、CUDAExtension 等    |
| paddle.utils.unique_name       | 工具类命名相关 API，比如 generate、guard 等                     |
| paddle.amp                     | paddle.amp 目录下包含飞桨框架支持的动态图自动混合精度(AMP)相关的 API，比如 GradScaler 等 |
| paddle.jit                     | paddle.jit 目录下包含飞桨框架支持动态图转静态图相关的 API，比如 to_static 等 |
| paddle.distribution            | paddle.distribution 目录下包含飞桨框架支持的概率分布相关的 API，比如 Normal 等 |
| paddle.regularizer             | 正则相关的 API，比如 L1Decay 等                                 |
| paddle.sysconfig               | Paddle 系统路径相关 API，比如 get_include、get_lib 等            |
| paddle.callbacks               | paddle.callbacks 目录下包含飞桨框架支持的回调函数相关的 API，比如 Callback 等 |
| paddle.hub                     | paddle.hub 目录下包含飞桨框架模型拓展相关的 API 以及支持的模型库列表，比如 list 等 |
| paddle.autograd                | 自动梯度求导相关，比如 grad、backward 等                       |
| paddle.inference               | paddle 预测相关，比如 Predictor 等                              |
| paddle.onnx                    | onnx 导出相关，比如 onnx.export                                |
| paddle.incubate                | 新增功能孵化目录                                             |



- 常用的 API 可以在更高层级建立别名，当前规则如下：
   1. paddle.tensor 目录下的 API，均在 paddle 根目录建立别名，其他所有 API 在 paddle 根目录下均没有别名。
   2. paddle.nn 目录下除了 functional 目录以外的所有 API，在 paddle.nn 目录下均有别名。

        ```python
        paddle.nn.functional.mse_loss # functional 下的函数不建立别名，使用完整名称
        paddle.nn.Conv2D # 为 paddle.nn.layer.conv.Conv2D 建立的别名
       ```
  1. 一些特殊情况比如特别常用的 API 会直接在 paddle 下建立别名
        ```python
        paddle.tanh # 为常用函数 paddle.tensor.math.tanh 建立的别名
        paddle.linspace# 为常用函数 paddle.fluid.layers.linspace 建立的别名
        ```


### API 行为定义规范

- 动静统一要求。除了 paddle.static 目录中的 API 外，其他目录的所有 API 原则上均需要支持动态图和静态图模式下的执行，且输入、输出要求一致。开发者使用相同的代码，可以在动态图和静态图模式下执行。

  ```python
  #静态图专用
  paddle.fluid.gradients(targets, inputs, target_gradients=None, no_grad_set=None)
  #动态图专用
  paddle.fluid.dygraph.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, only_inputs=True, allow_unused=False, no_grad_vars=None, backward_strategy=None)
  #动、静态图通用
  paddle.nn.functional.conv2d(x, weight, bias=None, padding=0, stride=1, dilation=1, groups=1, use_cudnn=True, act=None, data_format="NCHW", name=None)
  paddle.nn.Conv2D(num_channels, num_filters, filter_size, padding=0, stride=1, dilation=1, groups=1, param_attr=None, bias_attr=None, use_cudnn=True, act=None, data_format="NCHW", dtype='float32')
  ```


- API 不需要用户指定执行硬件，框架可以自动根据当前配置，选择执行的库。

- 设置缺省参数类型
  组网类 API 去除 dtype 参数，比如 Linear, Conv2d 等，通过使用 paddle.set_default_dtype 和 paddle.get_default_dtype 设置全局的数据类型。

- 数据类型转换规则

  1. 不支持 Tensor 和 Tensor 之间的隐式数据类型转换，隐藏类型转换虽然方便，但风险很高，很容易出现转换错误。如果发现类型不匹配，进行隐式类型转换，一旦转换造成精度损失，会导致模型的精度降低，由于没有任何提示，问题非常难以追查；而如果直接向用户报错或者警告，用户确认后，修改起来会很容易。避免了出错的风险。

     ```python
     import paddle
     a = paddle.randn([3, 1, 2, 2], dtype='float32')
     b = paddle.randint(0, 10, [3, 1, 2, 2], dtype='int32')
     c = a + b
     # 执行后会出现以下类型不匹配警告：
     # ......\paddle\fluid\dygraph\math_op_patch.py:239: UserWarning: The dtype of left and right variables are not the same, left dtype is paddle.float32, but right dtype is paddle.int32, the right dtype will convert to paddle.float32 format(lhs_dtype, rhs_dtype, lhs_dtype))
     ```

  2. 支持 Tensor 和 python Scalar 之间的隐式类型转换，当 Tensor 的数据类型和 python Scalar
  是同一类的数据类型时（都是整型，或者都是浮点型），或者 Tensor 是浮点型而 python Scalar 是
  整型的，默认会将 python Scalar 转换成 Tensor 的数据类型。而如果 Tensor 的数据类型是整型而 python Scalar 是浮点型时，计算结果会是 float32 类型的。
     ```python
     import paddle
     a = paddle.to_tensor([1.0], dtype='float32')
     b = a + 1  # 由于 python scalar 默认采用 int64, 转换后 b 的类型为'float32'
     c = a + 1.0  # 虽然 python scalar 是 float64, 但计算结果 c 的类型为'float32'
     a = paddle.to_tensor([1], dtype='int32')
     b = a + 1.0  # 虽然 python scalar 是 float64, 但计算结果 b 的类型为 'float32
     c = a + 1  # 虽然 python scalar 是 int64, 但计算结果 c 的类型为'int32'
     ```

### 数据类型规范

- 参数数据类型

  对于 loss 类 API，比如 cross_entropy, bce_loss 等，输入的 label 需要支持[int32, int64, float32, float64]数据类型。

- 返回值数据类型

  实现需要返回下标 indices 的 API，比如 argmax、argmin、argsort、topk、unique 等接口时，需要提供 dtype 参数，用于控制返回值类型是 int32 或者 int64，默认使用 dtype=’int64’（与 numpy, tf, pytorch 保持一致），主要目的是当用户在明确输入数据不超过 int32 表示范围时，可以手动设置 dtype=’int32’来减少显存的占用；对于 dtype=’int32’设置，需要对输入数据的下标进行检查，如果超过 int32 表示范围，通过报错提示用户使用 int64 数据类型。


## API 命名规范

**API 的命名应使用准确的深度学习相关英文术语，具体参考附录的中英术语表。**

### 类名与方法名的规范

- 类名的命名应采用驼峰命名法，通常采用名词形式

  ```python
  paddle.nn.Conv2D
  paddle.nn.BatchNorm
  paddle.nn.Embedding
  paddle.nn.LogSoftmax
  paddle.nn.SmoothL1Loss
  paddle.nn.LeakyReLU
  paddle.nn.Linear
  paddle.optimizer.lr.LambdaDecay
  ```

- 由多个单词组成的类名，最后一个单词应表示类型

    ```python
    # SimpleRNNCell 继承自 RNNCellBase
    paddle.nn.SimpleRNNCell
    # BrightnessTransform 继承 BaseTransform
    paddle.vision.BrightnessTransform
    ```

- 函数的名称应采用全小写，单词之间采用下划线分割

    ```python
    # 使用小写
    paddle.nn.functional.conv2d
    paddle.nn.functional.embedding
    # 如果由多个单词构成应使用下划线连接
    paddle.nn.functional.mse_loss
    paddle.nn.functional.batch_norm
    paddle.nn.functional.log_softmax
    ```

- 但一些约定俗成的例子可以保持不加下划线

    ```python
    paddle.isfinite
    paddle.isnan
    paddle.isinf
    paddle.argsort
    paddle.cumsum
    ```

- API 命名时，缩写的使用不应引起歧义或误解；在容易引起歧义或误解的情况下，需要使用全称，比如

    ```python
    # pytorch 使用 ge,lt 之类的缩写，可读性较差，应保留全称，与 numpy 和 paddle 保持一致
    paddle.tensor.greater_equal
    paddle.tensor.less_than
    # optimizer 不使用缩写
    paddle.optimizer.SGD
    # parameter 不使用缩写
    paddle.nn.create_parameter
    ```

- 在用于 API 命名时，常见的缩写列表如下：

    ```python
    conv、max、min、prod、norm、gru、lstm、add、func、op、num、cond
    ```

- 在用于 API 命名时，以下建议使用全称，不推荐使用缩写

    | 不规范命名 |   规范命名    |
    | :-------- | :----------- |
    |    div     |    divide     |
    |    mul     |   multiply    |
    |    sub     |   subtract    |
    | floor_div  | floor_divide  |
    |     lr     | learning_rate |
    |    act     |  activation   |
    |    eps     |    epsilon    |
    |    val     |     value     |
    |    var     |    varible    |
    |   param    |   parameter   |
    |    prog    |    program    |
    |    idx     |     index     |
    |    exe     |   executor    |
    |    buf     |    buffer     |
    |   trans    |   transpose   |
    |    img     |     image     |
    |    loc     |   location    |
    |    len     |    length     |



- API 命名不应包含版本号

    ```python
    # 不使用版本号
    paddle.nn.multiclass_nms2
    ```

- 常见的数学计算 API 中的逐元素操作不需要加上 elementwise 前缀，按照某一轴操作不需要加上 reduce 前缀，一些例子如下

    |  paddle2.0 之前  | pytorch |  numpy   | tensorflow  |   paddle2.0 之后   |
    | :------------- | :----- | :------ | :--------- | :--------------- |
    | elementwise_add |   add   |   add    |     add     |        add        |
    | elementwise_sub |   sub   | subtract |  subtract   |      subract      |
    | elementwise_mul |   mul   | multiply |  multiply   |     multiply      |
    | elementwise_div |   div   |  divide  |   divide    | divide |
    | elementwise_min |   min   | minimum  |   minimum   |      minimum      |
    | elementwise_max |   max   | maximum  |   maximum   |      maximum      |
    |   reduce_sum    |   sum   |   sum    | reduce_sum  |        sum        |
    |   reduce_prod   |  prod   |   prod   | reduce_prod |       prod        |
    |   reduce_min    |   min   |   min    | reduce_min  |        min        |
    |   reduce_max    |   max   |   max    | reduce_max  |        max        |
    |   reduce_all    |   all   |   all    | reduce_all  |        all        |
    |   reduce_any    |   any   |   any    | reduce_any  |        any        |
    |   reduce_mean   |  mean   |   mean   | reduce_mean |       mean        |



- 整数取模和取余

    目前整除和取余取模运算机器运算符重载在不同的语言和库中对应关系比较复杂混乱（取余运算中余数和被除数同号，取模运算中模和除数同号。取余整除是对商向 0 取整，取模整除是对商向负取整）

    | 库         | 取余整除                 | 取余                  | 取模整除         | 取模                                |
    | ---------- | :----------------------- | :-------------------- | :--------------- | :---------------------------------- |
    | tf         | truncatediv              | truncatemod           | //或 floordiv     | %或 floormod                         |
    | torch      | //或 floor_divide         | fmod                  | 无               | %或 remainder                        |
    | math       | 无                       | math.fmod             | 无               | math.remainder                      |
    | python     | 无                       | 无                    | //               | %                                   |
    | numpy      | 无                       | 无                    | //或 floor_divide | %或 mod remainder                    |
    | paddle     | //或 elementwise_div(int) | %或 elemtwise_mod(int) | 无               | %或 elemtwise_mod(float)             |
    | paddle 2.0 | truncate_divide(int)     | %或 truncate_mod(int)  | //或 floor_divide | %或 floor_mod(float)或 mod 或 remainder |

    |          | paddle2.0 之前            | torch           | numpy             | tensorflow    | math           | python | paddle2.0 之后                       |
    | :------: | :----------------------- | :-------------- | :---------------- | :------------ | :------------- | :----- | :---------------------------------- |
    | 取余整除 | //或 elementwise_div(int) | //或 floor_divid | 无                | truncatediv   | -              | 无     | truncate_divide(int)                |
    |   取余   | %或 elemtwise_mod(int)    | fmod            | 无                | truncatemod   | math.fmod      | 无     | %或 truncate_mod(int)                |
    | 取模整除 | -                        | -               | floor_divide      | //或 floordiv- | -              | //     | //或 floor_divide                    |
    |   取模   | %或 elemtwise_mod(float)  | %或 remainder    | %或 mod 或 remainder | %或 floormod   | math.remainder | %      | %或 floor_mod(float)或 mod 或 remainder |

- 常用组网 API 命名规范

    ```python
    # 卷积：
    paddle.nn.Conv2D #采用 2D 后缀，2D 表示维度时通常大写
    paddle.nn.Conv2DTranspose
    paddle.nn.functional.conv2d
    paddle.nn.functional.conv2d_transpose
    # 池化：
    paddle.nn.MaxPool2D
    paddle.nn.AvgPool2D
    paddle.nn.MaxUnpool2D
    paddle.nn.functional.max_pool2d
    paddle.nn.functional.avg_pool2d
    paddle.nn.functional.max_unpool2d
    # 归一化：
    paddle.nn.BatchNorm2D
    paddle.nn.functional.batch_norm
    ```


### 参数命名规范

- 参数名称全部使用小写

  ```python
  paddle.nn.functional.mse_loss: def mse_loss(input, label, reduction='mean', name=None):
  ```

- 参数名可以区分单复数形态，单数表示输入参数是一个或多个变量，复数表示输入明确是含有多个变量的列表

  ```python
  paddle.nn.Softmax(axis=-1) # axis 明确为一个 int 数
  paddle.squeeze(x, axis=None, dtype=None, keepdim=False, name=None): # axis 可以为一数也可以为多个
  paddle.strided_slice(x, axes, starts, ends, strides, name=None) #axis 明确是多个数，则参数用复数形式 axes
  ```

- 函数操作只有一个待操作的张量参数时，用 x 命名；如果有 2 个待操作的张量参数时，且含义明确时，用 x, y 命名

  ```python
  paddle.sum(x, axis=None, dtype=None, keepdim=False, name=None)
  paddle.divide(x, y, name=None)
  ```

- 原则上所有的输入都用 x 表示，包括 functional 下的 linear, conv2d, lstm, batch_norm 等

- 原则上都要具备 name 参数，用于标记 layer，方便调试和可视化

- loss 类的函数，使用`input` 表示输入，使用`label` 表示真实预测值/类别，部分情况下，为了更好的便于用户理解，可以选用其他更恰当的参数名称。如`softmax_with_logits`时，输入参数名可用`logits`

  ```python
  paddle.nn.functional.mse_loss(input, label, reduction='mean', name=None)
  paddle.nn.functional.cross_entropy(input,
                                     label,
                                     weight=None,
                                     ignore_index=-100,
                                     reduction='mean',
                                     soft_label=False,
                                     axis=-1,
                                     name=None):
  ```


- Tensor 名称和操作

   | 中文          | 英文                | 缩写 | 说明                             |
   | ------------- | ------------------- | ---- | -------------------------------- |
   | 张量          | tensor              |      |                                  |
   | 形状          | shape               |      |                                  |
   | 维数（阶）    | rank                |      |                                  |
   | 第几维(轴)    | axis/axes           |      | 从 0 开始编号                      |
   | 0 阶张量(标量) | scalar              |      |                                  |
   | 1 阶张量(向量) | vector              |      |                                  |
   | 2 阶张量(矩阵) | matrix/matrice      |      |                                  |
   | 矩阵转置      | transpose           |      |                                  |
   | 点积          | dot                 |      | 一维向量，内积；二维矩阵，矩阵乘 |
   | 内积          | inner               |      |                                  |
   | 矩阵乘        | matmul              |      |                                  |
   | 逐元素乘      | mul/elementwise_mul |      |                                  |
   | 逐元素加      | add/elementwise_add |      |                                  |
   | 按轴求和      | reduce_sum          |      |                                  |

- 常用参数表

   | 中文名         | 推荐           | 不推荐写法                            | 示例                                                         | 备注                                                         |
   | ------------ | ------------- | ------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
   | 算子名        | name          | input                                     | relu(x, inplace=False, name=None)                        | 调用 api 所创建的算子名称                                      |
   | 单个输入张量 | x         | x                                     | relu(x, inplace=False, name=None)                        | 单个待操作的张量                                             |
   | 两个输入张量 | x, y          | input, other/ X, Y                    | elementwise_add(x, y, axis=-1, activation=None, name=None)   | 两个待操作的张量                                             |
   | 数据类型     | dtype         | type, data_type                       | unique(x, dtype='int32')                                 |                                                              |
   | 输出张量     | out           | output                                |                                                              |                                                              |
   | 轴           | axis/axes     | dim/dims                              | concat(x, axis=0, name=None)                             | 虽然 pytorch 的 dim 单词比较简单，但 axis 跟 numpy, tf 和 paddle 历史一致。axis 通常从 0 开始编号，dim 一般从 1 开始编号，比如 3 维空间的第 1 维 |
   | 参数属性     | param_attr    |                                       | fc(param_attr=None, ...)                                     |                                                              |
   | 偏置属性     | bias_attr     |                                       | fc(bias_attr=None, ... )                                     |                                                              |
   | 激活函数     | activation    | act                                   | batch_norm(input, activation=None, ...)                      | act 简称不容易理解，跟 pytorch 保持一致                         |
   | 标签         | label         | target                                | def cross_entropy(input, label, soft_label=False, ignore_index=kIgnoreIndex) | label 容易理解，跟 paddle 历史保持一致                          |
   | 张量形状     | shape         | size                                  |                                                              |                                                              |
   | 程序         | program       | prog                                  |                                                              |                                                              |
   | 数据格式     | data_format   | data_layout                           | conv2d(x, weight, bias=None, padding=0, stride=1, dilation=1, groups=1,  activation=None, data_format="NCHW", name=None) | 跟 paddle 历史保持一致                                         |
   | 文件名       | filename      | file_name                             |                                                              | 跟 paddle 历史和 c 语言 fopen 函数保持一致                         |
   | 目录名       | path          | dirname                               |                                                              |                                                              |
   | 设备         | device        | place/force_cpu                       | ones(shape, dtype=None, out=None, device=None)               | device 比 place 更容易理解；跟 pytorch 一致                       |
   | 执行器       | executor      | exe                                   |                                                              |                                                              |
   | 下标         | index         | idx                                   |                                                              |                                                              |
   | 字母 epsilon  | epsilon       | eps                                   |                                                              |                                                              |
   | 值           | value         | val/v                                 |                                                              |                                                              |
   | 变量         | variable      | var/v                                 |                                                              |                                                              |
   | 批大小       | batch_size    | batch_num，batch_number               |                                                              |                                                              |
   | 隐层大小     | hidden_size   | hidden, hid_size, hid_dim, hidden_dim |                                                              |                                                              |
   | 卷积核大小   | filter_size   | filter                                |                                                              |                                                              |
   | 范围         | start, stop   | begin, end                            |                                                              | 跟 python 的 range 函数一致，numpy 的 arange                       |
   | 步数         | step          | num, count                            |                                                              |                                                              |
   | 条件         | cond          | condition                             |                                                              |                                                              |
   | 禁用梯度     | stop_gradient | require_grad                          |                                                              | 跟 tf 和 paddle 历史一致                                         |
   | 学习率       | learning_rate | lr                                    |                                                              |                                                              |
   | 保持维度     | keep_dim      | keepdim                               |                                                              |                                                              |
   | 禁用梯度     | no_grad_vars  | no_grad_set                           | gradients ( targets, inputs, target_gradients=None, no_grad_vars=None ) | 跟 dygraph.grad 保持一致                                       |
   | dropout 比例  | dropout_rate  | dropout_prob, dropout                 |                                                              |                                                              |
   | 传入         | feed          | feed_dict                             |                                                              |                                                              |
   | 取出         | fetch         | fetch_list, fetch_targets             |                                                              |                                                              |
   | 转置         | transpose     | trans, trans_x                        |                                                              |                                                              |
   | decay 步数    | decay_steps   | step_each_epoch                       |                                                              |                                                              |
   | 类别数       | num_classes   | class_nums                            |                                                              |                                                              |
   | 通道数       | num_channels  | channels                              |                                                              |                                                              |
   | 卷积核数     | num_filters   | filters                               |                                                              |                                                              |
   | 组数         | num_groups    | groups                                |                                                              |                                                              |
   | 操作输入     | inplace=True  | in_place                              |                                                              |                                                              |
   | 训练模式     | training=True | is_train, is_test                     |                                                              | 跟 pytorch 保持一致， tf 用 trainable                            |

## 附-中英术语表

以下深度学习相关术语主要来自 [deep learning](https://github.com/exacity/deeplearningbook-chinese) 这本书中的[术语表]( https://github.com/exacity/deeplearningbook-chinese/blob/master/terminology.tex )


| 中文 | 英文 | 缩写 |
| ---- | ---- | ---- |
| 深度学习 | deep learning | |
| 机器学习 | machine learning | |
| 机器学习模型 | machine learning model | |
| 逻辑回归 | logistic regression | |
| 回归 | regression | |
| 人工智能 | artificial intelligence | |
| 朴素贝叶斯 | naive Bayes | |
| 表示 | representation | |
| 表示学习 | representation learning | |
| 自编码器 | autoencoder | |
| 编码器 | encoder | |
| 解码器 | decoder | |
| 多层感知机 | multilayer perceptron | |
| 人工神经网络 | artificial neural network | |
| 神经网络 | neural network | |
| 随机梯度下降 | stochastic gradient descent | SGD |
| 线性模型 | linear model | |
| 线性回归 | linear regression | |
| 整流线性单元 | rectified linear unit | ReLU |
| 分布式表示 | distributed representation | |
| 非分布式表示 | nondistributed representation | |
| 非分布式 | nondistributed | |
| 隐藏单元 | hidden unit | |
| 长短期记忆 | long short-term memory | LSTM |
| 深度信念网络 | deep belief network | DBN |
| 循环神经网络 | recurrent neural network | RNN |
| 循环 | recurrence | |
| 强化学习 | reinforcement learning | |
| 推断 | inference | |
| 上溢 | overflow | |
| 下溢 | underflow | |
| softmax 函数 | softmax function | |
| softmax | softmax | |
| 欠估计 | underestimation | |
| 过估计 | overestimation | |
| 病态条件 | poor conditioning | |
| 目标函数 | objective function | |
| 目标 | objective | |
| 准则 | criterion | |
| 代价函数 | cost function | |
| 代价 | cost | |
| 损失函数 | loss function | |
| PR 曲线 | PR curve | |
| F 值 | F-score | |
| 损失 | loss | |
| 误差函数 | error function | |
| 梯度下降 | gradient descent | |
| 导数 | derivative | |
| 临界点 | critical point | |
| 驻点 | stationary point | |
| 局部极小点 | local minimum | |
| 极小点 | minimum | |
| 局部极小值 | local minima | |
| 极小值 | minima | |
| 全局极小值 | global minima | |
| 局部极大值 | local maxima | |
| 极大值 | maxima | |
| 局部极大点 | local maximum | |
| 鞍点 | saddle point | |
| 全局最小点 | global minimum | |
| 偏导数 | partial derivative | |
| 梯度 | gradient | |
| 样本 | example | |
| 二阶导数 | second derivative | |
| 曲率 | curvature | |
| 凸优化 | Convex optimization | |
| 非凸 | nonconvex | |
| 数值优化 | numerical optimization | |
| 约束优化 | constrained optimization | |
| 可行 | feasible | |
| 等式约束 | equality constraint | |
| 不等式约束 | inequality constraint | |
| 正则化 | regularization | |
| 正则化项 | regularizer | |
| 正则化 | regularize | |
| 泛化 | generalization | |
| 泛化 | generalize | |
| 欠拟合 | underfitting | |
| 过拟合 | overfitting | |
| 偏差 | biass | |
| 方差 | variance | |
| 集成 | ensemble | |
| 估计 | estimator | |
| 权重衰减 | weight decay | |
| 协方差 | covariance | |
| 稀疏 | sparse | |
| 特征选择 | feature selection | |
| 特征提取器 | feature extractor | |
| 最大后验 | Maximum A Posteriori | MAP |
| 池化 | pooling | |
| Dropout | Dropout | |
| 蒙特卡罗 | Monte Carlo | |
| 提前终止 | early stopping | |
| 卷积神经网络 | convolutional neural network | CNN |
| 小批量 | minibatch | |
| 重要采样 | Importance Sampling | |
| 变分自编码器 | variational auto-encoder | VAE |
| 计算机视觉 | Computer Vision | CV |
| 语音识别 | Speech Recognition | |
| 自然语言处理 | Natural Language Processing | NLP |
| 有向模型 | Directed Model | |
| 原始采样 | Ancestral Sampling | |
| 随机矩阵 | Stochastic Matrix | |
| 平稳分布 | Stationary Distribution | |
| 均衡分布 | Equilibrium Distribution | |
| 索引 | index of matrix | |
| 磨合 | Burning-in | |
| 混合时间 | Mixing Time | |
| 混合 | Mixing | |
| Gibbs 采样 | Gibbs Sampling | |
| 吉布斯步数 | Gibbs steps | |
| Bagging | bootstrap aggregating | |
| 掩码 | mask | |
| 批标准化 | batch normalization | |
| 参数共享 | parameter sharing | |
| KL 散度 | KL divergence | |
| 温度 | temperature | |
| 临界温度 | critical temperatures | |
| 并行回火 | parallel tempering | |
| 自动语音识别 | Automatic Speech Recognition | ASR |
| 级联 | coalesced | |
| 数据并行 | data parallelism | |
| 模型并行 | model parallelism | |
| 异步随机梯度下降 | Asynchoronous Stochastic Gradient Descent | |
| 参数服务器 | parameter server | |
| 模型压缩 | model compression | |
| 动态结构 | dynamic structure | |
| 隐马尔可夫模型 | Hidden Markov Model | HMM |
| 高斯混合模型 | Gaussian Mixture Model | GMM |
| 转录 | transcribe | |
| 主成分分析 | principal components analysis | PCA |
| 因子分析 | factor analysis | |
| 独立成分分析 | independent component analysis | ICA |
| 稀疏编码 | sparse coding | |
| 定点运算 | fixed-point arithmetic | |
| 浮点运算 | float-point arithmetic | |
| 生成模型 | generative model | |
| 生成式建模 | generative modeling | |
| 数据集增强 | dataset augmentation | |
| 白化 | whitening | |
| 深度神经网络 | DNN | |
| 端到端的 | end-to-end | |
| 图模型 | graphical model | |
| 有向图模型 | directed graphical model | |
| 依赖 | dependency | |
| 贝叶斯网络 | Bayesian network | |
| 模型平均 | model averaging | |
| 声明 | statement | |
| 量子力学 | quantum mechanics | |
| 亚原子 | subatomic | |
| 逼真度 | fidelity | |
| 信任度 | degree of belief | |
| 频率派概率 | frequentist probability | |
| 贝叶斯概率 | Bayesian probability | |
| 似然 | likelihood | |
| 随机变量 | random variable | |
| 概率分布 | probability distribution | |
| 联合概率分布 | joint probability distribution | |
| 归一化的 | normalized | |
| 均匀分布 | uniform distribution | |
| 概率密度函数 | probability density function | PDF |
| 累积函数 | cumulative function | |
| 边缘概率分布 | marginal probability distribution | |
| 求和法则 | sum rule | |
| 条件概率 | conditional probability | |
| 干预查询 | intervention query | |
| 因果模型 | causal modeling | |
| 因果因子 | causal factor | |
| 链式法则 | chain rule | |
| 乘法法则 | product rule | |
| 相互独立的 | independent | |
| 条件独立的 | conditionally independent | |
| 期望 | expectation | |
| 期望值 | expected value | |
| 样本 | example | |
| 特征 | feature | |
| 准确率 | accuracy | |
| 错误率 | error rate | |
| 训练集 | training set | |
| 解释因子 | explanatory factort | |
| 潜在 | underlying | |
| 潜在成因 | underlying cause | |
| 测试集 | test set | |
| 性能度量 | performance measures | |
| 经验 | experience | |
| 无监督 | unsupervised | |
| 有监督 | supervised | |
| 半监督 | semi-supervised | |
| 监督学习 | supervised learning | |
| 无监督学习 | unsupervised learning | |
| 数据集 | dataset | |
| 数据点 | data point | |
| 标签 | label | |
| 标注 | labeled | |
| 未标注 | unlabeled | |
| 目标 | target | |
| 强化学习 | reinforcement learning | |
| 设计矩阵 | design matrix | |
| 参数 | parameter | |
| 权重 | weight | |
| 均方误差 | mean squared error | MSE |
| 正规方程 | normal equation | |
| 训练误差 | training error | |
| 泛化误差 | generalization error | |
| 测试误差 | test error | |
| 假设空间 | hypothesis space | |
| 容量 | capacity | |
| 表示容量 | representational capacity | |
| 有效容量 | effective capacity | |
| 线性阈值单元 | linear threshold units | |
| 非参数 | non-parametric | |
| 最近邻回归 | nearest neighbor regression | |
| 最近邻 | nearest neighbor | |
| 验证集 | validation set | |
| 基准 | bechmark | |
| 基准 | baseline | |
| 点估计 | point estimator | |
| 估计量 | estimator | |
| 统计量 | statistics | |
| 无偏 | unbiased | |
| 有偏 | biased | |
| 异步 | asynchronous | |
| 渐近无偏 | asymptotically unbiased | |
| 标准差 | standard error | |
| 一致性 | consistency | |
| 统计效率 | statistic efficiency | |
| 有参情况 | parametric case | |
| 贝叶斯统计 | Bayesian statistics | |
| 先验概率分布 | prior probability distribution | |
| 最大后验 | maximum a posteriori | |
| 最大似然估计 | maximum likelihood estimation | |
| 最大似然 | maximum likelihood | |
| 核技巧 | kernel trick | |
| 核函数 | kernel function | |
| 高斯核 | Gaussian kernel | |
| 核机器 | kernel machine | |
| 核方法 | kernel method | |
| 支持向量 | support vector | |
| 支持向量机 | support vector machine | SVM |
| 音素 | phoneme | |
| 声学 | acoustic | |
| 语音 | phonetic | |
| 专家混合体 | mixture of experts | |
| 高斯混合体 | Gaussian mixtures | |
| 选通器 | gater | |
| 专家网络 | expert network | |
| 注意力机制 | attention mechanism | |
| 对抗样本 | adversarial example | |
| 对抗 | adversarial | |
| 对抗训练 | adversarial training | |
| 切面距离 | tangent distance | |
| 正切传播 | tangent prop | |
| 正切传播 | tangent propagation | |
| 双反向传播 | double backprop | |
| 期望最大化 | expectation maximization | EM |
| 均值场 | mean-field | |
| 变分推断 | variational inference | |
| 二值稀疏编码 | binary sparse coding | |
| 前馈网络 | feedforward network | |
| 转移 | transition | |
| 重构 | reconstruction | |
| 生成随机网络 | generative stochastic network | |
| 得分匹配 | score matching | |
| 因子 | factorial | |
| 分解的 | factorized | |
| 均匀场 | meanfield | |
| 最大似然估计 | maximum likelihood estimation | |
| 概率 PCA | probabilistic PCA | |
| 随机梯度上升 | Stochastic Gradient Ascent | |
| 团 | clique | |
| Dirac 分布 | dirac distribution | |
| 不动点方程 | fixed point equation | |
| 变分法 | calculus of variations | |
| 信念网络 | belief network | |
| 马尔可夫随机场 | Markov random field | |
| 马尔可夫网络 | Markov network | |
| 对数线性模型 | log-linear model | |
| 自由能 | free energy | |
| 局部条件概率分布 | local conditional probability distribution | |
| 条件概率分布 | conditional probability distribution | |
| 玻尔兹曼分布 | Boltzmann distribution | |
| 吉布斯分布 | Gibbs distribution | |
| 能量函数 | energy function | |
| 标准差 | standard deviation | |
| 相关系数 | correlation | |
| 标准正态分布 | standard normal distribution | |
| 协方差矩阵 | covariance matrix | |
| Bernoulli 分布 | Bernoulli distribution | |
| Bernoulli 输出分布 | Bernoulli output distribution | |
| Multinoulli 分布 | multinoulli distribution | |
| Multinoulli 输出分布 | multinoulli output distribution | |
| 范畴分布 | categorical distribution | |
| 多项式分布 | multinomial distribution | |
| 正态分布 | normal distribution | |
| 高斯分布 | Gaussian distribution | |
| 精度 | precision | |
| 多维正态分布 | multivariate normal distribution | |
| 精度矩阵 | precision matrix | |
| 各向同性 | isotropic | |
| 指数分布 | exponential distribution | |
| 指示函数 | indicator function | |
| 广义函数 | generalized function | |
| 经验分布 | empirical distribution | |
| 经验频率 | empirical frequency | |
| 混合分布 | mixture distribution | |
| 潜变量 | latent variable | |
| 隐藏变量 | hidden variable | |
| 先验概率 | prior probability | |
| 后验概率 | posterior probability | |
| 万能近似器 | universal approximator | |
| 饱和 | saturate | |
| 分对数 | logit | |
| 正部函数 | positive part function | |
| 负部函数 | negative part function | |
| 贝叶斯规则 | Bayes' rule | |
| 测度论 | measure theory | |
| 零测度 | measure zero | |
| Jacobian 矩阵 | Jacobian matrix | |
| 自信息 | self-information | |
| 奈特 | nats | |
| 比特 | bit | |
| 香农 | shannons | |
| 香农熵 | Shannon entropy | |
| 微分熵 | differential entropy | |
| 微分方程 | differential equation | |
| KL 散度 | Kullback-Leibler (KL) divergence | |
| 交叉熵 | cross-entropy | |
| 熵 | entropy | |
| 分解 | factorization | |
| 结构化概率模型 | structured probabilistic model | |
| 图模型 | graphical model | |
| 回退 | back-off | |
| 有向 | directed | |
| 无向 | undirected | |
| 无向图模型 | undirected graphical model | |
| 成比例 | proportional | |
| 描述 | description | |
| 决策树 | decision tree | |
| 因子图 | factor graph | |
| 结构学习 | structure learning | |
| 环状信念传播 | loopy belief propagation | |
| 卷积网络 | convolutional network | |
| 卷积网络 | convolutional net | |
| 主对角线 | main diagonal | |
| 转置 | transpose | |
| 广播 | broadcasting | |
| 矩阵乘积 | matrix product | |
| AdaGrad | AdaGrad | |
| 逐元素乘积 | element-wise product | |
| Hadamard 乘积 | Hadamard product | |
| 团势能 | clique potential | |
| 因子 | factor | |
| 未归一化概率函数 | unnormalized probability function | |
| 循环网络 | recurrent network | |
| 梯度消失与爆炸问题 | vanishing and exploding gradient problem | |
| 梯度消失 | vanishing gradient | |
| 梯度爆炸 | exploding gradient | |
| 计算图 | computational graph | |
| 展开 | unfolding | |
| 求逆 | invert | |
| 时间步 | time step | |
| 维数灾难 | curse of dimensionality | |
| 平滑先验 | smoothness prior | |
| 局部不变性先验 | local constancy prior | |
| 局部核 | local kernel | |
| 流形 | manifold | |
| 流形正切分类器 | manifold tangent classifier | |
| 流形学习 | manifold learning | |
| 流形假设 | manifold hypothesis | |
| 环 | loop | |
| 弦 | chord | |
| 弦图 | chordal graph | |
| 三角形化图 | triangulated graph | |
| 三角形化 | triangulate | |
| 风险 | risk | |
| 经验风险 | empirical risk | |
| 经验风险最小化 | empirical risk minimization | |
| 代理损失函数 | surrogate loss function | |
| 批量 | batch | |
| 确定性 | deterministic | |
| 随机 | stochastic | |
| 在线 | online | |
| 流 | stream | |
| 梯度截断 | gradient clipping | |
| 幂方法 | power method | |
| 前向传播 | forward propagation | |
| 反向传播 | backward propagation | |
| 展开图 | unfolded graph | |
| 深度前馈网络 | deep feedforward network | |
| 前馈神经网络 | feedforward neural network | |
| 前向 | feedforward | |
| 反馈 | feedback | |
| 网络 | network | |
| 深度 | depth | |
| 输出层 | output layer | |
| 隐藏层 | hidden layer | |
| 宽度 | width | |
| 单元 | unit | |
| 激活函数 | activation function | |
| 反向传播 | back propagation | backprop |
| 泛函 | functional | |
| 平均绝对误差 | mean absolute error | |
| 赢者通吃 | winner-take-all | |
| 异方差 | heteroscedastic | |
| 混合密度网络 | mixture density network | |
| 梯度截断 | clip gradient | |
| 绝对值整流 | absolute value rectification | |
| 渗漏整流线性单元 | Leaky ReLU | |
| 参数化整流线性单元 | parametric ReLU | PReLU |
| maxout 单元 | maxout unit | |
| 硬双曲正切函数 | hard tanh | |
| 架构 | architecture | |
| 操作 | operation | |
| 符号 | symbol | |
| 数值 | numeric value | |
| 动态规划 | dynamic programming | |
| 自动微分 | automatic differentiation | |
| 并行分布式处理 | Parallel Distributed Processing | |
| 稀疏激活 | sparse activation | |
| 衰减 | damping | |
| 学成 | learned | |
| 信息传输 | message passing | |
| 泛函导数 | functional derivative | |
| 变分导数 | variational derivative | |
| 额外误差 | excess error | |
| 动量 | momentum | |
| 混沌 | chaos | |
| 稀疏初始化 | sparse initialization | |
| 共轭方向 | conjugate directions | |
| 共轭 | conjugate | |
| 条件独立 | conditionally independent | |
| 集成学习 | ensemble learning | |
| 独立子空间分析 | independent subspace analysis | |
| 慢特征分析 | slow feature analysis | SFA |
| 慢性原则 | slowness principle | |
| 整流线性 | rectified linear | |
| 整流网络 | rectifier network | |
| 坐标下降 | coordinate descent | |
| 坐标上升 | coordinate ascent | |
| 预训练 | pretraining | |
| 无监督预训练 | unsupervised pretraining | |
| 逐层的 | layer-wise | |
| 贪心算法 | greedy algorithm | |
| 贪心 | greedy | |
| 精调 | fine-tuning | |
| 课程学习 | curriculum learning | |
| 召回率 | recall | |
| 覆盖 | coverage | |
| 超参数优化 | hyperparameter optimization | |
| 超参数 | hyperparameter | |
| 网格搜索 | grid search | |
| 有限差分 | finite difference | |
| 中心差分 | centered difference | |
| 储层计算 | reservoir computing | |
| 谱半径 | spectral radius | |
| 收缩 | contractive | |
| 长期依赖 | long-term dependency | |
| 跳跃连接 | skip connection | |
| 门控 RNN | gated RNN | |
| 门控 | gated | |
| 卷积 | convolution | |
| 输入 | input | |
| 输入分布 | input distribution | |
| 输出 | output | |
| 特征映射 | feature map | |
| 翻转 | flip | |
| 稀疏交互 | sparse interactions | |
| 等变表示 | equivariant representations | |
| 稀疏连接 | sparse connectivity | |
| 稀疏权重 | sparse weights | |
| 接受域 | receptive field | |
| 绑定的权重 | tied weights | |
| 等变 | equivariance | |
| 探测级 | detector stage | |
| 符号表示 | symbolic representation | |
| 池化函数 | pooling function | |
| 最大池化 | max pooling | |
| 池 | pool | |
| 不变 | invariant | |
| 步幅 | stride | |
| 降采样 | downsampling | |
| 全 | full | |
| 非共享卷积 | unshared convolution | |
| 平铺卷积 | tiled convolution | |
| 循环卷积网络 | recurrent convolutional network | |
| 傅立叶变换 | Fourier transform | |
| 可分离的 | separable | |
| 初级视觉皮层 | primary visual cortex | |
| 简单细胞 | simple cell | |
| 复杂细胞 | complex cell | |
| 象限对 | quadrature pair | |
| 门控循环单元 | gated recurrent unit | GRU |
| 门控循环网络 | gated recurrent net | |
| 遗忘门 | forget gate | |
| 截断梯度 | clipping the gradient | |
| 记忆网络 | memory network | |
| 神经网络图灵机 | neural Turing machine | NTM |
| 精调 | fine-tune | |
| 共因 | common cause | |
| 编码 | code | |
| 再循环 | recirculation | |
| 欠完备 | undercomplete | |
| 完全图 | complete graph | |
| 欠定的 | underdetermined | |
| 过完备 | overcomplete | |
| 去噪 | denoising | |
| 去噪 | denoise | |
| 重构误差 | reconstruction error | |
| 梯度场 | gradient field | |
| 得分 | score | |
| 切平面 | tangent plane | |
| 最近邻图 | nearest neighbor graph | |
| 嵌入 | embedding | |
| 近似推断 | approximate inference | |
| 信息检索 | information retrieval | |
| 语义哈希 | semantic hashing | |
| 降维 | dimensionality reduction | |
| 对比散度 | contrastive divergence | |
| 语言模型 | language model | |
| 标记 | token | |
| 一元语法 | unigram | |
| 二元语法 | bigram | |
| 三元语法 | trigram | |
| 平滑 | smoothing | |
| 级联 | cascade | |
| 模型 | model | |
| 层 | layer | |
| 半监督学习 | semi-supervised learning | |
| 监督模型 | supervised model | |
| 词嵌入 | word embedding | |
| one-hot | one-hot | |
| 监督预训练 | supervised pretraining | |
| 迁移学习 | transfer learning | |
| 学习器 | learner | |
| 多任务学习 | multitask learning | |
| 领域自适应 | domain adaption | |
| 一次学习 | one-shot learning | |
| 零次学习 | zero-shot learning | |
| 零数据学习 | zero-data learning | |
| 多模态学习 | multimodal learning | |
| 生成式对抗网络 | generative adversarial network | GAN |
| 前馈分类器 | feedforward classifier | |
| 线性分类器 | linear classifier | |
| 正相 | positive phase | |
| 负相 | negative phase | |
| 随机最大似然 | stochastic maximum likelihood | |
| 噪声对比估计 | noise-contrastive estimation | NCE |
| 噪声分布 | noise distribution | |
| 噪声 | noise | |
| 独立同分布 | independent identically distributed | |
| 专用集成电路 | application-specific integrated circuit | ASIC |
| 现场可编程门阵列 | field programmable gated array | FPGA |
| 标量 | scalar | |
| 向量 | vector | |
| 矩阵 | matrix | |
| 张量 | tensor | |
| 点积 | dot product | |
| 内积 | inner product | |
| 方阵 | square | |
| 奇异的 | singular | |
| 范数 | norm | |
| 三角不等式 | triangle inequality | |
| 欧几里得范数 | Euclidean norm | |
| 最大范数 | max norm | |
| 对角矩阵 | diagonal matrix | |
| 对称 | symmetric | |
| 单位向量 | unit vector | |
| 单位范数 | unit norm | |
| 正交 | orthogonal | |
| 正交矩阵 | orthogonal matrix | |
| 标准正交 | orthonormal | |
| 特征分解 | eigendecomposition | |
| 特征向量 | eigenvector | |
| 特征值 | eigenvalue | |
| 分解 | decompose | |
| 正定 | positive definite | |
| 负定 | negative definite | |
| 半负定 | negative semidefinite | |
| 半正定 | positive semidefinite | |
| 奇异值分解 | singular value decomposition | SVD |
| 奇异值 | singular value | |
| 奇异向量 | singular vector | |
| 单位矩阵 | identity matrix | |
| 矩阵逆 | matrix inversion | |
| 原点 | origin | |
| 线性组合 | linear combination | |
| 列空间 | column space | |
| 值域 | range | |
| 线性相关 | linear dependency | |
| 线性无关 | linearly independent | |
| 列 | column | |
| 行 | row | |
| 同分布的 | identically distributed | |
| 词嵌入 | word embedding | |
| 机器翻译 | machine translation | |
| 推荐系统 | recommender system | |
| 词袋 | bag of words | |
| 协同过滤 | collaborative filtering | |
| 探索 | exploration | |
| 策略 | policy | |
| 关系 | relation | |
| 属性 | attribute | |
| 词义消歧 | word-sense disambiguation | |
| 误差度量 | error metric | |
| 性能度量 | performance metrics | |
| 共轭梯度 | conjugate gradient | |
| 在线学习 | online learning | |
| 逐层预训练 | layer-wise pretraining | |
| 自回归网络 | auto-regressive network | |
| 生成器网络 | generator network | |
| 判别器网络 | discriminator network | |
| 矩 | moment | |
| 可见层 | visible layer | |
| 无限 | infinite | |
| 容差 | tolerance | |
| 学习率 | learning rate | |
| 轮数 | epochs | |
| 轮 | epoch | |
| 对数尺度 | logarithmic scale | |
| 随机搜索 | random search | |
| 分段 | piecewise | |
| 汉明距离 | Hamming distance | |
| 可见变量 | visible variable | |
| 近似推断 | approximate inference | |
| 精确推断 | exact inference | |
| 潜层 | latent layer | |
| 知识图谱 | knowledge graph | |
