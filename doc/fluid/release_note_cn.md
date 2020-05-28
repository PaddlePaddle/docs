# Release Note

## 重要更新

本版本深度优化了命令式编程模式（动态图）的功能、性能和体验，框架基础功能也进一步强化；原生推理库性能显著优化，轻量化推理引擎PaddleLite实现了对硬件支持的极大覆盖，新发布前端推理引擎Paddle.js，PaddleServing全面升级，提供功能强大简单易用的服务化部署能力。对应的开发套件和工具组件进一步丰富完善，即有套件组件的功能和体验持续提升，全新发布PaddleClas视觉分类套件和量桨Paddle Quantum量子机器学习框架。

**训练框架：** 深度优化了命令式编程（动态图）功能、性能和体验，特别是增强了动静转换的能力，能支持依赖数据的控制流的动态图实现进行静态存储部署，也可以转为静态图模式训练；Data Loader的功能和梯度裁剪的使用方式进一步优化；声明式编程模式下多卡运行时fetch不定长Tensor等问题得到解决，混合精度配合重计算显示出支持大Batch训练很好的成效。新增了大量API，并新增 ComplexVariable，支持复数张量的表示和常见的复数运算。

**预测部署：** Paddle inference 新增CUDA下多线程多流支持、TRT子图对动态shape输入的支持，强化量化推理，性能显著优化；Paddle Serving 全面升级，功能完善，易用性显著提升；Paddle Lite进一步优化编译安装体验，全面提升对支持芯片的覆盖度（包括RK、MTK、百度昆仑、寒武纪、比特大陆、华为NPU等等）以及对应的模型数量和性能；PaddleSlim量化、裁剪和NAS功能持续强化；发布国内首个开源JavaScript深度学习前端推理引擎Paddle.js，可以帮助用户实现网页端深度学习模型部署。

**开发套件：** 全新发布PaddleClas，包含23个图像分类网络实现，117个图像预训练模型，并添加了数据增广、SSLD蒸馏等辅助策略，以及特色应用案例；PaddleSeg人像分割系列模型全面升级，新增多种遥感相关的策略方案；PaddleDetection、PaddleOCR和语音合成套件Parakeet算法覆盖更全面，速度显著提升。

**工具组件：** PaddleHub新增包括一系列视觉预训练模型在内更多的模型，模型总数120+； PaddleFL发布1.0版本，开源基于Mulit-party Computation (MPC)的联邦学习，支持横向、纵向等多个联邦学习场景；PGL发布业界首个结合语义信息与结构信息的图神经网络模型ERNIESage；PARL开源工业界首个进化学习应用框架Evokit；全新发布量子机器学习框架量桨Paddle Quantum。

##  基础框架

### 新增API
- 新增`fluid.device_guard`：设置OP的运行设备为CPU或者GPU。
- 新增 `fluid.enable_imperative` 和 `fluid.disable_imperative` 接口，支持函数式启动关闭动态图模式，相对`with fluid.dygraph.guard()`的方式减少代码缩进。
- 在fluid.dygraph目录新增4个API（具体定义见文档）: BCELoss, L1Loss, MSELoss, NLLLoss, InstanceNorm
- 在fluid.layers目录新增30个API（具体定义见文档）: addmm, allclose, arange, bmm, clamp, cross, diag_embed, dist, dot, elementwise_equal, flip, full, full_like, index_select, interpolate, log1p, log_softmax, logsumexp, meshgrid, nonzero, randint, randn, randperm, resize_bicubic, resize_linear, roll, t, tril, triu

### 功能优化

- 命令式编程模式（动态图）：
	- 增强动转静的功能，新增了基于语法解析转换的ProgramTranslator，支持依赖数据的控制流的动态图模型进行部署；同时支持将动态图模型转换为静态图模型进行训练，提升RNN等任务的训练性能。
    - 重构动态图的变量生命周期管理机制，保证在train模式下不调用var.backward()接口也能正确地释放内存/显存资源。
    - 新增动态图下的double grad功能，支持依赖梯度惩罚的GAN模型训练。
    -  针对动态图下`no_grad`只能通过装饰器的方式使用的问题，新增了支持context manager使用方式，更方便动态图无梯度操作的代码编写。
    - 为了方便单独设置batchnorm和dropout两个op的train/eval模式设置，将train/eval模式信息从全局设置，变成Layer内部设置；新增Layer形式的Dropout，记录模式信息。
    - 支持 `cond` `switch` `while_loop` 控制流接口和 tensor array 的读写也可在动态图下使用 ，便于高层API的统一。
    - 修改`if var`在动态图模式下的行为（不兼容升级），按var中的值进行判断，解决动态图模式下 if x > y 行为与预期不符的问题；并支持将var转换为float/long/int/len/index的功能，提动态图升易用性。
    - 针对任务中强依赖hook的功能，新增Layer的forward pre-hook和forward post-hook接口，可以在不改变网络输入输出的结构的情况下方便地获取、改变网络中间层变量的值，提升动态图易用性。
    - 支持cudnn algorithm cache可以在动态图模式下生效，在waveflow模型上性能提升200%。

- 声明式编程模式（静态图）：
    - 执行器支持根据feed和fetch变量在运行时自动裁剪网络，去掉与当前feed和fetch无关的部分，提升运行效率，支持多任务学习网络。
    - 优化反向传播过程，对本身无需反向传播的变量进行自动裁剪，不再需要组网时对变量显式设置stop_gradient=True。
    - 执行器支持多卡运行时fetch不定长Tensor的功能，对使用不定长数据的任务（如NLP类部分任务等）提供更好的支持。
    - 解决单进程多卡预测阶段会丢弃尾部不足卡数的部分数据的问题，可以在DataLoader中设置drop_last=False来避免丢弃尾部数据。
    - 增加混合精度（AMP）与重计算（Recompute）配合的机制，在Bert-large模型上配合使用二者，最大batch size提升400%，吞吐提升17.5%~31.4%；
- DataLoader：
    - 新增多进程模式加速数据读取，对于Map-style类型的数据集，用户可以通过实现自定义Dataset和BatchSampler的方式来提高数据读取性能，对于数据读取量大及预处理复杂的任务速度提升明显，如在视频分类TSM模型上，使用多进程读取模式，在声明式编程模式（“静态图”）下训练性能提升419%，命令式编程模式（“动态图”）下训练性能提升89.6%。
- 梯度裁剪使用方式：
    - 裁剪类型统一由optimizer的grad_clip参数传入，支持全局参数裁剪和部分参数裁剪功能，原有set_gradient_clip接口不再推荐使用，并可能在后续版本中删除。同时，ParamAttr中取消了grad_clip参数（不兼容升级），无法再通过ParamAttr对单个参数进行梯度裁剪，对部分参数进行梯度裁剪的功能统一通过上述新接口实现。
- 动态图、静态图以及高层API支持一致的Collective Operators调用。
- Intel对Ngraph停止维护，移除NGraph库相关代码。
- 移除所有MKL-DNN相关op中未使用的或者不兼容的属性，如`is_test`属性。
- 新增对复数计算的支持：
    - 新增 ComplexVariable，支持复数张量的表示和常见的复数运算，包括四则基本运算、matmul、kron product、reshape、transpose 等；
- 性能分析工具（Profiler）功能升级：
    - 支持按照事件之间的嵌套调用关系，分层次统计和打印Profile结果。
    - 添加tracer_option参数，可配置为`Default`、`OpDetail`和`AllOpDetail`，支持用户选择不同程度的计时和分析对象。
    - 添加对框架开销、GpuMemcpy操作的统计功能。
- 报错信息全面优化
    - 累计优化数千条表意不清的报错信息，规范错误类型及错误描述。
    - 自动检测某些用户易错操作，给出明确的报错信息。
    - 优化GPU相关API报错信息，将不可读的错误代码转换为具体信息，与NVIDIA官网信息保持同步。

### 性能优化
- 命令式编程模式（“动态图”）：
    -  为降低框架overhead, 优化自动生成的OP函数的数据结构，ptb lm模型任务单卡训练速度提升4%。
    - 为降低框架overhead, 优化InferVarType接口设计，提升了InferVarType的速度，ptb lm模型任务训练速度提升超5%。
    - 为降低框架overhead, 减少了动态图op中非必要attribute的添加，在ptb lm模型训练任务上速度提升4%
    - 为提升数据加载性能，实现Tensor申请共享内存存储及Tensor序列化与反序列化机制，支持进程间传输Tensor，优化原动态图异步DataLoader性能，ResNet模型任务在P40机器上单卡训练速度进一步提升超15%
    - 优化了动态图 Variable slice 的性能，性能提升60%，并支持slice中step为负数。

- 声明式编程模式（“静态图”）：
    - 新增自动融合功能，支持将elementwise类、activation类、sum、cast、scale、fill_constant等逐元素计算类型的算子组成的子图进行融合，性能提升幅度依赖于在网络中匹配到的相关子图数量，目前对RNN语言模型训练速度有大幅提升。
- OP性能优化：
    - OP的执行过程增加对Prepare Data的缓存，在10+模型训练任务上平均加速2%，框架开销最高减少6%。
    - 优化depthwise_conv2d的GPU性能，常用参数设置下加速20%。
    - 优化elementwise_mul的GPU广播模式的实现，针对不同输入可加速2~50倍。
    - 优化conv2d_transpose的GPU实现，针对fp16有显著性能提升。
    - 优化shape OP实现，避免在不同设备间的不必要数据传输而引起的等待。

#### Bug修复

- 修复当数据量很大时，SGD报错`Xbyak::Error`的问题, 使得支持SGD大数据量可以成功运行。

- 修复Linux版本下MKL内存泄漏的问题。

- 修复动态图多卡启动时命令行参数解析的bug。

- 修复clone(for_test=True)接口处理含控制流op的网络时的问题。

- 修复动态图模块和静态图模块环依赖。

- 修正 python 2 & 3 之间 pickle dump/load 的兼容性问题。

- 修复动态图Layer不能注册或覆盖参数为None的问题。

- 修复不同Op name属性值相同时造成的Op输出Var命名冲突的问题。

- 修正concat op在axis=0时输出的LoD设置，应为输入LoD的拼接。

- 修复BatchNorm在eval模式下无法更新的mean和var的bug。

## 推理部署

###  Paddle Inference
#### 功能升级
-  新增TRT子图对动态shape输入的支持, 新加`config.SetTRTDynamicShapeInfo(min_input_shape, max_input_shape, opt_input_shape)`接口。此接口用来指定子图的输入的最小，最大，最优的shape信息（最优shape表示，TRT会在此shape选择运行时最优kernel）。指定shape信息后，Paddle-TRT运行期间会使用Dynamic shape模式，预测期间支持`min_input_shape`，`max_input_shape`间的任意shape的输入。该功能支持包括FCN，Faster RCNN，Ernie/Bert等动态shape输入的模型。
-  为满足用户预测时将计算流绑定在当前线程上的需求，重构了设备上下文数据结构支持 CUDA 计算流优先级，并增加一个线程本地的显存分配器 ThreadLocalAllocator。具备不同线程绑定不同 CUDA 流的能力。
-  MKL-DNN 量化功能全面支持所有量化模型，新增支持'weight_quantize_type'为range_abs_max和'channel_wise_abs_max'，支持out_threshold属性。
- 新增官网推理API reference

#### 性能优化
- CUDA Bert/Ernie针对性的优化, 添加了 `embedding_eltwise_layernorm` 融合实现，优化了 `multihead_matmul` ，`fc_elementwise_layernorm` 融合实现。相比上一版本，P4卡，cuda10，batch_size=1下，ernie fp32预测从10ms优化至8.7ms。提升13%.  
- TRT子图对Ernie/Bert模型动态shape支持， 在T4卡，cuda10， batch_size=1下，ernie fp16 预测性能为2.9ms， 相比fp32的6.6ms，加速56%。
- Paddle-TRT对mobilenet v3的优化，支持TRT hard sigmoid OP，以及新增hard swish plugin，batch_size = 1下，P4下预测从3.48ms 到2.29ms, 性能提升34%； V100下预测从2.76ms 到1.33ms, 性能提升51%。
- 增加 swish 激活函数 DNNL 支持，使得ShuffleNet 在 6248单核上性能提升了76%。
- 量化：新增支持matmul op量化；新增`matmul+transpose+reshape` fuse，`scale+matmul` fuse。经过matmul量化和新增fuses，Ernie fp32模型和量化后INT8模型都在原来基础上性能提升了~10%(在6271机器上)。
- 新增 DNNL inplace op支持：目前支持 `elementwise_add`和包括softmax, gelu，relu等大部分激活函数的inplace执行，使得Ernie性能在6248上提升了~2%
- 经过上述优化量化，目前Ernie INT8模型相比未经DNNL优化（包括fuses等）和量化的FP32模型提速~5.51倍。

#### Bug修复

- 修复Inference阶段在TRT int8离线量化中，因融合策略不稳定导致本地与服务端生成校准表名字不一致，从而本地生成的校准表在服务中无法识别加载，会重新生成校准表的问题。目前已经能够保证在多次运行TRT离线量化校准时，校准表名字保持一致。
- 修复Inference阶段TRT离线量化产生校准表过程中传参错误的问题。该问题会一定程度上影响最终的量化预测精度。

### Paddle Serving

#### 易用性提升
  - 使用pybind对c++代码进行封装，提供python api的使用方式，提供paddle_serving_server、paddle_serving_server_gpu、paddle_serving_client的python2和python3环境whl安装包，发布了0.2.1版本
  - 提供centos6/7环境的cpu和gpu Docker镜像，包含可执行镜像和可编译镜像
  - 提供直接保存Serving部署所需的模型和配置文件的api，与Paddle训练框架无缝衔接
  - 实现一行命令启动模型预测服务
#### 功能完善
  - 提供RPC和HTTP两种预测服务方式
  - 支持Python和Go语言客户端
  - 支持A/B test
  - 发布了paddle_serving_app 0.0.2版本，针对LAC分词分词预处理、中文BERT模型预处理、图像处理提供预处理api。
  - 支持预测服务Timeline可视化
#### 性能优化
  - RPC服务模式下，中文BERT语义向量表示预测服务比paddle_gpu_serving 0.8.2版本在单张P4卡batch size 1时预测速度提升2.04倍。
#### 文档和示例
  - 完善和添加中英文使用文档、中英文开发和部署文档、中文性能调优文档。
  - 提供7种模型预测服务示例，包含中文分词、英文情感分析、中文语义向量表示、CTR预估、图像分类等领域。

### Paddle Lite

#### 功能升级
- 编译安装
     - Paddle-Lite 编译脚本优化：Android\iOS\ArmLinux 平台各拆分出单独编译脚本，脚本提高易用性。
     - 支持Python安装：可以在PC Linux/Windows/Mac 上安装Paddle-Lite Python 库；Python 可以调用Lite opt 优化模型。
     - 支持windows 编译： 可以在windows环境编译Paddle-Lite ，当前windows环境只支持x86 编译。
- 基础功能
    - 增加分割子图功能。对于以子图接入方式lite的模型，通过配置文件手动切分子图，让指定OP跑在host端，以提高性能(ssd_mobilenet_v1，加速约4.3倍)。  
    - 优化支持无校准训练后量化方法产出的量化模型，常见分类模型量化到8bit，精度损失从2%减小到0.1%。
- 硬件支持
    - 新增RK 1808 NPU，支持全量化MobileNetV1模型；
    - 新增MTK MT8175 APU，支持全量化MobileNetV1模型；
    - 新增百度昆仑XPU Kernel接入方式，支持ERNIE、ResNet-50和BERT模型。
    - 新增寒武纪MLU270，支持一下模型：Resnet50（int8）、Senet101（int8）;
    - 新增特大陆BM1682，支持以下模型： Mobilenet、Yolov3、Mobilenet-ssd、Inceptionv4、Vgg16、DarkNet-YOLOv3、PyramidBox。
    - 移动端GPU（opencl）：支持模型mobilenetv1/v2、GAN相关、mnasnet、sqeueezenet、shufflenet、resnet、unet、vgg16
    - Nvidia GPU： 增加exec多流支持，对于存在并行性的模型结构，相对单流预计有5-15%的性能提升，对于常见视觉模型，一般不具有并行性结构，开启多流无收益。cuda平台下打开多流功能`config.set_multi_stream(true);`。
    - 对x86 平台的优化：降低预测库体积（200M---->16M），支持关闭LOG（--shutdown_log=ON）、full_api 支持多线程共享模型权重参数、新增x86 cxx_demo
    - 华为NPU：
      - benchmark模型(mobilenet_v1，mobilenet_v2，squeezenet_v1.1，mnasnet，shufflenet_v2)，提速5-10倍。
      -  支持缓存不同尺寸的NPU模型，提升可变输入尺寸模型的性能。
- Demo：
    - 新增基于相机预览的实时口罩检测Android Demo
    - 新增实时人脸关键点检测和美颜Android Demo
    - 新增移动端训练的波士顿房价预测Android Demo

#### 性能优化
- InferShape耗时降低： Predictor连续运行时，infershape总耗时从0.25ms 降低至0.08ms
- opencl部分kernel支持动态shape并将部分计算移到ReinitWhenNeeded。fc_buffer、elementwise_add、scale、activation、grid_sampler。
- 优化sgemm在低端机上的性能表现
- 优化Precision Profiler功能。排版优化，新增支持标准差属性、增长率属性（在均值和标准差一样时，可以比较顺序），支持对OpenCL的Image/Buffer的每层output的精度打印，支持将每层的精度结果（最终的precision summary）写入手机设备上，便于APP调试，将精度打印与原有统计耗时的profiler的依赖分开。

#### Bug修复

- 修复conv op的激活act_type未初始化导致的不同Predictor结果随机的问题。
- 修复opencl kernel。bilinear kernel在mali gpu上兼容性问题、instance norm计算结果不对的问题、reshape的kernel注册错误导致模型转换失败问题、exp和tanh找不到kernel的导致注册kernel名写错绑定模型op失败问题。
- 修复opencl在mali gpu的执行计算结束卡主的问题。
- 修复opencl的资源相关问题。隔离每个Predictor中每个cl::kernel/cl::program等资源。

### PaddleSlim
#### 量化
- 增加无校准数据训练后量化方法，int16精度无损，int8精度损失低于0.1%。
- 增强量化功能，完善量化OP的输出scale信息，支持CPU预测端全面适配量化模型。
#### 剪裁
- 新增FPGM和BN scale两种剪裁策略, 在MobileNetV3-YOLOV3-COCO任务上，同等压缩率下精度提升0.6% 。
- 新增自定义剪裁策略接口，方便开发者快速新增压缩策略。
- 剪裁功能添加对新增Operator的默认处理逻辑，扩展支持剪裁更多复杂网络。
#### NAS
- 新增DARTS系列搜索算法，并提供扩展接口，方便用户调研和实现新的模型结构搜索策略。
- 模型结构搜索添加早停机制，提升搜索功能易用性。
- 新增一种基于强化学习的模型结构搜索策略，并提供扩展接口，为用户调研实现新策略提供参考。
#### Pantheon
- 支持以 fp16 格式进行数据的传输和存储，支持在线蒸馏模式下用多个通道进行知识传输，加大知识数据的传输效率。
- 新增词法分析示例，方便用户基于此构建自己的蒸馏任务


## 开发套件

### PaddleDetection
- 模型丰富度提升
  - 添加Efficientdet-D0: COCO val2017精度较TF高0.3 (33.8 vs 33.5), 不含后处理预测速度基本持平或微弱优势（~13ms vs ~14ms，T4实测速度) 。
  - 添加实例分割模型HTC，V100下推理速度达到11.5FPS, 较竞品高7.4FPS，在COCO 2017下BBox mAP 42.1%, Mask 37.1。
  - 添加anchor-free模型FCOS:  COCO val2017精度较pytorch精度高1.1(39.8 vs 38.7)。
  - YOLOv3新增MobileNetV3骨干网络，COCO数据集精度达到31.6 。
  -  添加anchor-free模型CornernetSqueeze：COCO val2017精度34.5, 较竞品高0.1, 优化模型精度38.2, 提升3.7个点，速度较yolo_v3 darknet快5%  
  -  添加服务器端实用目标检测模型cascade_rcnn_resnet50_vd_fpn_dcn，V100上20FPS时，coco mAP 47.8%，优于竞品EfficientDet。

- 移动端推出3种模型
  - SSDLite系列模型：ssdlite-mobilenet_v3 large模型在COCO数据集上mAP：22.8%，在高通骁龙845上单线程推理速度95ms。ssdlite-mobilenet_v3 small模型在COCO数据集上mAP：16.6%，在高通骁龙845上单线程推理速度40ms，精度优于竞品。ssdlite-mobilenet_v1模型在COCO数据集上mAP：23.6%，在高通骁龙845上单线程推理速度140ms，精度优于竞品。
  - yolo v3: yolov3_mobilenet_v3裁剪模型在高通骁龙845上单线程推理速度91ms，精度24.6(输入尺寸320*320)，速度和精度均领先于竞品框架SSDLite模型。
  - Faster RCNN：基于COCO数据集，cascade_rcnn_mobilenet_v3 large_fpn在输入图片尺度为320x320时，mAP为25.0%，在高通骁龙845上单线程推理速度87ms；在输入图片尺度为640x640时，mAP为30.2%，在高通骁龙845上单线程推理速度351ms。

- 预测部署重构:
  - 新增Python预测部署流程，支持RCNN，YOLO，SSD，RetinaNet，人脸系列模型。支持视频预测。
  - 重构C++预测部署，提高易用性。

- 易用性提升及功能组件
   - 增加AutoAugment数据增强。
   - 升级检测库文档结构。
   - 支持迁移学习自动进行shape匹配。
   - 优化mask分支评估阶段内存占用。
   - 升级预测部署功能，增加python端图像与视频预测。

### PaddleSeg

- 新增Lovasz Loss损失函数，可有效提升多类别分割的精度
- 人像分割系列模型全面升级
  * 发布首个支持移动端实时人像分割模型HumanSeg-lite
  * 新增基于光流算法的视频级别的分割后处理方案

- 新增遥感图像分割解决方案
 * 新增多通道遥感图像的数据预处理方案  
 * 新增适用于多通道图像的数据增强策略
 * 提供积雪检测和云检测两种气象遥感领域分割教程

### PaddleClas

 - 新增MobileNetV3系列模型，并且对23个系列，117个预训练模型进行性能评估。
 - 新增SSLD知识蒸馏方案，识别准确率提升3%以上，并发布82.4%的resnet50_vd、78.9%的mobilenetv3等6个蒸馏模型。
 - 新增8种数据增广方式：AutoAugment，RandAugment，CutOutRandErasing，HideAndSeek，GridMask，Mixup，Cutmix，用于增加训练样本的多样性，提升模型的泛化性能。
 - 新增10万类图像分类预训练模型，针对图像分类业务应用场景，识别准确率最高可提升30%。

### PaddleOCR

 - 新增DB、EAST文本检测算法。
 - 新增Rosetta、CRNN、STAR-Net以及RARE文本识别算法。
 - 新增超轻量级OCR模型，总共模型大小仅8.6M（文本检测4.1M，文本识别4.5M），同时支持横排和竖排、长文本、中英文数字混合等场景文字的识别。


###  Parakeet

- 发布 WaveFlow (res channel=64/128)、ClariNet、WaveNet 等模型的英文预训练模型和音频样本；
- 修复 Conv2DTranspose 的 fp16 kernel 速度过慢的问题，简化 WaveFlow 在 fp16 模式下的预测逻辑；
- 显著提升模型训练速度，通过优化数据预处理和 OP 计算逻辑，在 DeepVoice3、TransformerTTS 等模型上均带来了成倍的速度提升；


## 工具组件

### PaddleHub
* 视觉模型丰富度提升，预训练模型总数，预训练模型总数达到120+。
	* 新增大规模视觉预训练模型，可大幅度提升图像分类和目标检测任务的Fine-tune效果
	* 新增工业级短视频分类模型VideoTag，支持3000类中文标签识别
	* 新增轻量级中文OCR模型，支持一键快速OCR识别
	* 新增行人检测、车辆检测、动物识别、Object365 2019大规模目标检测夺冠模型
* Fine-tune API升级
	* 文本分类任务新增5个预置网络，包括CNN, BOW, LSTM, BiLSTM, DPCNN等
* 动态图能力升级
	* BERT类预训练模型支持动态图模式下的一键加载

### PaddleX
* 全新发布PaddleX飞桨全流程开发工具
 - 打通从数据接入到预测部署的深度学习开发全流程、并提供简明易懂的Python API
 - 覆盖CV领域图像分类、目标检测、语义分割、实例分割四大主流任务场景，并集成PaddleHub、PaddleSlim、VisualDL、Paddle Lite等工具组件。
 - 预置产业实践精炼沉淀预训练模型及诸多飞桨特色优势模型26类，共计43个。
 - 提供自动数据分析、自动超参推荐、数据增强策略、模型裁剪训练、模型量化、预训练模型保存及复用、多平台发布部署、模型加密等进阶功能。
 - 创新集成模型可解释性分析功能
 - 提供官方实现可视化前端Demo，支持Windows、Linux、Mac系统一键安装。
### VisualDL
* 发布VisualDL 2.0 beta版本
 - 后端内核全新升级，更轻更快，兼容性更高，支持文件存储系统拓展
 - API全面升级，更少代码完成可视化分析，显著提升易用性
 - UI与交互全新升级，提供更好的本地化支持，可视化分析更清晰、直观，给用户沉浸式体验
 - 与飞桨开发套件与工具组件深度融合，提供更流畅的深度学习开发体验

### PaddleFL
 - 发布PaddleFL 1.0版本
	 - 开源基于Mulit-party Computation(MPC)的联邦学习，支持横向、纵向等多个联邦学习场景
	 - 原有框架重构，将新增联邦学习方案与原有方案整合并开源
	 - 新增由单机模型转变为FL可训练program的功能，支持更多模型及场景

### PGL
* 发布业界首个结合语义信息与结构信息的图神经网络模型ERNIESage
* 新增PGL-KE，目前PGL涵盖游走类、消息传递类以及知识嵌入类等25+图学习模型
* 新增Graph Batch、Graph Pooling等Graph类操作算子
* 全面支持Open Graph Benchmark基准测试集，并发布相应SOTA
* Model Zoo新增MetaPath2Vec++、Mulit-MetaPath2Vec++、STGCN、GIN、PinSage模型

### PARL
* 开源工业界首个进化学习应用框架EvoKit
* 新增Multi-Agent RL算法支持，包括MADDPG
* 新增多卡训练支持，发布多卡DQN算法示例
* 开源连续控制领域的SOTA算法TD3和SAC
* NeurIPS2019强化学习挑战赛事冠军模型以及训练方案开源

### Paddle Quantum（量子计算实验室）
* Paddle Quantum（量桨）初版发布。量桨是基于百度飞桨开发的量子机器学习工具集，支持量子神经网络的搭建与训练，提供易用的量子机器学习开发套件与量子优化、量子化学等前沿量子应用工具集，使得飞桨成为国内首个支持量子机器学习的深度学习平台。
	- 支持 QAOA 算法实现，完成最大割 (Max-Cut) 问题的解决
	- 支持 VQE 算法实现，计算 H_2 的最小特征值
	- 支持 SSVQE 算法实现，计算给定哈密顿量的特征谱
	- 支持 VQSD 算法实现，计算量子态对角化后的形式，给出量子态的特征分解
	- 支持 Gibbs 算法实现，生成给定哈密顿量在确定温度下的吉布斯态
	- 支持量子计算常用函数
	- 支持描述U_Ansatz量子电路
