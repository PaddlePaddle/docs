#  Release Note

## Important Updates

This version deeply optimizes the function, performance, and experience of the imperative programming mode (dynamic graph), and further strengthens the basic functions of the framework. It also significantly optimizes the performance of the native inference library, provides a lightweight inference engine Paddle Lite to achieve a great coverage of hardware support, rcomprehensively upgrades Paddle Serving, and has a powerful and simple service-oriented deployment capability. This version further enriches and improves the corresponding development kits and utility components, continues to improve the function and experience of the existing kits and components, and releases a new image classification kit, i.e., PaddleClas, and Paddle quantum machine learning framework.

**Training framework:** Deeply optimizes the function, performance, and experience of imperative programming (dynamic graph) and especially enhances the capability of converting dynamic graph to static graph. Supports to convert data-dependent control flow into static graph to save and deploy, or train under static graph mode. Further optimizes the function of Data Loader and the usage of gradient clipping. Fixes problems for declarative programming mode such as fetching tensors with different lengths between multi-cards. The combination of mixed precision and recomputation shows good results in large-batch training. Adds a number of APIs and ComplexVariable and supports complex number tensor expressions and common complex number operations.

**Inference Deployment:**  For Paddle inference, adds the multi-threaded multi-stream support under CUDA and the TRT sub-map's support for the input of dynamic shape, strengthens quantization inference, and significantly optimizes the performance. Fully upgrades Paddle Serving, improves its function, and significantly enhances its usability. Further optimizes the compilation and installation experience of Paddle Lite, comprehensively improves the coverage of supported chips (including RK, MTK, Baidu Kunlun, Cambricon, Bitmain, and Huawei NPU) as well as the corresponding model quantity and performance. Continues to strengthen the PaddleSlim quantization, pruning, and NAS functions. Releases the added Paddle.js which is the first open source front-end inference engine for deep learning of JavaScript in China and can help users to implement the deployment of deep learning models on the webpage side.

**Development kits:**  Releases PaddleClas including 23 image classification network implementations and 117 image pre-training models. Adds data augmentation, SSLD distillation, and other auxiliary strategies as well as characteristic application cases. Fully upgrades the PaddleSeg portrait segmentation series of models and adds multiple remote sensing related strategies. The coverage of PaddleDetection, PaddleOCR, and text-to-speech kit Parakeet algorithms is more comprehensive and the speed is increased significantly.

**Utility Components:**  For PaddleHub, adds more models including a series of vision pre-training models, total number of pre-trained models is more than 120. Releases PaddleFL Version 1.0, open sources federated learning based on mulit-party computation (MPC), and supports multiple federated learning scenarios such as horizontal and vertical layout. For PGL, releases industry's first graphical neural network model ERNIESage which combines semantic information with structural information. For PARL, open sources the industry's first evolutionary learning application framework Evokit. Releases a new quantum machine learning framework Paddle Quantum.

## Basic Framework

### New APIs

- Adds `fluid.device_guard`: Sets an OP's running device to CPU or GPU.
- Adds `fluid.enable_imperative` and `fluid.disable_imperative`, to enable and disable dynamic graph mode, and avoid code indentation relative to `with fluid.dygraph.guard()`.
- Adds four APIs in the fluid.dygraph directory (see the document for details): BCELoss, L1Loss, MSELoss, NLLLoss, and InstanceNorm
- Adds 30 APIs in the fluid.layers directory (see the document for details): addmm, allclose, arange, bmm, clamp, cross, diag\_embed, dist, dot, elementwise\_equal, flip, full, full\_like, index\_select, interpolate, log1p, log\_softmax, logsumexp, meshgrid, nonzero, randint, randn, randperm, resize\_bicubic, resize\_linear, roll, t, tril, and triu

### Function Optimization

- Imperative Programming Mode (Dynamic Graph):

  - Enhances the dynamic-to-static function, adds ProgramTranslator based on grammar analysis and transformation, and supports the deployment of dynamic graph model with data-dependent control flow; supports the transformation of the dynamic graph model into the static graph model for training and improves the training performance of tasks such as RNN.
  - Reconstitutes the variable life cycle management mechanism of the dynamic graph to ensure that memory/GPU memory resources can be released correctly in train mode without calling the var.backward() API.
  - Adds the double grad function in dynamic graph mode and supports the GAN model training relying on gradient penalty.
  - To solve that can only use decorator method to set `no_grad` in dynamic graph mode , adds the context manager method to facilitate the coding of gradientless operations of the dynamic graph.
  - To facilitate separate setting of the train/eval mode of batchnorm and dropout ops, changes the train/eval mode information to the internal setting of Layer from the global setting. Adds Layer-formed Dropout with the mode information.
  - Supports the use of the `cond` `switch` `while_loop` control flow interfaces and tensor array read-write in dynamic graph mode to facilitate the unification of high-level APIs.
  - Modifies the behavior of `if var` in the dynamic graph mode to make a judgment according to the value in var (incompatible upgrade). Fixes the problem `if x > y` behavior is not consistent with the expectation in dynamic graph mode.　Supports the function of converting var　into float/long/int/len/index　to enhance the usability of the dynamic graph.
  - For the functions that strongly rely on hook in tasks, adds the forward pre-hook and forward post-hook APIs for Layer to easily obtain and change the values of variables of the network without changing the structure of network input and output, thus improving the usability of the dynamic graph.
  - Supports the validity of the cudnn algorithm cache in dynamic graph mode and improves the performance by 200% on the waveflow model.

- Declarative Programming Mode (Static Graph):

  - The executor supports automatic pruning of the network during run time according to the feed and fetch variables. Remove the parts irrelevant to the current feed and fetch to improve the running efficiency. Supports the multi-task learning network.
  - Optimizes the back propagation process. Automatically prunes the variables that do not need to be back propagated. Explicitly sets `stop\_gradient=True` for variables when networking is not required.
  - The executor supports to fetch variable-length tensors of multi-card to provide the better support for tasks that use variable-length data (e.g. some NLP tasks).
  - Fixes the problem of discarding some tail data about the insufficient number of cards in the single-process multi-card inference phase by setting `drop\_last=False` in DataLoader to avoid discarding the tail data.
  - Adds a mixed precision (AMP) and recomputation combination mechanism. When they are jointly used for the Bert-large model, the maximum batch size and the throughput are increased by 400% and 17.5%-31.4% respectively.

- DataLoader:

  - Adds accelerated data reading in multi-process mode. For the Map-style type of dataset s, users can improve the data reading performance by implementing user-defined Dataset and BatchSampler. The speed is significantly increased for tasks with a large amount of data reading or complex pre-processing. For example, the multi-process reading mode is used for the video classification TSM model. The training performance is improved by 419% in declarative programming mode ("static graph") and 89.6% in imperative programming mode ("dynamic graph").

- Usage Method of Gradient Pruning:

  - The clipping type is passed in by the optimizer's `grad\_clip` parameter. The global and partial clipping functions are supported. The original `set\_gradient\_clip` API is no longer recommended and may be removed in subsequent versions. The `grad\_clip` parameter is removed in `ParamAttr` (incompatibility upgrade). Gradient clipping of a single parameter cannot be performed through ParamAttr. Gradient clipping of some parameters can only be implemented through the above new APIs.

- Dynamic graphs, static graphs, and high-level APIs support consistent call of collective operators.

- Intel stops maintenance on Ngraph and removes the codes related to the NGraph library.

- Removes unused or incompatible attributes such as the `is_test` attribute from all MKL-DNN-related ops.

- Adds the Support for Complex Number Computation:

  - Adds ComplexVariable and supports complex number tensor expressions and common complex number operations, including four basic operations, matmul, kron product, reshape, and transpose.

- Function Upgrade of the Performance Analysis Tool (Profiler):

  - Supports hierarchical statistics and printing of profile results based on nested call relationships between events.
  - Adds the tracer\_option parameter which can be configured as `Default`, `OpDetail`, and `AllOpDetail`. Supports users to select different levels of timing and analysis objects.
  - Adds the statistic function of framework overhead and GpuMemcpy operations.

- Full Optimization of Error Messages

  - Optimizes an accumulative total of thousands of vague error messages and standardizes the error type and description.
  - Automatically detects some user misoperations and gives clear error messages.
  - Optimizes GPU-related API error messages, converts unreadable error codes into specific messages, and keeps synchronous with information on NVIDIA's official website.

### Performance Optimization

- Imperative Programming Mode ("Dynamic Graph"):

  - Optimizes the data structure of the automatically generated OP function to reduce the framework overhead, the trainning speed of ptb lm model increased by 4% on single card.
  - Optimizes the design of the InferVarType interface to reduce the framework overhead, raises the speed of InferVarType, and increases the training speed of ptb lm model by over 5%.
  - Reduces the unnecessary addition of attributes in the dynamic graph ops to reduce the framework overhead,b and increases the training speed of ptb lm model by 4% .
  - To improve the data loading performance, i cmplements the tensor application shared memory storage and the tensor serialization and deserialization mechanism, supports the transmission of tensors between processes, optimizes the asynchronous performance of DataLoader in dynamic graph mode, and further increases the single card training speed of ResNet model on the P40 machine by over 15%.
  - Optimizes the performance of the dynamic graph variable slice by 60% and supports step in the slice to be negative.

- Declarative Programming Mode ("Static Graph"):

  - Adds the automatic fusion function. Supports the fusion of subgraphs composed of elementwise, activation, sum, cast, scale, fill\_constant, and other element-wise operators. The performance improvement depends on the number of related subgraphs matched in the network. Currently, the training speed of the RNN language model is greatly improved.

- OP Performance Optimization

  - Adds caches for Prepare Data during the OP execution process, accelerates by an average of 2% for 10+ model training tasks, and reduces the framework overhead by up to 6%.
  - Optimizes the GPU performance of depthwise\_conv2d to accelerate by 20% at the common parameter settings.
  - Optimizes the implementation of the GPU broadcast mode of elementwise\_mul to accelerate by 2-50 times for different inputs.
  - Optimizes the GPU implementation of conv2d\_transpose to achieve significant performance improvement for fp16.
  - Optimizes the implementation of shape OP to avoid waiting due to unnecessary data transfer between different devices.

#### Bug Fixes

- To ensure successful operation at a large SGD data size, fix the SGD error `Xbyak::Error` problem that occurs when the data size is very large.

- Fix the MKL memory leak problem under the Linux version.

- Fix the bug of command line parameter parsing at the dynamic graph multi-card startup.

- Fix the problem that occurs when the clone(for\_test=True) interface processes a network containing the control flow op.

- Fix cyclic dependency between the dynamic and static graph modules.

- Fix the compatibility problem of pickle dump/load between Python 2 \& 3.

- Fix the problem of the parameter not being registered or overridden as none for the dynamic graph layer.

- Fix the problem of Op output Var naming conflict caused when different Op names have the same attribute value.

- Fix the output LoD setting when axis=0, which shall be the splicing of input LoD.

- Fix the bug of BatchNorm mean and var that cannot be updated in eval mode.

## Inference Deployment

### Paddle Inference

#### Function Upgrade

- Adds TRT submap's support for dynamic shape input as well as the `config.SetTRTDynamicShapeInfo(min_input_shape, max_input_shape, opt_input_shape)` interface. This interface is used to specify the minimum, maximum, and optimal shape information of the input of the submap (Optimal shape means that TRT will select the runtime optimal kernel at this shape). After the shape information is specified, the Dynamic shape mode is used during the Paddle-TRT operation and the input of any shape between `max_input_shape` and `min_input_shape` is supported during the inference. This function supports FCN, Faster RCNN, Ernie/Bert, and other dynamic shape input models.
- To meet the need for binding the computation flow to the current thread during user inference, refactors the device context data structure to support the CUDA computation flow priority, and adds a thread local GPU memory allocator ThreadLocalAllocator. Has the ability to bind different threads to different CUDA streams.
- The MKL-DNN quantization function fully supports all quantitative models. Adds the support for 'weight\_quantize\_type' as range\_abs\_max and 'channel\_wise\_abs\_max'. Supports the out\_threshold attribute.
- Adds official website inference API reference

#### Performance Optimization

- For the targeted optimization of CUDA Bert/Ernie, adds `embedding_eltwise_layernorm` fusion implementation and optimizes the `multihead_matmul` and `fc_elementwise_layernorm` fusion implementation. Compared with the previous version, the ernie fp32 inference is optimized to 8.7 ms from 10 ms or by 13% under the conditions of P4 card, cuda10, and batch\_size=1.
- TRT submap's support for the dynamic shape of the Ernie/Bert model. Under the conditions of T4 card, cuda10, and batch\_size=1, the ernie fp16 inference performance is 2.9 ms, which is accelerated by 56%, compared with 6.6 ms for fp32.
- Optimization of mobilenet v3 by Paddle-TRT. Supports TRT hard sigmoid OP and adds a hard swish plugin. The inference is optimized to 2.29 ms from 3.48 ms or by 34% under the conditions of batch\_size = 1 and P4, or to 1.33 ms from 2.76 ms or by 51% under the conditions of V100.
- Adds the support for the swish activation function DNNL so that the ShuffleNet performance is improved by 76% on the 6248 single-core processor.
- Quantization: Adds the support for matmul op quantization; adds the `matmul+transpose+reshape` fuse and the `scale+matmul` fuse. After matmul quantization and fuse addition, the performance of the Ernie fp32 and quantized INT8 models is improved by about 10%(on the 6271 machine).
- Adds the support for DNNL inplace op: Currently, the execution of inplace of `elementwise_add` and most activation functions including softmax, gelu, and relu are supported so that the Ernie performance is improved by about 2% on 6248.
- After the above optimization and quantization, the speed of the current Ernie INT8 model is increased by about 5.51 times compared with the FP32 model on which DNNL optimization (including fuses) and quantization are not performed.

#### Bug Fixes

- Fixes the problem of failure to identify and load a locally generated calibration table in the service and regeneration of a calibration table due to inconsistency of locally and server generated calibration table names resulting from unstable fusion strategies in the TRT int8 off-line quantization in the inference phase. Currently, the calibration table name can remain consistent when TRT off-line quantization calibration runs for multiple times.
- Fix the problem of parameter transmission error during the generation of a calibration table in the TRT off-line quantization in the Inference phase. This problem will affect the final quantitative inference precision to some extent.

### Paddle Serving

#### Improved Usability

- Uses pybind to encapsulate c++ codes. Provids a usage method of the python API. Provides the python2 and python3 environment whl installation packages of paddle\_serving\_server, paddle\_serving\_server\_gpu, and paddle\_serving\_client. Releases Version 0.2.1
- Provides cpu and gpu Docker images in the centos6/7 environment, including executable images and compilable images
- Provides an API to directly save the models and configuration files required for Serving deployment. Seamlessly connects the Paddle training framework
- Implements the startup of the model inference service using one line of commands

#### Function Perfection

- Provides RPC and HTTP inference service methods
- Supports Python and Go language clients
- Supports the A/B test
- Releases Paddle\_serving\_app Version 0.0.2. Provides preprocessing APIs for LAC words segmentation preprocessing, Chinese BERT model preprocessing, and image processing
- Supports the timeline visualization of the inference service

#### Performance Optimization

- In RPC service mode, the Chinese BERT semantic vector indicates that the inference speed of the inference service is increased by 2.04 times compared with paddle\_gpu\_serving Version 0.8.2 under the conditions of a single P4 card and batch size 1.

#### Documents and Examples

- Improves and adds Chinese and English operating documents, Chinese and English development and deployment documents, and Chinese performance tuning documents.
- Provides seven types of model inference service examples, including Chinese word segmentation, English emotion analysis, Chinese semantic vector representation, CTR estimation, image classification, and other fields.

### Paddle Lite

#### Function Upgrade

- Compilation and Installation
  - Optimization of Paddle-Lite compilation scripts: Splits separate compilation scripts from the Android\\iOS\\ArmLinux platform to improve the script usability.
  - Support for Python installation: The Paddle-Lite Python library can be installed on PC Linux/Windows/Mac. Python can call the Lite opt optimization model.
  - Support for Windows compilation: Paddle-Lite can be compiled in the Windows environment. Currently, the Windows environment supports only the x86 compilation.
- Basic Functions
  - Adds the submap segmentation function. For models lited by submap access, manually segments a submap through configuration files so that a specified OP runs in the host to improve the performance (ssd\_mobilenet\_v1, accelerated by about 4.3 times).
  - Optimizes the support for quantitative models generated using the uncalibrated post-training quantization method. Quantizes common classification models to 8bit. Decreases the precision loss to 0.1% from 2%.
- Hardware Support
  - Adds RK 1808 NPU to support the fully quantitative MobileNetV1 model.
  - Adds MTK MT8175 APU to support the fully quantitative MobileNetV1 model.
  - Adds a method of access to Baidu Kunlun XPU Kernel to support ERNIE, ResNet-50, and BERT models.
  - Adds Cambricon MLU270 to support the following models: Resnet50 (int8) and Senet101 (int8).
  - Adds Bitmain BM1682 to support the following models: Mobilenet, Yolov3, Mobilenet-ssd, Inceptionv4, Vgg16, DarkNet-YOLOv3, and PyramidBox.
  - Mobile GPU (opencl): Supports mobilenetv1/v2, GAN correlation, mnasnet, sqeueezenet, shufflenet, resnet, unet, and vgg16 models
  - Nvidia GPU: Adds exec multi-stream support. For the model structure with parallelism, the performance is expected to be improved by 5-15%, compared with a single stream. Common visual models, generally have no parallel structure and will obtain no benefit from enabling multi-stream. The multi-streams function `config.set_multi_stream(true);` is enabled under the cuda platform.
  - Optimization of the x86 platform: Reduces the size of the inference library (200M---->16M), supports LOG shutdown (--shutdown\_log=ON), supports the multi-thread sharing model weight parameter by full\_api, and adds x86 cxx\_demo
  - Huawei NPU:
    - Increase the speed of Benchmark models (mobilenet\_v1, mobilenet\_v2, squeezenet\_v1.1, mnasnet, and shufflenet\_v2) by 5-10 times.
    - Supports caching different sizes of NPU models to improve the performance of models with a variable input size.
- Demo:
  - Adds an Android Demo for real-time mask detection based on camera preview
  - Adds an Android Demo for real-time face key point detection and beauty
  - Adds an Android Demo for Boston house price inference of mobile training

#### Performance Optimization

- Reduction in time consumption of InferShape: When the predictor continuously runs, the total time consumption of infershape is reduced to 0.08 ms from 0.25 ms.
- The kernel of the opencl part supports dynamic shape and transfers partial computation to ReinitWhenNeeded. fc\_buffer, elementwise\_add, scale, activation, and grid\_sampler.
- Optimizes the sgemm performance on the low-end machine.
- Optimizes the Precision Profiler function. Optimizes the type setting. Adds the support for the standard deviation and growth rate attributes (a sequence can be compared when the mean value is the same as the standard deviation). Supports the precision printing of output of the OpenCL image/buffer at every layer. Supports writing precision results (final precision summary) at every layer to mobile devices to facilitate APP debugging. Separates precision printing from dependency on the original profiler for time consumption statistics.

#### Bug Fixes

- Fix the problem that the predictor results are random because the act\_type of the conv op is not initialized.
- Fix the opencl kernel. The bilinear kernel compatibility problem on the mali gpu, the problem of incorrect computation results of the instance norm, and the kernel registration error of the reshape result in the model transformation failure. The problem that the exp and tanh cannot find the kernel results in kernel name error and model op binding failure.
- Fix the problem that the opencl gets stuck at the end of the mali gpu's computation.
- Fix the opencl resource-related problem. Isolates every cl::kernel/cl::program and other resources in every predictor.

### PaddleSlim

#### Quantization

- Adds a post-training quantization method without calibration data. The int16 precision is lossless. The int8 precision loss is smaller than 0.1%.
- Enhances the quantization function, improves the output scale information of the quantization OP, and supports the CPU inference-side comprehensive adaptive quantitative model.

#### Pruning

- Adds two pruning strategies including FPGM and BN scale. Improves the precision by 0.6% at the same compressibility on the MobileNetV3-YOLOV3-COCO task.
- Adds a user-defined pruning strategy API to facilitate developers to quickly add compression strategies.
- Adds the default processing logic of added operators in the pruning function and extends the support for pruning more complex networks.

#### NAS

- Adds the DARTS series of search algorithms and provides an extended interface to facilitate users to investigate and implement a new model structure search strategy.
- Adds an early stop mechanism in the model structure to improve the usability of the search function.
- Adds a model structure search strategy based on reinforcement learning and provides an extended interface to provide reference for users' investigation and implement of the new strategy.

#### Pantheon

- Supports data transmission and storage in fp16 format. Supports knowledge transmission with multiple channels in online distillation mode. Increases the transmission efficiency of knowledge data.
- Adds lexical analysis examples to facilitate users to build their own distillation tasks based on the examples

## Development Kits

### PaddleDetection

- Enhancement of the Richness of Models

  - Adds Efficientdet-D0: The COCO val2017 precision is 0.3 higher than the TF precision (33.8 vs 33.5), excluding the postprocessing inference speed that is basically equal or has a weak advantage (about 13 ms vs about 14 ms, T4 measured speed).
  - Adds the instance segmentation model HTC. The inference speed under the V100 is 11.5 FPS which is 7.4 FPS higher than that of the competing product. The BBox mAP under COCO 2017 is 42.1% and Mask is 37.1.
  - Adds the anchor-free model FCOS:  The COCO val2017 precision is 1.1 higher than the pytorch precision (39.8 vs 38.7).
  - Adds a MobileNetV3 backbone network in YOLOv3. The COCO dataset  precision is 31.6.
  - Adds the anchor-free model CornernetSqueeze: The COCO val2017 precision is 34.5 which is 0.1 higher than that of the competing product. The precision of the optimized model is 38.2, up 3.7 points. The speed is 5% faster than that of yolo\_v3 darknet
  - Adds the server-side practical object detection model cascade\_rcnn\_resnet50\_vd\_fpn\_dcn. The COCO mAP is 47.8% at 20 FPS on V100, better than that of the competing product EfficientDet.

- Launch of Three Mobile Models

  - SSDLite series of models: For the ssdlite-mobilenet\_v3 large model, mAP on the COCO dataset  is 22.8% and the single-thread inference speed on the Qualcomm Snapdragon 845 is 95 ms. For the ssdlite-mobilenet\_v3 small model, mAP on the COCO dataset  is 16.6%, the single-thread inference speed on the Qualcomm Snapdragon 845 is 40 ms, and the precision is better than that of the competing product. For the ssdlite-mobilenet\_v1 model, mAP on the COCO dataset  is 23.6%, the single-thread inference speed on the Qualcomm Snapdragon 845 is 140 ms, and the precision is better than that of the competing product.
  - yolo v3: For the yolov3\_mobilenet\_v3 pruning model, the single-thread inference speed on the Qualcomm Snapdragon 845 is 91 ms and the precision is 24.6 (input dimensions 320 \* 320). Both the speed and precision are better than the speed and precision of the framework SSDLite model, a competing product.
  - Faster RCNN: Based on the COCO dataset , mAP of cascade\_rcnn\_mobilenet\_v3 large\_fpn is 25.0% and the single-thread inference speed on the Qualcomm Snapdragon 845 is 87 ms at the input image dimensions 320 x 320; mAP is 30.2% and the single-thread inference speed on the Qualcomm Snapdragon 845 is 351 ms at the input image dimensions 640 x 640.

- Inference Deployment Refactoring:

  - Adds a Python inference deployment process and supports the RCNN, YOLO, SSD, RetinaNet, and face series of models. Supports video inference.
  - Refactors C++ inference deployment and improves the usability.

- Usability Improvement and Function Components

  - Adds strength of AutoAugment data.
  - Upgrades the PaddleDetection library document structure.
  - Supports migration learning to automatically perform shape matching.
  - Optimizes the memory usage in the mask branch evaluation phase.
  - Upgrades the inference deployment function and adds python scenario image and video inference.

### PaddleSeg

- Adds the Lovasz Loss function to effectively improve the precision of multi-class segmentation

- Fully upgrades human portrait segmentation series model

  * Releases the first mobile real-time portrait segmentation model HumanSeg-lite
  * Adds a video-level segmentation postprocessing solution based on optical flow algorithm

- Adds a remote sensing image segmentation solution

  * Adds a data pre-processing solution for multi-channel remote sensing images
  * Adds a data augmentation strategy for multi-channel images
  * Provides a tutorial on the segmentation of two meteorological remote sensing fields including snow detection and cloud detection

### PaddleClas

- Adds the MobileNetV3 series of models and performs performance evaluation on 23 series of and 117 pre-training models.
- Adds an SSLD knowledge distillation solution, improves the recognition accuracy by more than 3%, and releases six distillation models including resnet50\_vd (82.4%) and mobilenetv3 (78.9%).
- Adds eight data augmentation modes including AutoAugment, RandAugment, CutOutRandErasing, HideAndSeek, GridMask, Mixup, and Cutmix that are used to increase the diversity of training samples and improve the generalization performance of the models.
- Adds 100,000 types of image classification pre-training models and improves the recognition accuracy rate by up to 30% for image classification service application scenarios.

### PaddleOCR

- Adds DB and EAST text detection algorithms.
- Adds Rosetta, CRNN, STAR-Net, and RARE text recognition algorithms.
- Adds an ultra-lightweight OCR model with a total size of only 8.6M (4.1M for text detection and 4.5M for text recognition). Supports text recognition at scenarios such as horizontal and vertical layout, long text, and mixed Chinese and English numbers.

### Parakeet

- Releases English pre-training models and audio samples of WaveFlow (res channel=64/128), ClariNet, WaveNet, and other models.
- Fixes the problem of too slow speed of the Conv2DTranspose fp16 kernel and simplifies the WaveFlow inference logic in fp16 mode.
- Increases the model training speed significantly. Doubles the speed in DeepVoice3, TransformerTTS, and other models by optimizing the data preprocessing and OP calculation logic.

## Utility Components

### PaddleHub

* Enhances the vision models’ richness. The total number of pre-trained  models is 120+.
  * Adds the large-scale vision pre-training models and greatly improves the fine-tune effects of image classification and object detection tasks
  * Adds the industrial short video classification model VideoTag and supports the recognition of more than 3000 types of Chinese tags
  * Adds the lightweight Chinese OCR model and supports one-click quick OCR recognition
  * Adds pedestrian detection, vehicle detection, animal recognition, and Object365 2019 large-scale object detection models which win the first prize on detection contest.
* Fine-tune API Upgrade
  * Adds five predefined networks for the text classification tasks, including CNN, BOW, LSTM, BiLSTM and DPCNN.

### PaddleX

* Releases PaddleX 1.0, the Entire Process Development Toolkit

- Opens up the entire process of deep learning development from data access to inference deployment and provides a easy-to-use Python API
- Covers the four mainstream task scenarios including image classification, object detection, semantic segmentation, and instance segmentation in the CV field. Integrates utility components such as PaddleHub, PaddleSlim and VisualDL.
- Presets 26 types of a total of 43 models including the industrial practice refining precipitation pre-training model and a number of characteristic and advantageous Paddle models.
- Provides advanced functions such as automatic data analysis, automatic hyper-parameter recommendation, data argumentation strategy, model pruning training, model quantization, pre-training model saving and reuse, multi-platform  deployment, model interpretation and encryption.
- Innovatively integrates the model explainability analysis function
- Provides an official implemented GUI and supports one-click installation of Windows, Linux, and Mac systems.

### VisualDL

* Releases VisualDL Version 2.0 beta

- Upgrades the back-end kernel, is lighter, faster, and more compatible, and supports file storage system expansion
- Fully upgrades APIs and uses less codes to finish visual analysis, significantly improving the usability
- Upgrades UI and interaction, provides better localization support, achieves clearer and more intuitive visual analysis, and gives users immersive experience
- Deeply integrates with Paddle development kits and utility components and provides a smoother deep learning development experience

### PaddleFL

- Releases PaddleFL Version 1.0
  - Open sources federated learning based on mulit-party computation (MPC) to supports horizontal, vertical, and other federated learning scenarios
  - Refactors the original framework to integrate and open source the new and original federated learning solutions
  - Adds the function of converting a single-machine model into an FL trainable program to support more models and scenarios

### PGL

* Releases the industry's first graphical neural network model ERNIESage which combines semantic information with structural information
* Adds PGL-KE. Currently, PGL covers 25+ graph learning models including walk, messaging, and knowledge embedding
* Adds graph batch, graph pooling, and other graph operators
* Fully supports the Open Graph Benchmark benchmark test set and releases the corresponding SOTA
* Adds MetaPath2Vec++, Mulit-MetaPath2Vec++, STGCN, GIN, and PinSage models in the model zoo

### PARL

* Open sources the industry's first evolutionary learning application framework EvoKit
* Adds the support for the Multi-Agent RL algorithm including MADDPG
* Adds the support for multi-card training and releases an example of a multi-card DQN algorithm
* Open sources SOTA algorithms TD3 and SAC in the continuous control field
* Open sources the NeurIPS2019 reinforcement learning challenge champion model and training solution

### Paddle Quantum (Quantum Computation Laboratory)

* First release of Paddle Quantum. Paddle Quantum is a quantum machine learning tool set developed based on Baidu Paddle. It supports the setup and training of the quantum neural network and provides easy-to-use quantum machine learning development kits and cutting-edge quantum application tool sets such as quantum optimization and quantum chemistry, making Paddle the first deep learning platform supporting quantum machine learning in China.
  - Supports a QAOA algorithm to solve the max-cut problem
  - Supports a VQE algorithm to calculate the minimum characteristic value of H\_2
  - Supports an SSVQE algorithm to calculate the characteristic spectrum of a given Hamiltonian
  - Supports a VQSD algorithm to calculate the diagonalized form of the quantum state and give the eigendecomposition of the quantum state
  - Supports a Gibbs algorithm to generate the Gibbs state of a given Hamiltonian at a certain temperature
  - Supports common functions in quantum computation
  - Supports the description of the U\_Ansatz quantum circuit
