==============
Release Notes
==============

Table of Contents
#####################################
* Highlights
* Fundamental framework updates
    * Installation
    * Optimization on Intermediate Representation IR and Pass
    * IO optimization
    * Execution optimization
    * Video memory optimization
    * Refine CPU JITKernel
    * Low-level Intel CPU computing optimization
    * Intel nGraph graph compiling engine integration
    * Adjustments to basic framework functionality
    * Accomplished basic functions in the preview version of dynamic graph Inference engine
* Inference engine
    * Server-side Inference Engine
    * Mobile Inference Engine
    * Deployment tools
* Distributed training
* Model construction
    * PaddleCV Intelligent Vision
    * PaddleNLP intelligent text processing
    * PaddleRec intelligent recommendation
* Tools and Components
* Bug fixes notes

Highlights
#####################################
* Significant improvement has been made on training speed and memory management of the fundamental framework. Full support for quantitative training has been incorporated. Integration of Intel nGraph is also accomplished. Besides, the basic functions of single-card and single-node in the preview version of dynamic graph are perfectly implemented.
* We have officially released the model compression toolkit `PaddleSlim <https://github.com/PaddlePaddle/models/tree/develop/PaddleSlim>`_ and the model inference service `Paddle Serving <https://github.com/PaddlePaddle/Serving>`_ to broadly enhance the PaddlePaddle deployment capabilities.
* Boosted distributed IO interfaces and the stream read capability of remote file systems. Synchronous multi-machine multi-card GPU training promotes bandwidth-insensitive training through enabling sparse communication. For low-bandwidth network, such as network of 10G, synchronous training is 10 times faster.
* Support for the K8S ecosystem is smoothened through Paddle-K8S-Operator support in industrial environments; Kubeflow supports paddle-job.
* We have officially released the `video classification toolkit <https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/video>`_ which covers mainstream video classification models, including Non-Local, TSM, Attention Cluster, NeXtVLAD, Attention LSTM, StNet, TSN.
* `ERNIE <https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE>`_ , a Chinese semantic representation model is introduced, which attains accuracy with absolute 1-2 percentage points higher than BERT on multiple Chinese language tasks. Generic dialogue comprehension model DGU is incorporated, with support for 5 types of dialogue tasks, and reaches SOTA in 3 public datasets.
* The Recommendation Model Based on `Graph Neural Network <https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/gnn>`_ (GNN) is carried out, for which Benchmark expectation has been reproduced on public dataset.
* `PaddleHub <https://github.com/PaddlePaddle/PaddleHub>`_ , a management tool for pre-trained models, has been officially released, offering three functions: pre-trained model management, command-line one-click manipulation and transfer learning. It strives to facilitate model management and conduct transfer learning more efficiently.
* Open source `AutoDL Design <https://github.com/PaddlePaddle/AutoDL/tree/master/AutoDL%20Design>`_ is officially released to enable automatic network design.
* Latest upgrades on the parallelization-oriented `PARL1.1 <https://github.com/PaddlePaddle/PARL>`_ . Users are allowed to implement parallelized reinforcement learning algorithms by using a decorator.
* The model conversion tool `X2Paddle <https://github.com/PaddlePaddle/X2Paddle>`_ has been officially published, which enables transfer of inference models in other deep learning frameworks to PaddlePaddle without any compromise.

Fundamental Framework Updates
#####################################
* Installation
    * install\_check.run\_check() interface is introduced to provide a more graceful check on whether the installation was successful.
* Optimization on Intermediate Representation IR and Pass
    * The encapsulation is fulfilled of IrGraph, IrNode, IrVarNode, and IrOpNode. IR Passes scripted in Python is also enabled.
* IO optimization
    * PyReader optimization: the brand new interface reader = fluid.io.PyReader (..., iterable=True, ...) makes it possible to create an iterable (by 'for' loop) reader and the data will be sent to the network through the 'feed' method.
* Execution optimization
    * The 'place' parameter in with\_data\_parallel can be set to specify to run model on which GPU cards to execute single-process multi-training tasks.
    * Scheduling strategy applied on the multi-card executor is optimized, which is proved on the performance that execution speed on the ResNet50 and Transformer models has witnessed a increase of 8%~19%.
    * For Multi-card environment, grouped Fuse for AllReduce is developed. With this manner in place, ResNet model on multi-card is accelerated by 8%~30% (the figure varies with the number of cards). Moreover, Transformer model running on multiple cards picks up speed by 4%.
* Video Memory optimization
    * GC strategy optimization: Eager Deletion strategy supports timely deletion of internal while\_op variables; supports non-full-quantity Eager Deletion strategy, users can set FLAGS\_memory\_fraction\_of\_eager\_deletion=0.xx to control the percentage of immediate deletion memory/memory\_space in real time.
    * Op optimization: Optimize the backward registration mechanism of cross entropy, expand, layer\_norm, dropout, etc., and remove irrelevant variable dependencies, and improve the video memory performance.
    * Two new FLAGS (FLAGS\_initial\_gpu\_memory\_in\_mb and FLAGS\_reallocate\_gpu\_memory\_in\_mb) to allow the users to specify the initial memory pool capacity and the reallocated memory pool capacity.
    * Adjust the inplace\_op\_pass strategy to increase the coverage of the inplace strategy.
    * Removed the logic for doing activation op inplace optimization on the python side, and included it to inplace\_op\_pass.
    * Memory Profile function is provided.
* Refine CPU JITKernel
    * Modify the manner to call JITKernel, employ cache mechanism and interfaces to get all functions of the same type, which is convenient for developers to flexibly call desired interfaces.
    * As JITKernel is adopted to optimize the SGD algorithm, the equivalent OP part speed is increased by 44% and the overall training speed is increased by 12% in the PyramidDNN model; On the other hand, JITKernel is used to optimize fused\_embedding\_seq\_pool, and the backward versions of corresponding ops in the PyramidDNN model is accelerated by 18% and overall training speeds up by 6%.
* low-level Intel CPU computing optimization
    * MKLDNN is upgraded to v0.18 and includes various performance boosts (e.g. GEMM-based convolution operations/INT8 convolution operations, etc.).
    * GELU OP is accelerated by MKL. After optimization, the OP performance attains 3 times of the previous.
    * Unit testing of MKLDNN-related Kernels are refined.
* Intel nGraph graph compiling engine integration is to facilitate the support for more hardware backends for PaddlePaddle
    * The subgraphs are transferred to the nGraph core via ngraph\_engine OP, and then optimized with graph algorithms, after which they will be dispatched to execute on CPUs. nGraph can be called at runtime with the environment variable set as FLAGS\_use\_ngraph=true.
    * Training and inference of the ResNet50 model on the CPU is fulfilled. The performance of the ResNet50 training and inference on CPU gains notable increase compared with the direct optimization by MKLDNN.
* Adjustments to basic framework functionality
    * Synchronized Batch Norm operation becomes available; specifying axis in softmax is allowed; new operators are in place: spectral norm, rang, acos, asin, atanh; Npair Loss is adopted for feature learning.
    * cosine\_decay , a new learning rate strategy, is implemented.
    * Users can use sampled\_softmax\_with\_cross\_entropy to improve training efficiency in large dictionaries.
    * Fuse is possible between SGD and Adam optimization algorithms. If enabled, on the Transformer model, the speed can increase by 2%, while on the Cycle GAN model, the gain turns out to be 6%.
    * A more sophisticated lsmtp, which is able to perform clipping internal cell, initializing cell state and hidden state.
    * A more adjustable adagrad by which users can initialize cumulative momentum.
    * Users are allowed to handle Tensor through \_\_getitem\_\_ method.
    * QuantizationFreezePass, ConvertToInt8Pass, and TransformForMobilePass are introduced with comprehensive support for both dynamic and static quantitative training methods and saving corresponding model.
* Accomplished basic functions in the preview version of dynamic graph
    * Basic functions: LRDecay, single GPU card and single-node CPU model training and evaluation.
    * API: expose the rudimentary interfaces of dynamic graph to users; reconstruct current Layers; build Layers such as GRU, LayerNorm, NCE, PRelu.
    * Performance: performance evaluated on the ResNet, MNIST model is essentially the same as the static graph.
    * Dynamic graph implementation of models such as Transformer, MNIST, SE-ResNeXt.

Inference Engine
#####################################
Server-side Inference Engine
+++++++++++++++++++++++++++++++++++++
* Inference library is currently integrated with PaddlePaddle/Anakin to unify interfaces for a more efficient inference process
    * able to handle Anakin GPU submaps and CPU submaps.
    * The Python inference interface has accepted Anakin subgraph.
    * significant Inference acceleration on ResNet, VGG, GoogleNet, MobileNet, ShuffleNet, Faster R-CNN, YOLO, SSD and other models
* Inference framework optimization. Inference of small models expedites noticeably
    * Through configuring runtime\_context\_cache\_pass, focal models have obtained a speed-up of 17%.
    * The infershape of 5 OPs are refined, so that the focal models accelerate by 13%.
    * The ZeroCopy interface is upgraded to avoid redundant CPU copies when using AnalysisPredictor.
* Reinforce INT8 quantitative Inference
    * More inclusive support for INT8 Quantization through TensorRT, applicable for AlexNet, Googlenet, VGG, MobileNet, ShuffleNet and more. Utilize the information on TensorRT in an optimal manner to perform the serialization and deserialization so that a model will be initialized more speedily.
    * Implement the INT8 quantization framework based on C++ Pass. A few new INT8 OP Kernel: Transpose, Contact, Requantize. By fine-tuning the quantization strategy in MkldnnQuantizerConfig, users can promptly get the INT8 quantization model that meets the accuracy requirements. The INT8 quantized ResNet-50/MobileNet v1 model achieved a performance 7 times/3 times higher compared with the original FP32 model (tested on the Xeon 6271 server supporting the AVX512-DL Boost instruction set).

Mobile Inference Engine
+++++++++++++++++++++++++++++++++++++
* ARM CPU
    * Paddle Mobile has reconstructed and enhanced efficiency of the matrix operation library sgemm and sgemv, which gives rise to performance boost of 10%~100% on most models.
    * 19 new operators are provided in this version such as while, sequence\_expand, sequence\_pool, sequence\_softmax, gru\_unit, beam\_search, and beam\_search\_decode. Apart from that, there has also been a large amount of optimization, and the support attention-based end-to-end Model prediction.
    * arm v8 of winograd implementation: higher inference performance on v8 hardware on IOS; winograd support for operator fusion to ensure higher efficiency after operator fusion.
    * Direct convolution for kernel with a 3x3 sliding window, which will be more efficient than winograd and gemm on the condition that the number of channels is small.
    * Reconstructed and optimized depthwise convolution with the kernel size 3x3: in contrast to previous versions, it supports arbitrary padding, and attains better performance and returns more reliable calculation results.
    * Depthwise convolution with the kernel size 5x5 on armv8: the NAS model prediction speeds up by more than 30%.
    * Complete the efficiency optimization of the deconvolution conv2d\_transpose.
    * Consolidated with memory reuse strategy based on graph optimization. When the strategy is applied, most models can reduce memory usage by nearly 50%. It is automatically turned on for the ARM CPU (not compatible with FPGA and GPU).
* ARM GPU
    * Paddle Mobile completes the convolution optimization for the kernel with size 1x1, and MobileNet v1 has an average inference performance improvement of 35% on Qualcomm Adreno GPUs.
    * Paddle Inference has preliminarily unified of Paddle Mobile and Anakin interfaces. Further integration is pending.

Deployment Tools
+++++++++++++++++++++++++++++++++++++
* Model compression toolkit PaddleSlim
    * Model clipping compression strategy: users can select sensitivity or uniform modes, apply it for various models such as VGG, ResNet, MobileNet, and customize clipping range.
    * Quantitative training model compression strategy: there are two two quantitative training modes, dynamic mode and static mode. Channel quantization or overall quantization of parameters are also selectable. Users can save models with float type simulating int8 value domain, with int8 type, or with formats compatible with Paddle Mobile .
    * Model distillation compression strategy: users are permitted to add combined loss at any layer in the teacher network and student network. FSP Loss, L2 Loss, Softmax with Cross-entropy Loss are all available methods.
    * Other functions: Users can configure hyper-parameters of file compression task, and are allowed to combine multiple compression strategies. Moreover, checkpoints function is also applicable for distillation and clipping compression process.
* Paddle Serving
    * Remote paddle inference deployment is accomplished.
    * The server allows users to add data processing Operator, or define inference logic, and it supports model hot-loading.
    * The client side offers a C++ SDK which can be called business logic if needed. Users are allowed to customize protobuf to define network data transfer protocols, and A/B testing capabilities.
    * Provides sample templates for classic tasks in paddle serving, including text classification and image classification tasks.
    * Benchmarks for latency and throughput for text classification tasks.

Distributed training
#####################################
* Distributed IO optimization
    * Pipe Reader Interface Optimization: high-efficiency IO methods are in place as maintaining flexibility of data pre-processing. Enterprise-class Linux system customization is supported. High-performance IO components are implemented. Unified maintenance is carried out in the procedure of off-line data preprocessing. Remote file system stream read capability is enhanced to support the modes in which data are loaded to memory and distributed shuffling.
* Integration of Executor and distributed IO
    * AsyncExecutor is integrated into Executor, equipped with a new train\_from\_dataset/infer\_from\_dataset interface. It supports Pipe Reader-based training, and accepts user-defined PipeLine program on the condition of maintaining multi-queue IO function, and provides flexible python-side data processing.
* bandwidth insensitive training ability of synchronous multi-node multi-card GPU training
    * Sync GPU training is capable of sparse communication and adopts sparse all reduce.
    * Guarantee model convergence from the algorithm perspective and introduce DGCOptimizer through control of communication sparsity.
    * Experiments on ResNet50 on imagenet prove that: in terms of model convergence, for 90 rounds of ResNet50, convergence remains stable; in high-speed interconnected network environment, sparse communication does not compromise training speed; for low network bandwidth network environment (such as 10G network) ), sparse communication has notable advantages in training speed, where the speed of synchronous training is 10 times faster than that of dense communication.
* Collective Operator mode
    * Collective Operator mode is available. Multiple all reduce operations are allowed under GPU. Incorporating collective op into Program through the Python API makes the development of distributed optimization algorithms much more flexible.
* Convergence speed optimization for ResNet50 on Imagenet
    * Dynamic BatchSize, dynamic ImageSize, and rectangular crop can be used. With FP32 precision, on v100 single-node 8 card testing environment, the convergence speed increases by 68% (acc1\>=75.9%, acc5=93.0%).
* K8S Ecosystem Support
    * Kubeflow has supported paddle-job and contributed to the kubeflow community.
    * The Paddle-K8S-Operator for industrial application is supported. It can collaborate with kubeflow.
    * The K8S environment is suitable for beginners to submit task scripts, of which reproducible tutorials are given on Baidu Cloud.

Model Construction
#####################################
* PaddleCV Intelligent Vision
    * Video Classification Toolkit is formally released. It covers mainstream video classification models, including Non-Local, TSM, Attention Cluster, NeXtVLAD, Attention LSTM, StNet, TSN, and attains the level of mainstream implementations.
    * New pre-trained ImageNet-based model: GoogleNet, ShuffleNetv2, ResNet18, ResNet34.
    * New target detection YOLOv3 model. The effect is equivalent to the finest open implementation (mAP is 7 percentage points higher than the original author).
    * The Simple Baselines human pose estimation model based on COCO and MPII data is realized. The effect is able to parallel mainstream implementation.
    * npair loss is introduced to feature learning models, and raises recall@1 to 79.03% (+0.78%) based on the pre-trained model (arcmargin loss).
* PaddleNLP intelligent text processing
    * The Chinese semantic representation ELMo model is available. It supports multi-card training, and the training speed is twice as fast as mainstream implementation. It has been verified that the F1 value is increased by absolute 1.1% in Chinese lexical analysis tasks, and the Rouge-L value increases by 1% in Chinese reading comprehension tasks.
    * The Chinese semantic representation model ERNIE is implemented, which has improved the accuracy by absolute 1% ~ 2% compared with the BERT Chinese model in Chinese tasks such as natural language inference, semantic similarity, named entity recognition, sentiment analysis, and question and answer matching.
    * The read understanding model is upgraded by optimizing data pre-processing and document selection. The effect is that Rouge-L was upgraded to 65 (baseline 39.29) on DuReader validation datasets.
    * A knowledge-aware dialogue model is added. Compared with the baseline generation dialog model, it outperforms by an average of 1 percentage point on the F1, BLEU1, and BLEU2 metrics.
    * The dialogue model toolkit is available. It consists of Deep Attention Matching Net, a new automatic dialogue assessment tool and the BERT-based generic dialog understanding model DGU (Dialogue General Understanding), which supports five types of dialogue tasks, namely dialogue semantic matching, DA, DST, slot analysis and intention recognition, and attains the effect of SOTA on three public datasets.
    * The PaddleNLP toolkit is released to unify the modeling of NLP tasks such as text classification, text matching, sequence labeling, reading comprehension, and intelligent dialogue. And their corresponding industrial pre-trained models are also open to use.
* PaddleRec intelligent recommendation
    * Deep Interest Network (DIN): DIN is fulfilled in this version. reproduce effect on public dataset and support single/multi-card training in both cpu and gpu mode. DIN is appropriate for the sorting scenarios in recommendation (such as ctr prediction). The main feature is the combination of the estimated target information in the process of modeling the historical sequence.
    * Graph Neural Network (GNN): a session-based graph neural network recommendation model is introduced. Effect has been reproduced on public dataset. It supports single-node single-card training in both CPU and GPU mode. The model is suitable for the recall scenario in the recommendation. Using GNN to model the user's historical information can capture more complex transformation relationships underlying item sequences.
    * Word2vec: word2vec sampling strategy is adjusted. The effect is reproduced on the public dataset. Multi-machine training support is included as well.

Tools and Components
#####################################
* Open source AutoDL Design is officially released to enable automatic network design
    * A series of neural networks generated with the AutoDL Design, and a total of six models trained on CIFAR10 data have saved the network structures and involved weights. Therefore, any developer or researcher interested in deep learning can easily work on PaddlePaddle and public CIFAR10 data to perform inference and model fusion on these six models, which have attained an accuracy over 98%.
    * The source code for the encoder and the critic is made open source. The source code is based on the PaddlePaddle platform and the PARL framework developed entirely by Baidu. The code also comes with Chinese documentation and some brief demos that make it easier for users to run effortlessly. (for example, with "How many 1s is generated by RNN" as a standard, you can quickly verify the correctness of the entire framework). Moreover, users can download, install, run, and try to generate your own original neural network structure.
* Latest upgrades on the parallelization-oriented PARL1.1. Users are allowed to implement parallelized reinforcement learning algorithms by using a decorator
    * Parallelization can be achieved simply with a modifier (@parl.remote_class). After computing-intensive tasks, such as the data-preprocessing and simulator simulation tasks, have encountered this decorator, the data will be automatically deployed to the specified computing resources, and no longer occupy the computing resources of the main thread.
    * Support parallelization algorithms such as IMPALA, A2C, and GA3C.
* PaddleHub, a pre-trained model management tool, is released and strives to help users manage models and conduct transfer learning more efficiently
    * **Pre-trained model management:**  Pre-trained model download, search, version management and other functions in the PaddlePaddle ecosystem can be completed through the hub command line.
    * **One-click command line:**  Free from code, you can use the pre-trained model to infer straight through the command line, and quickly examine the effect of the training model. The current version supports the following models: lexical analysis LAC; sentiment analysis Senta; target detection SSD; image classification ResNet, MobileNet.
    * **Transfer Learning:**  Provides a Finetune API based on pre-trained models. Users can complete transfer learning with a small amount of code. The API mainly includes BERT/ERNIE text classification, sequence labeling, image classification transfer.
* The X2Paddle model conversion tool is officially released to transfer prediction models implemented in other deep learning frameworks to PaddlePaddle without loss. The tool is also attached with detailed comparison documents of TensorFlow, the Caffe framework's API , to help users transform the model to PaddlePaddle more easily

BUG fixes notes
#####################################
* Fixed precision inconsistency in BFS occurred in backward computation.
* Fixed redundant backward inputs created by optimizer minimize.
* Fixed Paddle-TRT occupying too much video memory.
* Fixed bugs in AllReduceDepPass.
* Fixed bugs in FastThreadedExecutor.
* Fixed bugs in Op such as Reshape, cross\_entropy, arg\_min\_max, recurrent, etc.
* Fixed problems with VarBase construction
* Fixed a number of problems and bugs in memory\_optimize\_pass: Adjusted the multiplexing logic from \>= to =, reduced fragmentation caused by Variable multiplexing, removing the dependency of memory\_opitmize\_pass on BlockDesc. Fixed a bug that different types of Variables would be reused mutually.
* Fixed an issue with util.plot in python3.
* Improved the stability of the Profiler and introduced Memory Profile function.
* Fixed the problem that multithreading was effective only when C++ inference had been cloned within the thread.
* fix bugs of some ops in InferShape.
* fix bugs of some ops with input LoD length = 0.
* fix bugs of recurrent op for StaticRNN.
* fix bugs of dygraph when saving and loading model checkpoint.