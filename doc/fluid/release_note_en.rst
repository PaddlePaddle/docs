==============
Release Notes
==============

Table of Contents
#####################################
* Highlights
* Fundamental framework updates
    * Installation
    * Dynamic Diagram Preview Version
    * Performance Optimization
    * Optimization of Memory
    * Execution optimization
    * Framework basic functions enhancements
    * OP perfect
* Inference engine
    * Server-side Deployment Library
    * Paddle Serving
    * PaddleSlim
* Distributed training
* Model construction
    * Image classification
    * PaddleDetection
    * PaddleGAN
    * PaddleVideo
    * PaddleNLP
* Tools and Components
* Bug fixes notes

Highlights
#####################################
* The training performance has been greatly optimized in data reading, execution scheduling optimization, Op computing logic and base cuDNN API call, CUDA kernel and MKLDNN. Further optimize the memory occupation, the whole has the leading advantage.
* Add LSTM and GRU based on Padding, which is more convenient for users to learn and use. And add the new language model and the example model of seq2seq translation model based on corresponding API ; Enhanced partial OP functionality to better support Tensor multiple dimension-variable tasks in NLP.
* Release the dynamic Preview version and provide the relevant API documents, and provide the official implementation of the seven model dynamic versions.
* The official model library publishes the uniform framework of PaddleDetection object detection, which covers the mainstream target detection algorithm and is easy to be extended and modular. Publish image generation library, cover mainstream GAN algorithm, can run one-click; Launch Paddle NLP - Research, which includes Baidu's latest research in the NLP field.
* Model compression framework PaddleSlim adds auto-shear strategy based on simulated annealing and lightweight model structure auto-search function (Light-NAS).
* Distributed training releases High-Level API Fleet, single machine to distributed training cost significantly reduced; The multi-card performance of GPU is improved significantly. The speed of 4x8 v100 configuration in ResNet50, BERT and ERNIE models is more than 50% higher than that of Benchmark.
* PaddleHub added 29 pre-training models, covering 40 models in three fields, including text, image and video.
* Paddle Graph Learning (PGL) Preview is released to provide the most advanced graphic learning algorithms based on two computational paradigms: Wandering and messaging.

Fundamental Framework Updates
#####################################
* Installation
    * Add support to CUDA 10 under Linux; add support to CUDA 9 under Windows; unify cuDNN dependency to 7.3+ on all operating systems.
    * Installation packages no longer differentiate based on whether the AVX instruction set is supported by the CPU; include new automated judgment and selection of whether to use the AVX instruction set or not.
    * Limit the versions of dependent packages to avoid the potential version conflicts under Python2 and Python3.
    * Provide a new Docker mirror that supports offline installation of PaddlePaddle.
    * Add installation tests for multi-card GPU.
    * Remove single-card training GPU’s dependency on NCCL.
* Dynamic Diagram Preview Version
    * Release APIs and documentations related to dynamic diagram.
    * Perfect fundamental functions; optimize memory and speed; support single multi-card GPU training.
    * Add dynamic graph version implementations of 7 models including transformer, ocr recognition, resnet, and language model that have equivalent performance.
* Performance Optimization
    * Optimization of Reading Data
        * Use multi-thread to optimize data reading and pre-processing; DeepLab V3 + single GPU training achieves a 63% performance improvement.
    * Optimization of Op Computing Logistics
        * Optimize the implementation of concat/split op with number of input/output <= 4, avoiding 1 CPU -> GPU data transmission
        * Optimize the calling method of the executor in recurrent op: now it calls ``executor.Prepare`` before each iteration, and perform ``executor.RunPreparedContext`` during the iteration, thus avoiding the repetition of creating op in each iteration. This optimization brings 23% and 15% performance improvements to the PaddingRNN padding small and large models, respectively.
        * Merge the calculation of the optimizer Momentum op, bringing 1.6% and 10.6% performance improvement to Resnet50 single GPU and 4 GPU training respectively.
    * Optimization of cuDNN’s Utilization Strategy 
        * Use the new algorithm selection API in cuDNN v7--cudnnGetConvolutionForwardAlgorithm_v7—to optimize the algorithm selection strategy of conv_cudnn op, bringing 32% and 11% acceleration to Mask-RCNN and YoloV3 single GPU training, respectively.
        * Some ops’ cuDNN implementations are slower than the CUDA counterparts, such as conv2d_transpose、pool2d (with ``global_pooling=True``). Set ``use_cudnn = False`` to improve performance of Cycle GAN, SE-ResNeXt single GPU training by 33%, 34%, respectively.
    * Optimization of Op CUDA Kernel
        * Use the optimized CUDA kernel to optimize the sum op, bringing in 3.3 times acceleration to GPU execution. The effect is particularly obvious for multiple LoDTensor summation.
        * Optimize elementwise_mul grad op with a 2D thread block configuration to speed up the Broadcast operation in its CUDA Kernel.
    * Optimization of the Bottom-level Computing of Intel CPU
        * Add new OP to merge Pass（conv+relu6，conv_transpose+elementwise_add）
        * Add new FP32 MKLDNN kernel (FC)，INT8 MKLDNN kernel (Concat)
        * Optimize several OPs, including sequence_reverse (forward), sequence_padding (forward), sequence_unpad (reverse), and bilinear interpolate (forward).
        * Optimize MKLDNN integration (such as re-using reorder primitives to reduce the time to create a new primitive each time).
* Optimization of Memory
    * Optimize the Op layer memory (saving 1G or more memories on the Transformer, Mask-RCNN and other models).
        * Improve the coverage of the inplace strategy, supporting the inplace calculation of op such as sum, softmax, softmax_with_cross_entropy, etc.
        * Fix the reverse registration of dropout, conv_transpose, and activation op, reducing op memory usage.
    * Memory Allocation and Memory Reuse Strategy Refactoring
        * Refactors the underlying architecture of the Allocator to provide the foundation for subsequent extended Allocator policies.
        * Refactors the Inplace strategy to make its code easy to maintain, and to rule out variables in previous strategies that may produce bugs such as inplace, graph existence, etc.
    * Optimization of Configuration
        * The user can use the environment variable ``FLAGS_conv_workspace_size_limit`` to set the maximum workspace size of the conv layer in MB.
* Execution optimization
    * Update the default configuration of CPU_NUM to 1, which is previously the total number of logical cores of the device.
    * Cache the OpKernel in the Operator to avoid repeatedly selecting the kernel for each run.
    * ParallelExecutor execution mode (CompiledProgram.with_data_parallel()) optimization: reduce synchronization operation; optimize the speed at num_thread=1 — the speed increase for small models is more obvious  (16% increase for PaddingRNN small model).
* Framework basic functions enhancements
    * Add mkldnn_enabled_op_types option to build_strategy, giving users the flexibility to control which ops need to use the mkldnn kernel for acceleration.
    * Add drop_local_exe_scopes interface under ParallelExecutor. The setting of num_iteration_per_drop_scope that controls when the data in the local scope is cleaned is still valid.
    * Add automatic mixed precision training interface ``fluid.contrib.mixed_precision.decorate()`` that supports image classification, BERT and other model training.
    * Add ``fluid.gradients()`` interface with 11 operations supporting secondary reversal, used by gradient penalty for image generation.
    * Enhance the support for the Intel nGraph compilation engine; add the op support required by the Bert model. The BERT model can be trained by the Intel nGraph compilation engine, and the convergence effect is comparable.
* OP perfect
    * Enhance the fused_elewise_activation op function; add support for x+sigmoid(y), x+tanh(y) calculation modes.
    * Add a new index, Exponential Moving Average, which makes model training smoother and more stable.
    * Add sigmoid_focal_loss loss function
    * Add deformable RoI pooling operation
    * Add deformable convolution v2 operation
    * Provide unfold operation (i.e. im2col) operation

Inference Engine
#####################################
* Server-side Deployment Library
    * Optimize “video memory optimization” function. DAM’s video memory occupation decreases from 4G to 940M; MobileNet’s video memory occupation decreases from 1G to 500M.
    * The Paddle-TRT optimization process is migrated to model initialization to solve the problem that the Paddle-TRT initial prediction time is too long. For example, make MobileNet first predicted time drop from second level to millisecond level.
    * Fix the issue that ``AnalysisPredictor`` allocate memory repeatedly when it loads models from memory.
    * Enhance Python interference API; include the related user manual under “Deploy Inference Model” section on  Paddle’s documentation page.
    * Intel INT8 Quantization Interference Improvements
        * Continuously optimize the INT8 quantization framework (quantization after training); add five models (GoogLeNet, MobileNetV2, VGG16, VGG19, ResNet101); compared with the FP32 model, achieve a less than 1% accuracy loss and improve performance 2 to 3.7 times.
        * Run the model that supports QAT (Quantization as Training) on the INT8 kernel; Modify the QAT model with Pass to enable it to run on the INT8 kernel (currently supports quantization/dequantization/convolution); compared to the simulation that runs on the FP32 kernel, achieve a less than 1% accuracy loss with 7 models (GoogleNet, MobileNetV1, MobileNetV2, VGG16, VGG19, ResNet50, ResNet101).
* Paddle Serving
    * Support GPU devices; support multi-card parallel inference.
    * Provide the SE_ResNeXt50_32x4d model as a standard example; give image classification task benchmark of single card multiple concurrency, multi-card multi-concurrency, etc.
    * Support large-scale sparse parameter tasks: storage and online access for very large-scale embedding in scenarios such as CTR estimation; release a stand-alone version in the first phase, supporting billion-level embedding access.
    * Provide easy to use API interface and API demo examples.
* PaddleSlim
    * Integrated INT8 quantization framework
    * New automatic shearing strategy based on simulated annealing algorithm to search for optimal shearing rate: 50% reduction in FLOPS compared to MobileNet V1 on ImageNet 1000 classification task; Top1 - Accuracy = 69.7%
    * New Light-NAS feature: 17% reduction in FLOPS compared to MobileNet V1 for ImageNet 1000 classification tasks with no loss of accuracy

Distributed training
#####################################
* Distributed High-Level API Fleet
    * Distributed Training Unified API, which supports Parameter Server and Collective mode training, greatly reducing the number of new codes for users to switch from single computer to multi-computer training
    * Users can invoke different parallel training methods by configuring distributed policies, supporting multiple built-in RoleMaker for different distributed environments to facilitate user calls
* New Communicator Design for Parameter Server Training
    * Independent communication logic to Communicator to simplify asynchronous training logic
    * Provides controllable communication switches that can be tuned to different models
* GPU multi-computer multi-card add multi-boosting extensible feature, NLP/CV classic model multi-computer multi-card training speed up 50%
    * Add Fused All Reduce: Reduce the number of parameter sync times by automatically merging gradient tensor
    * New Hierachical All Reduce: Hierarchical all reduce operation
    * New All Reduce communication concurrent capability: Increased capacity for network wave tolerance under multi-machine training
    * Added dependency analysis between reverse and optimization algorithms: Improving the ability to communicate and compute overlap concurrency
    * The above-mentioned new capability convergence enables more than 50 percent faster training on Bert Large (batch 16x128) and Resnet 50 (batch 32) computers (v1008 * 4 cards) than PaddlePaddle1.4.1.
* GPU Multi-computer Multi-card Benchmark Update
    * Speed comparisons on ResNet50, VGG16, Transformer and Bert, and reproducible benchmarks scripts.
* Pipeline parallel capability support for CPU-GPU heterogeneous equipment
    * Add pipeline parallel capability to support user-defined allotment calculation OP in heterogeneous hardware, exchange data through pipeline, thus realize collocation of heterogeneous computing equipment and free allocation of computing resources, and improve training speed.
    * In the case of large IO and small computation, such as CTR prediction, Graph Neural Network has obvious speed advantage over pure GPU training.

Model Construction
#####################################
* Image classification
    * 9 ImageNet pre-training models published, including ResNet50_vc, ResNet50_vd, ResNet101_vd, ResNet 152_vd, ResNet 200_vd, ResNeXt101_64x4d, ResNeXt101_vd_64x4d, SENet 154_vd, InceptionV4
    * ResNet50_vd is 2.62% higher than the published ResNet50, and the accuracy of ResNet101 is achieved. ResNet101_vd 1.88% better than ResNet101
* PaddleDetection
    * Publish a unified framework for detecting PaddleDetection objects, including Faster-RCNN (support FPN), Mask-RCNN (support FPN), Cascade-RCNN, RetinaNet, Yolo v3, SSD, FPN, Cascade RCNN and RetinaNet.
    * Releases a series of pre-training models in which RCNN series models support ResNet, ResNet_vd, ResNeXt, ResNeXt_vd, SEResNeXt backbone networks. Yolo v3 continues to add lighter ResNet 34, MobileNet backbone networks and release pre-training models
* PaddleGAN
    * Release the PaddleGAN Image Generation Library, which includes CGAN, DCGAN, CycleGAN, Pix2 Pix, StarGAN, AttGAN, STGAN, supporting a variety of datasets and supporting classic GAN network structures. STGAN is an arbitrary image attribute editing model developed by Baidu Visual Technology Department.
* PaddleVideo
    * Optimize the already published classification model, NeXt VLAD training speed 60%, TSM speed 39%
    * Add published model backbone networks and Nonlocal models add ResNet101 and I3d network structures
    * Added motion positioning model C-TCN, Baidu 2018 ActivityNet Championship Scheme
* PaddleNLP
    * ERNIE/BERT support dynamic mixed precision training; Supporting multi-card task training in a multi-process manner, increasing the multi-card acceleration ratio; To optimize the speedup ratio of multi-machine and multi-card training, the speedup efficiency of 6 machines to 76% on V100 GPU cluster compared to single machine FP32 training is improved.
    * Launch of PaddleNLP-Research, open source MRQA2019, Paddle Fluid baseline, DuConv (ACL2019), ARNOR (ACL2019), MMPMS (IJCAI 2019), MPM (NAACL2019) and other recent Baidu work in the NLP academic field

Tools and Components
#####################################
* PaddleHub
    * New release of PaddleHub official web site, enhanced ease of use
        * New website http://hub.paddlepaddle.org.cn, including introduction to pre-training models for PaddlePaddle ecology
        * Migrate learning Demo to AI Studio and AI Book for quick experience without installation
        * New PaddleHub back-end services to support model retrieval, download and privatization deployment
    * 29 new pre-training models covering three areas: Text, image and video; 40 pre-training models currently available
        * CV pre-training model
            * 11 new pre-training models for image classification: SE_ResNeXt, GoogleNet, ShuffleNet, etc.
            * Added target detection models Faster-RCNN and YOLOv3
            * New image generation model CycleGAN
            * New face detection model Pyramidbox
            * 4 new video classification models: TSN, TSM, StNet, Non-Local
        * NLP pre-training model
            * New semantic model ELMo
            * 3 new emotion analysis models: Senta-BOW, Senta-CNN, Senta-GRNN
            * New Chinese Emotional Recognition Model EmoTect
            * New Chinese Semantic Similarity Analysis Model Simnet
            * Upgrading the LAC lexical analysis model, adding dictionary intervention to support user-defined segmentation
    * Fine-tune API upgrades, flexibility and performance upgrades
        * Support for multi-card parallel, PyReader multi-threaded IO, ERNIE Text Classification Fine-tune 60% faster
        * Simplified use logic for finetune, evaluuate, predict, etc., for ease of use
        * Add event callback to facilitate users to quickly implement custom migration learning tasks
        * New Tag Classification Task Fine-tune
* Figure Learning Framework `PGL <https://github.com/PaddlePaddle/PGL>`_  (Paddle Graph Learning)
    * The PaddlePaddle-based Graphics Framework PGL Preview is released to provide the most advanced Graphics algorithms based on Walk Based and Message Passing. PGL takes full advantage of Paddle LoD Tensor to greatly improve the efficiency of information aggregation in Message-Passing paradigm, which takes into account flexibility and efficiency.
        * New GCN and GAT based on PGL to reach SOTA level in multiple datasets
        * New Graphsage model based on large-scale subgraph sampling model with 50 million nodes and 2 billion edges
        * Added node2vec, deep walk and other chart sign learning methods to reach SOTA level
        * New PGL documentation, APIs, Tutorial, etc.

BUG fixes notes
#####################################
* Repair issues where ignore_label does not support labels in the version of softmax_with_cross_entropy operation CPU
* Repair Logging.basicConfig setup failure after import paddle
* Repair the problem of python/paddle/fluid/layers/ops.py reporting errors under python3
* Repair of sequence unpad op instability during training
* Repair the problem of dropping when the concat op attribute axis is a negative number
* Fixed potential bugs for enable_inplace and memory_optimize to ensure that some of the op's output variables are not reused incorrectly
* Fix the bug of Eager Deletion strategy which may erroneous delete variable storage space in advance and improve the stability of Eager Deletion strategy.
* Fixes the case of different model graph generation with the same model input due to bugs in topology sorting in model graph analysis
* Fixed a problem with other service thread OMP thread conflicts after the prediction ends. The fix is that in CPU mode, the prediction engine sets the number of global OMP threads to 1 after the prediction ends.
