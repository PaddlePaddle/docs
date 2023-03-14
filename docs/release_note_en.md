# 2.4.2 Release Note

 V2.4.2 fixed known bugs, and added a tiny set of features.

## Training Framework (distributed included)

 - Fix the problem while using paddle.utils.dlpack.to_dlpack API to create dlpack objects multiple times in the for loop, and fix the bug that the reference counting error causes the memory actually pointed by dlpack to be destructed unexpectedly. [#50138](https://github.com/PaddlePaddle/Paddle/pull/50138)
 - Fixed the issue of out-of-bounds memory access when the input tensor is multi-dimensional in paddle.multiplex API. [#49368](https://github.com/PaddlePaddle/Paddle/pull/49368)
 - Fix the occasional compilation error caused by incorrect referencing of the Eigen header file. [#48157](https://github.com/PaddlePaddle/Paddle/pull/48157)
 - Fixed the bug that the output value of the backward operator may be None when the output gradient parameter order of the custom operator is not continuous.[#48656](https://github.com/PaddlePaddle/Paddle/pull/48656)
 - Add cutlass and implement the fusion kernel of gather+gemm+scatter; Optimize training and inference performance of sparse convolution; Optimize inference performance of batch_norm under 1D input data.[#50118](https://github.com/PaddlePaddle/Paddle/pull/50118)
 - Fix compilation failure in gcc54 environment caused by using constexpr. [#50421](https://github.com/PaddlePaddle/Paddle/pull/50421)
 - Move sum op kernel to PHI and fix bug that can't get correct SelectedRows' dims when run infermeta.[#49342](https://github.com/PaddlePaddle/Paddle/pull/49342)
 - Fixed the issue that the fold operator accesses memory out of bounds under large bs input.[#49491](https://github.com/PaddlePaddle/Paddle/pull/49491)
 - Fix the problem that no parameter Layer cannot call backward under dynamic to static mode.[#49812](https://github.com/PaddlePaddle/Paddle/pull/49812)
 - Fix the compile problem of CUDA11.8 on windows platform.[#50205](https://github.com/PaddlePaddle/Paddle/pull/50205)
 - Fix the unsupported error for `FusedDropoutActBiasGrad` on H100.[#47285](https://github.com/PaddlePaddle/Paddle/pull/47285)
 - Add `debug_graphviz_path` option into `build_strategy`.[#46531](https://github.com/PaddlePaddle/Paddle/pull/46531)
 - Fix the not closed `popen` object.[#47053](https://github.com/PaddlePaddle/Paddle/pull/47053)

##  Deployment Direction (Paddle Inference)

 - Improve the functionality and stability of mixed-precision inference. Reconstruct the implementation of interface convert_to_mixed_precision and add parameter precision to interface enable_use_gpu.[#49077](https://github.com/PaddlePaddle/Paddle/pull/49077)、[#49239](https://github.com/PaddlePaddle/Paddle/pull/49239)、[#49477](https://github.com/PaddlePaddle/Paddle/pull/49477)
 - Support compilation under jetson ampere architecture.[#49364](https://github.com/PaddlePaddle/Paddle/pull/49364)
 - Fixed fc kernel diff.[#49781](https://github.com/PaddlePaddle/Paddle/pull/49781)
 - Fixed the error of trt workspace parameter type under CAPI. [#48350](https://github.com/PaddlePaddle/Paddle/pull/48350)
 - Fixed the error caused by arg_max/arg_min without flatten dtype parameter in Paddle 1.x version. [#49771](https://github.com/PaddlePaddle/Paddle/pull/49771)
 - Fixed the bug of missing information about lod logic after split infermeta's refactoring. [#49745](https://github.com/PaddlePaddle/Paddle/pull/49745)
 - Fixed the bug of the constant-folding pass, which causes the conv2d weight to be non-persistent after folding and not enter the TensorRT engine. [#50105](https://github.com/PaddlePaddle/Paddle/pull/50105)

# 2.4.1 Release Note


Remove the dependence of the Paddle on python.so, and fix the bug that fails to execute due to the inability to find python.so in specific environments, including conda.


# 2.4.0 Release Note

## 1. Important Updates

- **New dynamic graph architecture is officially effective**: The new dynamic graph framework has significantly improved the scheduling performance. The scheduling performance of more than 90% APIs is improved by over 50%, and the model performance of more than 50% kits is improved by over 5%. The functional architecture is clearer, and the secondary development capability and experience are significantly enhanced.

- **Comprehensive improvement of the dynamic-static unification ability of the PaddlePaddle**: The dynamic-to-static function is provided with richer Python syntax support. The Python syntax coverage of the PaddlePaddle reaches 90%. The syntax transcription logic is mainly optimized to completely support the control flow syntax, with providing smooth dynamic-to-static graph experiences by pressing one key. With the newly upgraded static graph executor, the dynamic-to-static training has better acceleration capability, and the key model test shows that it is close to the best level of the static graph. The dynamic-to-static scalability is improved, with newly supporting multi-function merge export and inference. Users can use the PHI operator library for secondary development and flexible deployment. This can effectively support the custom decoding of U2++ featured models in the speech domain.

- **Add sparse computing APIs**: Add 55 sparse APIs `paddle.sparse.*` and support mainstream sparse computing scenarios. The APIs have been applied to sparse training and inference deployment for 3D point cloud target detection, Sparse Transformers, and other tasks, with a speedup of 105.75% compared to DenseTensor in high sparse scenarios. In contrast to similar products, the speed of sparse computing is increased by 4.01%-58.55%. Support the computing of a variety of sparse Tensors (SparseCoo and SparseCsr). This is the ultimate saving of video memory. Meanwhile, it maintains a consistent usage experience, with the same usage method of the dense Tensor API.

- **Large-scale graph neural network GPU training engine**: Through the heterogeneous hierarchical storage technology of SSD, memory, and video memory, it breaks through the video memory bottleneck and supports all-GPU storage and training of super-large-scale graphs. It realizes the all-GPU integrated solution of walk, sampling and training. This can increase the training speed by more than 10x under the same costs, compared to the traditional distributed CPU solution.

- **Environment adaptation**: Add pre-compiled installer adapted to CUDA version 11.7. It newly supports the running in Ubuntu 22.04 or later.

### Forward-looking forecast

- PaddlePaddle Framework will deprecate support for python 3.6 in version 2.5.
- The PaddlePaddle framework will gradually deprecate the API under the `paddle.fluild` namespace on the python side, and some of the APIs under this namespace will be directly removed in version 2.5.

## 2. Incompatibility upgrade

- The pre-compiled installer for CUDA version 10.1 is cancelled.
- The -Tensor.clear_gradient(bool set_to_zero) interface will not take the value passed by kwargs, and will have to pass the bool variable of set_to_zero through args.
- In order to improve the utilization efficiency of video memory, only the gradients of forward leaf node variables, such as the gradients of network parameters in training, are retained in the dynamic graph by default, instead of the gradients of non-leaf nodes. If you need to preserve a specific Tensor gradient, you can call the Tensor.retain_grads() interface before reverse execution.
- paddle.autograd. PyLayer will no longer support the case where the input is tuple, pass in a list of Tensor if you want a group of them.

## 3. Training framework (including the distributed feature)

### （1）New APIs and enhanced API functions
- **Add the sparse computing class API**：paddle.sparse
  - Add 55 sparse APIs and support mainstream sparse computing scenarios. The APIs have been applied to sparse training and inference deployment for 3D point cloud target detection, Sparse Transformers, and other tasks, with a speedup of 105.75% compared to DenseTensor in high sparse scenarios. In contrast to similar products, the speed of sparse computing is increased by 4.01%-58.55%. Support the computing of a variety of sparse Tensors (SparseCoo and SparseCsr). This is the ultimate saving of video memory. Meanwhile, it maintains a consistent usage experience, with the same usage method of the dense Tensor API.[#45849](https://github.com/PaddlePaddle/Paddle/pull/45849), [#46694](https://github.com/PaddlePaddle/Paddle/pull/46694), [#45086](https://github.com/PaddlePaddle/Paddle/pull/45086), [#41857](https://github.com/PaddlePaddle/Paddle/pull/41857), [#42935](https://github.com/PaddlePaddle/Paddle/pull/42935), [#43475](https://github.com/PaddlePaddle/Paddle/pull/43475), [#43668](https://github.com/PaddlePaddle/Paddle/pull/43668), [#43966](https://github.com/PaddlePaddle/Paddle/pull/43966), [#44022](https://github.com/PaddlePaddle/Paddle/pull/44022), [#44346](https://github.com/PaddlePaddle/Paddle/pull/44346), [#44432](https://github.com/PaddlePaddle/Paddle/pull/44432), [#44451](https://github.com/PaddlePaddle/Paddle/pull/44451), [#44743](https://github.com/PaddlePaddle/Paddle/pull/44743), [#42013](https://github.com/PaddlePaddle/Paddle/pull/42013), [#43520](https://github.com/PaddlePaddle/Paddle/pull/43520), [#41434](https://github.com/PaddlePaddle/Paddle/pull/41434), [#42130](https://github.com/PaddlePaddle/Paddle/pull/42130), [#41276](https://github.com/PaddlePaddle/Paddle/pull/41276), [#41857](https://github.com/PaddlePaddle/Paddle/pull/41857), [#41356](https://github.com/PaddlePaddle/Paddle/pull/41356)
- **Add the audio field API：** paddle.audio
  - Add the feature extraction APIs such as MFCC, Spectrogram, and LogMelSpectrogram. Support the GPU computing. The performance increases by more than 15x compared to the CPU. This can significantly improve the GPU utilization in speech model training.[#45424](https://github.com/PaddlePaddle/Paddle/pull/45424)
  - Add the feature extraction basic APIs such as Window Function and Discrete Cosine Transform. This can facilitate users to customize the speech feature extraction.[#45424](https://github.com/PaddlePaddle/Paddle/pull/45424)
  - Add the speech I/O module. It provides 2 types of audio I/O backend and supports 6 types of codecs for convenient loading of speech data. [#45939](https://github.com/PaddlePaddle/Paddle/pull/45939)
  - Add TESS and ESC50 speech classification datasets. It is convenient for users to complete the classical speech classification model.[#45939](https://github.com/PaddlePaddle/Paddle/pull/45939)
- **Add the graph learning domain API:** paddle.geometric
  - Graph learning is gradually becoming a key technology in the field of machine learning. The new paddle.geometric module of PaddlePaddle provides a better modeling and training development experience of graph learning.
    - Message passing: The message passing mechanism of the graph learning is the basis of graph modeling. We add 7 graph learning message passing APIs to make it more convenient to complete the modeling of the graph learning. Among them, 3 newly added message passing fusion operators can significantly reduce the GPU memory consumption in the GNN model training. In the dense graph scenarios, more than 50% of GPU memory can be saved in the models of GCN series, and the training speed can increase by more than 20%.[#44848](https://github.com/PaddlePaddle/Paddle/pull/44848), [#44580](https://github.com/PaddlePaddle/Paddle/pull/44580), [#43174](https://github.com/PaddlePaddle/Paddle/pull/43174), [#44970](https://github.com/PaddlePaddle/Paddle/pull/44970)
    - Graph sampling: Graph sampling is the performance bottleneck of GNN model training. This newly added high-performance graph sampling operator supports high concurrent graph sampling. It can increase the sampling speed of GraphSage by more than 32 times and the model training speed by more than 12 times.[#44970](https://github.com/PaddlePaddle/Paddle/pull/44970)
- **Add the vision domain API**
  - The paddle.vision is added with target detection domain operators.([#43736](https://github.com/PaddlePaddle/Paddle/pull/43736)), paddle.vision.generate_proposals([#43611](https://github.com/PaddlePaddle/Paddle/pull/43611)), paddle.vision.matrix_nms([#44357](https://github.com/PaddlePaddle/Paddle/pull/44357)), paddle.vision.prior_box 和 paddle.vision.box_coder( [#47282](https://github.com/PaddlePaddle/Paddle/pull/47282) ).

- - **Add other API**
  - Add the iinfo([#45321](https://github.com/PaddlePaddle/Paddle/pull/45321)), count_nonzero([#44169](https://github.com/PaddlePaddle/Paddle/pull/44169)), nanmedian([#42385](https://github.com/PaddlePaddle/Paddle/pull/42385)), remainder\_ ([#45266](https://github.com/PaddlePaddle/Paddle/pull/45266)), take([#44741](https://github.com/PaddlePaddle/Paddle/pull/44741)), triu_indices([#45168](https://github.com/PaddlePaddle/Paddle/pull/45168)), sgn([#44568](https://github.com/PaddlePaddle/Paddle/pull/44568)), bucketize([#44195](https://github.com/PaddlePaddle/Paddle/pull/44195)), nanquantile([#41343](https://github.com/PaddlePaddle/Paddle/pull/41343)), frac([#41226](https://github.com/PaddlePaddle/Paddle/pull/41226)), logcumsumexp([#42267](https://github.com/PaddlePaddle/Paddle/pull/42267)), pairwise_distance([#44161](https://github.com/PaddlePaddle/Paddle/pull/44161)), heaviside([#41872](https://github.com/PaddlePaddle/Paddle/pull/41872)), logspace([#41261](https://github.com/PaddlePaddle/Paddle/pull/41261)), corrcoef([#40690](https://github.com/PaddlePaddle/Paddle/pull/40690))
  - Add the RReLU([#41823](https://github.com/PaddlePaddle/Paddle/pull/41823)), CyclicLR([#40698](https://github.com/PaddlePaddle/Paddle/pull/40698)), OneCycleLR([#41825](https://github.com/PaddlePaddle/Paddle/pull/41825)), Softmax2D([#40910](https://github.com/PaddlePaddle/Paddle/pull/40910)), SoftMarginLoss([#42364](https://github.com/PaddlePaddle/Paddle/pull/42364)), MultiLabelSoftMarginLoss([#41183](https://github.com/PaddlePaddle/Paddle/pull/41183)), TripletMarginLoss([#40487](https://github.com/PaddlePaddle/Paddle/pull/40487)), TripletMarginWithDistanceLoss([#40545](https://github.com/PaddlePaddle/Paddle/pull/40545)), CosineEmbeddingLoss 和 cosine_embedding_loss([#41680](https://github.com/PaddlePaddle/Paddle/pull/41680)), PixelUnshuffle([#40728](https://github.com/PaddlePaddle/Paddle/pull/40728)), ChannelShuffle([#40743](https://github.com/PaddlePaddle/Paddle/pull/40743))
- **Enhanced API functions**
  - Add the large batch_size calculation function of BatchNorm1D [#43072](https://github.com/PaddlePaddle/Paddle/pull/43072)
- **Optimize the collective communications distributed training API**
  - Optimize the `fleet.init` function, and add the `log_level` parameter to facilitate users to view logs during operation  [#45909](https://github.com/PaddlePaddle/Paddle/pull/45909)
  - Add the `paddle.distributed.fleet.recompute_sequential paddle.distributed.fleet.recompute_hybrid` interface. It is convenient for users to use the recompute function [#45348](https://github.com/PaddlePaddle/Paddle/pull/45348)
  - Add the `paddle.distributed.fleet.layers.mpu` package. It is convenient for users to use tensor parallel function [#45803](https://github.com/PaddlePaddle/Paddle/pull/45803)
  - Add the communication API `paddle.distributed.destroy_process_group paddle.distributed.isend paddle.distributed.irecv paddle.distributed.all_to_all_single`. It improves the completeness and ease of use of communication [#43918](https://github.com/PaddlePaddle/Paddle/pull/43918)
  - Add the `paddle.distributed.stream` package. The performance is increased by 5% to 10% compared to the base version[#46023](https://github.com/PaddlePaddle/Paddle/pull/46023) [#45282](https://github.com/PaddlePaddle/Paddle/pull/45282)
  - The communication API is added with the support of multiple data types such as `Char/Byte/Bool`. It improves the completeness and ease of use of communication [#45574](https://github.com/PaddlePaddle/Paddle/pull/45574) [#45440](https://github.com/PaddlePaddle/Paddle/pull/45440)
  - The communication API asynchronous parameter is changed from`use_calc_stream` to `sync_op`, It enhances the semantic readability of the interface [#46493](https://github.com/PaddlePaddle/Paddle/pull/46493)
- **Enhanced high-level API**
  - The visual model ResNeXt in the high-level API implements the reuse of the ResNet code for refactoring. [#40588](https://github.com/PaddlePaddle/Paddle/pull/40588)
  - The visual models Inceptionv3, MobileNetv1, MobileNetv2, and ShuffleNetv2 in the high level API are improved.[#40431](https://github.com/PaddlePaddle/Paddle/pull/40431)

### （2）New functions and important upgrades

- **The new dynamic graph architecture is officially launched**：The scheduling performance of the new dynamic graph framework is greatly improved. Compared with the original architecture, the scheduling performance is significantly enhanced. The scheduling performance of more than 90% APIs is improved by over 50%, and the model performance of more than 50% of kits is improved by over 5%. The new dynamic graph architecture is clear, and the coupling is low. The learning and development costs of extension modules such as Hook and PyLayer are significantly reduced based on the new architecture. [#37550](https://github.com/PaddlePaddle/Paddle/pull/37550) , [#37574](https://github.com/PaddlePaddle/Paddle/pull/37574) , [#37813](https://github.com/PaddlePaddle/Paddle/pull/37813)  ,  [#37926](https://github.com/PaddlePaddle/Paddle/pull/37926) , [#39192](https://github.com/PaddlePaddle/Paddle/pull/39192) , [#37599](https://github.com/PaddlePaddle/Paddle/pull/37599) , [#37406](https://github.com/PaddlePaddle/Paddle/pull/37406) , [#37466](https://github.com/PaddlePaddle/Paddle/pull/37466) , [#37599](https://github.com/PaddlePaddle/Paddle/pull/37599) , [#40945](https://github.com/PaddlePaddle/Paddle/pull/40945) , [#39989](https://github.com/PaddlePaddle/Paddle/pull/39989)

- **High-order auto-differentiation mechanism**：In order to better support scientific computing and other scenarios, the PaddlePaddle framework has been further improved and optimized for higher-order auto-differentiation capabilities. At present, the `paddle.incubate.autograd`  directory has provided relevant trial functions and APIs for forward/reverse higher-order auto-differentiation (Currently they are in incubation, and related functions and API signatures may change).If you intend to implement related models and explore the auto-differentiation mechanism by yourself, please read the [usage and limitations of higher-order auto-differentiation](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/incubate/autograd/Overview_cn.html) carefully. Specific upgrades include：
  1. Static graph higher-order differentiation mechanism upgrade. Through the base operator system and program transformation, it supports higher-order forward and reverse differentiation, with the availability of the compiler and distributed functions.[#41919](https://github.com/PaddlePaddle/Paddle/pull/41919), [#41201](https://github.com/PaddlePaddle/Paddle/pull/41201)
  2. Add the forward and reverse higher-order auto-differentiation API, `paddle.incubate.autograd.forward_grad`, `paddle.incubate.autograd.grad`. [#43354](https://github.com/PaddlePaddle/Paddle/pull/43354)
  3. Add 18 higher-order auto-differentiation operators:`sin`, `cos`, `exp`, `erf`, `abs`, `log`, `cast`, `where`, `equal`, `not_equal`, `greater_than`, `greater_equal`, `elementwise_pow` `square`, `elementwise_max`, `gelu`, `reduce_mean`, `size`. [#46184](https://github.com/PaddlePaddle/Paddle/pull/46184), [#46024](https://github.com/PaddlePaddle/Paddle/pull/46024), [#45888](https://github.com/PaddlePaddle/Paddle/pull/45888), [#45338](https://github.com/PaddlePaddle/Paddle/pull/45338), [#44345](https://github.com/PaddlePaddle/Paddle/pull/44345)
  4. Fix the existing bugs of the operators such as`elementwise_div`, `reduce_sum`, `p_norm`. [#46514](https://github.com/PaddlePaddle/Paddle/pull/46514), [#46184](https://github.com/PaddlePaddle/Paddle/pull/46184)
- **Generic heterogeneous parameter server architecture**：
  - Parameter server GPUGraph infrastructure upgraded to meet the implementation needs of large-scale applications: The storage and training of large-scale graph neural networks based on the traditional CPU feature high cost, low stability, and less performance. To overcome these problems, we have built a pure GPU graph training engine (PGLBox). Through the heterogeneous hierarchical storage technology of SSD, memory and video memory, it supports the training of ultra-large scale graph models. The training performance is improved by more than 10x compared with CPU graph training engine on the premise of equal cost. The task failure rate is extremely low.[#44594](https://github.com/PaddlePaddle/Paddle/pull/44594)
  - Large-scale federation parameter server architecture: For large-scale personalized recommendation scenarios, the large-scale federation parameter server training is developed based on the heterogeneous PS infrastructure, to support horizontal and vertical federation under hundreds of billions of parameters. It includes two features: User private parameters updated locally and public parameters updated remotely. Users can flexibly configure the slicing policy for private and public parameters. A new central scheduling node Coordinator is added. Users can perform secondary development from the base class to customize the Client selection policy. [#42682](https://github.com/PaddlePaddle/Paddle/pull/42682) , [#44864](https://github.com/PaddlePaddle/Paddle/pull/44864) , [#44327](https://github.com/PaddlePaddle/Paddle/pull/44327)
- **Adaptive parallel**
  - Design and launch a complete automatic parallelism interface system: Support automatic dynamic-to-static distributed training, automatic distributed data loading, automatic distributed saving and loading, automatic parameter conversion, custom slice marker and custom execution process. Users can easily obtain the automatic distributed training capability based on a single machine networking. It supports data parallel, model parallel, pipeline parallel, and hybrid parallel. [#45776](https://github.com/PaddlePaddle/Paddle/pull/45776) ，[#46552](https://github.com/PaddlePaddle/Paddle/pull/46552) , [#44202](https://github.com/PaddlePaddle/Paddle/pull/44202) , [#45840](https://github.com/PaddlePaddle/Paddle/pull/45840) , [#45518](https://github.com/PaddlePaddle/Paddle/pull/45518) , [#40528](https://github.com/PaddlePaddle/Paddle/pull/40528), [#42838](https://github.com/PaddlePaddle/Paddle/pull/42838), [#43093](https://github.com/PaddlePaddle/Paddle/pull/43093), [#43312](https://github.com/PaddlePaddle/Paddle/pull/43312), [#45053](https://github.com/PaddlePaddle/Paddle/pull/45053).
  - Improve the underlying adaptive parallel mechanism, including the upgrade of the distributed costmodel design and implementation, to provide better evaluation of the slice policy. Add the native distributed properties to ProgramIR and enrich the Cluster functions. [#40457](https://github.com/PaddlePaddle/Paddle/pull/40457) , [#42601](https://github.com/PaddlePaddle/Paddle/pull/42601) , [#42727](https://github.com/PaddlePaddle/Paddle/pull/42727) , [#42874](https://github.com/PaddlePaddle/Paddle/pull/42784) , [#43114](https://github.com/PaddlePaddle/Paddle/pull/43114) , [#44095](https://github.com/PaddlePaddle/Paddle/pull/44095) , [#44146](https://github.com/PaddlePaddle/Paddle/pull/44146) , [#44701](https://github.com/PaddlePaddle/Paddle/pull/44701) , [#44973](https://github.com/PaddlePaddle/Paddle/pull/44973) , [#45002](https://github.com/PaddlePaddle/Paddle/pull/45002) , [#45118](https://github.com/PaddlePaddle/Paddle/pull/45118) , [#45237](https://github.com/PaddlePaddle/Paddle/pull/45237) , [#42576](https://github.com/PaddlePaddle/Paddle/pull/42576) , [#41722](https://github.com/PaddlePaddle/Paddle/pull/41722) , [#44150](https://github.com/PaddlePaddle/Paddle/pull/44150) ,  [#44989](https://github.com/PaddlePaddle/Paddle/pull/44989), [#44951](https://github.com/PaddlePaddle/Paddle/pull/44951),  [#44963](https://github.com/PaddlePaddle/Paddle/pull/44963) .
  - Add the Shardingstage1/2/3 AutoTuning feature under data parallel. This allows to automatically select the highest throughput Shardingstage policy while ensuring that the video memory constraints are met.  [#43782](https://github.com/PaddlePaddle/Paddle/pull/43782) .

- **Training hardware access - Plug-in solutions**：Add custom Runtime/Kernel/CCL/Graph/Pass solutions. The hardware vendors can choose which modules to implement on-demand based on hardware characteristics.

- **ONNX format export**
  - Support the quantized model export. The exported ONNX model uses TensorRT or ONNXRuntime to load inference. About 1.5~4 times inference acceleration can be obtained [#856](https://github.com/PaddlePaddle/Paddle2ONNX/pull/856), [#782](https://github.com/PaddlePaddle/Paddle2ONNX/pull/782)
  - Add the export of a large model greater than 2GB [#942](https://github.com/PaddlePaddle/Paddle2ONNX/pull/942)

### （3）Function optimization
- **Comprehensive increase of dynamic-to-static analysis conversion & extension capabilities**
  - In order to improve the success rate and experience of model dynamic-to-static conversion, the transcription logic of control flow syntax is reconstructed. The core syntax has been upgraded to JIT (just-in-time) paradigm to achieve equivalent transcription with Python codes. The syntax functions such as break, return and continue are improved.[#43666](https://github.com/PaddlePaddle/Paddle/pull/43666) , [#43846](https://github.com/PaddlePaddle/Paddle/pull/43846) , [#43848](https://github.com/PaddlePaddle/Paddle/pull/43848) , [#43880](https://github.com/PaddlePaddle/Paddle/pull/43880) , [#43957](https://github.com/PaddlePaddle/Paddle/pull/43957) , [#43328](https://github.com/PaddlePaddle/Paddle/pull/43328) , [#43348](https://github.com/PaddlePaddle/Paddle/pull/43348) , [#43998](https://github.com/PaddlePaddle/Paddle/pull/43998) , [#44465](https://github.com/PaddlePaddle/Paddle/pull/44465) , [#44504](https://github.com/PaddlePaddle/Paddle/pull/44504) , [#43713](https://github.com/PaddlePaddle/Paddle/pull/43713) , [#43864](https://github.com/PaddlePaddle/Paddle/pull/43864) , [#43967](https://github.com/PaddlePaddle/Paddle/pull/43967) , [#44155](https://github.com/PaddlePaddle/Paddle/pull/44155) , [#44487](https://github.com/PaddlePaddle/Paddle/pull/44487) , [#44527](https://github.com/PaddlePaddle/Paddle/pull/44527) , [#45105](https://github.com/PaddlePaddle/Paddle/pull/45105) , [#45900](https://github.com/PaddlePaddle/Paddle/pull/45900)
  - In order to support the voice custom decoding flexible deployment scenarios, the jit.save/load interface function is extended to support user multi-function merge and export. A new JITLayer component is added to support the invocation of class functions. Meanwhile, the custom inference deployment function is implemented with the PHI operator library C++ API. [#44283](https://github.com/PaddlePaddle/Paddle/pull/44283), [#41783](https://github.com/PaddlePaddle/Paddle/pull/41783), [#43607](https://github.com/PaddlePaddle/Paddle/pull/43607),  [#43754](https://github.com/PaddlePaddle/Paddle/pull/43754), [#43758](https://github.com/PaddlePaddle/Paddle/pull/43758),  [#43798](https://github.com/PaddlePaddle/Paddle/pull/43798),  [#44010](https://github.com/PaddlePaddle/Paddle/pull/44010), [#44351](https://github.com/PaddlePaddle/Paddle/pull/44351), [#44465](https://github.com/PaddlePaddle/Paddle/pull/44465), [#44504](https://github.com/PaddlePaddle/Paddle/pull/44504),  [#44597](https://github.com/PaddlePaddle/Paddle/pull/44597),  [#44738](https://github.com/PaddlePaddle/Paddle/pull/44738), [#44984](https://github.com/PaddlePaddle/Paddle/pull/44984), [#46249](https://github.com/PaddlePaddle/Paddle/pull/46249)
  -  In order to unify API dynamic and static behaviors, 20 operators are upgraded to support variable attribute information of Op in static graphs, to ensure consistent dynamic and static behaviors and improve the success rate of dynamic-to-static conversion of models. Include `pad2d`,`depthwise_conv2d_transpose`,`conv2d_transpose`,`adaptive_avg_pool2d`,`reverse`,`bincount`,`multinomial`,`reduce_sum`,`reduce_mean`,`reduce_prod`,`reduce_min`,`reduce_max`,`uniform`,`squeeze`,`max_unpool2d`,`dropout`,`cumsum`,`eye`,`argmin`,`argmax`. [#44737](https://github.com/PaddlePaddle/Paddle/pull/44737), [#45084](https://github.com/PaddlePaddle/Paddle/pull/45084), [#45189](https://github.com/PaddlePaddle/Paddle/pull/45189), [#45391](https://github.com/PaddlePaddle/Paddle/pull/45391), [#45417](https://github.com/PaddlePaddle/Paddle/pull/45417), [#45427](https://github.com/PaddlePaddle/Paddle/pull/45427), [#45514](https://github.com/PaddlePaddle/Paddle/pull/45514), [#45525](https://github.com/PaddlePaddle/Paddle/pull/45525), [#45543](https://github.com/PaddlePaddle/Paddle/pull/45543), [#45660](https://github.com/PaddlePaddle/Paddle/pull/45660), [#46352](https://github.com/PaddlePaddle/Paddle/pull/46352/), [#46433](https://github.com/PaddlePaddle/Paddle/pull/46433), [#45078](https://github.com/PaddlePaddle/Paddle/pull/45078), [#45342](https://github.com/PaddlePaddle/Paddle/pull/45342), [#45372](https://github.com/PaddlePaddle/Paddle/pull/45372), [#45453](https://github.com/PaddlePaddle/Paddle/pull/45453), [#45522](https://github.com/PaddlePaddle/Paddle/pull/45522), [#45620](https://github.com/PaddlePaddle/Paddle/pull/45620)
  - In order to solve the problem of occasional loss of error reporting stack for user dynamic-to-static, the logic of the error reporting module is optimized to improve the readability of the error reporting stack and the user debugging experience. [#44054](https://github.com/PaddlePaddle/Paddle/pull/44054), [#44083](https://github.com/PaddlePaddle/Paddle/pull/44083), [#44781](https://github.com/PaddlePaddle/Paddle/pull/44781), [#44996](https://github.com/PaddlePaddle/Paddle/pull/44996)
  - Add the TypeHint syntax recognition and transcription module to fully support Python Type Hint syntax. [#47121](https://github.com/PaddlePaddle/Paddle/pull/47121)

- **PHI operator library covers the full amount of arithmetic class operators**：Continuously build the highly reusable operator library PHI. The remaining PaddlePaddle 2.x arithmetic class PythonAPI-associated operators and related kernels are migrated to the PHI operators library and rewritten as functional expression. Add about 180 forward/reverse operator CPU&GPU kernels, and 170 Kunlun-specific arithmetic kernels. This further enhances the kernel function sets that can be reused when new operators are added. In addition, add more than 100 C++ arithmetic class APIs. These APIs can be used in the custom operators, further enhancing the ease of use for external extension development based on the PaddlePaddle. [#44577](https://github.com/PaddlePaddle/Paddle/pull/44577), [#44631](https://github.com/PaddlePaddle/Paddle/pull/44631), [#44434](https://github.com/PaddlePaddle/Paddle/pull/44434), [#44605](https://github.com/PaddlePaddle/Paddle/pull/44605), [#44676](https://github.com/PaddlePaddle/Paddle/pull/44676), [#44742](https://github.com/PaddlePaddle/Paddle/pull/44742), [#44436](https://github.com/PaddlePaddle/Paddle/pull/44436) , [#45887](https://github.com/PaddlePaddle/Paddle/pull/45887), [#45851](https://github.com/PaddlePaddle/Paddle/pull/45851), [#45623](https://github.com/PaddlePaddle/Paddle/pull/45623), [#45397](https://github.com/PaddlePaddle/Paddle/pull/45397), [#45863](https://github.com/PaddlePaddle/Paddle/pull/45863)

- **Normalized operator definitions with significantly improving the model simplicity**：For the problems of many redundant parameters in the historical operator definitions of PaddlePaddle 1.x and the high cost of understanding the adaptation, the redundant parameters of about 150 high-frequency operators are cleaned up centrally. Basically, the mathematically irrelevant parameters are removed. After these redundant parameters are cleaned up, the amount of information in the inference model stored in the PaddlePaddle is significantly reduced. Generally, about 40% of the attribute variables are removed, significantly improving the clarity of the PaddlePaddle operator definition, and improving the experience of model analysis and debugging. Meanwhile, the size of the inference model stored in the PaddlePaddle is also significantly reduced by more than 70%. As a result, this can significantly improve the lightweight of the PaddlePaddle model. [#44310](https://github.com/PaddlePaddle/Paddle/pull/44310) , [#45613](https://github.com/PaddlePaddle/Paddle/pull/45613) , [#45684](https://github.com/PaddlePaddle/Paddle/pull/45684) , [#45708](https://github.com/PaddlePaddle/Paddle/pull/45708) , [#45758](https://github.com/PaddlePaddle/Paddle/pull/45758) , [#45786](https://github.com/PaddlePaddle/Paddle/pull/45786) , [#45772](https://github.com/PaddlePaddle/Paddle/pull/45772) , [#45845](https://github.com/PaddlePaddle/Paddle/pull/45845) , [#45984](https://github.com/PaddlePaddle/Paddle/pull/45984) , [#46218](https://github.com/PaddlePaddle/Paddle/pull/46218) , [#46553](https://github.com/PaddlePaddle/Paddle/pull/46553)

### （4）Performance optimization

- AMP performance and accuracy optimization
  - More operators are added with the support of FP16 data types, including elementwise series operators, compare series operators, strided_slice, set_value, uniform_ramdom, etc.（[#45504](https://github.com/PaddlePaddle/Paddle/pull/45504) [#44405](https://github.com/PaddlePaddle/Paddle/pull/44405) [#45496](https://github.com/PaddlePaddle/Paddle/pull/45496) [#46641](https://github.com/PaddlePaddle/Paddle/pull/46641),  [#46906](https://github.com/PaddlePaddle/Paddle/pull/46906) ）
  - Optimize the implementation scheme of the hard_swish operator FP16 Kernel to guarantee the accuracy without loss. （ [35386](https://github.com/PaddlePaddle/Paddle/pull/35386) ）
  - More operators are added with the support of BF16 data types, including fused_linear, empty, selu, pow, adam, clip, embedding, gelu, pad3d, pixel_shuffle, tile, where, etc. [#46364](https://github.com/PaddlePaddle/Paddle/pull/46364), [#47177](https://github.com/PaddlePaddle/Paddle/pull/47177)
- AutoTuning of single machine training performance
  - Transpose OP supports automatic Kernel selection mechanism. This allows the automatic search for the best Kernel implementation for different model configurations, improving the model performance. [#43310](https://github.com/PaddlePaddle/Paddle/pull/43310) (Transpose Op access AutoTuning function)
  - AMP Layout auto-switching supports the new dynamic graph mode. For the ResNet50, TSM, and DeepLabV3 models, the performance increases by 9%-21% by Layout AutoTuning in the new dynamic graph. ([#45409](https://github.com/PaddlePaddle/Paddle/pull/45409), [#45751](https://github.com/PaddlePaddle/Paddle/pull/45751), [#45826](https://github.com/PaddlePaddle/Paddle/pull/45826), [#46880](https://github.com/PaddlePaddle/Paddle/pull/46880))
- Generic performance optimization of GPU single machine training
  - Optimize the Cache scheme of the Conv operator cuDNN algorithm and Cache the results in all algorithm acquisition methods. This can significantly reduce the CPU overhead of the operator.（[#41891](https://github.com/PaddlePaddle/Paddle/pull/41891) [#47197](https://github.com/PaddlePaddle/Paddle/pull/47197) ）
  - Further optimize the GPU Kernel and Python side performance of multiple operators, including dist, poisson, depthwise_conv2d, transpose, eigh, broadcast computation, reduce computation, layer_norm, cross_entropy, etc. This can achieve better performance in more configuration scenarios. （[#44946](https://github.com/PaddlePaddle/Paddle/pull/44946), [#45057](https://github.com/PaddlePaddle/Paddle/pull/45057), [#45160](https://github.com/PaddlePaddle/Paddle/pull/45160), [#42491](https://github.com/PaddlePaddle/Paddle/pull/42491), [#42704](https://github.com/PaddlePaddle/Paddle/pull/42704), [#42853](https://github.com/PaddlePaddle/Paddle/pull/42853), [#46287](https://github.com/PaddlePaddle/Paddle/pull/46287), [#46362](https://github.com/PaddlePaddle/Paddle/pull/46362), [#46490](https://github.com/PaddlePaddle/Paddle/pull/46490), [#46412](https://github.com/PaddlePaddle/Paddle/pull/46412), [#46623](https://github.com/PaddlePaddle/Paddle/pull/46623),  [#40051](https://github.com/PaddlePaddle/Paddle/pull/40051) ）
- Performance optimization of distributed training for collective communications
  - To improve pipeline parallel scheduling efficiency, support the dynamic graph Interleaving1F1B scheduling policy. In the GPT-3 model, the performance is improved by 3%-4%.  [#45797](https://github.com/PaddlePaddle/Paddle/pull/45797) , [#45869](https://github.com/PaddlePaddle/Paddle/pull/45869) , [#45922](https://github.com/PaddlePaddle/Paddle/pull/45922) , [#46209](https://github.com/PaddlePaddle/Paddle/pull/46209) , [#45402](https://github.com/PaddlePaddle/Paddle/pull/45402) , [#45444](https://github.com/PaddlePaddle/Paddle/pull/45444) , [#45497](https://github.com/PaddlePaddle/Paddle/pull/45497) , [#45797](https://github.com/PaddlePaddle/Paddle/pull/45797) , [#45869](https://github.com/PaddlePaddle/Paddle/pull/45869) , [#45922](https://github.com/PaddlePaddle/Paddle/pull/45922), [#46209](https://github.com/PaddlePaddle/Paddle/pull/46209), [#46399](https://github.com/PaddlePaddle/Paddle/pull/46399) , [#46483](https://github.com/PaddlePaddle/Paddle/pull/46483) , [#46876](https://github.com/PaddlePaddle/Paddle/pull/46876) , [#47242](https://github.com/PaddlePaddle/Paddle/pull/47242) , [#47249](https://github.com/PaddlePaddle/Paddle/pull/47249) , [#47497](https://github.com/PaddlePaddle/Paddle/pull/47497) , [#47517](https://github.com/PaddlePaddle/Paddle/pull/47517)
  - To improve the distributed training performance of the MLPerfBERT model, the DistributedFusedLamb distributed optimizer supports hierarchical AllReduce. It improves MLPerfBERT performance by 17% on the DCU1024 card.  [#44821](https://github.com/PaddlePaddle/Paddle/pull/44821) , [#44843](https://github.com/PaddlePaddle/Paddle/pull/44843)
  - To optimize the video memory footprint when using DataParallel, the Buffer Lazy initialization policy for Tensor Fusion is supported, thus reducing the video memory footprint by an amount equal to the number of model parameters. [#45631](https://github.com/PaddlePaddle/Paddle/pull/45631).
  - Distributed parallel policies DataParallel and Sharding support BF16 training. [#46846](https://github.com/PaddlePaddle/Paddle/pull/46846) , [#47246](https://github.com/PaddlePaddle/Paddle/pull/47246)
  - To support the Sequence Parallel policy, the Distributed Pipeline Parallel supports enable_partial_send_recv policy, and supports the tensor after slice of the transmission sequence parallel.  [#46992](https://github.com/PaddlePaddle/Paddle/pull/46992) , [#47083](https://github.com/PaddlePaddle/Paddle/pull/47083)
  - To improve the performance of sharding stage 2 policy, implement the overlap of sharding stage 2 optimizer broadcast parameters with next step forward and use multi-CUDA Stream for communication. In the GPT 6.7B model, the 16-card training performance is improved by 11%.  [#46495](https://github.com/PaddlePaddle/Paddle/pull/46495) , [#46656](https://github.com/PaddlePaddle/Paddle/pull/46656) , [#47061](https://github.com/PaddlePaddle/Paddle/pull/47061)

### （5）Bug fix

- Dynamic-to-static
  - Fix the bug of reporting an error in dynamic-to-static of the model in a Parameter no-gradient scenario during multi-card training. [#44485](https://github.com/PaddlePaddle/Paddle/pull/44485)
  - Fix the bug of where redundant frame logs are mistakenly output by the terminal in the dynamic-to-static. [#45754](https://github.com/PaddlePaddle/Paddle/pull/45754), [#46800](https://github.com/PaddlePaddle/Paddle/pull/46800)
  - Fix the bug of reporting an error in the dynamic-to-static training when the control flow in the model contains a Tensor that does not require a gradient. [#43034](https://github.com/PaddlePaddle/Paddle/pull/43034)
  - Fix the bug of incorrect computation value during gradient aggregation in the dynamic-to-static training. [#44893](https://github.com/PaddlePaddle/Paddle/pull/44893)
  - Fix the bug of reporting an error in the dynamic-to-static when the function is decorated with @staticmethod. [#44983](https://github.com/PaddlePaddle/Paddle/pull/44983), [#45268](https://github.com/PaddlePaddle/Paddle/pull/45268), [#45277](https://github.com/PaddlePaddle/Paddle/pull/45277)
  - Fix the bug of too much video memory footprint in some scenarios where the model contains the dynamic-to-static training. [#45380](https://github.com/PaddlePaddle/Paddle/pull/45380)
  - Fix the bug of reporting an error of dynamic-to-static shape derivation in the networking phase when the model contains a complex control flow. [#45916](https://github.com/PaddlePaddle/Paddle/pull/45916), [#46020](https://github.com/PaddlePaddle/Paddle/pull/46020)
- Fix the error report mechanism
  - Replace self.assertTrue(np.allclose(...)) with np.testing.assert_allclose to get fuller error reporting information  ( [#44947](https://github.com/PaddlePaddle/Paddle/pull/44947),  [#44988](https://github.com/PaddlePaddle/Paddle/pull/44988), [#45213](https://github.com/PaddlePaddle/Paddle/pull/45213))
- Distributed training in collective communications
  - Fix several bugs in communication library initialization and communication process, and enhance the system operation stability.  [#44964](https://github.com/PaddlePaddle/Paddle/pull/44964) [#45100](https://github.com/PaddlePaddle/Paddle/pull/45100) [#44758](https://github.com/PaddlePaddle/Paddle/pull/44758)
  - Fix the bug of frequent occurrences of hang in pipeline parallel, and enhance the ease of use of the policy [#47201](https://github.com/PaddlePaddle/Paddle/pull/47201); enhance the pipeline function to support unbalanced input.  [#47199](https://github.com/PaddlePaddle/Paddle/pull/47199)
  - Fix the bug that the performance of the new dynamic graph MP/PP policy is lower than the old dynamic graph.  [#47071](https://github.com/PaddlePaddle/Paddle/pull/47071)
  - Fix the bug that the shardingstage2 policy incorrectly maintains the parameter trainable property. [#47240](https://github.com/PaddlePaddle/Paddle/pull/47240)
  - Fix the bug that tensornumel is greater than INT32_MAX in series of OPs. [#45711](https://github.com/PaddlePaddle/Paddle/pull/45711), [#45741](https://github.com/PaddlePaddle/Paddle/pull/45741), [#45897](https://github.com/PaddlePaddle/Paddle/pull/45897), [#46158](https://github.com/PaddlePaddle/Paddle/pull/46158), [#46767](https://github.com/PaddlePaddle/Paddle/pull/46767), [#47191](https://github.com/PaddlePaddle/Paddle/pull/47191), [#46045](https://github.com/PaddlePaddle/Paddle/pull/46045), [#46160](https://github.com/PaddlePaddle/Paddle/pull/46160)
  - Fix the bug of too much video memory footprint in FusedAttention and Fused FeedForward OP.[#47236](https://github.com/PaddlePaddle/Paddle/pull/47236), [#47235](https://github.com/PaddlePaddle/Paddle/pull/47235)
  - Fix the bug of incorrect parameter update in multi_tensor_adam and multi_tensor_momentumOP when the parameters passed in are listofdict. [#47352](https://github.com/PaddlePaddle/Paddle/pull/47352), [#47372](https://github.com/PaddlePaddle/Paddle/pull/47372)

## 4. Deployment direction (Paddle Inference)

### （1）New features

- Optimize the back-end graph engine integration scheme
  - In order to reduce Paddle-TensorRT plugin code development and reduce the number of Paddle-TensorRT subgraphs and thus reducing resource usage, a generic plugin mechanism has been developed, to automatically provide a unified TensorRT plugin interface for rich Phi operators in the framework. As a result, the video memory footprint can be effectively reduced in most scenarios.  [#46970](https://github.com/PaddlePaddle/Paddle/pull/46070), [#46179](https://github.com/PaddlePaddle/Paddle/pull/46179), [#46580](https://github.com/PaddlePaddle/Paddle/pull/46580)
  - In order to facilitate users to customize operators in the framework and make Paddle-TensorRT perform efficient inference, the function is upgraded to support the framework custom Paddle-TensorRT plugin. [#46970](https://github.com/PaddlePaddle/Paddle/pull/46070)
- Optimize the Inference library build system. The size can be pruned on demand
  - Pre-compiled installer supports TensorRT by default: The pre-compiled installer for training and the pre-compiled installer for deployment (Paddle Inference) are unified into one pre-compiled installer. The build system is optimized so that the pre-compiled installer supports TensorRT by default, reducing the switching cost for users using PaddleTensorRT. [#46008](https://github.com/PaddlePaddle/Paddle/pull/46008), [#45824](https://github.com/PaddlePaddle/Paddle/pull/45824), [#46058](https://github.com/PaddlePaddle/Paddle/pull/46058)
  - The size can be pruned on demand: Pruned according to the model operator. [#47033](https://github.com/PaddlePaddle/Paddle/pull/47033) , [#47049](https://github.com/PaddlePaddle/Paddle/pull/47049) , [#47047](https://github.com/PaddlePaddle/Paddle/pull/47047)
- Inference supports native AMP
  - In order to make full use of GPUTensorCore computation capability and improve the model inference performance, a model accuracy conversion tool has been developed. The InferenceGPU natively supports the inference of the mixed precision model. For the usages, refer to the documentation. [documentation](https://github.com/PaddlePaddle/Paddle-Inference-Demo/blob/release/v2.4/docs-official/guides/nv_gpu_infer/gpu_mixed_precision.md), [#43814](https://github.com/PaddlePaddle/Paddle/pull/43814), [#43881](https://github.com/PaddlePaddle/Paddle/pull/43881), [#44057](https://github.com/PaddlePaddle/Paddle/pull/44057), [#44307](https://github.com/PaddlePaddle/Paddle/pull/44307), [#44457](https://github.com/PaddlePaddle/Paddle/pull/44457), [#44866](https://github.com/PaddlePaddle/Paddle/pull/44866), [#45050](https://github.com/PaddlePaddle/Paddle/pull/45050), [#45346](https://github.com/PaddlePaddle/Paddle/pull/45346), [#45379](https://github.com/PaddlePaddle/Paddle/pull/45379), [#45406](https://github.com/PaddlePaddle/Paddle/pull/45406), [#45882](https://github.com/PaddlePaddle/Paddle/pull/45882)
  - In order to improve the inference performance of the mixed precision model, the FP16kernel of high-frequency operators that do not support FP16 computation is supplemented, thus reducing the possibility of inserting the cast operator due to input precision mismatch. The inference performance is improved. [#44642](https://github.com/PaddlePaddle/Paddle/pull/44642), [#45061](https://github.com/PaddlePaddle/Paddle/pull/45061), [#44653](https://github.com/PaddlePaddle/Paddle/pull/44653), [#45504](https://github.com/PaddlePaddle/Paddle/pull/45504), [#45061](https://github.com/PaddlePaddle/Paddle/pull/45061), [#44969](https://github.com/PaddlePaddle/Paddle/pull/44969), [#44558](https://github.com/PaddlePaddle/Paddle/pull/44558), [#44710](https://github.com/PaddlePaddle/Paddle/pull/44710), [#43871](https://github.com/PaddlePaddle/Paddle/pull/43871), [#44792](https://github.com/PaddlePaddle/Paddle/pull/44792)
- Upgrade the compression and inference engine
  - Upgrade the quantization model storage format. The new format supports PaddleInference, PaddleLite and Paddle2ONNX 3 deployment methods. The supported chips include X86 CPU, NVIDIA GPU, and Arm CPU. （[#46305](https://github.com/PaddlePaddle/Paddle/pull/46305), [#462832](https://github.com/PaddlePaddle/Paddle/pull/46283), [#46022](https://github.com/PaddlePaddle/Paddle/pull/46022) ）
  - Add the INT8 full quantization function compatible with SoC/NPU chips. This can ensure the output INT8 quantization model has the best inference acceleration and precision on SoC/NPU chips.
- Add the INT8 full quantization function compatible with SoC/NPU chips. This can ensure the output INT8 quantization model has the best inference acceleration and precision on SoC/NPU chips.
    - Upgrade the interface module between the PaddlePaddle framework and compiler, to support inference models to access the compiler for optimization via Paddle Inference. （[#44499](https://github.com/PaddlePaddle/Paddle/pull/44499) [#44708](https://github.com/PaddlePaddle/Paddle/pull/44708) ）

### （2）Underlying optimization

- **GPU performance optimization**
  - Add the TensorRT mapping for operators such as matmul_v2, LSTM, reshape, fill_constant, swish, mulitclass_nms3, bilinear_interp_v2, split, silu, shuffle_channel operators. Optimize the support for the dynamic shape. Performance improved by 7% to 90% for multi-class focused models. ([#46177](https://github.com/PaddlePaddle/Paddle/pull/46177), [#44678](https://github.com/PaddlePaddle/Paddle/pull/44678), [#44314](https://github.com/PaddlePaddle/Paddle/pull/44314), [#44561](https://github.com/PaddlePaddle/Paddle/pull/44561), [#45166](https://github.com/PaddlePaddle/Paddle/pull/45166),  [#44411](https://github.com/PaddlePaddle/Paddle/pull/44411), [#43424](https://github.com/PaddlePaddle/Paddle/pull/43424), [#44516](https://github.com/PaddlePaddle/Paddle/pull/44516))
  - Add constant folding PASS for inference performance optimization, to improve the performance of SwinTransformer, HifiGAN, FastSpeech2, and other models.（[#45494](https://github.com/PaddlePaddle/Paddle/pull/45494))
  - Add cache of conv_fusionworkspacesize, to improve the computation performance of conv_fusion. ([#45902](https://github.com/PaddlePaddle/Paddle/pull/45902))
- **Vision ViT model optimization**
  - Add the ViT model Attention structure fusion PASS, and support OSSPlugin and auto padding. The ViT inference speed increases by 30%-40%.  [#45019](https://github.com/PaddlePaddle/Paddle/pull/45019) [#45506](https://github.com/PaddlePaddle/Paddle/pull/45506)
- **Inference performance optimization of large model**
  - To improve the inference speed of very large generative models and save the video memory, add INT8 implementation (fused_multi_transformer_int8_op) to the multi-layer Transformer fusion operator (fused_multi_transformer_op), and support quantized inference of generative models. Use the matrix multiplication algorithm to select, quantize/de-quantize the kernel fusion for performance optimization.  [#46169](https://github.com/PaddlePaddle/Paddle/pull/46169)
  - Add Pass for automatic matching fusion in order to improve the ease of use of fused_multi_transformer fusion for large model inference.
- **CPU performance optimization**
  - Optimize the speech U2++ model. The FP32 model inference speed is improved by 35%. The INT8 model inference speed is improved by 69%.  ([#47592](https://github.com/PaddlePaddle/Paddle/pull/47592), [#47127](https://github.com/PaddlePaddle/Paddle/pull/47127), [#47391](https://github.com/PaddlePaddle/Paddle/pull/47391), [#47234](https://github.com/PaddlePaddle/Paddle/pull/47234), [#47009](https://github.com/PaddlePaddle/Paddle/pull/47009), [#47080](https://github.com/PaddlePaddle/Paddle/pull/47080))


### （3）Bug fix

- TensorRT workspace size supports int64. （[#44469](https://github.com/PaddlePaddle/Paddle/pull/44469) ）
- In Paddle-TRT, fully support Op's input as weight.（[#45545](https://github.com/PaddlePaddle/Paddle/pull/45545) ）
- In Paddle-TRT, support conv2d_transpose/conv3d_transpose to have the output_padding attribute.（[#45004](https://github.com/PaddlePaddle/Paddle/pull/45004) ）
- In Paddle-TRT, enhance the strided_slice support for dynamic shape. （[#46819](https://github.com/PaddlePaddle/Paddle/pull/46819) ）
- In Paddle-TRT, optimize the video memory footprint of context when running in multi-thread scenarios.（[#45468](https://github.com/PaddlePaddle/Paddle/pull/45468) ）
- In Paddle-TRT, fix the bug of repeatedly generating serialization files in case of change of initialization sequences when multiple models run in the same process.（[#43942](https://github.com/PaddlePaddle/Paddle/pull/43942) ）
- Fix the bug of occasional crash when Predictor is initialized to run for multiple times in the same process.（[#45203](https://github.com/PaddlePaddle/Paddle/pull/45203) ）
- Fix the bug of abnormal inference accuracy of quantization models such as MobileNetV3_large, ERNIE 3.0-Medium and bert ([#45416](https://github.com/PaddlePaddle/Paddle/pull/45416), [#46283](https://github.com/PaddlePaddle/Paddle/pull/46283), [#45920](https://github.com/PaddlePaddle/Paddle/pull/45920) [#47573](https://github.com/PaddlePaddle/Paddle/pull/47574))

## 5. Environment adaptation

- The pre-compiled installer for training and the pre-compiled installer for deployment (Paddle Inference) are unified into one pre-compiled installer. The build system is optimized so that the pre-compiled installer supports TensorRT by default.
- The pre-compiled installer for CUDA version 10.1 is cancelled.
- Add the pre-compiled installer for CUDA 11.7.
- Decrease of source code compilation time: Reduce inter-module dependencies, improve the parallel, and optimize the compilation speed of some modules. The full compilation time is reduced by about 20 minutes in total.
- Support the running of PaddlePaddle on windows 11, Centos 8, Ubuntu 22.04, Jetson 5.02 system environment. Support to run PaddlePaddle linux installer in windows system by using the WSL 2 tool.
- Fix the running error bug of the PaddlePaddle in glibc2.34+ environment.
- Optimize the code style of C++, Python, CMake in the whole code repository. Introduce or upgrade the following code style checking tools.
  - pre-commit is upgraded from 1.10.4 to 2.17.0： [#43103](https://github.com/PaddlePaddle/Paddle/pull/43103)
  - pylint is changed from default version to specify as： [#43103](https://github.com/PaddlePaddle/Paddle/pull/43103)
  - remove-crlf is upgraded from 1.0.1 to 1.1.14 ： [#43103](https://github.com/PaddlePaddle/Paddle/pull/43103)
  - cpplint is changed from default version to specify as 1.6.0 ： [#43175](https://github.com/PaddlePaddle/Paddle/pull/43175), [#43978](https://github.com/PaddlePaddle/Paddle/pull/43978), [#43673](https://github.com/PaddlePaddle/Paddle/pull/43673), [#43679](https://github.com/PaddlePaddle/Paddle/pull/43679), [#43695](https://github.com/PaddlePaddle/Paddle/pull/43695), [#43733](https://github.com/PaddlePaddle/Paddle/pull/43733), [#43740](https://github.com/PaddlePaddle/Paddle/pull/43740)
  - clang-format is upgrade from 3.8 to 13.0 ： [#42840](https://github.com/PaddlePaddle/Paddle/pull/42840), [#43248](https://github.com/PaddlePaddle/Paddle/pull/43248), [#43329](https://github.com/PaddlePaddle/Paddle/pull/43329), [#43333](https://github.com/PaddlePaddle/Paddle/pull/43333), [#43633](https://github.com/PaddlePaddle/Paddle/pull/43633), [#43678](https://github.com/PaddlePaddle/Paddle/pull/43678)
  - Introduce the black tool for python code style checking ：[#46014](https://github.com/PaddlePaddle/Paddle/pull/46014)
  - Introduce the cmakelint tool for cmake file code checking. Version is 1.4.2 ： [#43222](https://github.com/PaddlePaddle/Paddle/pull/43222), [#43406](https://github.com/PaddlePaddle/Paddle/pull/43406), [#43414](https://github.com/PaddlePaddle/Paddle/pull/43414), [#43428](https://github.com/PaddlePaddle/Paddle/pull/43428)
  - Introduce cmake-format for automatic formatting of cmake files. Version is 0.6.13 ： [#43057](https://github.com/PaddlePaddle/Paddle/pull/43057)

## 6. Hardware adaptation
### Hygon DCU
- Add the Profiler function on DCU, to collect, count and display performance data of model running process on DCU, and support DCU occupancy display at kernel level.
### Kunlunxin Chip
- Add Profiler function on Kunlunxin 2 generation chip, which can collect, count and display the performance data of model running process on Kunlunxin 2 generation chip, and support occupancy display of Kunlunxin 2 generation chip at kernel level.
- Training/reasoning support for Kunlunxin 2 generation chips (Kunlunxin AI accelerator cards R200, R300, R200-8F, R200-8FS, RG800), a total of 51 models such as PPYOLOE, PP-OCR, ERNIE3.0, PP-TSM, PP-TTS, DLRM, PPO, etc. have been verified, supporting static graph + dynamic graph training, supporting mixed precision training, support single machine single card and single machine multi-card training, covering 5 fields of intelligent vision, natural language processing, intelligent speech, intelligent recommendation, reinforcement learning.
### Cambricon
-  Support the training/inference of Cambricon MLU chip (MLU370 series of boards): The ResNet50, BERT, YoloV3, OCR-DB, Deeplabv3 and many other models are verified. Support the static graph + dynamic graph training. Support mixed precision training. Support the single machine single card and single machine multi-card training.
### Graphcore
- Support the training/inference of Graphcore IPU chip (including IPU Mk2 GC200 and Bow IPU). Support ResNet50, BERT and other models. Support the static graph and dynamic-to-static graph mode training. Support the single chip, single machine, and multi-machine distributed training.
- Add the support of more operators
- Upgrade to Poplar SDK v3.0.0  [#46892](https://github.com/PaddlePaddle/Paddle/pull/46892)
* Support the training models by using the dynamic-to-static graph mode. Add a new paddle.incubate.identity_loss op to assist with composition [#43770](https://github.com/PaddlePaddle/Paddle/pull/43770)
* Support the Paddle native distributed training API: paddle.distributed.launch [#43311](https://github.com/PaddlePaddle/Paddle/pull/43311)
* Support the training models with the mixed precision [#41733](https://github.com/PaddlePaddle/Paddle/pull/41733)
* Paddle Inference supports custom operators by using PopART [#45235](https://github.com/PaddlePaddle/Paddle/pull/45235)

### Intel
- Migrate oneDNN operators : transpose2_grad([#46139](https://github.com/PaddlePaddle/Paddle/pull/46139)), relu6_grad([#46501](https://github.com/PaddlePaddle/Paddle/pull/46501)), gaussian_random([#46747](https://github.com/PaddlePaddle/Paddle/pull/46747), [#45481](https://github.com/PaddlePaddle/Paddle/pull/45481)), sgd and stack([#46374](https://github.com/PaddlePaddle/Paddle/pull/46374)), concat+grad, expand+grad,fill_constant([#45863](https://github.com/PaddlePaddle/Paddle/pull/45863)), slice, slice_grad, split,pad and pad3d([#46101](https://github.com/PaddlePaddle/Paddle/pull/46101)), softmax_grad([#46257](https://github.com/PaddlePaddle/Paddle/pull/46257)), Shape([#46051](https://github.com/PaddlePaddle/Paddle/pull/46051)), Sum([#46239](https://github.com/PaddlePaddle/Paddle/pull/46239)), Transpose2_grad([#46139](https://github.com/PaddlePaddle/Paddle/pull/46139)), Cast, clip+grad andpool+grad([#45775](https://github.com/PaddlePaddle/Paddle/pull/45775)), Reduce sum+grad,mean+grad, min and max([#45536](https://github.com/PaddlePaddle/Paddle/pull/45536)), Relu and abs([#45397](https://github.com/PaddlePaddle/Paddle/pull/45397)), Gelu([#45596](https://github.com/PaddlePaddle/Paddle/pull/45596)), Scale([#45537](https://github.com/PaddlePaddle/Paddle/pull/45537))
- Optimize kernels of fill_constant, fc, conv, and a number of operators
- Add several Pass fusion optimizations
- Optimize the Adam-W CPU FP32 optimizer  ([#42522](https://github.com/PaddlePaddle/Paddle/pull/42522))
- Optimize pad3d fp32 onednn operator kernel implementation ([#43990](https://github.com/PaddlePaddle/Paddle/pull/43990))
- Optimize the concurrent execution of matmul, FC andlookup_v2 kernels ([#44023](https://github.com/PaddlePaddle/Paddle/pull/44023), [#44078](https://github.com/PaddlePaddle/Paddle/pull/444078), [#44640](https://github.com/PaddlePaddle/Paddle/pull/44640), [#44744](https://github.com/PaddlePaddle/Paddle/pull/44744), [#45249](https://github.com/PaddlePaddle/Paddle/pull/45249))
- FC onednn operator kernel supports bf16 ( [#42758](https://github.com/PaddlePaddle/Paddle/pull/42758), [#43154](https://github.com/PaddlePaddle/Paddle/pull/43154), [#43109](https://github.com/PaddlePaddle/Paddle/pull/43109))
- Add the fusion of matrix multiplication and activation functions ([#43519](https://github.com/PaddlePaddle/Paddle/pull/43519), [#43198](https://github.com/PaddlePaddle/Paddle/pull/43198))
- Support convolution operator int8 parameter production IR passes ( [#44680](https://github.com/PaddlePaddle/Paddle/pull/44680), [#42625](https://github.com/PaddlePaddle/Paddle/pull/42625))
- Add pool/avg quantization and scales correction  ([#44186](https://github.com/PaddlePaddle/Paddle/pull/44186))
- Add the matmul and elementwise onednn operator kernel fusion ([#45077](https://github.com/PaddlePaddle/Paddle/pull/45077))
- Fix the QAT precision bug ([#43693](https://github.com/PaddlePaddle/Paddle/pull/43693), [#45936](https://github.com/PaddlePaddle/Paddle/pull/45936), [#46378](https://github.com/PaddlePaddle/Paddle/pull/46378))
- Migrate 42 oneDNN operator kernels to PHI operator library ([#46374](https://github.com/PaddlePaddle/Paddle/pull/46374), [#46101](https://github.com/PaddlePaddle/Paddle/pull/46101), [#45989](https://github.com/PaddlePaddle/Paddle/pull/45989), [#45863](https://github.com/PaddlePaddle/Paddle/pull/45863), [#45775](https://github.com/PaddlePaddle/Paddle/pull/45775), [#45626](https://github.com/PaddlePaddle/Paddle/pull/45626), [#45536](https://github.com/PaddlePaddle/Paddle/pull/45536), [#46501](https://github.com/PaddlePaddle/Paddle/pull/46501), [#46257](https://github.com/PaddlePaddle/Paddle/pull/46257), [#45596](https://github.com/PaddlePaddle/Paddle/pull/45596), [#45537](https://github.com/PaddlePaddle/Paddle/pull/45537), [#45481](https://github.com/PaddlePaddle/Paddle/pull/45481), [#45397](https://github.com/PaddlePaddle/Paddle/pull/45397), [#46239](https://github.com/PaddlePaddle/Paddle/pull/46239), [#46139](https://github.com/PaddlePaddle/Paddle/pull/46139), [#46051](https://github.com/PaddlePaddle/Paddle/pull/46051))
- Quantize the elementwise_sub and shape operator kernels ([#42854](https://github.com/PaddlePaddle/Paddle/pull/42854), [#44124](https://github.com/PaddlePaddle/Paddle/pull/44124))

## Thanks to our Contributors

This release contains contributions from:

0x45f, Aganlengzi, Ainavo, Allen Guo, Asthestarsfalll, Aurelius84, Baibaifan, baoachun, BiynXu, Bo Zhang, BrilliantYuKaimin, cambriconhsq, caozhou, carryyu, ccrrong, ceci3, chalsliu, Chang Xu, Charles-hit, Chen Long, Chen Weihang, chenjian, chentianyu03, Chenxiao Niu, cifar10, crystal, csy0225, danleifeng, David Nicolas, dc-cheny, denglin-github, dongfangshenzhu, duanboqiang, duanyanhui, engineer, enzodechine, Fan Zhang, feifei-111, Feiyu Chan, Feng Ni, feng_shuai, FlyingQianMM, freeliuzc, furnace, fuyou765, fwenguang, Ghost Screaming, gongweibao, Guanghua Yu, guguguzi, Guoxia Wang, Haipeng Wang, handiz, Haohongxiang, haosicheng, helen88, heliqi, hong, HongyuJia, houj04, huangxu96, Hui Zhang, Huihuang Zheng, huzhiqiang, Jacek Czaja, Jack Zhou, jack603047588, Jackwaterveg, jakpiase, james, Jiabin Yang, jiangcheng, Jiaqi Liu, JingZhuangzhuang, joanna.wozna.intel, JYChen, JZ-LIANG, Kaipeng Deng, kangguangli, kuizhiqing, Leo Chen, Leo Guo, levi131, Li Min, Li-fAngyU, lidanqing, LielinJiang, Ligoml, Lijunhui, lilong12, limingshu, Lin Manhui, Linjie Chen, liqitong-a, littletomatodonkey, liu zhengxi, Liu-xiandong, liutiexing, Liyulingyue, LiYuRio, Lux et Veritas, lyq, Matsumoto Ruko, MayYouBeProsperous, mengqingchun02, Ming-Xu Huang, ming1753, minghaoBD, moyan, mrcangye, Netpunk, niuliling123, Nyakku Shigure, OccupyMars2025, onecatcn, pangyoki, parap1uie-s, peachlcy, piotrekobi, Qi Li, QingshuChen, qipengh, Rayman, Regan Yue, RichardWooSJTU, risemeup1, Roc, ronnywang, Rui Li, Ruibiao Chen, seemingwang, Shang Zhizhou, shangliang Xu, ShenLiang, shentanyue, Shijie, ShiningZhang, shixingbo, shiyutang, Shuangchi He, Siming Dai, Sing_chan, Skr Bang, SmirnovKol, sneaxiy, sprouteer, Sylwester Fraczek, Sławomir Siwek, taixiurong, Tao CHANG, TeFeng Chen, Thomas Young, thunder95, Thunderbrook, tiancaishaonvjituizi, tianshuo78520a, Tomasz Socha, TTerror, USTCKAY, Vigi Zhang, Walter, Wang Bojun, wangguanqun, wangguanzhong, wanghuancoder, wangna11BD, WangXi, wangxinxin08, Wangzheee, WangZhen, wangzhen38, wawltor, wbn, Wei Shengyu, Weilong Wu, weishengying, Wen Sun, wenbin, whs, Wilber, WJJ1995, wuhuachaocoding, wuhuanzhou, wuyefeilin, XiaoguangHu, xiaoguoguo626807, xiaohemaikoo, xiaoting, xiaoxiaohehe001, Xiaoxu Chen, xiayanming, Xingyuan Zhang, xiongkun, yang131313, yangguohao, YangZhou, Yanxing Shi, Yao Zihang, yaoxuefeng, yaozhixin, yeliang2258, Yilingyelu, Yiqun Liu, ykkk2333, Yuang Liu, Yuanle Liu, YuanRisheng, yuguo, Yulong Ao, Yulv-git, YUNSHEN XIE, Zhang Jun, Zhang Ting, Zhang Zheng, zhangbo9674, zhangbopd, zhangchunle, Zhangjingyu06, zhangkaihuo, zhangxiaoci, zhangyikun02, zhangzhenguo, Zhanlue Yang, zhaocaibei123, zhaoying9105, zhaoyingli, Zhen Wang, Zhengyang Song, zhiboniu, Zhong Hui, Zhou Wei, zhoutianzi666, zhupengyang, ziyoujiyi, zlsh80826, zmxdream, zn, Zuza Gawrysiak, zyfncg, 傅剑寒, 六个骨头, 津, 熊峻峰, 王明冬, 石晓伟

# 2.3.1 Release Note

## **1. Important Updates**

- V2.3.1 is built on V2.3 by fixing known issues and releasing  precompiled binary that supports CUDA 11.6.

## **2. Training Framework (distributed included)**

### **(1) Function Optimization**

#### API

- Modify two initialization modes of `paddle.nn.initializer.KaimingUniform` and `paddle.nn.initializer.KaimingNormal`, to support multiple types of activation functions. ([#43721](https://github.com/PaddlePaddle/Paddle/pull/43721), [#43827](https://github.com/PaddlePaddle/Paddle/pull/43827))
- Optimize the data pre-fetching function of `paddle.io.DataLoader`, so that it can support the setting of the `prefetch_factor` to set the cache size of pre-fetched data. This can avoid IO blocking when reading large blocks of data. ([#43674](https://github.com/PaddlePaddle/Paddle/pull/43674))

#### **New dynamic graph execution mechanism**

- Modify the initialization method of optional type Tensor in the new dynamic graph API logic to prevent data exceptions caused by early destruction. ([#42561](https://github.com/PaddlePaddle/Paddle/pull/42561))

#### **New static graph executor**

- Defer initialization of the thread pools in the  executor, to avoid creating thread pools for `programs` that execute only once (e.g.,`save, load, startup_program`, etc.). ([#43768](https://github.com/PaddlePaddle/Paddle/pull/43768))

#### **Mixed precision training**

- Disabling `state_dict` hook in `set_state_dict` in `paddle.nn.Layer`. ([#43407](https://github.com/PaddlePaddle/Paddle/pull/43407))

#### **Distributed training**

- Enabling tensor parallelism in `paddle.incubate.nn.functional.fused_attention` and `paddle.incubate.nn.functional.fused_feedforward`. ([#43505](https://github.com/PaddlePaddle/Paddle/pull/43505))

#### **Others**

- Adjust print format of the framework operator kernels to facilitate automated splitting and parsing. ([#42931](https://github.com/PaddlePaddle/Paddle/pull/42931))
- Update the model quantization API to support the round-off in `rounding to nearest ties to even`, and support quantization in the range [-128, 127]. ([#43829](https://github.com/PaddlePaddle/Paddle/pull/43829))
- Support AMP mixed precision training in quantization-aware training. ([#43689](https://github.com/PaddlePaddle/Paddle/pull/43689))
- Add the `progress bar` at the beginning of quantization-aware training, so that it is easy to check the progress of quantization initialization. Skip the scale op when counting out_threshold to speed up the initialization process. ([#43454](https://github.com/PaddlePaddle/Paddle/pull/43454))
- Support `conv` and `bn` fusion in the dynamic graph quantization training. Support the settings of skip_tensor_list in the static graph offline quantization, to skip some layers without quantization. ([#43301](https://github.com/PaddlePaddle/Paddle/pull/43301))

### **(2) Performance Optimization**

- Optimize`paddle.incubate.nn.functional.fused_attention` and `paddle.incubate.nn.functional.fused_feedforward`operators. Add `add_residual` property to control whether to perform add-`residual` operation in the last step. The performance of CAE model is improved by 7.7%. ([#43719](https://github.com/PaddlePaddle/Paddle/pull/43719))
- Optimize `linspace` operator. Initialize three input Tensor of `start`,`stop` and `num` on CPU, to avoid GPU->CPU copy in the operator. This can speed up SOLOv2 model performance by 6%. ([#43746](https://github.com/PaddlePaddle/Paddle/pull/43746))

### **(3) Bug Fix**

#### API

- Fix the error reported by `paddle.io.DataLoader` when `return_list=True` due to multi-thread conflict. ([#43691](https://github.com/PaddlePaddle/Paddle/pull/43691))
- Fix the error that the `to` method reports NoneType does not have the device attribute when the `paddle.nn.Layer` parameter has the `None` type parameter. ([#43597](https://github.com/PaddlePaddle/Paddle/pull/43597))
- Fix the bug that the calculation result of cumsum op is wrong in some `shape` settings. ([#42500](https://github.com/PaddlePaddle/Paddle/pull/42500), [#43777](https://github.com/PaddlePaddle/Paddle/pull/43777))
- Fix the bug that the output result dimension of `Tensor.__getitem__` is 0 in the networking stage when using `bool` index in the static graph. ([#43246](https://github.com/PaddlePaddle/Paddle/pull/43246))
- Fix the bug occurred when `paddle.slice` and `paddle.strided_slice` handle negative parameters. ([#43432](https://github.com/PaddlePaddle/Paddle/pull/43432))
- Fix the bug that the assignment result of set_value op is abnormal when the processing slice `step` is negative. ([#43694](https://github.com/PaddlePaddle/Paddle/pull/43694))
- Fix the bug that the `copy` interface in C++ cannot copy between multiple cards. ([#43728](https://github.com/PaddlePaddle/Paddle/pull/43728))
- Fix the bug in inference stage caused by attribute naming in `paddle.incubate.nn.functional.fused_attention`and `paddle.incubate.nn.functional.fused_feedforward`. ([#43505](https://github.com/PaddlePaddle/Paddle/pull/43505))
- Fix an exception in ConditionalBlockGrad op when processing Tensor that does not require `grad`. ([#43034](https://github.com/PaddlePaddle/Paddle/pull/43034))
- Fix the bug of device memory increase caused by einsum op in the speed optimization of backward computation. By default, this optimization is enabled. ([#43397](https://github.com/PaddlePaddle/Paddle/pull/43397))
- Fix the bug that data fails to be fixed when `paddle.io.DataLoader` multi-process data reads the fixing random seeds under a single card. ([#43702](https://github.com/PaddlePaddle/Paddle/pull/43702))
- Fix the bug that softmax op triggers CUDNN_STATUS_NOT_SUPPORT when the Tensor exceeds 2G. ([#43719](https://github.com/PaddlePaddle/Paddle/pull/43719))
- Fix the bug that the trace op `Event` string is indistinguishable among different operators that cause the inconvenient performance analysis. ([#42789](https://github.com/PaddlePaddle/Paddle/pull/42789))

#### **Others**

- Fix the bug of overflowing device memory caused by multiple deepcopy and saving in case of dynamic-to-static. ([#43141](https://github.com/PaddlePaddle/Paddle/pull/43141))
- Fix the bug that the device id introduced by the upgrade of PlaceType used in the custom operator is wrong in the multi-card scenario. ([#43830](https://github.com/PaddlePaddle/Paddle/pull/43830))
- Optimize the `paddle.profiler.Profiler` timeline visualization logic, move events customized in python scripts from C++ folding display to python folding display. ([#42790](https://github.com/PaddlePaddle/Paddle/pull/42790))

## **3.** Deployment Direction (Paddle Inference)

### **(1) New Features**

#### **New functions**

- Add the support of the PaddleSlim quantization model for ONNX Runtime backends on CPUs. ([#43774](https://github.com/PaddlePaddle/Paddle/pull/43774), [#43796](https://github.com/PaddlePaddle/Paddle/pull/43796))

### **(2) Underlying Optimization**

#### **CPU performance optimization**

- Remove `gpu_cpu_reshape2_matmul_fuse_pass` from EnableMkldnn configuration to fix the bug of ResNet50 performance degradation. ([#43750](https://github.com/PaddlePaddle/Paddle/pull/43750))

#### **GPU performance optimization**

- Add the support of `bilinear_interp_v2` TensorRT convert. ([#43618](https://github.com/PaddlePaddle/Paddle/pull/43618))
- Add `matmul_scale_fuse_pass` and `multihead_matmul_fuse_pass_v3` to GPU pass. ([#43765](https://github.com/PaddlePaddle/Paddle/pull/43765))
- Add the support of the GPU handle deferred initialization. ([#43661](https://github.com/PaddlePaddle/Paddle/pull/43661))

### **(3) Bug Fixing**

#### **Framework and API fixing**

- Fix the compile error problem when binding Paddle-Lite XPU. ([#43178](https://github.com/PaddlePaddle/Paddle/pull/43178))
- Fix the bug of false trigger of ERNIE 3.0 pass. ([#43948](https://github.com/PaddlePaddle/Paddle/pull/43948))
- Fix the bug that int8 quantization attribute in multihead op cannot be read. ([#43020](https://github.com/PaddlePaddle/Paddle/pull/43020))

#### **Backend capability fixing**

- Fix the bug that two ops of elementwise_mul and matmul in MKLDNN are crashed during quantitative inference. ([#43725](https://github.com/PaddlePaddle/Paddle/pull/43725))
- Fix a bug where TensorRT subgraph serialization files are repeatedly generated for the same model during inference. ([#42945](https://github.com/PaddlePaddle/Paddle/pull/43945), [#42633](https://github.com/PaddlePaddle/Paddle/pull/42633))
- Fix a conflict between the ONNX Runtime backend and the externally use of protobuf. ([#43159](https://github.com/PaddlePaddle/Paddle/pull/43159), [#43742](https://github.com/PaddlePaddle/Paddle/pull/43742))
- Fix an error reported by python prediction library when using ONNX Runtime backend in case of multiple inputs. ([#43621](https://github.com/PaddlePaddle/Paddle/pull/43621))

## **4. Environment Adaptation**

### **Compile and install**

- Complete verification and adaptation of CUDA 11.6, and release CUDA 11.6 precompiled binary. ([#43935](https://github.com/PaddlePaddle/Paddle/pull/43935), [#44005](https://github.com/PaddlePaddle/Paddle/pull/44005))
- Fix a cub error when compiling with CUDA 11.6 on Windows. ([#43935](https://github.com/PaddlePaddle/Paddle/pull/43935), [#44005](https://github.com/PaddlePaddle/Paddle/pull/44005))
- Fix the bug of long compilation time for elementwise and reduce op. ([#43202](https://github.com/PaddlePaddle/Paddle/pull/43202), [#42779](https://github.com/PaddlePaddle/Paddle/pull/42779), [#43205](https://github.com/PaddlePaddle/Paddle/pull/43205))

### **New hardware adaptation**

- Cambricon MLU supports PaddlePaddle Profiler. ([#42115](https://github.com/PaddlePaddle/Paddle/pull/42115))
- GraphCore IPU supports visualization of compilation progress. ([#42078](https://github.com/PaddlePaddle/Paddle/pull/42078))

# 2.3.0 Release Note

## 1. **Important Updates**

We are excited to release the PaddlePaddle Framework V2.3.0. This version contains the following highlights.

### API

- Added more than 100 new APIs, covering automatic differentiation, linear algebra, probability distribution, sparse tensor, framework performance analysis, hardware device management, vision domain, etc.

- Added 4 new automatic differentiation APIs, 11 new linear algebra APIs, and 21 new probability distribution APIs to better support use cases in scientific computing, reinforcement learning, xand other application areas.

- Added 11 new Sparse Tensor APIs including basic functions of sparse tensor construction and conversion. The COO and CSR formats are supported.

- Added 9 new framework performance analysis APIs. The new performance profiling APIs, centered around Paddle.Profiler.Profiler, help users collect and analyze performance statistics during training and inference.

- Added 7 APIs for device management, facilitating hardware information acquistion.

- Added several visual and text domain APIs to facilitate ~~the~~ reusability of MobileNetV3, ResNeXt and other backbone networks, to achieve the fast networking.


### **Paddle** HIgh reusability operator l**ibrary**

- We announce PHI as the new Paddle HIgh reusability operator library. PHI provides Primitive API, enabling kernel reuse for operator development. As a refactored functional operator library, PHI aims to solve legacy problems that harm the framework's performance and reusability, in particular on the operator development. Such problems include inefficient ways of cross using operators, unclear operator interfaces and lacking direct calls to the operator library in C++. With PHI, new operators can be easily implemented by composing functions available in the functional library. The library provides over 200 C++ operator class APIs and nearly 500 kernels. Composing new operators through these built-in functions can greatly reduce the user's development effort. PHI supports different types of hardware (e.g., GPU and XPU). In addition, PHI is extensible with plugins for accommodating third party accelerators (such as NPU) in a low cost and reusable fashion. In short, PHI supports low level operator composability, the reuse of kernels through Primitives, and accelerators through plugins.

### **Distributed Training**

- Fully upgrade the adaptive distributed training architecture, including multiple modules such as elastic resource management, asynchronous pipelined executor, heterogeneous communication, and automatic parallelism, and support the hard-aware distributed training and inference under a variety of heterogeneous hardware.

- Add MoE parallel strategy, GroupSharded parallel strategy, and Pure FP16 under dynamic graph hybrid Parallelism, which further supports the efficient distributed training of large models under the dynamic graph.

- Comprehensively upgrade and optimize the architecture of general heterogeneous parameter server, and simplify each module, such as communication and storage, to improve the secondary development experience of parameter server. The performance of GPU parameter server is improved by 2.38 times under 100 billion parameters and 10 billion data.


### **Compile and Install**

- From version 2.3.0, PaddlePaddle upgrades GPU architectures supported.


### **Inference Deployment**

- Add the Java API and ONNX Runtime CPU backend.

- Support the TensorRT 8.0 / 8.2 and structured sparsity, with deep performance optimization for ERNIE-like structural models.


### **Hardware Backend Extention**

- Add custom device support: provide a plug-in way to extend PaddlePaddle hardware backend.

- Add training/inference support for multiple heterogeneous chips such as HUAWEI Ascend 910 / GraphCore IPU / Cambricon MLU / KUNLUNXIN 2.


### **Framework Architecture**

- In this version, we did a lot of work on the framework executor. For details, please see [New Dynamic Graph Execution Mechanism](#new-dynamic-graph-execution-mechanism) and [New Static Graph Executor](#new-static-graph-executor).

## **2. Incompatibility Upgrade**

- Due to limitation of the binary size, sm35 CUDA ARCH is dropped in pre-compiled binaries. ([#41754](https://github.com/PaddlePaddle/Paddle/pull/41754))

- When `paddle.to_tensor` converts a python int scalar to a Tensor, the default data type on Windows changes from int32 to int64, thus alignment with Linux/Mac. ([#39662](https://github.com/PaddlePaddle/Paddle/pull/39662))

- To keep consistency with division behavior under python3, the division symbol `/` has been changed from “rounding divide” to “true divide”, and the data type of the computed output has been switched from int to float. ([#40890](https://github.com/PaddlePaddle/Paddle/pull/40890))


<table>
<tr>
<th>
2.2
</th>
<th>
2.3.0
</th>
</tr>

<tr>
<td>
<pre>

```python
>>> import paddle
>>> a = paddle.to_tensor([327])
>>> b = paddle.to_tensor([80])
>>> a / b
Tensor(shape=[1], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
      [4])
```
</pre>
</td>
<td>
<pre>

```python
>>> import paddle
>>> a = paddle.to_tensor([327])
>>> b = paddle.to_tensor([80])
>>> a / b
Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
      [4.08750010])
```
</pre>
</td>
</tr>
</table>

- Revise the ELU's formula. The computing method in case of alpha <0 aligns with the original paper, thus fixing a small number of cases where the results are incorrectly calculated. Meanwhile, elu_ will report an error in case of alpha <0, because it is not mathematically possible to compute the inverse gradient from the output only at alpha <0. ([#37316](https://github.com/PaddlePaddle/Paddle/pull/37316))

<table>
<tr>
<th>
2.2
</th>
<th>
2.3.0
</th>
</tr>

<tr>
<td>
<pre>

```python
# elu(x) = max(0, x) + min(0, α ∗ (e^x − 1))
>>> import paddle
>>> x = paddle.to_tensor([-1., 6.])
>>> m = paddle.nn.ELU(-0.2)
>>> out = m(x)
>>> out
Tensor(shape=[2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
       [ 0.        , -74.48576355])
>>> out = paddle.nn.functional.elu_(x, alpha=-0.2, name=None)
>>> out
Tensor(shape=[2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
       [ 0.        , -74.48576355])
```
</pre>
</td>
<td>
<pre>

```python
# elu(x) = x, if x > 0
# elu(x) = α ∗ (e^x − 1), if x <= 0
>>> import paddle
>>> x = paddle.to_tensor([-1., 6.])
>>> m = paddle.nn.ELU(-0.2)
>>> out = m(x)
>>> out
Tensor(shape=[2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
       [0.12642412,  6.        ])
>>> out = paddle.nn.functional.elu_(x, alpha=-0.2, name=None)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/lib/python3.7/dist-packages/decorator.py", line 232, in fun
    return caller(func, *(extras + args), **kw)
  File "/usr/local/lib/python3.7/dist-packages/paddle/fluid/wrapped_decorator.py", line 25, in __impl__
    return wrapped_func(*args, **kwargs)
  File "/usr/local/lib/python3.7/dist-packages/paddle/fluid/dygraph/inplace_utils.py", line 34, in __impl__
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.7/dist-packages/paddle/nn/functional/activation.py", line 89, in elu_
    assert alpha >= 0., "elu_ only support alpha >= 0, please use elu instead."
AssertionError: elu_ only support alpha >= 0, please use elu instead.
```
</pre>
</td>
</tr>
</table>

## **3. Training Framework (with the distributed function)**

### **(1) New functions**

#### API

- Add 4 new automatic differentiation APIs to support scientific computing, as listed below: ([#40692](https://github.com/PaddlePaddle/Paddle/pull/40692))

  - `paddle.incubate.autograd.vjp`, compute vector-Jacobi matrix product.

  - `paddle.incubate.autograd.jvp`, compute Jacobi matrix-vector product.

  - `paddle.incubate.autograd.Jacobian`, compute Jacobi matrix.

  - `paddle.incubate.autograd.Hessian`, compute Hessian matrix.

- Add linear algebra class API

  - Add `paddle.linalg.triangular_solve`, to compute a system of linear equations with unique solutions through a triangular coefficient. ([#36714](https://github.com/PaddlePaddle/Paddle/pull/36714))

  - Add `paddle.linalg.eig`, to compute the characteristic decomposition of the general square matrix. ([#35764](https://github.com/PaddlePaddle/Paddle/pull/35764))

  - Add `paddle.linalg.sovle`, to compute solutions to systems of linear equations. ([#35715](https://github.com/PaddlePaddle/Paddle/pull/35715))

  - Add `paddle.linalg.lstsq`, to compute least-squares solutions to systems of linear equations. ([#38585](https://github.com/PaddlePaddle/Paddle/pull/38585), [#38621](https://github.com/PaddlePaddle/Paddle/pull/38621))

  - Add `paddle.linalg.qr`, compute QR decomposition of matrix. ([#35742](https://github.com/PaddlePaddle/Paddle/pull/35742), [#38824](https://github.com/PaddlePaddle/Paddle/pull/38824))

  - Add `paddle.inner`, to compute inner product of a matrix. ([#37706](https://github.com/PaddlePaddle/Paddle/pull/37706))

  - Add `paddle.outer`, to compute outer product of a matrix. ([#37706](https://github.com/PaddlePaddle/Paddle/pull/37706))

  - Add `paddle.linalg.cov`, to compute covariance between vectors. ([#38392](https://github.com/PaddlePaddle/Paddle/pull/38392))

  - Add `paddle.linalg.cholesky_sovle`, to compute the cholesky solution of the equation. ([#38167](https://github.com/PaddlePaddle/Paddle/pull/38167))

  - Add `paddle.linalg.lu` and `paddle.linalg.lu_unpack`, to compute matrix lu decomposition, and decompress lu matrix. ([#38617](https://github.com/PaddlePaddle/Paddle/pull/38617), [#38559](https://github.com/PaddlePaddle/Paddle/pull/38559), [#38616](https://github.com/PaddlePaddle/Paddle/pull/38616))

- Add 21 new probability distribution class APIs for reinforcement learning, variation inference, scientific computing, and other scenarios. Including 6 random variable distributions, 13 random variable transformations, and 2 KL divergence computing. as listed below: ([#40536](https://github.com/PaddlePaddle/Paddle/pull/40536), [#38820](https://github.com/PaddlePaddle/Paddle/pull/38820), [#38558](https://github.com/PaddlePaddle/Paddle/pull/38558/files), [#38445](https://github.com/PaddlePaddle/Paddle/pull/38445), [#38244](https://github.com/PaddlePaddle/Paddle/pull/38244), [#38047](https://github.com/PaddlePaddle/Paddle/pull/38047))

  - `paddle.distribution.ExponentialFamily`, exponential distribution family base class.

  - `paddle.distribution.Beta`, `Beta` distribution.

  - `paddle.distribution.Dirichlet`, `Dirichlet` distribution.

  - `paddle.distribution.Independent`, Independent distribution, used to create higher order distributions.

  - `paddle.distribution.TransformedDistribution`, Transform distribution, used to generate higher-order distributions through the base distribution and a series of transformations.

  - `paddle.distribution.Multionmial`, a multinomial distribution.

  - `paddle.distribution.Transform`, base class for transforming random variables.

  - `paddle.distribution.AbsTransform`, take absolute value transform.

  - `paddle.distribution.AffineTransform`, affine transform.

  - `paddle.distribution.ChainTransform`, chain combination of the transform.

  - `paddle.distribution.ExpTransform`, exponential transform.

  - `paddle.distribution.IndependentTransform`, independent transform, used to extend the `event_dim` of the transform definition field.

  - `paddle.distribution.PowerTransform`, power transform.

  - `paddle.distribution.ReshapeTransform`, `reshape` transform.

  - `paddle.distribution.SigmoidTransform`, `sigmoid` transform.

  - `paddle.distribution.SoftmaxTransform`, `softmax` transform.

  - `paddle.distribution.StackTransform`, `stack` transform, used to combine multiple transforms in a `stack` method.

  - `paddle.distribution.StickBreakingTransform`, `stickbreaking` transform.

  - `paddle.distribution.TanhTransform`, `tanh` transform.

  - `paddle.distribution.kl_divergence`, compute KL divergence.

  - `paddle.distribution.register_kl`, register user-defined KL divergence calculation function.

- Add high-level API

  - Add `paddle.vision.models.AlexNet` and `paddle.vision.models.alexnet`, to use AlexNet models directly. ([#36058](https://github.com/PaddlePaddle/Paddle/pull/36058))

  - Add `paddle.vision.models.DenseNet`, `paddle.vision.models.densenet121`, `paddle.vision.models.densenet161`, `paddle.vision.models. densenet169`, `paddle.vision.models.densenet201`, and `paddle.vision.models.densenet264`, to use DenseNet models directly. ([#36069](https://github.com/PaddlePaddle/Paddle/pull/36069))

  - Add `paddle.vision.models.GoogLeNet` and `paddle.vision.models.googlenet`, to use GoogLeNet models directly. ([#36034](https://github.com/PaddlePaddle/Paddle/pull/36034))

  - Add `paddle.vision.models.InceptionV3`, `paddle.vision.models.inception_v3`, to use InceptionV3 models directly. ([#36064](https://github.com/PaddlePaddle/Paddle/pull/36064))

  - Add `paddle.vision.models.MobileNetV3Small`, `paddle.vision.models.MobileNetV3Large`, `paddle.vision.models.mobilenet_v3_small`, and `paddle.vision.models.mobilenet_v3_large`, to use MobileNetV3 models directly. ([#38653](https://github.com/PaddlePaddle/Paddle/pull/38653))

  - Add `paddle.vision.models.resnext50_32x4d`, `paddle.vision.models.resnext50_64x4d`, `paddle.vision.models. paddle.vision.models.resnext101_32x4d`, `paddle.vision.models.resnext101_64x4d`, `paddle.vision.models.resnext152_32x4d`, and `paddle.vision.models.resnext152_64x4d`, to use ResNeXt models directly. ([#36070](https://github.com/PaddlePaddle/Paddle/pull/36070))

  - Add `paddle.vision.models.ShuffleNetV2`, `paddle.vision.models.shufflenet_v2_x0_25`, `paddle.vision.models.shufflenet_v2_x0_33`, `paddle.vision.models.shufflenet_v2_x0_5`, `paddle.vision.models.shufflenet_v2_x1_0`, `paddle.vision.models.shufflenet_v2_x1_5`, `paddle.vision.models.shufflenet_v2_x2_0`, and `paddle.vision.models.shufflenet_v2_swish`, to use ShuffleNetV2 models directly ([#36067](https://github.com/PaddlePaddle/Paddle/pull/36067))

  - Add `paddle.vision.models.SqueezeNet`, `paddle.vision.models.squeezenet1_0`, and `paddle.vision.models.squeezenet1_1`, to use SqueezeNet models directly. ([#36066](https://github.com/PaddlePaddle/Paddle/pull/36066))

  - Add `paddle.vision.models.wide_resnet50_2`, and `paddle.vision.models.wide_resnet101_2`, to use WideResNet models directly. ([#36952](https://github.com/PaddlePaddle/Paddle/pull/36952))

  - Add `paddle.vision.ops.nms` API, to support single-category and multi-category non-maximum suppression (NMS) algorithms for target detection and prediction task acceleration ([#40962](https://github.com/PaddlePaddle/Paddle/pull/40962))

  - Add `paddle.vision.ops.roi_pool` and `paddle.vision.ops.RoIPool`, to support RoI region pooling operations in detection tasks. ([#36154](https://github.com/PaddlePaddle/Paddle/pull/36154))

  - Add `paddle.vision.ops.roi_align` and `paddle.vision.ops.RoIAlign`, to support RoI Align operations in detection tasks. ([#35102](https://github.com/PaddlePaddle/Paddle/pull/36154))

  - Add `paddle.text.ViterbiDecoder`, and `paddle.text.viterbi_decode` Viterbi decoding API, mainly for sequence tagging model prediction. ([#35778](https://github.com/PaddlePaddle/Paddle/pull/35778))

- Add 11 Sparse class APIs, to support basic functions, such as creating Sparse Tensor in COO and CSR formats, and add C++ inter-converting with Tensor.

  - `paddle.sparse.sparse_coo_tensor`，create Sparse Tensor in COO format. ([#40780](https://github.com/PaddlePaddle/Paddle/pull/40780))

  - `paddle.sparse.sparse_csr_tensor`，create Sparse Tensor in CSR format. ([#40780](https://github.com/PaddlePaddle/Paddle/pull/40780))

  - `paddle.sparse.ReLU`，support ReLU activation layer for SparseCooTensor. ([#40959](https://github.com/PaddlePaddle/Paddle/pull/40959))

  - `paddle.sparse.functional.relu`，support ReLU function of SparseCooTensor. ([#40959](https://github.com/PaddlePaddle/Paddle/pull/40959))

  - `Tensor.values()`，c++ method to get non-zero elements of a SparseCooTensor or SparseCsrTensor. ([#40608](https://github.com/PaddlePaddle/Paddle/pull/40608))

  - `Tensor.indices()`，c++ method to get the coordinate information of a SparseCooTensor. ([#40608](https://github.com/PaddlePaddle/Paddle/pull/40608))

  - `Tensor.crows()`，c++ method to get information about the compressed row information of the SparseCsrTensor. ([#40608](https://github.com/PaddlePaddle/Paddle/pull/40608))

  - `Tensor.cols()`，c++ method to get the column information of the SparseCsrTensor ([#40608](https://github.com/PaddlePaddle/Paddle/pull/40608))

  - `Tensor.to_sparse_coo()`，c++ method to convert a DenseTensor or SparseCsrTensor to a SparseCooTensor. ([#40780](https://github.com/PaddlePaddle/Paddle/pull/40780))

  - `Tensor.to_sparse_csr()`，c++ convert a DenseTensor or SparseCooTensor to a SparseCsrTensor. ([#40780](https://github.com/PaddlePaddle/Paddle/pull/40780))

  - `Tensor.to_dense()`，c++ convert a SparseCooTensor or SparseCsrTensor to a DenseTensor. ([#40780](https://github.com/PaddlePaddle/Paddle/pull/40780))

- Add hardware related APIs

  - Add four GPU memory monitoring related APIs: `paddle.device.cuda.max_memory_allocated`, `paddle.device.cuda.max_memory_reserved`, `paddle.device.cuda.memory_allocated`, and `paddle.device.cuda.memory_reserved`, to view and analyze the GPU memory usage in real-time. ([#38657](https://github.com/PaddlePaddle/Paddle/pull/38657))

  - Add `paddle.device.cuda.get_device_properties`, to return the properties of the GPU device. ([#35661](https://github.com/PaddlePaddle/Paddle/pull/35661))

  - Add `paddle.device.cuda.get_device_name` and `paddle.device.cuda.get_device_capability`, to return the name and compute capability of the GPU device. ([#35672](https://github.com/PaddlePaddle/Paddle/pull/35672))

- Add Tensor operation API

  - Add `paddle.nansum`, to sum input Tensor along `axis` with ignoring the `NaNs` values. ([#38137](https://github.com/PaddlePaddle/Paddle/pull/38137))

  - Add `paddle.nanmean`,to average input Tensor along `axis` with ignoring the `NaNs` values. ([#40472](https://github.com/PaddlePaddle/Paddle/pull/40472))

  - Add `paddle.clone`, to return a copy of the input Tensor and provide gradient calculation. ([#38020](https://github.com/PaddlePaddle/Paddle/pull/38020))

  - Add `paddle.Tensor.element_size`, to return the number of bytes allocated for a single element in a Tensor. ([#38020](https://github.com/PaddlePaddle/Paddle/pull/38020))

  - Add `paddle.Tensor.to_uva_tensor`, to convert the numpy objects to be accessed by CUDA objects with virtual addresses, which are stored in CPU memory physically. ([#39146](https://github.com/PaddlePaddle/Paddle/pull/39146), [#38950](https://github.com/PaddlePaddle/Paddle/pull/38950))

  - Add `paddle.rot90`, to rotate the n-dimensional Tensor by 90 degrees along the plane specified by `axes`. ([#37634](https://github.com/PaddlePaddle/Paddle/pull/37634))

  - Add `paddle.logit` and `paddle.Tensor.logit`, to compute the logit function values for input Tensor. ([#37844](https://github.com/PaddlePaddle/Paddle/pull/37844))

  - Add `paddle.repeat_interleave`, to copy the input along the specified axis, and return a new Tensor. ([#37981](https://github.com/PaddlePaddle/Paddle/pull/37981))

  - Add `paddle.renorm`, to split the Tensor into multiple pieces at the specified `axis` and then perform p norm operations separately. ([#38130](https://github.com/PaddlePaddle/Paddle/pull/38130), [#38459](https://github.com/PaddlePaddle/Paddle/pull/38459))

  - Add `paddle.mode` and `paddle.Tensor.mode`, to search the values and indices of the input Tensor along the specified axis. ([#38446](https://github.com/PaddlePaddle/Paddle/pull/38446))

  - Add `paddle.quantile` and `paddle.Tensor.quantile`, to compute the q-quantile of a Tensor along the specified axis. ([#38567](https://github.com/PaddlePaddle/Paddle/pull/38567))

  - Add `paddle.kthvalue` and `paddle.Tensor.kthvalue`, to find the values and indices of the kth smallest at the specified axis. ([#38386](https://github.com/PaddlePaddle/Paddle/pull/38386))

  - Add `paddle.is_floating_point` and `paddle.Tensor.is_floating_point`, to determine if the input Tensor is the floating point type. ([#37885](https://github.com/PaddlePaddle/Paddle/pull/37885))

  - Add `paddle.erfinv` and `paddle.Tensor.erfinv`, to compute the inverse error function of the input Tensor. ([#38295](https://github.com/PaddlePaddle/Paddle/pull/38295))

  - Add `paddle.lerp` and `paddle.Tensor.lerp`, to compute linear interpolation among the input Tensors based on the given weights. ([#37253](https://github.com/PaddlePaddle/Paddle/pull/37253))

  - Add `paddle.angle`, to compute the phase angle of a complex Tensor. ([#37689](https://github.com/PaddlePaddle/Paddle/pull/37689))

  - Add `paddle.rad2deg` and `paddle.Tensor.rad2deg`, to convert each of the elements of input from the angles in radians to the degrees. ([#37598](https://github.com/PaddlePaddle/Paddle/pull/37598))

  - Add `paddle.deg2rad` and `paddle.Tensor.deg2rad`, to convert each of the elements of input from the degrees in radians to the angles. ([#37598](https://github.com/PaddlePaddle/Paddle/pull/37598))

  - Add `paddle.gcd` and `paddle.Tensor.gcd`, to compute the greatest common divisors of the absolute values of two inputs by element. ([#37819](https://github.com/PaddlePaddle/Paddle/pull/37819))

  - Add `paddle.lcm` and `paddle.Tensor.lcm`, to compute the least common multiple of the absolute value of two inputs by element. ([#37819](https://github.com/PaddlePaddle/Paddle/pull/37819))

  - Add `paddle.amax` and `paddle.Tensor.amax`, to get the maximum value of Tensor elements along the specified dimension. ([#38417](https://github.com/PaddlePaddle/Paddle/pull/38417))

  - Add `paddle.amin` and `paddle.Tensor.amin`, to get the minimum value of Tensor elements along the specified dimension. ([#38417](https://github.com/PaddlePaddle/Paddle/pull/38417))

  - Add `paddle.isclose`, to determine if each element of two Tensors is close to each other. ([#37135](https://github.com/PaddlePaddle/Paddle/pull/37135))

  - Add `paddle.put_along_axis` and `paddle.take_along_axis`, for extracting or placing elements with specified index subscripts. ([#38608](https://github.com/PaddlePaddle/Paddle/pull/38608))

  - Add `paddle.bincount` and `paddle.Tensor.bincount`, for counting the number of occurrences of each element in a Tensor. ([#36317](https://github.com/PaddlePaddle/Paddle/pull/36317))

  - Add `paddle.fmax` and `paddle.fmin`, to extend the max/min function to support the case of NaN values in the two Tensors. If there is one NaN value in the corresponding position, return that non-NaN value; if there are two NaN values in the corresponding position, return the NaN value. ([#37826](https://github.com/PaddlePaddle/Paddle/pull/37826))

  - Add `paddle.diff`, for computing the nth forward difference along a given dimension. It currently supports n=1. ([#37441](https://github.com/PaddlePaddle/Paddle/pull/37441))

  - Add inverse hyperbolic functions: `paddle.asinh`, `paddle.acosh`, and `paddle.atanh`. ([#37076](https://github.com/PaddlePaddle/Paddle/pull/37076))

  - Add `paddle.as_real` and `paddle.as_complex` for conversion between real Tensor and complex Tensor. ([#37784](https://github.com/PaddlePaddle/Paddle/pull/37784))

  - Add `paddle.complex`, for constructing a complex Tensor with the given real and imaginary parts. ([#37918](https://github.com/PaddlePaddle/Paddle/pull/37918), [#38272](https://github.com/PaddlePaddle/Paddle/pull/38272))

  - Add `paddle.det` and `paddle.slogdet`, to compute the determinant of a matrix and the natural logarithm of the determinant. ([#34992](https://github.com/PaddlePaddle/Paddle/pull/34992))

  - Add `paddle.nn.utils.parameters_to_vector`, to flatten parameters to a 1-D Tensor. ([#38020](https://github.com/PaddlePaddle/Paddle/pull/38020))

  - Add `paddle.nn.utils.vector_to_parameters`, to transform a Tensor with 1-D shape to the parameters. ([#38020](https://github.com/PaddlePaddle/Paddle/pull/38020))

- Add networking class APIs

  - Add `paddle.nn.Fold` and `paddle.nn.functional.fold`, to extract sliding local area blocks for the Tensors of a batch. ([#38613](https://github.com/PaddlePaddle/Paddle/pull/38613))

  - Add `paddle.nn.CELU` and `paddle.nn.functional.celu`, to support the CELU activation layer. ([#36088](https://github.com/PaddlePaddle/Paddle/pull/36088))

  - Add `paddle.nn.HingeEmbeddingLoss`. Add a way to compute hinge embedding loss. It is usually used for nonlinear embedding or semi-supervised learning. ([#37540](https://github.com/PaddlePaddle/Paddle/pull/37540))

  - Add `paddle.nn.ZeroPad2D` API, for zero-padding according to the padding property. ([#37151](https://github.com/PaddlePaddle/Paddle/pull/37151))

  - Add `paddle.nn.MaxUnPool3D` and `paddle.nn.MaxUnPool1D`, for computing 3D maximum inverse pooling and 1D maximum inverse pooling. ([#38716](https://github.com/PaddlePaddle/Paddle/pull/38716))

  - Add `paddle.incubate.graph_khop_sampler`, `paddle.incubate.graph_sample_neighbors`, and `paddle.incubate.graph_reindex` APIs, to support graph multi-order neighbor sampling and graph reindexing operations. They are mainly used for graph neural network model training. ([#39146](https://github.com/PaddlePaddle/Paddle/pull/39146), [#40809](https://github.com/PaddlePaddle/Paddle/pull/40809))

- Add random number class APIs

  - Add `paddle.poisson`, to generate a Tensor that obeys Poisson distributed with the lambda parameter. ([#38117](https://github.com/PaddlePaddle/Paddle/pull/38117))

  - Add `paddle.randint_like` API, to generate a new Tensor that obeys uniform distribution in the range [low, high), with the shape of the output matching the shape of the input. ([#36169](https://github.com/PaddlePaddle/Paddle/pull/36169))

  - Add `paddle.Tensor.exponential_`. It is an inplace style API that populates the input Tensor with exponentially distributed random numbers. ([#38256](https://github.com/PaddlePaddle/Paddle/pull/38256))

- Add parameter initialization class APIs

  - Add `paddle.nn.initializer.Dirac`, to initialize 3D/4D/5D parameters with Dirac delta functions. It is commonly used for initialization of Conv1D/Conv2D/Conv3D parameters in the convolution layer. ([#37389](https://github.com/PaddlePaddle/Paddle/pull/37389))

  - Add `paddle.nn.initializer.Orthogonal` for orthogonal matrix initialization. The initialized parameter is the (semi-) orthogonal vector. ([#37163](https://github.com/PaddlePaddle/Paddle/pull/37163))

  - Add `paddle.nn.initializer.calculate_gain`, to get the recommended gain value for the activation function. The gain value can be used to set certain initialization APIs to adjust the initialization range. ([#37163](https://github.com/PaddlePaddle/Paddle/pull/37163))

- Add learning rate class API

  - Add `paddle.optimizer.lr.MultiplicativeDecay`, to provide the `lambda` function to set the learning rate. ([#38250](https://github.com/PaddlePaddle/Paddle/pull/38250))
- Add distributed-related APIs

  - Add `paddle.incubate.optimizer.DistributedFusedLamb`, to allow the Lamb optimizer to update parameters distributedly. ([#40011](https://github.com/PaddlePaddle/Paddle/pull/40011), [#39972](https://github.com/PaddlePaddle/Paddle/pull/39972), [#39900](https://github.com/PaddlePaddle/Paddle/pull/39900), [#39747](https://github.com/PaddlePaddle/Paddle/pull/39747), [#39148](https://github.com/PaddlePaddle/Paddle/pull/39148), [#39416](https://github.com/PaddlePaddle/Paddle/pull/39416))
- Add new optimizer-related APIs([#40710](https://github.com/PaddlePaddle/Paddle/pull/40710))

  - `paddle.incubate.optimizer.functional.minimize_bfgs`，add second-order optimizer BFGS.

  - `paddle.incubate.optimizer.functional.minimize_lbfgs`，add second-order optimizer L-BFGS.

- Add `paddle.incubate.multiprocessing` module, to provide Tensor (CPU/GPU) data transfer between python processes. ([#37302](https://github.com/PaddlePaddle/Paddle/pull/37302), [#41339](https://github.com/PaddlePaddle/Paddle/pull/41339))

- Add `paddle.incubate.autotune.set_config` API, to support multi-version Kernel auto-selection, mixed precision data layout auto-conversion, and num_workers auto-selection for DataLoader to automatically improve model performance. ([#42301](https://github.com/PaddlePaddle/Paddle/pull/42301))

- Add `paddle.incubate.nn.FusedMultiTransformer` and `paddle.incubate.nn.functional.fused_multi_transformer` API, to fuse multiple layers of transformers into a single op to improve model inference performance. It should be noted that only forward is supported.  ([#42311](https://github.com/PaddlePaddle/Paddle/pull/42311))

- Add einsum_v2 operators for consistent interface between dynamic graph mode and static graph mode. It is compatible with the `paddle.einsum` implementation at the original python side, while supporting dynamic to static export and more complete Infershape inference. ([#42495](https://github.com/PaddlePaddle/Paddle/pull/42495), [#42327](https://github.com/PaddlePaddle/Paddle/pull/42327), [#42397](https://github.com/PaddlePaddle/Paddle/pull/42397), [#42105](https://github.com/PaddlePaddle/Paddle/pull/42105))


#### IR(Intermediate Representation)

- Dynamic graph to static graph

  - For the variable type StaticAnalysis module, add support for type tag similar to `a, b = paddle.shape(x)`. ([#39245](https://github.com/PaddlePaddle/Paddle/pull/39245))

  - Add a computed field, supporting `InputSpec.name` as the Program cache hash key. ([#38273](https://github.com/PaddlePaddle/Paddle/pull/38273))

  - Add syntax for supporting `dict['key'] = x.shape`. ([#40611](https://github.com/PaddlePaddle/Paddle/pull/40611))

  - Add the support for Pure FP16 training. ([#36944](https://github.com/PaddlePaddle/Paddle/pull/36944))

  - Add the support `for i in [x,y,z]` syntax. ([#37259](https://github.com/PaddlePaddle/Paddle/pull/37259))

  - Add the support for type hint syntax of python3. ([#36544](https://github.com/PaddlePaddle/Paddle/pull/36544))

- Pass development

  - Add forward and backward fusion for FC + [relu|gelu] based on NVIDIA cuBlasLt Epilogue. ([#39437](https://github.com/PaddlePaddle/Paddle/pull/39437))
- Kernel Primitive API

  - Add KP operators on GPU platform, including cast, scale, clip, bce_loss, abs_grad, reduce_sum_grad, reduce_mean_grad, clip, bce_loss, full, full_like, distribution, random, masked_select_kernel, where_index, masked_select_grad, dropout, sigmoid, where, and abs_grad. ([#36203](https://github.com/PaddlePaddle/Paddle/pull/36203), [#36423](https://github.com/PaddlePaddle/Paddle/pull/36423), [#39390](https://github.com/PaddlePaddle/Paddle/pull/39390), [#39734](https://github.com/PaddlePaddle/Paddle/pull/39734), [#38500](https://github.com/PaddlePaddle/Paddle/pull/38500), [#38959](https://github.com/PaddlePaddle/Paddle/pull/38959), [#39197](https://github.com/PaddlePaddle/Paddle/pull/39197/), [#39563](https://github.com/PaddlePaddle/Paddle/pull/39563), [#39666](https://github.com/PaddlePaddle/Paddle/pull/39666), [#40517](https://github.com/PaddlePaddle/Paddle/pull/40517), [#40617](https://github.com/PaddlePaddle/Paddle/pull/40617), [#40766](https://github.com/PaddlePaddle/Paddle/pull/40766), [#39898](https://github.com/PaddlePaddle/Paddle/pull/39898), [#39609](https://github.com/PaddlePaddle/Paddle/pull/39609))

  - Add the support for XPU2 source code compilation mode. ([#37254](https://github.com/PaddlePaddle/Paddle/pull/37254), [#40397](https://github.com/PaddlePaddle/Paddle/pull/40397), [#38455](https://github.com/PaddlePaddle/Paddle/pull/38455))

  - Add the support for KP operator reuse on XPU2 and GPU, including reduce, broadcast, elementwise_add, `exp、log、relu、sigmoid、leaky_relu、softplus、hard_swish、reciprocal`。([#36904](https://github.com/PaddlePaddle/Paddle/pull/36904), [#37226](https://github.com/PaddlePaddle/Paddle/pull/37226), [#38918](https://github.com/PaddlePaddle/Paddle/pull/38918), [#40560](https://github.com/PaddlePaddle/Paddle/pull/40560/), [#39787](https://github.com/PaddlePaddle/Paddle/pull/39787), [#39917](https://github.com/PaddlePaddle/Paddle/pull/39917), [#40002](https://github.com/PaddlePaddle/Paddle/pull/40002), [#40364](https://github.com/PaddlePaddle/Paddle/pull/40364))

  - Add unit tests of KP operators on the XPU2 platform, including `brelu、ceil、celu、elu、floor、hard_shrink、hard_sigmoid、log1p、logsigmoid、relu6、silu、soft_relu、softsign、sqrt、square、swish、thresholded_relu、softshrink`。([#40448](https://github.com/PaddlePaddle/Paddle/pull/40448), [#40524](https://github.com/PaddlePaddle/Paddle/pull/40524))

  - Add the support for XPU2 KP models, including resnet50, deepfm, wide_deep, yolov3-darknet53, det_mv3_db, bert, transformer, mobilenet_v3, and GPT2.


#### **Mixed Precision Training**

- Split the `paddle.amp.GradScaler.unscale_` method from the `minimize` of the mixed precision training `paddle.amp.GradScaler`, to provide a separate interface for recovering the loss. ([#35825](https://github.com/PaddlePaddle/Paddle/pull/35825))

- Add the FP16 support for `paddle.nn.ClipByGlobalNorm` dynamic graph mode. Add FP16 Kernel for clip op to enable clip-related operations to support FP16 compute. ([#36198](https://github.com/PaddlePaddle/Paddle/pull/36198), [#36577](https://github.com/PaddlePaddle/Paddle/pull/36577))

- Support the case that the `optimizer` parameter transferred from `paddle.amp.decorate` is Nan. ([#37541](https://github.com/PaddlePaddle/Paddle/pull/37541))

- For the merged_momentum op，add the support of input multiple learning rates, the computing for use_nesterov policy and the regularization computing. ([#37527](https://github.com/PaddlePaddle/Paddle/pull/37527))

- Add multi_tensor policy to `paddle.optimizer.Momentum` optimizer. Add `set_to_zero` branch to `clear_grad` of `Optimzizer` class. ([#37564](https://github.com/PaddlePaddle/Paddle/pull/37564))

- Add multi_tensor policy to `paddle.optimizer.Adam`. ([#38010](https://github.com/PaddlePaddle/Paddle/pull/38010))

- Add multi_precision policy to `paddle.optimizer.SGD` optimizer. ([#38231](https://github.com/PaddlePaddle/Paddle/pull/38231))

- Add the storage `master weight` parameter to the optimizer `state_dict` method. ([#39121](https://github.com/PaddlePaddle/Paddle/pull/39121))

- Add support for op CUDA bfloat16 mixed precision training. Support for O1 and O2 modes. Enable the above training modes via `paddle.amp.auto_cast`. ([#39029](https://github.com/PaddlePaddle/Paddle/pull/39029), [#39815](https://github.com/PaddlePaddle/Paddle/pull/39815))

- Add bfloat16 CUDA Kernel for the following ops: matmul, concat, split, dropout, reshape, slice, squeeze, stack, transpose, unbind, elementwize_max, elementwize_add, elementwize_mul, elementwize_sub, scale, sum, layer_norm, p_norm, reduce_sum, softmax, log_softmax, sigmoid, sqrt, softplus, square, gaussian_random, fill_constant, and fill_any_like. ([#39485](https://github.com/PaddlePaddle/Paddle/pull/39485), [#39380](https://github.com/PaddlePaddle/Paddle/pull/39380), [#39395](https://github.com/PaddlePaddle/Paddle/pull/39380), [#39402](https://github.com/PaddlePaddle/Paddle/pull/39402), [#39457](https://github.com/PaddlePaddle/Paddle/pull/39457), [#39461](https://github.com/PaddlePaddle/Paddle/pull/39461), [#39602](https://github.com/PaddlePaddle/Paddle/pull/39602), [#39716](https://github.com/PaddlePaddle/Paddle/pull/39716), [#39683](https://github.com/PaddlePaddle/Paddle/pull/39683), [#39843](https://github.com/PaddlePaddle/Paddle/pull/39843), [#39999](https://github.com/PaddlePaddle/Paddle/pull/39999), [#40004](https://github.com/PaddlePaddle/Paddle/pull/40004), [#40027](https://github.com/PaddlePaddle/Paddle/pull/40027))

- Add bfloat16 CPU Kernel for the following ops: dropout, reshape, slice, squeeze, unsqueeze, stack, transpose, unbind, elementwize_max, elementwise_mul, elementwise_sub, and gather. ([#39380](https://github.com/PaddlePaddle/Paddle/pull/39380), [#39395](https://github.com/PaddlePaddle/Paddle/pull/39380), [#39402](https://github.com/PaddlePaddle/Paddle/pull/39402), [#39457](https://github.com/PaddlePaddle/Paddle/pull/39457), [#39461](https://github.com/PaddlePaddle/Paddle/pull/39461), [#39602](https://github.com/PaddlePaddle/Paddle/pull/39602), [#39716](https://github.com/PaddlePaddle/Paddle/pull/39716), [#39683](https://github.com/PaddlePaddle/Paddle/pull/39683))

- Support printing of Tensor with data of bfloat16. ([#39375](https://github.com/PaddlePaddle/Paddle/pull/39375), [#39370](https://github.com/PaddlePaddle/Paddle/pull/39370))

- Add support for FP16 computation for `p_norm`, `elementwise_max`, and `fill_constant_batch_size_like ``scatter`. ([#35888](https://github.com/PaddlePaddle/Paddle/pull/35888), [#39907](https://github.com/PaddlePaddle/Paddle/pull/39907), [#38136](https://github.com/PaddlePaddle/Paddle/pull/38136), [#38499](https://github.com/PaddlePaddle/Paddle/pull/38499))

- Add support for int16_t for the following ops: cumsum, less_than, less_equal, greater_than, greater_equal, equal, not_equal, fill_any_like, grather_nd reduce_sum, where_index, reshape, and unsqueeze. ([#39636](https://github.com/PaddlePaddle/Paddle/pull/39636))

- Add support for int16_t label type for cross_entropy op. ([#39409](https://github.com/PaddlePaddle/Paddle/pull/39409))

- Add support for int16_t id type for embedding op. ([#39381](https://github.com/PaddlePaddle/Paddle/pull/39381))

- Add support for FP16 type for reduce_mean op. ([#38289](https://github.com/PaddlePaddle/Paddle/pull/38289))

- Add support for FP16 type for elementwise_min op. ([#38123](https://github.com/PaddlePaddle/Paddle/pull/38123))

- Update bfloat16 AMP oneDNN default support list. ([#39304](https://github.com/PaddlePaddle/Paddle/pull/39304))


#### **Paddle HIgh reusability operator library**

We announce PHI as the new Paddle HIgh reusability operator library. PHI provides Primitive API, enabling kernel reuse for operator development. As a refactored functional operator library, PHI aims to solve legacy problems that harm the framework's performance and reusability, in particular on the operator development. Such problems include inefficient ways of cross using operators, unclear operator interfaces and lacking direct calls to the operator library in C++. With PHI, new operators can be easily implemented by composing functions available in the functional library. The library provides over 200 C++ operator class APIs and nearly 500 kernels. Composing new operators through these built-in functions can greatly reduce the user's development effort. PHI supports different types of hardware (e.g., GPU and XPU). In addition, PHI is extensible with plugins for accommodating third party accelerators (such as NPU) in a low cost and reusable fashion. In short, PHI supports low level operator composabilty, the reuse of kernels through Primitives, and accelerators through plugins.The main contents include six parts as below:

- **The implementation of the operator library infrastructure, core components and mechanisms**: The directory structure of the new operator library is reasonably planned, design and implement the common base data structure of the new operator library, the new functional InferMeta and Kernel development paradigm and the corresponding registration and management components. Support the automated compilation object generation and compilation dependency generation of Kernel files, allowing developers to focus only on the functional Kernel implementation, and making the development paradigm clear and concise. ([#34425](https://github.com/PaddlePaddle/Paddle/pull/34425), [#37107](https://github.com/PaddlePaddle/Paddle/pull/37107), [#36946](https://github.com/PaddlePaddle/Paddle/pull/36946), [#36948](https://github.com/PaddlePaddle/Paddle/pull/36948), [#37876](https://github.com/PaddlePaddle/Paddle/pull/37876), [#37916](https://github.com/PaddlePaddle/Paddle/pull/37916), [#37977](https://github.com/PaddlePaddle/Paddle/pull/37977), [38078](https://github.com/PaddlePaddle/Paddle/pull/38078), [#38861](https://github.com/PaddlePaddle/Paddle/pull/38861), [#39123](https://github.com/PaddlePaddle/Paddle/pull/39123), [#39131](https://github.com/PaddlePaddle/Paddle/pull/39131), [#39748](https://github.com/PaddlePaddle/Paddle/pull/39748), [#39790](https://github.com/PaddlePaddle/Paddle/pull/39790), [#39941](https://github.com/PaddlePaddle/Paddle/pull/39941), [#40239](https://github.com/PaddlePaddle/Paddle/pull/40239), [#40635](https://github.com/PaddlePaddle/Paddle/pull/40635), [#41091](https://github.com/PaddlePaddle/Paddle/pull/41091), [#37409](https://github.com/PaddlePaddle/Paddle/pull/37409), [#37942](https://github.com/PaddlePaddle/Paddle/pull/37942), [#39002](https://github.com/PaddlePaddle/Paddle/pull/39002), [#38109](https://github.com/PaddlePaddle/Paddle/pull/38109), [#37881](https://github.com/PaddlePaddle/Paddle/pull/37881), [#37517](https://github.com/PaddlePaddle/Paddle/pull/37517), [#39870](https://github.com/PaddlePaddle/Paddle/pull/39870), [#40975](https://github.com/PaddlePaddle/Paddle/pull/40975), [#39475](https://github.com/PaddlePaddle/Paddle/pull/39475), [#37304](https://github.com/PaddlePaddle/Paddle/pull/37304), #36910, #37120, #37146, #37215, #37255, #37369, #38258, #38257, #38355, #38853, #38937, #38977, #38946, #39085, #39153, #39228, #38301, #38275, #38506, #38607, #38473, #38632, #38811, #38880, #38996, #38914, #39101)

- **Operator library C++ API system construction**: design and implement yaml configuration file-based operator definition paradigm, to automatically generate more than 200 C++ operator class APIs for internal and external developers to reuse. This reduces the cost of repeated development of basic operators. ([#37668](https://github.com/PaddlePaddle/Paddle/pull/37668), [#36938](https://github.com/PaddlePaddle/Paddle/pull/36938), [#38172](https://github.com/PaddlePaddle/Paddle/pull/38172), [#38182](https://github.com/PaddlePaddle/Paddle/pull/38182), [#38311](https://github.com/PaddlePaddle/Paddle/pull/38311), [#38438](https://github.com/PaddlePaddle/Paddle/pull/38438), [#39057](https://github.com/PaddlePaddle/Paddle/pull/39057), [#39229](https://github.com/PaddlePaddle/Paddle/pull/39229), [#39281](https://github.com/PaddlePaddle/Paddle/pull/39281), [#39263](https://github.com/PaddlePaddle/Paddle/pull/39263), [#39408](https://github.com/PaddlePaddle/Paddle/pull/39408), [#39436](https://github.com/PaddlePaddle/Paddle/pull/39436), [#39482](https://github.com/PaddlePaddle/Paddle/pull/39482), [#39497](https://github.com/PaddlePaddle/Paddle/pull/39497), [#39651](https://github.com/PaddlePaddle/Paddle/pull/39651), [#39521](https://github.com/PaddlePaddle/Paddle/pull/39521), [#39760](https://github.com/PaddlePaddle/Paddle/pull/39760), [#40060](https://github.com/PaddlePaddle/Paddle/pull/40060), [#40196](https://github.com/PaddlePaddle/Paddle/pull/40196), [#40218](https://github.com/PaddlePaddle/Paddle/pull/40218), [#40640](https://github.com/PaddlePaddle/Paddle/pull/40640), [#40732](https://github.com/PaddlePaddle/Paddle/pull/40732), [#40729](https://github.com/PaddlePaddle/Paddle/pull/40729), [#40840](https://github.com/PaddlePaddle/Paddle/pull/40840), [#40867](https://github.com/PaddlePaddle/Paddle/pull/40867), [#41025](https://github.com/PaddlePaddle/Paddle/pull/41025), [#41368](https://github.com/PaddlePaddle/Paddle/pull/41368))

- **Operator library compatible with various execution systems**: Implement new InferMeta and Kernel to access the original dynamic and static graph execution system. Support the safe removal of the original OpKernel registration and migration to the new Kernel form. ([#34425](https://github.com/PaddlePaddle/Paddle/pull/34425), [#38825](https://github.com/PaddlePaddle/Paddle/pull/38825), [#38837](https://github.com/PaddlePaddle/Paddle/pull/38837), [#38842](https://github.com/PaddlePaddle/Paddle/pull/38842), [#38976](https://github.com/PaddlePaddle/Paddle/pull/38976), [#39134](https://github.com/PaddlePaddle/Paddle/pull/39134), [#39140](https://github.com/PaddlePaddle/Paddle/pull/39140), [#39135](https://github.com/PaddlePaddle/Paddle/pull/39135), [#39252](https://github.com/PaddlePaddle/Paddle/pull/39252), [#39222](https://github.com/PaddlePaddle/Paddle/pull/39222), [#39351](https://github.com/PaddlePaddle/Paddle/pull/39351))

- **Decouple the underlying data structures and tool functions of the operator library from the framework**: Relieve PHI's dependence on the framework for core data structures, lay the foundation for subsequent independent compilation of PHI, and support infrt, custom Kernel, and a series of Phi-based construction work ([#38583](https://github.com/PaddlePaddle/Paddle/pull/38583), [#39188](https://github.com/PaddlePaddle/Paddle/pull/39188), [#39560](https://github.com/PaddlePaddle/Paddle/pull/39560), [#39931](https://github.com/PaddlePaddle/Paddle/pull/39931), [#39169](https://github.com/PaddlePaddle/Paddle/pull/39169), [#38951](https://github.com/PaddlePaddle/Paddle/pull/38951), [#38898](https://github.com/PaddlePaddle/Paddle/pull/38898), [#38873](https://github.com/PaddlePaddle/Paddle/pull/38873), [#38696](https://github.com/PaddlePaddle/Paddle/pull/38696), [#38651](https://github.com/PaddlePaddle/Paddle/pull/38651), [#39359](https://github.com/PaddlePaddle/Paddle/pull/39359), [#39305](https://github.com/PaddlePaddle/Paddle/pull/39305), [#39234](https://github.com/PaddlePaddle/Paddle/pull/39234), [#39098](https://github.com/PaddlePaddle/Paddle/pull/39098), [#39120](https://github.com/PaddlePaddle/Paddle/pull/39120), [#38979](https://github.com/PaddlePaddle/Paddle/pull/38979), [#38899](https://github.com/PaddlePaddle/Paddle/pull/38899), [#38844](https://github.com/PaddlePaddle/Paddle/pull/38844), [#39714](https://github.com/PaddlePaddle/Paddle/pull/39714), [#39729](https://github.com/PaddlePaddle/Paddle/pull/39729), [#39889](https://github.com/PaddlePaddle/Paddle/pull/39889), [#39587](https://github.com/PaddlePaddle/Paddle/pull/39587), [#39558](https://github.com/PaddlePaddle/Paddle/pull/39558), [#39514](https://github.com/PaddlePaddle/Paddle/pull/39514), [#39502](https://github.com/PaddlePaddle/Paddle/pull/39502), [#39300](https://github.com/PaddlePaddle/Paddle/pull/39300), [#39246](https://github.com/PaddlePaddle/Paddle/pull/39246), [#39124](https://github.com/PaddlePaddle/Paddle/pull/39124))

- **Integration between custom operator mechanism and Phi with improvement**: support for calling over 200 C++ operator class APIs automatically generated by PHI when writing custom operators. This reduces custom operator development costs. A series of bugs are fixed. ([#37122](https://github.com/PaddlePaddle/Paddle/pull/37122), [#37276](https://github.com/PaddlePaddle/Paddle/pull/37276), [#37281](https://github.com/PaddlePaddle/Paddle/pull/37281), [#37262](https://github.com/PaddlePaddle/Paddle/pull/37281), [#37415](https://github.com/PaddlePaddle/Paddle/pull/37415), [#37423](https://github.com/PaddlePaddle/Paddle/pull/37423), [#37583](https://github.com/PaddlePaddle/Paddle/pull/37683), [#38776](https://github.com/PaddlePaddle/Paddle/pull/38776), [#39353](https://github.com/PaddlePaddle/Paddle/pull/39353), [#41072](https://github.com/PaddlePaddle/Paddle/pull/41072))

- **Operator scale migration and refactoring**: migrate about 250 high-frequency forward and backward operator Kernel to the new operator library and refactor them as a single function. Achieve the high-performance operator by encapsulating multiple base Kernel functions on the C++ side for the fast combination. Meanwhile, add the corresponding yaml operator definition, and access to the new dynamic graph execution system to improve the python API scheduling performance. The migrated and refactored operators include:

  - sqrt ([#40727](https://github.com/PaddlePaddle/Paddle/pull/40727))

  - square([#40727](https://github.com/PaddlePaddle/Paddle/pull/40727))

  - sin ([#40175](https://github.com/PaddlePaddle/Paddle/pull/40175))

  - sinh ([#40175](https://github.com/PaddlePaddle/Paddle/pull/40175))

  - elementwise_fmax([#40140](https://github.com/PaddlePaddle/Paddle/pull/40140))

  - elementwise_fmin([#40140](https://github.com/PaddlePaddle/Paddle/pull/40140))

  - pool2d([#40208](https://github.com/PaddlePaddle/Paddle/pull/40208), [#41053](https://github.com/PaddlePaddle/Paddle/pull/41053))

  - max_pool2d_with_index([#40208](https://github.com/PaddlePaddle/Paddle/pull/40208), [#41053](https://github.com/PaddlePaddle/Paddle/pull/41053))

  - pool3d([#40208](https://github.com/PaddlePaddle/Paddle/pull/40208), [#41053](https://github.com/PaddlePaddle/Paddle/pull/41053))

  - max_pool3d_with_index([#40208](https://github.com/PaddlePaddle/Paddle/pull/40208), [#41053](https://github.com/PaddlePaddle/Paddle/pull/41053))

  - fill_constant ([#36930](https://github.com/PaddlePaddle/Paddle/pull/36930), [#39465](https://github.com/PaddlePaddle/Paddle/pull/39465))

  - p_norm ([#40819](https://github.com/PaddlePaddle/Paddle/pull/40819))

  - fill_constant_batch_size_like ([#40784](https://github.com/PaddlePaddle/Paddle/pull/40784))

  - conv2d([#39354](https://github.com/PaddlePaddle/Paddle/pull/39354))

  - conv2d_transpose([#40675](https://github.com/PaddlePaddle/Paddle/pull/40675), [#41053](https://github.com/PaddlePaddle/Paddle/pull/41053))

  - conv3d([#39354](https://github.com/PaddlePaddle/Paddle/pull/39354))

  - conv3d_transpose([#40675](https://github.com/PaddlePaddle/Paddle/pull/40675), [#41053](https://github.com/PaddlePaddle/Paddle/pull/41053))

  - mish([#40727](https://github.com/PaddlePaddle/Paddle/pull/40727))

  - gather_nd ([#40090](https://github.com/PaddlePaddle/Paddle/pull/40090), [#40043](https://github.com/PaddlePaddle/Paddle/pull/40043))

  - gather ([#40500](https://github.com/PaddlePaddle/Paddle/pull/40500))

  - scatter ([#40090](https://github.com/PaddlePaddle/Paddle/pull/40090), [#40043](https://github.com/PaddlePaddle/Paddle/pull/40043))

  - scatter_nd_add ([#40090](https://github.com/PaddlePaddle/Paddle/pull/40090), [#40043](https://github.com/PaddlePaddle/Paddle/pull/40043))

  - sgd([40045](https://github.com/PaddlePaddle/Paddle/pull/40045))

  - momentum ([#41319](https://github.com/PaddlePaddle/Paddle/pull/41319))

  - rmsprop([#40994](https://github.com/PaddlePaddle/Paddle/pull/40994))

  - index_sample([#38130](https://github.com/PaddlePaddle/Paddle/pull/38130), [#38459](https://github.com/PaddlePaddle/Paddle/pull/38459),[#39905](https://github.com/PaddlePaddle/Paddle/pull/39905))

  - adam ([#40351](https://github.com/PaddlePaddle/Paddle/pull/40351))

  - layer_norm([#40193](https://github.com/PaddlePaddle/Paddle/pull/40193))

  - adagrad([#40994](https://github.com/PaddlePaddle/Paddle/pull/40994/))

  - adamax ([#40173](https://github.com/PaddlePaddle/Paddle/pull/40173))

  - adadelta ([#40173](https://github.com/PaddlePaddle/Paddle/pull/40173))

  - clip([#40602](https://github.com/PaddlePaddle/Paddle/pull/40602), [#41661](https://github.com/PaddlePaddle/Paddle/pull/41661), [#41675](https://github.com/PaddlePaddle/Paddle/pull/41675))

  - ceil ([#40913](https://github.com/PaddlePaddle/Paddle/pull/40913))

  - cos ([#40175](https://github.com/PaddlePaddle/Paddle/pull/40175))

  - atan ([#40175](https://github.com/PaddlePaddle/Paddle/pull/40175))

  - cosh ([#40175](https://github.com/PaddlePaddle/Paddle/pull/40175))

  - erf([#40388](https://github.com/PaddlePaddle/Paddle/pull/40388))

  - asin ([#40175](https://github.com/PaddlePaddle/Paddle/pull/40175))

  - acos ([#40175](https://github.com/PaddlePaddle/Paddle/pull/40175))

  - scale ([#39278](https://github.com/PaddlePaddle/Paddle/pull/39278))

  - elementwise_pow ([#40993](https://github.com/PaddlePaddle/Paddle/pull/40993))

  - elementwise_sub ([#39225](https://github.com/PaddlePaddle/Paddle/pull/39225), [#37260](https://github.com/PaddlePaddle/Paddle/pull/37260))

  - round ([#40913](https://github.com/PaddlePaddle/Paddle/pull/40913))

  - floor ([#40913](https://github.com/PaddlePaddle/Paddle/pull/40913))

  - pow ([#40913](https://github.com/PaddlePaddle/Paddle/pull/40913))

  - elementwise_floordiv ([#40993](https://github.com/PaddlePaddle/Paddle/pull/40993))

  - reciprocal([#40727](https://github.com/PaddlePaddle/Paddle/pull/40727))

  - log1p ([#40785](https://github.com/PaddlePaddle/Paddle/pull/40785))

  - allclose ([#40469](https://github.com/PaddlePaddle/Paddle/pull/40469))

  - mul ([#40833](https://github.com/PaddlePaddle/Paddle/pull/40833))

  - elementwise_max ([#40590](https://github.com/PaddlePaddle/Paddle/pull/40590))

  - elementwise_min ([#40590](https://github.com/PaddlePaddle/Paddle/pull/40590))

  - elementwise_mod ([#40590](https://github.com/PaddlePaddle/Paddle/pull/40590))

  - elementwise_add ([#39048](https://github.com/PaddlePaddle/Paddle/pull/39048), [#37043](https://github.com/PaddlePaddle/Paddle/pull/37043))

  - matmul_v2 ([#36844](https://github.com/PaddlePaddle/Paddle/pull/36844), [#38713](https://github.com/PaddlePaddle/Paddle/pull/38713))

  - elementwise_mul ([#41042](https://github.com/PaddlePaddle/Paddle/pull/41042), [#40252](https://github.com/PaddlePaddle/Paddle/pull/40252), [#37471](https://github.com/PaddlePaddle/Paddle/pull/37471))

  - elementwise_div ([#40172](https://github.com/PaddlePaddle/Paddle/pull/40172), [#40039](https://github.com/PaddlePaddle/Paddle/pull/40039), [#37418](https://github.com/PaddlePaddle/Paddle/pull/37418))

  - SelectedRows ([#39037](https://github.com/PaddlePaddle/Paddle/pull/39037), [#39087](https://github.com/PaddlePaddle/Paddle/pull/39087), [#39128](https://github.com/PaddlePaddle/Paddle/pull/39128), [#39162](https://github.com/PaddlePaddle/Paddle/pull/39162), [#39236](https://github.com/PaddlePaddle/Paddle/pull/39236))

  - fill_any_like ([#39807](https://github.com/PaddlePaddle/Paddle/pull/39807))

  - dot([#38359](https://github.com/PaddlePaddle/Paddle/pull/38359))

  - sum ([#40873](https://github.com/PaddlePaddle/Paddle/pull/40873))

  - cumsum ([#39976](https://github.com/PaddlePaddle/Paddle/pull/39976), [#40200](https://github.com/PaddlePaddle/Paddle/pull/40200))

  - diag_v2 ([#39914](https://github.com/PaddlePaddle/Paddle/pull/39914))

  - auc ([#39976](https://github.com/PaddlePaddle/Paddle/pull/39976), [#40200](https://github.com/PaddlePaddle/Paddle/pull/40200))

  - log_loss ([#39976](https://github.com/PaddlePaddle/Paddle/pull/39976), [#40200](https://github.com/PaddlePaddle/Paddle/pull/40200))

  - one_hot_v2([39876](https://github.com/PaddlePaddle/Paddle/pull/39876))

  - sigmoid_cross_entropy_with_logits ([#39976](https://github.com/PaddlePaddle/Paddle/pull/39976), [#40200](https://github.com/PaddlePaddle/Paddle/pull/40200))

  - bce_loss ([#39868](https://github.com/PaddlePaddle/Paddle/pull/39868))

  - argsort ([#40151](https://github.com/PaddlePaddle/Paddle/pull/40151))

  - arg_max ([#40222](https://github.com/PaddlePaddle/Paddle/pull/40222))

  - arg_min ([#40222](https://github.com/PaddlePaddle/Paddle/pull/40222))

  - segment_pool ([#40099](https://github.com/PaddlePaddle/Paddle/pull/40099))

  - frobenius_norm([#40707](https://github.com/PaddlePaddle/Paddle/pull/40707), [#41053](https://github.com/PaddlePaddle/Paddle/pull/41053))

  - dist ([#40178](https://github.com/PaddlePaddle/Paddle/pull/40178))

  - isnan_v2 ([#40076](https://github.com/PaddlePaddle/Paddle/pull/40076))

  - logical_and ([#39942](https://github.com/PaddlePaddle/Paddle/pull/39942))

  - logical_not ([#39942](https://github.com/PaddlePaddle/Paddle/pull/39942))

  - isfinite_v2 ([#40076](https://github.com/PaddlePaddle/Paddle/pull/40076))

  - logical_or ([#39942](https://github.com/PaddlePaddle/Paddle/pull/39942))

  - isinf_v2 ([#40076](https://github.com/PaddlePaddle/Paddle/pull/40076))

  - is_empty ([#39919](https://github.com/PaddlePaddle/Paddle/pull/39919))

  - logical_xor ([#39942](https://github.com/PaddlePaddle/Paddle/pull/39942))

  - less_than([#39970](https://github.com/PaddlePaddle/Paddle/pull/39970))

  - not_equal([#39970](https://github.com/PaddlePaddle/Paddle/pull/39970))

  - equal([#39970](https://github.com/PaddlePaddle/Paddle/pull/39970))

  - less_equal([#39970](https://github.com/PaddlePaddle/Paddle/pull/39970))

  - equal_all([#39970](https://github.com/PaddlePaddle/Paddle/pull/39970))

  - uniform_random ([#39937](https://github.com/PaddlePaddle/Paddle/pull/39937))

  - randint ([#39876](https://github.com/PaddlePaddle/Paddle/pull/39876), [#41375](https://github.com/PaddlePaddle/Paddle/pull/41375))

  - randperm ([#41265](https://github.com/PaddlePaddle/Paddle/pull/41265))

  - unbind ([#39789](https://github.com/PaddlePaddle/Paddle/pull/39789))

  - bernoulli ([#39590](https://github.com/PaddlePaddle/Paddle/pull/39590))

  - increment ([#39858](https://github.com/PaddlePaddle/Paddle/pull/39858), [#39913](https://github.com/PaddlePaddle/Paddle/pull/39913))

  - multinomial ([#39858](https://github.com/PaddlePaddle/Paddle/pull/39858), [#39913](https://github.com/PaddlePaddle/Paddle/pull/39913))

  - addmm ([#39858](https://github.com/PaddlePaddle/Paddle/pull/39858), [#39913](https://github.com/PaddlePaddle/Paddle/pull/39913))

  - cholesky ([#39858](https://github.com/PaddlePaddle/Paddle/pull/39858), [#39913](https://github.com/PaddlePaddle/Paddle/pull/39913))

  - where ([#39811](https://github.com/PaddlePaddle/Paddle/pull/39811))

  - log10 ([#40785](https://github.com/PaddlePaddle/Paddle/pull/40785))

  - log2 ([#40785](https://github.com/PaddlePaddle/Paddle/pull/40785))

  - expm1([#40727](https://github.com/PaddlePaddle/Paddle/pull/40727))

  - atan2 ([#39806](https://github.com/PaddlePaddle/Paddle/pull/39806))

  - gaussian_random ([#39932](https://github.com/PaddlePaddle/Paddle/pull/39932), [#40122](https://github.com/PaddlePaddle/Paddle/pull/40122), [#40191](https://github.com/PaddlePaddle/Paddle/pull/40191))

  - empty ([#38334](https://github.com/PaddlePaddle/Paddle/pull/38334))

  - truncated_gaussian_random ([#39971](https://github.com/PaddlePaddle/Paddle/pull/39971), [#40191](https://github.com/PaddlePaddle/Paddle/pull/40191))

  - mv ([#39861](https://github.com/PaddlePaddle/Paddle/pull/39861), [#39954](https://github.com/PaddlePaddle/Paddle/pull/39954))

  - tan ([#40175](https://github.com/PaddlePaddle/Paddle/pull/40175))

  - set_value ([#40195](https://github.com/PaddlePaddle/Paddle/pull/40195), [#40478](https://github.com/PaddlePaddle/Paddle/pull/40478), [#40636](https://github.com/PaddlePaddle/Paddle/pull/40636))

  - bitwise_and ([#40031](https://github.com/PaddlePaddle/Paddle/pull/40031))

  - bitwise_not([#40031](https://github.com/PaddlePaddle/Paddle/pull/40031))

  - bitwise_or([#40031](https://github.com/PaddlePaddle/Paddle/pull/40031))

  - poisson([#39814](https://github.com/PaddlePaddle/Paddle/pull/39814))

  - cholesky_solve([#40387](https://github.com/PaddlePaddle/Paddle/pull/40387))

  - bitwise_xor([#40031](https://github.com/PaddlePaddle/Paddle/pull/40031))

  - triangular_solve([#40417](https://github.com/PaddlePaddle/Paddle/pull/40417))

  - sigmoid ([#40626](https://github.com/PaddlePaddle/Paddle/pull/40626))

  - atanh ([#40175](https://github.com/PaddlePaddle/Paddle/pull/40175))

  - softsign([#40727](https://github.com/PaddlePaddle/Paddle/pull/40727))

  - thresholded_relu ([#40385](https://github.com/PaddlePaddle/Paddle/pull/40385))

  - tanh_shrink ([#40565](https://github.com/PaddlePaddle/Paddle/pull/40565))

  - stanh([#40727](https://github.com/PaddlePaddle/Paddle/pull/40727))

  - reduce_mean ([#37559](https://github.com/PaddlePaddle/Paddle/pull/37559))

  - reduce_max([#40225](https://github.com/PaddlePaddle/Paddle/pull/40225))

  - reduce_min ([#40374](https://github.com/PaddlePaddle/Paddle/pull/40374))

  - mean ([#40872](https://github.com/PaddlePaddle/Paddle/pull/40872), [#41319](https://github.com/PaddlePaddle/Paddle/pull/41319))

  - reduce_all ([#40374](https://github.com/PaddlePaddle/Paddle/pull/40374))

  - reduce_any ([#40374](https://github.com/PaddlePaddle/Paddle/pull/40374))

  - logsumexp ([#40790](https://github.com/PaddlePaddle/Paddle/pull/40790))

  - softshrink([#40565](https://github.com/PaddlePaddle/Paddle/pull/40565))

  - range ([#41265](https://github.com/PaddlePaddle/Paddle/pull/41265), [#40581](https://github.com/PaddlePaddle/Paddle/pull/40851))

  - stack([#40581](https://github.com/PaddlePaddle/Paddle/pull/40851))

  - tile ([#40371](https://github.com/PaddlePaddle/Paddle/pull/40371))

  - unique([#40581](https://github.com/PaddlePaddle/Paddle/pull/40851))

  - unstack([#40581](https://github.com/PaddlePaddle/Paddle/pull/40851))

  - slice([#40736](https://github.com/PaddlePaddle/Paddle/pull/40736))

  - transpose2([#39327](https://github.com/PaddlePaddle/Paddle/pull/39327))

  - unsqueeze2( [#40596](https://github.com/PaddlePaddle/Paddle/pull/40596))

  - squeeze2( [#40596](https://github.com/PaddlePaddle/Paddle/pull/40596))

  - strided_slice ([#40708](https://github.com/PaddlePaddle/Paddle/pull/40708))

  - softmax ([#39547](https://github.com/PaddlePaddle/Paddle/pull/39547))

  - leaky_relu ([#40385](https://github.com/PaddlePaddle/Paddle/pull/40385))

  - gelu ([#40393](https://github.com/PaddlePaddle/Paddle/pull/40393))

  - prelu ([#40393](https://github.com/PaddlePaddle/Paddle/pull/40393))

  - log_softmax ([#40393](https://github.com/PaddlePaddle/Paddle/pull/40393))

  - elu ([#40565](https://github.com/PaddlePaddle/Paddle/pull/40565))

  - logsigmoid ([#40626](https://github.com/PaddlePaddle/Paddle/pull/40626))

  - psroi_pool ([#40353](https://github.com/PaddlePaddle/Paddle/pull/40353), [#41173](https://github.com/PaddlePaddle/Paddle/pull/41173))

  - kthvalue([#40575](https://github.com/PaddlePaddle/Paddle/pull/40575))

  - mode ([#40571](https://github.com/PaddlePaddle/Paddle/pull/40571))

  - yolo_box([#40112](https://github.com/PaddlePaddle/Paddle/pull/40112))

  - yolov3_loss ([#40944](https://github.com/PaddlePaddle/Paddle/pull/40944))

  - temporal_shift([#40727](https://github.com/PaddlePaddle/Paddle/pull/40727))

  - depthwise_conv2d([#39354](https://github.com/PaddlePaddle/Paddle/pull/39354))

  - pad3d ([#40701](https://github.com/PaddlePaddle/Paddle/pull/40701))

  - pad( [#40012](https://github.com/PaddlePaddle/Paddle/pull/40012))

  - greater_equal([#39970](https://github.com/PaddlePaddle/Paddle/pull/39970))

  - kldiv_loss ([#39770](https://github.com/PaddlePaddle/Paddle/pull/39770))

  - isclose ([#39770](https://github.com/PaddlePaddle/Paddle/pull/39770))

  - silu ([#40565](https://github.com/PaddlePaddle/Paddle/pull/40565))

  - unfold ([#39778](https://github.com/PaddlePaddle/Paddle/pull/39778))

  - batch_norm([39347](https://github.com/PaddlePaddle/Paddle/pull/39347))

  - norm([#39324](https://github.com/PaddlePaddle/Paddle/pull/39324))

  - roi_pool ([#40574](https://github.com/PaddlePaddle/Paddle/pull/40574), [#40682](https://github.com/PaddlePaddle/Paddle/pull/40682), [#41173](https://github.com/PaddlePaddle/Paddle/pull/41173))

  - roi_align ([#40382](https://github.com/PaddlePaddle/Paddle/pull/40382), [#40556](https://github.com/PaddlePaddle/Paddle/pull/40556), [#41402](https://github.com/PaddlePaddle/Paddle/pull/41402))

  - deformable_conv ([#40700](https://github.com/PaddlePaddle/Paddle/pull/40700), [#40794](https://github.com/PaddlePaddle/Paddle/pull/40794), [#41644](https://github.com/PaddlePaddle/Paddle/pull/41644))

  - deformable_conv_v1 ([#40794](https://github.com/PaddlePaddle/Paddle/pull/40794), [#41644](https://github.com/PaddlePaddle/Paddle/pull/41644))

  - label_smooth ([#39796](https://github.com/PaddlePaddle/Paddle/pull/39796))

  - grid_sampler ([#40585](https://github.com/PaddlePaddle/Paddle/pull/40585))

  - greater_than([#39970](https://github.com/PaddlePaddle/Paddle/pull/39970))

  - pixel_shuffle ([#39949](https://github.com/PaddlePaddle/Paddle/pull/39949), [#39712](https://github.com/PaddlePaddle/Paddle/pull/39712))

  - nearest_interp_v2 ([#40855](https://github.com/PaddlePaddle/Paddle/pull/40855))

  - bilinear_interp_v2 ([#40855](https://github.com/PaddlePaddle/Paddle/pull/40855))

  - softmax_with_cross_entropy ([#40832](https://github.com/PaddlePaddle/Paddle/pull/40832))

  - rnn ([#41007](https://github.com/PaddlePaddle/Paddle/pull/41007))

  - reverse ([#40791](https://github.com/PaddlePaddle/Paddle/pull/40791))

  - trace ([#39510](https://github.com/PaddlePaddle/Paddle/pull/39510))

  - kron([#40427](https://github.com/PaddlePaddle/Paddle/pull/40427))

  - accuracy([#39982](https://github.com/PaddlePaddle/Paddle/pull/39982))

  - gather_tree ([#40082](https://github.com/PaddlePaddle/Paddle/pull/40082), [#39844](https://github.com/PaddlePaddle/Paddle/pull/39844))

  - dropout([#40148](https://github.com/PaddlePaddle/Paddle/pull/40148))

  - bincount ([#39947](https://github.com/PaddlePaddle/Paddle/pull/39947))

  - warpctc ([#41389](https://github.com/PaddlePaddle/Paddle/pull/41389), [#40023](https://github.com/PaddlePaddle/Paddle/pull/https://github.com/PaddlePaddle/Paddle/pull/40023))

  - multiplex([#40007](https://github.com/PaddlePaddle/Paddle/pull/40007), [#40102](https://github.com/PaddlePaddle/Paddle/pull/40102))

  - qr([#40007](https://github.com/PaddlePaddle/Paddle/pull/40007), [#40007](https://github.com/PaddlePaddle/Paddle/pull/40007))

  - assign_value ([#40967](https://github.com/PaddlePaddle/Paddle/pull/40967))

  - assign ([#40022](https://github.com/PaddlePaddle/Paddle/pull/40022))

  - cast ([#37610](https://github.com/PaddlePaddle/Paddle/pull/37610))

  - tril_triu([#40007](https://github.com/PaddlePaddle/Paddle/pull/40007), [#41053](https://github.com/PaddlePaddle/Paddle/pull/41053))

  - where_index ([#40255](https://github.com/PaddlePaddle/Paddle/pull/40255))

  - index_select ([#40260](https://github.com/PaddlePaddle/Paddle/pull/40260), [#41053](https://github.com/PaddlePaddle/Paddle/pull/41053))

  - roll ([#40257](https://github.com/PaddlePaddle/Paddle/pull/40257), [#41053](https://github.com/PaddlePaddle/Paddle/pull/41053))

  - cumprod (Xiong Kun [#39770](https://github.com/PaddlePaddle/Paddle/pull/39770))

  - shard_index ([#40254](https://github.com/PaddlePaddle/Paddle/pull/40254))

  - reshape2 ([#40914](https://github.com/PaddlePaddle/Paddle/pull/40914), [#39631](https://github.com/PaddlePaddle/Paddle/pull/39631), [#38833](https://github.com/PaddlePaddle/Paddle/pull/38833), [#37164](https://github.com/PaddlePaddle/Paddle/pull/37164))

  - flip ([#39822](https://github.com/PaddlePaddle/Paddle/pull/39822), [#40974](https://github.com/PaddlePaddle/Paddle/pull/40974))

  - eye ([#39712](https://github.com/PaddlePaddle/Paddle/pull/39712), [#40105](https://github.com/PaddlePaddle/Paddle/pull/40105), [#41476](https://github.com/PaddlePaddle/Paddle/pull/41476))

  - lookup_table_v2([#39901](https://github.com/PaddlePaddle/Paddle/pull/39901))

  - searchsorted([#40520](https://github.com/PaddlePaddle/Paddle/pull/40520), [#41053](https://github.com/PaddlePaddle/Paddle/pull/41053))

  - adamw ([#40351](https://github.com/PaddlePaddle/Paddle/pull/40351))

  - tanh ([#40385](https://github.com/PaddlePaddle/Paddle/pull/40385))

  - cross ([#39829](https://github.com/PaddlePaddle/Paddle/pull/39829))

  - concat ([#38955](https://github.com/PaddlePaddle/Paddle/pull/38955), [#41112](https://github.com/PaddlePaddle/Paddle/pull/41112))

  - split ([#39060](https://github.com/PaddlePaddle/Paddle/pull/39060))

  - linspace ([#40124](https://github.com/PaddlePaddle/Paddle/pull/40124))

  - huber_loss ([#39761](https://github.com/PaddlePaddle/Paddle/pull/39761))

  - hierarchical_sigmoid([#40553](https://github.com/PaddlePaddle/Paddle/pull/40553))

  - nll_loss ([#39936](https://github.com/PaddlePaddle/Paddle/pull/https://github.com/PaddlePaddle/Paddle/pull/39936))

  - graph_send_recv ([#40092](https://github.com/PaddlePaddle/Paddle/pull/40092), [#40320](https://github.com/PaddlePaddle/Paddle/pull/40320))

  - abs([#39492](https://github.com/PaddlePaddle/Paddle/pull/39492), [#39762](https://github.com/PaddlePaddle/Paddle/pull/39762))

  - exp([#40727](https://github.com/PaddlePaddle/Paddle/pull/40727))

  - rsqrt([#40727](https://github.com/PaddlePaddle/Paddle/pull/40727))

  - viterbi_decode ([#40186](https://github.com/PaddlePaddle/Paddle/pull/40186))

  - conj ([#38247](https://github.com/PaddlePaddle/Paddle/pull/38247))

  - real ([#39777](https://github.com/PaddlePaddle/Paddle/pull/39777), [#41173](https://github.com/PaddlePaddle/Paddle/pull/41173))

  - imag ([#39777](https://github.com/PaddlePaddle/Paddle/pull/39777), [#41173](https://github.com/PaddlePaddle/Paddle/pull/41173))

  - take_along_axis ([#39959](https://github.com/PaddlePaddle/Paddle/pull/39959), [#40270](https://github.com/PaddlePaddle/Paddle/pull/40270), [#40974](https://github.com/PaddlePaddle/Paddle/pull/40974))

  - put_along_axis ([#39959](https://github.com/PaddlePaddle/Paddle/pull/39959), [#40974](https://github.com/PaddlePaddle/Paddle/pull/40974))

  - lgamma ([#39770](https://github.com/PaddlePaddle/Paddle/pull/39770))

  - relu ([#40175](https://github.com/PaddlePaddle/Paddle/pull/40175))

  - maxout ([#39959](https://github.com/PaddlePaddle/Paddle/pull/39959), [#40974](https://github.com/PaddlePaddle/Paddle/pull/40974))

  - log ([#40785](https://github.com/PaddlePaddle/Paddle/pull/40785))

  - bilinear_tensor_product([#39903](https://github.com/PaddlePaddle/Paddle/pull/39903))

  - flatten_contiguous_range ([#38712](https://github.com/PaddlePaddle/Paddle/pull/38712), [#36957](https://github.com/PaddlePaddle/Paddle/pull/36957), [#41345](https://github.com/PaddlePaddle/Paddle/pull/41345))

  - matrix_rank ([#40074](https://github.com/PaddlePaddle/Paddle/pull/40074), [#40519](https://github.com/PaddlePaddle/Paddle/pull/40519), [#41466](https://github.com/PaddlePaddle/Paddle/pull/41466))

  - logit ([#37844](https://github.com/PaddlePaddle/Paddle/pull/37844))

  - lerp ([#40105](https://github.com/PaddlePaddle/Paddle/pull/40105), [#39524](https://github.com/PaddlePaddle/Paddle/pull/39524))

  - erfinv ([#39949](https://github.com/PaddlePaddle/Paddle/pull/39949), [#39712](https://github.com/PaddlePaddle/Paddle/pull/39712))

  - broadcast_tensors([#40047](https://github.com/PaddlePaddle/Paddle/pull/40047))

  - gumbel_softmax([#39873](https://github.com/PaddlePaddle/Paddle/pull/39873))

  - diagonal ([#39575](https://github.com/PaddlePaddle/Paddle/pull/39575))

  - trunc ([#39543](https://github.com/PaddlePaddle/Paddle/pull/39543), [#39772](https://github.com/PaddlePaddle/Paddle/pull/39772))

  - multi_dot ([#40038](https://github.com/PaddlePaddle/Paddle/pull/40038))

  - matrix_power ([#40231](https://github.com/PaddlePaddle/Paddle/pull/40231))

  - digamma([#39240](https://github.com/PaddlePaddle/Paddle/pull/39240))

  - masked_select([#39193](https://github.com/PaddlePaddle/Paddle/pull/39193))

  - determinant ([#40539](https://github.com/PaddlePaddle/Paddle/pull/40539))

  - eigh ([#40213](https://github.com/PaddlePaddle/Paddle/pull/40213))

  - size ([#39949](https://github.com/PaddlePaddle/Paddle/pull/39949), [#39712](https://github.com/PaddlePaddle/Paddle/pull/39712))

  - shape ([#40248](https://github.com/PaddlePaddle/Paddle/pull/40248))

  - reduce_sum([#37559](https://github.com/PaddlePaddle/Paddle/pull/37559), [#41295](https://github.com/PaddlePaddle/Paddle/pull/41295))

  - reduce_prod ([#39844](https://github.com/PaddlePaddle/Paddle/pull/39844))

  - histogram([#39496](https://github.com/PaddlePaddle/Paddle/pull/39496))

  - meshgrid ([#41411](https://github.com/PaddlePaddle/Paddle/pull/41411))

  - brelu ([#40385](https://github.com/PaddlePaddle/Paddle/pull/40385))

  - hard_swish ([#40913](https://github.com/PaddlePaddle/Paddle/pull/40913))

  - hard_shrink ([#40565](https://github.com/PaddlePaddle/Paddle/pull/40565))

  - selu ([#39819](https://github.com/PaddlePaddle/Paddle/pull/39819))

  - expand_v2 ([#39471](https://github.com/PaddlePaddle/Paddle/pull/39471))

  - top_k_v2([#40064](https://github.com/PaddlePaddle/Paddle/pull/40064))

  - expand_as_v2([#40373](https://github.com/PaddlePaddle/Paddle/pull/40373))

  - swish ([#40913](https://github.com/PaddlePaddle/Paddle/pull/40913))

  - hard_sigmoid ([#40626](https://github.com/PaddlePaddle/Paddle/pull/40626))

  - exp, det, assign, gaussian_random, matrix_rank, eye, and deformable_conv. ([#41755](https://github.com/PaddlePaddle/Paddle/pull/41755), [#41737](https://github.com/PaddlePaddle/Paddle/pull/41737))

#### **New Dynamic Graph Execution Mechanism**

To improve scheduling performance and custom development capability of the dynamic graph execution mechanism of the PaddlePaddle, we have reconstructed the underlying execution mechanism of the dynamic graph. With the new execution method, the PHI operator library can be used for efficient runtime execution. For the operators supported by the PHI operator library, switching to the new dynamic graph mode will get a significant improvement in scheduling performance. However, due to the huge workload required in the upgrade of the overall framework execution mechanism and this part of the work is coupled with a lot on the PHI operator library, we still do not use this execution method by default in this version. If you want to try it, you can switch to it by setting the environment variable `FLAGS_enable_eager_mode=1`.The details are as follows:

- **Implementation of dynamic graph execution infrastructure, core components and mechanism**: By staticizing dynamic graph-related execution codes, the original homogeneous operators constructing converted to specific calling for different PHI APIs, thus greatly optimizing the scheduling overhead. ([#36059](https://github.com/PaddlePaddle/Paddle/pull/36059), [#37323](https://github.com/PaddlePaddle/Paddle/pull/37323), [#37556](https://github.com/PaddlePaddle/Paddle/pull/37556), [#37555](https://github.com/PaddlePaddle/Paddle/pull/37555), [#37478](https://github.com/PaddlePaddle/Paddle/pull/37478), [#37458](https://github.com/PaddlePaddle/Paddle/pull/37458), [#37479](https://github.com/PaddlePaddle/Paddle/pull/37479), [#37599](https://github.com/PaddlePaddle/Paddle/pull/37599), [#37659](https://github.com/PaddlePaddle/Paddle/pull/37659), [#37654](https://github.com/PaddlePaddle/Paddle/pull/37654), [#39200](https://github.com/PaddlePaddle/Paddle/pull/39200), [#39309](https://github.com/PaddlePaddle/Paddle/pull/39309), [#39319](https://github.com/PaddlePaddle/Paddle/pull/39319), [#39414](https://github.com/PaddlePaddle/Paddle/pull/39414), [#39504](https://github.com/PaddlePaddle/Paddle/pull/39504), [#39526](https://github.com/PaddlePaddle/Paddle/pull/39526), [#39878](https://github.com/PaddlePaddle/Paddle/pull/39878), [#39963](https://github.com/PaddlePaddle/Paddle/pull/39963))

- **New dynamic graph execution mechanism sub-function development and adaptation**: support more flexible and complete dynamic graph sub-functions such as hook, pylayer, double_grad, inplace, amp, etc. ([#41396](https://github.com/PaddlePaddle/Paddle/pull/41396), [#40400](https://github.com/PaddlePaddle/Paddle/pull/40400), [#40695](https://github.com/PaddlePaddle/Paddle/pull/40695), [#41043](https://github.com/PaddlePaddle/Paddle/pull/41043), [#40915](https://github.com/PaddlePaddle/Paddle/pull/40915), [#41104](https://github.com/PaddlePaddle/Paddle/pull/41104), [#41350](https://github.com/PaddlePaddle/Paddle/pull/41350), [#41209](https://github.com/PaddlePaddle/Paddle/pull/41209), [#40830](https://github.com/PaddlePaddle/Paddle/pull/40830), [#40891](https://github.com/PaddlePaddle/Paddle/pull/40891), [#36814](https://github.com/PaddlePaddle/Paddle/pull/36814), [#37377](https://github.com/PaddlePaddle/Paddle/pull/37377), [#37193](https://github.com/PaddlePaddle/Paddle/pull/37193), [#36965](https://github.com/PaddlePaddle/Paddle/pull/36965), [#37810](https://github.com/PaddlePaddle/Paddle/pull/37810), [#36837](https://github.com/PaddlePaddle/Paddle/pull/36837), [#38488](https://github.com/PaddlePaddle/Paddle/pull/38488), [#39282](https://github.com/PaddlePaddle/Paddle/pull/39282), [#39449](https://github.com/PaddlePaddle/Paddle/pull/39449), [#39531](https://github.com/PaddlePaddle/Paddle/pull/39531), [#39638](https://github.com/PaddlePaddle/Paddle/pull/39638), [#39674](https://github.com/PaddlePaddle/Paddle/pull/39674), [#39893](https://github.com/PaddlePaddle/Paddle/pull/39893), [#40170](https://github.com/PaddlePaddle/Paddle/pull/40170), [#40693](https://github.com/PaddlePaddle/Paddle/pull/40693), [#40937](https://github.com/PaddlePaddle/Paddle/pull/40937), [#41016](https://github.com/PaddlePaddle/Paddle/pull/41016), [#41051](https://github.com/PaddlePaddle/Paddle/pull/41051), [#41121](https://github.com/PaddlePaddle/Paddle/pull/41121), [#41198](https://github.com/PaddlePaddle/Paddle/pull/41198), [#41287](https://github.com/PaddlePaddle/Paddle/pull/41287), [#41380](https://github.com/PaddlePaddle/Paddle/pull/41380), [#41306](https://github.com/PaddlePaddle/Paddle/pull/41306), [#41387](https://github.com/PaddlePaddle/Paddle/pull/41387), [#40623](https://github.com/PaddlePaddle/Paddle/pull/40623), [#40945](https://github.com/PaddlePaddle/Paddle/pull/40945), [#39282](https://github.com/PaddlePaddle/Paddle/pull/39282), [#39449](https://github.com/PaddlePaddle/Paddle/pull/39449), [#38488](https://github.com/PaddlePaddle/Paddle/pull/38488))

- **Automatic code generation mechanism for new dynamic graph execution**: When we are trying to split the computation and scheduling logic of a large number of homogeneous operators into different specific scheduling logics, we find that it is a huge workload. So we introduce a new automatic code generation logic to generate code and thus simplify the runtime logic of dynamic graphs. Meanwhile, in order to adapt to the various types of runtime logic in the previous framework, we also use some complicated compilation techniques to obtain information at runtime to generate more accurate scheduling code. ([#37574](https://github.com/PaddlePaddle/Paddle/pull/37574), [#37575](https://github.com/PaddlePaddle/Paddle/pull/37575), [#37639](https://github.com/PaddlePaddle/Paddle/pull/37639), [#37723](https://github.com/PaddlePaddle/Paddle/pull/37723), [#37753](https://github.com/PaddlePaddle/Paddle/pull/37753), [#37812](https://github.com/PaddlePaddle/Paddle/pull/37812), [#37837](https://github.com/PaddlePaddle/Paddle/pull/37837), [#37910](https://github.com/PaddlePaddle/Paddle/pull/37910), [#37943](https://github.com/PaddlePaddle/Paddle/pull/37943), [#37992](https://github.com/PaddlePaddle/Paddle/pull/37992), [#37959](https://github.com/PaddlePaddle/Paddle/pull/37959), [#38017](https://github.com/PaddlePaddle/Paddle/pull/38017), [#37969](https://github.com/PaddlePaddle/Paddle/pull/37969), [#38160](https://github.com/PaddlePaddle/Paddle/pull/38160), [#38085](https://github.com/PaddlePaddle/Paddle/pull/38085), [#38562](https://github.com/PaddlePaddle/Paddle/pull/38562), [#38573](https://github.com/PaddlePaddle/Paddle/pull/38573), [#39192](https://github.com/PaddlePaddle/Paddle/pull/39192), [#39215](https://github.com/PaddlePaddle/Paddle/pull/39215), [#39355](https://github.com/PaddlePaddle/Paddle/pull/39355), [#39358](https://github.com/PaddlePaddle/Paddle/pull/39358), [#39328](https://github.com/PaddlePaddle/Paddle/pull/39328), [#39233](https://github.com/PaddlePaddle/Paddle/pull/39233), [#39628](https://github.com/PaddlePaddle/Paddle/pull/39628), [#39767](https://github.com/PaddlePaddle/Paddle/pull/39767), [#39743](https://github.com/PaddlePaddle/Paddle/pull/39743), [#39897](https://github.com/PaddlePaddle/Paddle/pull/39897), [#39797](https://github.com/PaddlePaddle/Paddle/pull/39797), [#39997](https://github.com/PaddlePaddle/Paddle/pull/39997), [#40058](https://github.com/PaddlePaddle/Paddle/pull/40058), [#40080](https://github.com/PaddlePaddle/Paddle/pull/40080), [#40107](https://github.com/PaddlePaddle/Paddle/pull/40107), [#39962](https://github.com/PaddlePaddle/Paddle/pull/39962), [#40132](https://github.com/PaddlePaddle/Paddle/pull/40132), [#40276](https://github.com/PaddlePaddle/Paddle/pull/40276), [#40266](https://github.com/PaddlePaddle/Paddle/pull/40266), [#40480](https://github.com/PaddlePaddle/Paddle/pull/40480), [#40482](https://github.com/PaddlePaddle/Paddle/pull/40482), [#40368](https://github.com/PaddlePaddle/Paddle/pull/40368), [#40650](https://github.com/PaddlePaddle/Paddle/pull/40650), [#40815](https://github.com/PaddlePaddle/Paddle/pull/40815), [#40907](https://github.com/PaddlePaddle/Paddle/pull/40907), [#40935](https://github.com/PaddlePaddle/Paddle/pull/40935), [#41089](https://github.com/PaddlePaddle/Paddle/pull/41089))

- **New dynamic graph execution mechanism accessed into the main framework and Integration test**: we currently use some environment variables to distinguish between static graph mode and dynamic graph mode (including new dynamic graph and old dynamic graph mode). We have adapted most logics of dynamic graphs in these modes. However, there are still a lot of problems being fixed. ([#37638](https://github.com/PaddlePaddle/Paddle/pull/37638), [#37643](https://github.com/PaddlePaddle/Paddle/pull/37643), [#37653](https://github.com/PaddlePaddle/Paddle/pull/37653), [#38314](https://github.com/PaddlePaddle/Paddle/pull/38314), [#38337](https://github.com/PaddlePaddle/Paddle/pull/38337), [#38338](https://github.com/PaddlePaddle/Paddle/pull/38338), [#39164](https://github.com/PaddlePaddle/Paddle/pull/39164), [#39326](https://github.com/PaddlePaddle/Paddle/pull/39326), [#40391](https://github.com/PaddlePaddle/Paddle/pull/40391), [#40201](https://github.com/PaddlePaddle/Paddle/pull/40201), [#40854](https://github.com/PaddlePaddle/Paddle/pull/40854), [#40887](https://github.com/PaddlePaddle/Paddle/pull/40887))

- **Update some judgment logics under dynamic graphs, to support fast execution paths for dynamic graphs in compatible forms**：([#40786](https://github.com/PaddlePaddle/Paddle/pull/40786))

  - Non-static graph mode (current transition scheme): `_non_static_mode()`。

  - Determined as new dynamic graph in dynamic graph mode (recommended judgment logic): `_in_dygrah_mode()`。

  - Determined as old dynamic graph in dynamic graph mode (Not recommended. It will be deprecated in future versions): `_in_legacy_dygraph()`。

  - Enable old dynamic graph and disable new dynamic graph in dynamic graph mode: `_enable_legacy_dygraph()` or exit `_test_eager_guard()`。

  - Enable new dynamic graph and disable old dynamic graph in dynamic graph mode: `_disable_legacy_dygraph()` or with `with _test_eager_guard()`。

  - Determine in new dynamic graph in static or dynamic graph mode: `_in_eager_without_dygraph_check()`。

- **Support inplace after dynamic graph reconstruction**: input and output are the same Tensor.

  - Adapt the inplace strategy for dynamic graph reconstruction intermediate states. ([#40400](https://github.com/PaddlePaddle/Paddle/pull/40400))

  - Adapt the inplace strategy to the final state of the dynamic graph reconstruction. ([#40695](https://github.com/PaddlePaddle/Paddle/pull/40695))

  - Add inplace strategy to PyLayer function after dynamical graph reconstruction. ([#41043](https://github.com/PaddlePaddle/Paddle/pull/41043))

  - Add inplace strategy for Tensor's setitem function after dynamical graph reconstruction. ([#40915](https://github.com/PaddlePaddle/Paddle/pull/40915))

  - Add `_reset_grad_inplace_version` interface after dynamic graph reconstruction, to set the inplace version of the Tensor's gradient to 0. ([#41101](https://github.com/PaddlePaddle/Paddle/pull/41101))

  - If the value of the forward Tensor is not needed during the inverse computation (no need buffer property), the inplace version detection operation is not needed for that Tensor. For Tensor with no_need_buffer, skip the inplace version check. ([#41350](https://github.com/PaddlePaddle/Paddle/pull/41350))

  - Unify error messages for inplace version checks after and before reconstruction of dynamic graphs. ([#41209](https://github.com/PaddlePaddle/Paddle/pull/41209))

- **Support view strategy after dynamical graph reconstruction**: input and output Tensor share underlying data.

  - Adapt the view strategy for dynamic graph reconstruction intermediate states. Include `reshape`, `squeeze`, `unsqueeze`, and `flatten` APIs. ([#40830](https://github.com/PaddlePaddle/Paddle/pull/40830))

  - Adapt the view strategy for dynamic graph reconstruction final state. Include `reshape` API. ([#40891](https://github.com/PaddlePaddle/Paddle/pull/40891))

- **Add support for weakref on the python side of the new dynamic graph eager Tensor.** ([#41797](https://github.com/PaddlePaddle/Paddle/pull/41797))

- **Enhance the new dynamic graph DoubleGrad function** to support the basic DoubleGrad feature. ([#41893](https://github.com/PaddlePaddle/Paddle/pull/41893), [#41894](https://github.com/PaddlePaddle/Paddle/pull/41894), [#41895](https://github.com/PaddlePaddle/Paddle/pull/41895))

- **Add `core.eager.StringTensor` interface**, to support the construction of StringTensor on python side and the use of the StringTensor related APIs. ([#41039](https://github.com/PaddlePaddle/Paddle/pull/41039))

- **Add `_grad_name` and `_grad_value`*to `core.eager.Tensor` to return the name and value of a gradient.  ([#41990](https://github.com/PaddlePaddle/Paddle/pull/41990))

- **Add the processing of the no_need_buffer attribute for dynamic graph intermediate state.** The Tensor with the no_need_buffer attribute is skipped in the inplace backward check operation. ([#41720](https://github.com/PaddlePaddle/Paddle/pull/41720))


#### **New Static Graph Executor**

In order to solve the problem that the original static graph executor of the PaddlePaddle is not good enough for scheduling in some scenarios and it is not easy to use multiple streams, we have implemented a new static graph executor with superior performance. It is easy to take advantage of the asynchronous scheduling capabilities of multi-streams and multi-threads. The new executor is a compatible upgrade of the original executor. At present, it is used by default in single-card scenarios. Users do not need to make any changes in the training codes. It can be used automatically. Of course, we also provide an interface to switch back to the original executor. Users can switch back to the original executor by setting the environment variable: `FLAGS_USE_STANDALONE_EXECUTOR=false`. ([#41179](https://github.com/PaddlePaddle/Paddle/pull/41179)) The main contents are as follows.

- Basic components: High-performance thread pool for multi-threaded scheduling in the executor ([#35470](https://github.com/PaddlePaddle/Paddle/pull/35470), [#35930](https://github.com/PaddlePaddle/Paddle/pull/35930), [#36030](https://github.com/PaddlePaddle/Paddle/pull/36030), [#36480](https://github.com/PaddlePaddle/Paddle/pull/36480), [#36688](https://github.com/PaddlePaddle/Paddle/pull/36688), [#36740](https://github.com/PaddlePaddle/Paddle/pull/36740), [#38335](https://github.com/PaddlePaddle/Paddle/pull/38335), [#40770](https://github.com/PaddlePaddle/Paddle/pull/40770)) and thread co-op component ([#38779](https://github.com/PaddlePaddle/Paddle/pull/38779), [#40876](https://github.com/PaddlePaddle/Paddle/pull/40876), [#40912](https://github.com/PaddlePaddle/Paddle/pull/40912)). There is the timely memory recovery after operator execution ([#37642](https://github.com/PaddlePaddle/Paddle/pull/37642), [#39617](https://github.com/PaddlePaddle/Paddle/pull/39617), [#40859](https://github.com/PaddlePaddle/Paddle/pull/40859)). There is the new dependency analysis algorithm for parallel executor ([#37231](https://github.com/PaddlePaddle/Paddle/pull/37231)) etc.

- Scheduling logic: Optimize the scheduling method of operator in the executor. Support multi-stream multi-threaded asynchronous scheduling mechanism. Change transforms such as data type, device, and layout to the operator scheduling to improve performance. Support caching the selection of operator Kernel. Support the selection of new PHI operator. ([#35024](https://github.com/PaddlePaddle/Paddle/pull/35024), [#34922](https://github.com/PaddlePaddle/Paddle/pull/34922), [#35711](https://github.com/PaddlePaddle/Paddle/pull/35711), [#35928](https://github.com/PaddlePaddle/Paddle/pull/35928), [#39458](https://github.com/PaddlePaddle/Paddle/pull/39458)，[#36899](https://github.com/PaddlePaddle/Paddle/pull/36899))。

- Interface compatibility: Compatible with the user interface and functionality of the original executor, such as alignment with python interface Executor.run(), support for managing Tensor in Scope, etc. This ensures that users can switch to the new executor without perception. ([#37278](https://github.com/PaddlePaddle/Paddle/pull/37278), [#37379](https://github.com/PaddlePaddle/Paddle/pull/37379), [#37445](https://github.com/PaddlePaddle/Paddle/pull/37445), [#37510](https://github.com/PaddlePaddle/Paddle/pull/37510), [#40955](https://github.com/PaddlePaddle/Paddle/pull/40955), [#41778](https://github.com/PaddlePaddle/Paddle/pull/41178), [#41058](https://github.com/PaddlePaddle/Paddle/pull/41058), [#38584](https://github.com/PaddlePaddle/Paddle/pull/38584), [#37957](https://github.com/PaddlePaddle/Paddle/pull/37957), [#37672](https://github.com/PaddlePaddle/Paddle/pull/37672), [#37474](https://github.com/PaddlePaddle/Paddle/pull/37474), [#37085](https://github.com/PaddlePaddle/Paddle/pull/37085), [#37061](https://github.com/PaddlePaddle/Paddle/pull/37061), [#36945](https://github.com/PaddlePaddle/Paddle/pull/36945))

- Enhance debugging and error reporting in multi-threaded scenarios by capturing error reports from sub-threads and throwing them uniformly in the main thread. This can improve user experience. ([#36692](https://github.com/PaddlePaddle/Paddle/pull/36692)，[#36802](https://github.com/PaddlePaddle/Paddle/pull/36802))

- Fix the bug with the new executor communication flow resetting stream cache information in the allocator, to reduce RecordStream overhead in cross-stream scenarios. This improves performance of DeepFM models by about 8% after optimization. ([#42046](https://github.com/PaddlePaddle/Paddle/pull/42046))

- Optimize the dependency analysis method between new executor operators to improve runtime performance. Establish correct dependencies for send/recv communication operators to support pipeline parallel. ([#42009](https://github.com/PaddlePaddle/Paddle/pull/42009))



#### **Distributed Training**

- Basic functions of multi-machine multi-card parallel training based on collective communication

  - Add support for elastic training, enables scaling up and down the number of workers, enables training process resuming when node failure，to improve the fault tolerance of distributed training. ([#36684](https://github.com/PaddlePaddle/Paddle/pull/36684), [#37177](https://github.com/PaddlePaddle/Paddle/pull/37177), [#37781](https://github.com/PaddlePaddle/Paddle/pull/37781))

  - Refactor launch startup module, add `master` collaboration and node number `nnodes` definition, to improve the ease of using the distributed startup. ([#40086](https://github.com/PaddlePaddle/Paddle/pull/40086), [#40568](https://github.com/PaddlePaddle/Paddle/pull/40568), [#40782](https://github.com/PaddlePaddle/Paddle/pull/40782), [#40844](https://github.com/PaddlePaddle/Paddle/pull/40844), [#40936](https://github.com/PaddlePaddle/Paddle/pull/40936), [#41190](https://github.com/PaddlePaddle/Paddle/pull/41190), [#41314](https://github.com/PaddlePaddle/Paddle/pull/41314))

  - Add support for GPU/NPU/XPU multi-hardware heterogeneous training. ([#37613](https://github.com/PaddlePaddle/Paddle/pull/37613), [#37998](https://github.com/PaddlePaddle/Paddle/pull/37998))

  - Add fleet_executor asynchronous pipeline executor. ([#36966](https://github.com/PaddlePaddle/Paddle/pull/36966), [#37049](https://github.com/PaddlePaddle/Paddle/pull/37049), [#37087](https://github.com/PaddlePaddle/Paddle/pull/37087), [#37126](https://github.com/PaddlePaddle/Paddle/pull/37126), [#37150](https://github.com/PaddlePaddle/Paddle/pull/37150), [#37203](https://github.com/PaddlePaddle/Paddle/pull/37203), [#37167](https://github.com/PaddlePaddle/Paddle/pull/37167), [#37282](https://github.com/PaddlePaddle/Paddle/pull/37282), [#37319](https://github.com/PaddlePaddle/Paddle/pull/37319), [#37462](https://github.com/PaddlePaddle/Paddle/pull/37462), [#37507](https://github.com/PaddlePaddle/Paddle/pull/37507), [#37533](https://github.com/PaddlePaddle/Paddle/pull/37533), [#37576](https://github.com/PaddlePaddle/Paddle/pull/37576), [#37605](https://github.com/PaddlePaddle/Paddle/pull/37605), [#37691](https://github.com/PaddlePaddle/Paddle/pull/37691), [#37742](https://github.com/PaddlePaddle/Paddle/pull/37742), [#37783](https://github.com/PaddlePaddle/Paddle/pull/37783), [#37809](https://github.com/PaddlePaddle/Paddle/pull/37809), [#37862](https://github.com/PaddlePaddle/Paddle/pull/37862), [#37882](https://github.com/PaddlePaddle/Paddle/pull/37882), [#37934](https://github.com/PaddlePaddle/Paddle/pull/37934), [#38024](https://github.com/PaddlePaddle/Paddle/pull/38024), [#38083](https://github.com/PaddlePaddle/Paddle/pull/38083), [#38164](https://github.com/PaddlePaddle/Paddle/pull/38164), [#38261](https://github.com/PaddlePaddle/Paddle/pull/38261), [#38290](https://github.com/PaddlePaddle/Paddle/pull/38290), [#40607](https://github.com/PaddlePaddle/Paddle/pull/40607), [#37093](https://github.com/PaddlePaddle/Paddle/pull/37093), [#37106](https://github.com/PaddlePaddle/Paddle/pull/37106), [#37143](https://github.com/PaddlePaddle/Paddle/pull/37143), [#37338](https://github.com/PaddlePaddle/Paddle/pull/37338), [#37376](https://github.com/PaddlePaddle/Paddle/pull/37376), [#37485](https://github.com/PaddlePaddle/Paddle/pull/37485), [#37531](https://github.com/PaddlePaddle/Paddle/pull/37531), [#37623](https://github.com/PaddlePaddle/Paddle/pull/37623), [#37693](https://github.com/PaddlePaddle/Paddle/pull/37693), [#37755](https://github.com/PaddlePaddle/Paddle/pull/37755), [#37807](https://github.com/PaddlePaddle/Paddle/pull/37807), [#37889](https://github.com/PaddlePaddle/Paddle/pull/37889), [#38420](https://github.com/PaddlePaddle/Paddle/pull/38420), [#38539](https://github.com/PaddlePaddle/Paddle/pull/38539), [#36892](https://github.com/PaddlePaddle/Paddle/pull/36892), [#37084](https://github.com/PaddlePaddle/Paddle/pull/37084), [#37158](https://github.com/PaddlePaddle/Paddle/pull/37158), [#37361](https://github.com/PaddlePaddle/Paddle/pull/37361), [#37509](https://github.com/PaddlePaddle/Paddle/pull/37509), [#37603](https://github.com/PaddlePaddle/Paddle/pull/37603), [#37703](https://github.com/PaddlePaddle/Paddle/pull/37703), [#37824](https://github.com/PaddlePaddle/Paddle/pull/37824), [#38114](https://github.com/PaddlePaddle/Paddle/pull/38114), [#38322](https://github.com/PaddlePaddle/Paddle/pull/38322), [#38535](https://github.com/PaddlePaddle/Paddle/pull/38535), [#38650](https://github.com/PaddlePaddle/Paddle/pull/38650), [#38709](https://github.com/PaddlePaddle/Paddle/pull/38709), [#38799](https://github.com/PaddlePaddle/Paddle/pull/38799), [#38839](https://github.com/PaddlePaddle/Paddle/pull/38839), [#38904](https://github.com/PaddlePaddle/Paddle/pull/38904))

  - Add distributed inference function for large-scale model. ([#38795](https://github.com/PaddlePaddle/Paddle/pull/38795), [#39012](https://github.com/PaddlePaddle/Paddle/pull/39012), [#39032](https://github.com/PaddlePaddle/Paddle/pull/39032), [#39076](https://github.com/PaddlePaddle/Paddle/pull/39076), [#39194](https://github.com/PaddlePaddle/Paddle/pull/39194), [#39207](https://github.com/PaddlePaddle/Paddle/pull/39207), [#39241](https://github.com/PaddlePaddle/Paddle/pull/39241), [#39603](https://github.com/PaddlePaddle/Paddle/pull/39603), [#39758](https://github.com/PaddlePaddle/Paddle/pull/39758), [#39992](https://github.com/PaddlePaddle/Paddle/pull/39992)).

- Dynamic graph hybrid parallelism

  - Reconstruct `paddle.distributed.fleet.utils.recompute`, to support new dynamic computational graph. ([#41396](https://github.com/PaddlePaddle/Paddle/pull/41396))

  - Add pure FP16 training to support data parallelism. ([#36420](https://github.com/PaddlePaddle/Paddle/pull/36420))

  - Add MoE (Mixture of Experts) parallel strategy, to support large-scale MoE model training. ([#41092](https://github.com/PaddlePaddle/Paddle/pull/41092), [#40895](https://github.com/PaddlePaddle/Paddle/pull/40895), [#40850](https://github.com/PaddlePaddle/Paddle/pull/40580), [#39224](https://github.com/PaddlePaddle/Paddle/pull/39224))

  - Add GroupSharded parallel strategy. Support stage1, stage2, stage3, and it supports synchronous and asynchronous communication. It can be used together with the basic function combinations such as Recompute, AMP O1\O2, Offload, GroupShardedClipGrad, and GroupShardedScaler. ([#37489](https://github.com/PaddlePaddle/Paddle/pull/37489), [#37568](https://github.com/PaddlePaddle/Paddle/pull/37568), [#37707](https://github.com/PaddlePaddle/Paddle/pull/37707), [#37836](https://github.com/PaddlePaddle/Paddle/pull/37836), [#37947](https://github.com/PaddlePaddle/Paddle/pull/37947), [#38151](https://github.com/PaddlePaddle/Paddle/pull/38151), [#38407](https://github.com/PaddlePaddle/Paddle/pull/38407), [#38052](https://github.com/PaddlePaddle/Paddle/pull/38052), [#39112](https://github.com/PaddlePaddle/Paddle/pull/39112), [#38989](https://github.com/PaddlePaddle/Paddle/pull/38989), [#39171](https://github.com/PaddlePaddle/Paddle/pull/39171), [#39285](https://github.com/PaddlePaddle/Paddle/pull/39285), [#39334](https://github.com/PaddlePaddle/Paddle/pull/39334), [#39397](https://github.com/PaddlePaddle/Paddle/pull/39397), [#39581](https://github.com/PaddlePaddle/Paddle/pull/39581), [#39668](https://github.com/PaddlePaddle/Paddle/pull/39668), [#40129](https://github.com/PaddlePaddle/Paddle/pull/40129), [#40396](https://github.com/PaddlePaddle/Paddle/pull/40396), [#40488](https://github.com/PaddlePaddle/Paddle/pull/40488), [#40601](https://github.com/PaddlePaddle/Paddle/pull/40601)，[#37725](https://github.com/PaddlePaddle/Paddle/pull/37725)，[#37904](https://github.com/PaddlePaddle/Paddle/pull/37904), [#38064](https://github.com/PaddlePaddle/Paddle/pull/38064))

- Static graph hybrid parallelism

  - Add `scale_gradient` flag bit to `gradient_scale_configs` to control the position where the gradient aggregation operation averages the gradients under pipeline parallelism. ([#36384](https://github.com/PaddlePaddle/Paddle/pull/36384))

  - Under tensor parallelism, the dropout op supports the settings of deterministic random seed generators, to ensure random consistency for non-distributed variables and randomness of distributed variables. ([#36228](https://github.com/PaddlePaddle/Paddle/pull/36228))

  - NPU hybrid parallelism supports Offload, with saving 40% of NPU memory. ([#37224](https://github.com/PaddlePaddle/Paddle/pull/37224))

  - Add `force_cpu` optional parameter to the seed op, to allow dropout to read seed values directly from CPU. ([#35820](https://github.com/PaddlePaddle/Paddle/pull/35820))

  - Improve the Automatic Sparsity (ASP) sharding strategy and support the selection of sharding strategy according to the program. ([#40028](https://github.com/PaddlePaddle/Paddle/pull/40028))

- Automatic parallel

  - Add the process restart (relaunch) after automatic mapping between logical processes and physical devices. ([#37523](https://github.com/PaddlePaddle/Paddle/pull/37523), [#37326](https://github.com/PaddlePaddle/Paddle/pull/37326))

  - Improve the underlying mechanism and interface for automatic parallel to facilitate the unification of modules and add the optimized pass. ([#36617](https://github.com/PaddlePaddle/Paddle/pull/36617), [#38132](https://github.com/PaddlePaddle/Paddle/pull/38132))

  - Add unified resource representation, to support for automatic mapping between logical processes and physical devices. ([#37091](https://github.com/PaddlePaddle/Paddle/pull/37091), [#37482](https://github.com/PaddlePaddle/Paddle/pull/37482), [#37094](https://github.com/PaddlePaddle/Paddle/pull/37094))

  - Improve the distributed attribute complementation for the backward and update parts of the computation graph. ([#36744](https://github.com/PaddlePaddle/Paddle/pull/36744))

  - Add data slicing function. ([#36055](https://github.com/PaddlePaddle/Paddle/pull/36055))

  - Add tensor resharding function to reshard the tensor according to the distributed properties of the tensor and operator. ([#40865](https://github.com/PaddlePaddle/Paddle/pull/40865), [#41106](https://github.com/PaddlePaddle/Paddle/pull/41106))

  - Add the automatic conversion pass of distributed parameters when the number of resources or parallel policy changes. ([#40434](https://github.com/PaddlePaddle/Paddle/pull/40434))

  - Add GradientMerge pass to reduce the number of communications and improve training efficiency. ([#38259](https://github.com/PaddlePaddle/Paddle/pull/38259), [#40737](https://github.com/PaddlePaddle/Paddle/pull/40737))

  - Add Recompute pass to reduce the activation memory storage. ([#38920](https://github.com/PaddlePaddle/Paddle/pull/38920))

  - Add Sharding optimization pass, to support p-g-os 3 stage optimization. ([#38502](https://github.com/PaddlePaddle/Paddle/pull/38502))

  - Add AMP + FP16 optimization pass. ([#38764](https://github.com/PaddlePaddle/Paddle/pull/38764), [#40615](https://github.com/PaddlePaddle/Paddle/pull/40615))

  - Add fused QKV parallelization for Transformer class model. ([#39080](https://github.com/PaddlePaddle/Paddle/pull/39080))

  - Improve the sharding propagation for while op to ensure convergence of the fix-point algorithm. ([#39939](https://github.com/PaddlePaddle/Paddle/pull/39939), [#39086](https://github.com/PaddlePaddle/Paddle/pull/39086), [#39014](https://github.com/PaddlePaddle/Paddle/pull/39014))

  - Support training and inference for sub-block and while op control flow. ([#39612](https://github.com/PaddlePaddle/Paddle/pull/39612), [#39895](https://github.com/PaddlePaddle/Paddle/pull/39895), [#40077](https://github.com/PaddlePaddle/Paddle/pull/40077))

- Parameter Server

  - Add NaN/Inf value checking tool under GPUPS. ([#38131](https://github.com/PaddlePaddle/Paddle/pull/38131))

  - Under GPUPS, add set_date interface to adapt incremental training. ([#36194](https://github.com/PaddlePaddle/Paddle/pull/36194))

  - Under GPUPS, add asynchronous release dataset function. ([#37790](https://github.com/PaddlePaddle/Paddle/pull/37790))

  - Under GPUPS, support the Dump parameters and intermediate layers([#36157](https://github.com/PaddlePaddle/Paddle/pull/36157))；

  - Under GPUPS, support the optimizer parameter configuration. ([#39783](https://github.com/PaddlePaddle/Paddle/pull/39783), [#39849](https://github.com/PaddlePaddle/Paddle/pull/39849))

  - Under the Unified Parameter Server, refactor the base classes of each module such as communication and storage, to improve the ease of secondary development of each module. ([#41207](https://github.com/PaddlePaddle/Paddle/pull/41207), [#41022](https://github.com/PaddlePaddle/Paddle/pull/41022), [#40702](https://github.com/PaddlePaddle/Paddle/pull/40702), [#39341](https://github.com/PaddlePaddle/Paddle/pull/39341) [#39377](https://github.com/PaddlePaddle/Paddle/pull/39377), [#39191](https://github.com/PaddlePaddle/Paddle/pull/39191), [#39064](https://github.com/PaddlePaddle/Paddle/pull/39064))

  - Add evaluation metrics module under the Unified Parameter Server, to support AUC/WuAUC/MaskAUC and other evaluation metrics calculation and customizable extensions. ([#38789](https://github.com/PaddlePaddle/Paddle/pull/38789))

  - Supports XPU parameter server training on KUNLUNXIN 2. ([#41917](https://github.com/PaddlePaddle/Paddle/pull/41917), [#42266](https://github.com/PaddlePaddle/Paddle/pull/42266), [#41916](https://github.com/PaddlePaddle/Paddle/pull/41916))

#### Profiler

- Add the performance analysis module `paddle.profiler` in the Python layer: Provide the ability to collect, export, and count performance data during the training push. ([#40065](https://github.com/PaddlePaddle/Paddle/pull/40065), [#40357](https://github.com/PaddlePaddle/Paddle/pull/40357), [#40888](https://github.com/PaddlePaddle/Paddle/pull/40888))

  - `paddle.profiler.Profiler`: performance analyzer, interface for user interaction. ([#41029](https://github.com/PaddlePaddle/Paddle/pull/41029), [#41524](https://github.com/PaddlePaddle/Paddle/pull/41524), [#41157](https://github.com/PaddlePaddle/Paddle/pull/41157), [#40249](https://github.com/PaddlePaddle/Paddle/pull/40249), [#40111](https://github.com/PaddlePaddle/Paddle/pull/40111), [#39964](https://github.com/PaddlePaddle/Paddle/pull/39964), [#40133](https://github.com/PaddlePaddle/Paddle/pull/40133))

  - `paddle.profiler.RecordEvent`: provide custom punches to record time. ([#39693](https://github.com/PaddlePaddle/Paddle/pull/39693), [#39694](https://github.com/PaddlePaddle/Paddle/pull/39694), [#39695](https://github.com/PaddlePaddle/Paddle/pull/39695), [#39675](https://github.com/PaddlePaddle/Paddle/pull/39675),[#41445](https://github.com/PaddlePaddle/Paddle/pull/41445), [#41132](https://github.com/PaddlePaddle/Paddle/pull/41132))

  - `paddle.profiler.ProfilerTarget`: specify the target device for performance analysis.

  - `paddle.profiler.ProfilerState`: indicate the state of the performance analyzer.

  - `paddle.profiler.SortedKeys`: specify the sorting method of the data within the statistics form.

  - `paddle.profiler.make_scheduler`: the scheduler generating the performance analyzer state and implement the periodic control of the collection scope.

  - `paddle.profiler.export_chrome_tracing`: save performance data to a google chrome tracing file viewable by the chrome://tracing plugin. ([#39316](https://github.com/PaddlePaddle/Paddle/pull/39316), [#39984](https://github.com/PaddlePaddle/Paddle/pull/39984), [#41029](https://github.com/PaddlePaddle/Paddle/pull/41029))

  - `paddle.profiler.export_protobuf`: save performance data to a protobuf file represented by internal structure. ([#39519](https://github.com/PaddlePaddle/Paddle/pull/39519), [#39109](https://github.com/PaddlePaddle/Paddle/pull/39109), [#39474](https://github.com/PaddlePaddle/Paddle/pull/39474))

  - `paddle.profiler.load_profiler_result`: load the performance data saved to a protobuf file.

  - `paddle.profiler.Profiler` generate statistics for data reading, step overhead and throughput for the model training by specifying the `timer_only` parameter. ([#40386](https://github.com/PaddlePaddle/Paddle/pull/40386))

- Refactor Profiler underlying infrastructure in C++ layer

  - Refactor the Profiler's controller architecture. ([#38826](https://github.com/PaddlePaddle/Paddle/pull/38826), [#39230](https://github.com/PaddlePaddle/Paddle/pull/39230), [#39779](https://github.com/PaddlePaddle/Paddle/pull/39779) )

  - Add Host Tracer to collect host-side performance metrics. ([#37629](https://github.com/PaddlePaddle/Paddle/pull/39629), [#37766](https://github.com/PaddlePaddle/Paddle/pull/37766), [#37944](https://github.com/PaddlePaddle/Paddle/pull/37944), [#38280](https://github.com/PaddlePaddle/Paddle/pull/38280), [#39975](https://github.com/PaddlePaddle/Paddle/pull/39975), [#40460](https://github.com/PaddlePaddle/Paddle/pull/40460))

  - Add CUDA Tracer to collect device-side performance metrics. ([#39488](https://github.com/PaddlePaddle/Paddle/pull/39488))

  - Profiler support for grading. ([#39926](https://github.com/PaddlePaddle/Paddle/pull/39926))

- Modify the name and type of logging for op under new dynamic graph. ([#41771](https://github.com/PaddlePaddle/Paddle/pull/41771/)

- Add Kernel running statistics into profilers' summarization and optimize the summarization. ([#41989](https://github.com/PaddlePaddle/Paddle/pull/41989)

- Remove side-effect to performance in forward computing forward when Profiler is off. ([#42142](https://github.com/PaddlePaddle/Paddle/pull/42142))

#### **CINN compiler adoption**

With the recent development of PaddlePaddle's compiler, a.k.a, CINN([GitHub - PaddlePaddle/CINN: Compiler Infrastructure for Neural Networks](https://github.com/PaddlePaddle/CINN)), paddle framework has also been changed to adapt the compiler CINN features. These include the subgraph management related functions for the Paddle-CINN runtime, optimization of memory and speed performance, and bug fixing during development.

- Functions developed:

  - Subgraph op related functions:

    - Add the function to find and generate CINN subgraphs from computational graphs. ([#36345](https://github.com/PaddlePaddle/Paddle/pull/36345))

    - Add cinn_launch op as a runtime entry point to CINN. It is responsible for scheduling CINN to compile the subgraph, to initialize the data, and to execute the generated kernels. ([#36600](https://github.com/PaddlePaddle/Paddle/pull/36600))

    - Add a helper class `CinnLaunchContext` to the kernel implementation of cinn_launch op to manage the intermediate data for compiling and running subgraphs, to improve scalability and code readability. ([#37938](https://github.com/PaddlePaddle/Paddle/pull/37938))

    - Add additional fetch nodes to CINN subgraphs, thus ensuring that CINN external nodes can fetch the values of variables. ([#37172](https://github.com/PaddlePaddle/Paddle/pull/37172), [#37190](https://github.com/PaddlePaddle/Paddle/pull/37190))

    - Add the function to symbolize a CINN subgraph, which is used to topologically sort the subgraphs and return the CINN execution sequence. ([#36417](https://github.com/PaddlePaddle/Paddle/pull/36417)

    - Add `CinnCompiler` class for involking subgraphs in the CINN compiled graph that can be replaced by using CINN operators. ([#36562](https://github.com/PaddlePaddle/Paddle/pull/36562), [#36975](https://github.com/PaddlePaddle/Paddle/pull/36975))

    - Add the interface to CINN symbolization class to get the names of subgraph fetched variables to prevent fetched variables from being eliminated in compilation optimizations. ([#37218](https://github.com/PaddlePaddle/Paddle/pull/37218))

  - Checking, debugging, and PI changes related:

    - Synchronize the update of NetBuilder API name changes in CINN. ([#40392](https://github.com/PaddlePaddle/Paddle/pull/40392))

    - Add necessary log information to Paddle-CINN for better debugging. ([#36867](https://github.com/PaddlePaddle/Paddle/pull/36867))

    - Add the bidirectional conversion function between Paddle desc and CINN desc. ([#36100](https://github.com/PaddlePaddle/Paddle/pull/36100))

    - The operator implemented in CINN may not use some input variables compared to Paddle. Therefore, remove the check that the input variables must be used in the cinn_launch op. ([#37119](https://github.com/PaddlePaddle/Paddle/pull/37119))

    - Added cinn_instruction_run op for invoking CINN to execute a single generation instruction, facilitating the construction of scheduling run subgraphs on the Paddle side. ([#39435](https://github.com/PaddlePaddle/Paddle/pull/39435), [#39576](https://github.com/PaddlePaddle/Paddle/pull/39576))

    - Add control macros to Paddle for CUDA/CUBLAS/MKL/CINN pass application required to compile CINN. ([#37066](https://github.com/PaddlePaddle/Paddle/pull/37066), [#36660](https://github.com/PaddlePaddle/Paddle/pull/36660))

    - Add two control flags FLAGS_allow_cinn_ops and FLAGS_deny_cinn_ops to control the categories of CINN operators used to replace native operators during Paddle training. ([#36842](https://github.com/PaddlePaddle/Paddle/pull/36842))

- Performance optimization:

  - Speed optimization

    - Optimize the computational time consumed by CinnCacheKey. ([#37786](https://github.com/PaddlePaddle/Paddle/pull/37786), [#37317](https://github.com/PaddlePaddle/Paddle/pull/37317))

    - Cache variable scope for CINN compiled subgraphs to reduce runtime parameter construction overhead. ([#37983](https://github.com/PaddlePaddle/Paddle/pull/37983))

    - Utilize CINN's auto-tuning in case of subgraph compilation, could be enabled by flag, for further tuning of training performance. ([#41795](https://github.com/PaddlePaddle/Paddle/pull/41795))

    - Refactor the correctness check of compilation results in case of subgraph compilation to avoid repeated checks at runtime and reduce the scheduling overhead. ([#41777](https://github.com/PaddlePaddle/Paddle/pull/41777))

    - Enable TransposeFolding and GemmRewriter optimization passes by default in Paddle-CINN training. ([#41084](https://github.com/PaddlePaddle/Paddle/pull/41084))

    - Pass the cuda stream created in Paddle into CINN so that Paddle and CINN can use the same CUDA stream in cuda computing. ([#37337](https://github.com/PaddlePaddle/Paddle/pull/37337))

    - Move CINN optimization pass application logic from Paddle to CINN. ([#42047](https://github.com/PaddlePaddle/Paddle/pull/42047), [#42070](https://github.com/PaddlePaddle/Paddle/pull/42070))

  - Device memory optimization

    - Add NoNeedBufferVars to cinn_launch op to declare a list of input variables that do not require a buffer, so that the memory can be freed in advance. ([#38367](https://github.com/PaddlePaddle/Paddle/pull/38367))

    - Pass in reference count information for external variables to the subgraph, so that subgraphs within cinn_launch can reuse memory optimization passes and reduce the memory overhead in using CINN. ([#39209](https://github.com/PaddlePaddle/Paddle/pull/39209), [#39622](https://github.com/PaddlePaddle/Paddle/pull/39622))

    - Add the function to convert a collection of executable instructions generated by CINN compilation to a Paddle Graph, supporting reuse of the Paddle scheduler and memory optimization pass, further reducing the memory overhead in using CINN. ([#39724](https://github.com/PaddlePaddle/Paddle/pull/39724), [#39911](https://github.com/PaddlePaddle/Paddle/pull/39911))

    - Add Kernel of cinn_instruction_run op, to support dynamic device memory requests based on data types inferred from compilation results. ([#40920](https://github.com/PaddlePaddle/Paddle/pull/40920))

- Bug fixing:

  - Fix and optimize the generation logic of CINN subgraphs. ([#36503](https://github.com/PaddlePaddle/Paddle/pull/36503))

  - Fix the bug that Paddle-CINN does not support no-input subgraphs. ([#40814](https://github.com/PaddlePaddle/Paddle/pull/40814))

  - Fix an error reported due to CINN not being able to handle useless outputs in operators such as batch_norm. ([#36996](https://github.com/PaddlePaddle/Paddle/pull/36996))

  - Fix several bugs in CINN subgraph partitioning and symbolization, and solve problems with Paddle training accessing the CINN. ([#36739](https://github.com/PaddlePaddle/Paddle/pull/36739), [#36698](https://github.com/PaddlePaddle/Paddle/pull/36698) )

  - CINN does not yet support the control flow yet. Add logic to skip control flow when encountered. ([#40812](https://github.com/PaddlePaddle/Paddle/pull/40812))

#### **Other**

- Model quantization

  - Upgrade quantization storage format to unify quantization formats for dynamic and static graphs. ([#41041](https://github.com/PaddlePaddle/Paddle/pull/41041))

  - Add new post training quantization (PTQ): EMD and Adaround. ([#40421](https://github.com/PaddlePaddle/Paddle/pull/40421), [#38460](https://github.com/PaddlePaddle/Paddle/pull/38460))

  - Support to quantize more operations in PTQ and QAT, such as crop, split, ab, unsqueeze etc. ([#40083](https://github.com/PaddlePaddle/Paddle/pull/40083))

  - Support to quantize operators in control flow. ([#37498](https://github.com/PaddlePaddle/Paddle/pull/37498))

  - Support quantization of matmul_v2 operator. ([#36469](https://github.com/PaddlePaddle/Paddle/pull/36469))

  - Add support for quantized matmul_v2 inference on TensorRT. ([#36594](https://github.com/PaddlePaddle/Paddle/pull/36594))

- CUDA memory optimization

  - Implement multi-stream safe Allocator to support safe and efficient use of CUDA memory in asynchronous computing scenarios. ([#37290](https://github.com/PaddlePaddle/Paddle/pull/37290))

  - Add new APIs (paddle.device.cuda.max_memory_allocated, paddle.device.cuda.max_memory_reserved, paddle.device.cuda.memory_allocated and paddle.device.cuda.memory_reserved) for GPU memory monitoring in runtime. ([#38657](https://github.com/PaddlePaddle/Paddle/pull/38657))

  - Support allocate CUDA Managed Memory to train super large models in memory-constrained scenarios. ([#39075](https://github.com/PaddlePaddle/Paddle/pull/39075))

  - Add GetBasePtr interface in C++ to get device address created with *cudaMalloc*. ([#37978](https://github.com/PaddlePaddle/Paddle/pull/37978))

  - Reduce the number of free blocks in AutoGrowth Allocator to improve memory allocation performance. ([#35732](https://github.com/PaddlePaddle/Paddle/pull/35732))

  - Remove redundant Float32 temporary tensor and cast operation for tensor with data type FP16 in`initializer.Normal` and `initializer.Constant`to save 2x memory. ([#38818](https://github.com/PaddlePaddle/Paddle/pull/38818))

- High-order derivative testing for models in dynamic graphs.

  - Add third-order derivative testing for network in dynamic graphs. ([#36814](https://github.com/PaddlePaddle/Paddle/pull/36814), [#37377](https://github.com/PaddlePaddle/Paddle/pull/37377))
- Custom op: Support to custom op in ROCm(HIP) platform. ([#36771](https://github.com/PaddlePaddle/Paddle/pull/36771))

- Cost Model: Add basic Cost Model based on profiling infomation. ([#35774](https://github.com/PaddlePaddle/Paddle/pull/35774))

- Added a function to allow user to add their own layer and correspond pruning way to ASP support. ([#40253](https://github.com/PaddlePaddle/Paddle/pull/40253))

- Add string tensor data structure, allowing the framework to have the ability to represent and process string. ([#39830](https://github.com/PaddlePaddle/Paddle/pull/39830), [#40992](https://github.com/PaddlePaddle/Paddle/pull/40992))

- Add or upgrade oneDNN FP32/int8/bfloat16 Kernel, including:

  - ELU ([#37149](https://github.com/PaddlePaddle/Paddle/pull/37149))

  - exp ([#38624](https://github.com/PaddlePaddle/Paddle/pull/38624))

  - stack ([#37002](https://github.com/PaddlePaddle/Paddle/pull/37002))

  - softplus ([#36382](https://github.com/PaddlePaddle/Paddle/pull/36382))

  - round ([#39653](https://github.com/PaddlePaddle/Paddle/pull/39653))

  - shape ([#36033](https://github.com/PaddlePaddle/Paddle/pull/36033))

  - flatten and flatten2 ([#35892](https://github.com/PaddlePaddle/Paddle/pull/35892))

  - slice ([#37630](https://github.com/PaddlePaddle/Paddle/pull/37630))

  - elementwise_mul ([#40546](https://github.com/PaddlePaddle/Paddle/pull/40546))

  - elementwise_add ([#38176](https://github.com/PaddlePaddle/Paddle/pull/38176))

  - ementwise_div ([#36158](https://github.com/PaddlePaddle/Paddle/pull/36158))

  - elementwise_sub ([#35662](https://github.com/PaddlePaddle/Paddle/pull/35662))

  - roi_align ([#37848](https://github.com/PaddlePaddle/Paddle/pull/37848))

  - nearest_interp and nearest_interp_v2 ([#37985](https://github.com/PaddlePaddle/Paddle/pull/37985)，[#38622](https://github.com/PaddlePaddle/Paddle/pull/38622)，[#39490](https://github.com/PaddlePaddle/Paddle/pull/39490))

  - assembly optimized Adam ([#39158](https://github.com/PaddlePaddle/Paddle/pull/39158))

  - logsoftmax ([#39793](https://github.com/PaddlePaddle/Paddle/pull/39793))

  - activation ([#40721](https://github.com/PaddlePaddle/Paddle/pull/40721))

  - mul ([#38552](https://github.com/PaddlePaddle/Paddle/pull/38552))

  - mean ([#37104](https://github.com/PaddlePaddle/Paddle/pull/37104))

  - relu ([#36265](https://github.com/PaddlePaddle/Paddle/pull/36265))

  - pool2d ([#37081](https://github.com/PaddlePaddle/Paddle/pull/37081))

  - concat ([#35889](https://github.com/PaddlePaddle/Paddle/pull/35889))

  - conv2d ([#38507](https://github.com/PaddlePaddle/Paddle/pull/38507)，[#38938](https://github.com/PaddlePaddle/Paddle/pull/38938)，[#36284](https://github.com/PaddlePaddle/Paddle/pull/36284))

  - LayerNorm ([#40418](https://github.com/PaddlePaddle/Paddle/pull/40418))

- Add the 3-stage storage graph retrieval engine based on SSD - host memory - GPU device memory, to support large-scale graph neural network training. ([#42472](https://github.com/PaddlePaddle/Paddle/pull/42472), [#42321](https://github.com/PaddlePaddle/Paddle/pull/42321), [#42027](https://github.com/PaddlePaddle/Paddle/pull/42027))

- Add heterogeneous multi-cloud training communication module switch, implement the Send/Recv interface function, and support multiple heterogeneous cloud communication. ([#40965](https://github.com/PaddlePaddle/Paddle/pull/40965) [40911](https://github.com/PaddlePaddle/Paddle/pull/40911))

### **(2) Function optimization**

#### API

- Add backward implementation of `paddle.linalg.det `. ([#36013](https://github.com/PaddlePaddle/Paddle/pull/36013))

- Add support for mixed precision training O2 mode for `paddle.Model`, i.e., support for Pure FP16 training mode of the original dynamic/static graphs. ([#36441](https://github.com/PaddlePaddle/Paddle/pull/40962441))

- Support for self chain calls for `paddle.nn.Layer`. ([#36609](https://github.com/PaddlePaddle/Paddle/pull/36609))

- Add settings of `is_distributed` property for the `to` method of `paddle.nn.Layer` to ensure that the distributed properties remain consistent before and after network parameter transform. ([#36221](https://github.com/PaddlePaddle/Paddle/pull/36221))

- Improve the parameter conversion logic of the `to` method of `paddle.nn.Layer`, to reduce the peak memory consumption of the conversion process and improve the conversion success rate. ([#36862](https://github.com/PaddlePaddle/Paddle/pull/36862))

- Support settings of the shape of the output Tensor for `paddle.incubate.graph_send_recv` to reduce the memory usage during the actual computation. ([#40509](https://github.com/PaddlePaddle/Paddle/pull/40509))

- Add the support of int32 and int64 data types for `paddle.incubate.segment_sum`, `segment_mean`, `segment_max`, and `segment_min`. ([#40577](https://github.com/PaddlePaddle/Paddle/pull/40577))

- Add the support of the bool type for transpose op. ([#35886](https://github.com/PaddlePaddle/Paddle/pull/35886))

- Switch the `paddle.mm` underlying operator from matmul to matmul_v2. ([#35770](https://github.com/PaddlePaddle/Paddle/pull/35770))

- Support static graph mode and support the unknown shape for `paddle.einsum`. ([#40360](https://github.com/PaddlePaddle/Paddle/pull/40360))

- Support data`parallelism for paddle.nn.functional.margin_cross_entropy` and `paddle.nn.functional.class_center_sample`. ([#39852](https://github.com/PaddlePaddle/Paddle/pull/39852))

- Support input of shape [1] for `paddle.nn.functional.grid_sample`. ([#36183](https://github.com/PaddlePaddle/Paddle/pull/36183))

- Support NHWC data format for `paddle.nn.PRelu`. ([#37019](https://github.com/PaddlePaddle/Paddle/pull/37019))

- Support the fixed random state using `paddle.seed` for `paddle.nn.functional.class_center_sample`. ([#38248](https://github.com/PaddlePaddle/Paddle/pull/38248))

- Add ROCM backend support for all APIs under `paddle.fft`, and optimize CUFFT backend error messages. ([#36415](https://github.com/PaddlePaddle/Paddle/pull/36415), [#36114](https://github.com/PaddlePaddle/Paddle/pull/36114/files))

- Support the function that the slicing dimension i 0, that is, allow slicing index results to be empty. ([#37313](https://github.com/PaddlePaddle/Paddle/pull/37313))

- Support int and bool type Tensor with using bool index for `Tensor.setitem`. ([#37761](https://github.com/PaddlePaddle/Paddle/pull/37761))

- Support nearest mode for `paddle.nn.functional.interpolate` when the input shape is 5D. ([#38868](https://github.com/PaddlePaddle/Paddle/pull/38868))

- Add the support of int16 for `paddle.nn.Embedding`and`paddle.gather`. ([#40964](https://github.com/PaddlePaddle/Paddle/pull/40964), [#40052](https://github.com/PaddlePaddle/Paddle/pull/40052))

- Support data`parallelism on single machine on``CPU platform``in paddle.distributed.spawn`. ([#35745](https://github.com/PaddlePaddle/Paddle/pull/35745), [#36758](https://github.com/PaddlePaddle/Paddle/pull/36758), [#36637](https://github.com/PaddlePaddle/Paddle/pull/36637))

- Add `depthwise_conv2d` MKLDNN operator. ([#38484](https://github.com/PaddlePaddle/Paddle/pull/38484))

- Add complex types check in the static graph model for API`paddle.abs`, `paddle.transpose`, `paddle.squeeze`, `paddle.unsqueeze`, `paddle.matmul`, and `paddle.full`. ([#40113](https://github.com/PaddlePaddle/Paddle/pull/40113))

- Support tuple and list type arguments for `paddle.autograd.PyLayer`. ([#38146](https://github.com/PaddlePaddle/Paddle/pull/38146))

- Add check whether tensor is inplace and leaf when calculate gradient. ([#37931](https://github.com/PaddlePaddle/Paddle/pull/37931))

- Support HIP library for `paddle.autograd.PyLayer`. ([#38184](https://github.com/PaddlePaddle/Paddle/pull/38184))

- Support more size inputs for `paddle.take_along_axis` and `paddle.put_along_axis`, and allow index matrix shape size to be larger than array matrix shape size. ([#39072](https://github.com/PaddlePaddle/Paddle/pull/39072))

- Optimize the error report message of API `paddle.nn.Pad2D` when replicate is 0. ([#36510](https://github.com/PaddlePaddle/Paddle/pull/36510/files))

- Support pad input in tuple format for API `paddle.nn.Pad2D`. ([#35985](https://github.com/PaddlePaddle/Paddle/pull/35985/files))

- Add tdm_sample API in `paddle.distributed.InMemoryDataset` to support sampling operations in TDM algorithms. ([#37044](https://github.com/PaddlePaddle/Paddle/pull/37044))

- Add Pre-saving Hooks mechanism for `paddle.jit.save`. ([#38186](https://github.com/PaddlePaddle/Paddle/pull/38186))

- Add new higher-order differentiation-related APIs.

  - `elementwise_add`: add third-order Kernel, to support computation of third-order differentiation. ([#36508](https://github.com/PaddlePaddle/Paddle/pull/36508), [#36618](https://github.com/PaddlePaddle/Paddle/pull/36618))

  - `matmul_v2`: add third-order Kernel, to support computation of third-order differentiation. ([#36459](https://github.com/PaddlePaddle/Paddle/pull/36459))

  - `elementwise_mul`: Add third-order Kernel, to support computation of third-order differentiation. ([#37152](https://github.com/PaddlePaddle/Paddle/pull/37547))

- Improve the logic of the `paddle.amp.GradScaler` to call check_finite_and_unscale op, to eliminate the cudaMemcpy introduced by the creation of the bool variable. ([#37770](https://github.com/PaddlePaddle/Paddle/pull/37770))

- Add check for unstack and unique op in case of input Tensor with 0 elements. ([#36021](https://github.com/PaddlePaddle/Paddle/pull/36021))

- Add new multi-layer, bi-directional LSTM function that supports KUNLUNXIN 2, to improve RNN forward/backward ops, and support the use of temporal model training. ([#](https://github.com/PaddlePaddle/Paddle/pull/41781)[42076](https://github.com/PaddlePaddle/Paddle/pull/42076))

- Add bce_loss forward/backward ops for KUNLUNXIN 2. ([#41610](https://github.com/PaddlePaddle/Paddle/pull/41610))

- Add backward implementation of `paddle.linalg.det `. ([#36013](https://github.com/PaddlePaddle/Paddle/pull/36013))

#### IR(Intermediate Representation)

- Dynamic Graphs to Static Graphs

  - Optimize the behavior of the `ProgramCache.last` interface for dynamic graph to static graph so that it returns the most recently used Program instead of the final generated Program. ([#39541](https://github.com/PaddlePaddle/Paddle/pull/39541))

  - Optimize the error report message for the `paddle.reshape` API for dynamic graph to static graph, and add a new recommended usage hint. ([#40599](https://github.com/PaddlePaddle/Paddle/pull/40599))

  - Optimize the type of exception catch in the `is_api_in_module` function when transcribing dynamic code to static code. ([#40243](https://github.com/PaddlePaddle/Paddle/pull/40243))

  - Optimize the hint of error message for dynamic graph to static graph，hide warning information by default. ([#39730](https://github.com/PaddlePaddle/Paddle/pull/https://github.com/PaddlePaddle/Paddle/pull/39730))

  - Add the support of type hint syntax for dynamic graph to static graph to improve the accuracy of variable type analysis. ([#39572](https://github.com/PaddlePaddle/Paddle/pull/39572))

  - Optimize the `paddle.cond` function to allow values are equal for basic types such as bool and int. ([#37888](https://github.com/PaddlePaddle/Paddle/pull/37888))

  - Optimize the decorate function `@to_static` to allow the switch of the train/eval mode. ([#37383](https://github.com/PaddlePaddle/Paddle/pull/37383))

  - Optimize the stack of error report for dynamic graph to static graph, to highlight user-related codes and reduce the framework redundant error stack. ([#36741](https://github.com/PaddlePaddle/Paddle/pull/36741))

  - Remove `no_value` placeholder from the return value of `paddle.cond`. ([#36513](https://github.com/PaddlePaddle/Paddle/pull/36513)、[#36826](https://github.com/PaddlePaddle/Paddle/pull/36826))

  - Adapt the run_program op to the new dynamic graph mode. ([#40198](https://github.com/PaddlePaddle/Paddle/pull/40198), [#40355](https://github.com/PaddlePaddle/Paddle/pull/40355))

  - Add check for zip syntax. ([#37846](https://github.com/PaddlePaddle/Paddle/pull/https://github.com/PaddlePaddle/Paddle/pull/37846))

  - Fix the dynamic graph to static graph failure due to the error of dimension and type judgment in the `paddle.signal.frame`, `paddle.signal.stft` and `paddle.signal.istft`. ([#40113](https://github.com/PaddlePaddle/Paddle/pull/40113))

  - Add registration of plural type Kernel for mean, pad3d ops. ([#40113](https://github.com/PaddlePaddle/Paddle/pull/40113))


#### **Mixed Precision Training**

- Add GPU Compute Capability environment check for amp. Add the usage warning for GPU environments that the fail acceleration for training. ([#38086](https://github.com/PaddlePaddle/Paddle/pull/38086))

- Add check of calling order when using `paddle.amp.decorate` and `paddle.DataParallel` at the same time. ([#38785](https://github.com/PaddlePaddle/Paddle/pull/38785))


#### **Distributed Training**

- Basic functions of the distributed training

  - Optimize Fleet API and DistributedStrategy configuration to use dynamic graph parallel function conveniently. ([#40408](https://github.com/PaddlePaddle/Paddle/pull/40408))

  - Optimize Dynamic Graph mixed parallel HybridParallelClipGrad strategy, support 4D hybrid parallel and Pure FP16 training. ([#36237](https://github.com/PaddlePaddle/Paddle/pull/36237), [#36555](https://github.com/PaddlePaddle/Paddle/pull/36555))

  - Restructure dynamic graph data parallel strategy, to support new dynamic graph and communication. ([#40389](https://github.com/PaddlePaddle/Paddle/pull/40389), [#40593](https://github.com/PaddlePaddle/Paddle/pull/40593), [#40836](https://github.com/PaddlePaddle/Paddle/pull/40836), [#41119](https://github.com/PaddlePaddle/Paddle/pull/41119), [#41413](https://github.com/PaddlePaddle/Paddle/pull/41413), [#39987](https://github.com/PaddlePaddle/Paddle/pull/39987))

  - Support distributed tensor model parallel for fused_attention op. ([#40101](https://github.com/PaddlePaddle/Paddle/pull/40101))

  - Support the distributed tensor model parallel for fused_feedforward op. ([#40160](https://github.com/PaddlePaddle/Paddle/pull/40160))

- Graph retrieval engine

  - Optimize the data format returned by the graph sampling interface of the graph engine, with a 3x improvement of the sampling speed. ([#37315](https://github.com/PaddlePaddle/Paddle/pull/37315))

  - Reduce the amount of graph engine threads to improve performance. ([#37098](https://github.com/PaddlePaddle/Paddle/pull/37098))

  - Optimize graph engine data transfer to improve performance. ([#37341](https://github.com/PaddlePaddle/Paddle/pull/37341))

  - Optimize the merge logic of embedding op to improve performance by exploiting the topological relationship of embedding op in the model. [(#35942)](https://github.com/PaddlePaddle/Paddle/pull/35942)

- Communication library: restructure the communication library to improve the scalability and development of the communication library, and support heterogeneous communication. ([#41398](https://github.com/PaddlePaddle/Paddle/pull/41398), [#39720](https://github.com/PaddlePaddle/Paddle/pull/39720), [#40911](https://github.com/PaddlePaddle/Paddle/pull/40911), [#40579](https://github.com/PaddlePaddle/Paddle/pull/40579), [#40629](https://github.com/PaddlePaddle/Paddle/pull/40629), [#40437](https://github.com/PaddlePaddle/Paddle/pull/40437), [#40430](https://github.com/PaddlePaddle/Paddle/pull/40430), [#40228](https://github.com/PaddlePaddle/Paddle/pull/40228), [#40181](https://github.com/PaddlePaddle/Paddle/pull/40181), [#40100](https://github.com/PaddlePaddle/Paddle/pull/40100), [#40097](https://github.com/PaddlePaddle/Paddle/pull/40097), [#39892](https://github.com/PaddlePaddle/Paddle/pull/39892), [#39384](https://github.com/PaddlePaddle/Paddle/pull/39384), [#39737](https://github.com/PaddlePaddle/Paddle/pull/39737), [#40040](https://github.com/PaddlePaddle/Paddle/pull/40040))

- Support the publication of MoE-related interfaces in `paddle.incubate.distributed.models.moe ` (`moe.GShardGate `, `moe.BaseGate `, `moe.SwitchGate `, `moe.MoELayer `, and `moe. ClipGradForMOEByGlobalNorm `). ([#42300](https://github.com/PaddlePaddle/Paddle/pull/42300))

- Fix the error report in the use of recomputing in `paddle.incubate.distributed.models.moe.MoELayer `. ([#42128](https://github.com/PaddlePaddle/Paddle/pull/42128))

- Fix the error report in the new dynamic graph pipeline parallel caused by different data types  ([#41937](https://github.com/PaddlePaddle/Paddle/pull/41937) [#42053](https://github.com/PaddlePaddle/Paddle/pull/42053))

- Fix the error report in the new dynamic graph tensor model parallel due to different data types([#41960](https://github.com/PaddlePaddle/Paddle/pull/41960))

#### **Custom operator**

- Enhance the C++ custom operator mechanism for writing second-order gradient operators, to support adding suffixes to the gradient input variables of second-order gradient operators for use as outputs. ([#41781](https://github.com/PaddlePaddle/Paddle/pull/41781))

- Remove the use of the deprecated enumeration type `PlaceType` from the Tensor API member methods, make it compatible, and add a deprecation warning. ([#41882](https://github.com/PaddlePaddle/Paddle/pull/41882))

- Add deprecated warning for a number of deprecated interfaces of the original Tensor API, including the incomplete constructor, reshape, mutable_data, and copy_to methods. ([#41882](https://github.com/PaddlePaddle/Paddle/pull/41882))

#### **Other**

- Error report and debugging optimization

  - Optimize `the error message of the label` boundary check for the cross_entropy op. ([#40001](https://github.com/PaddlePaddle/Paddle/pull/40001))

  - Add profile record for `infer_shape` and `compute` methods of op execution of dynamic graphs, show their cost in timeline. ([#39023](https://github.com/PaddlePaddle/Paddle/pull/39023))

  - Replace `pybind::index_error` error hint on Windows for unknown exceptions. ([#40538](https://github.com/PaddlePaddle/Paddle/pull/40538))

  - Add the error message in the out-of-bounds checks for user scatter op. ([#37429](https://github.com/PaddlePaddle/Paddle/pull/37429))

- Download tool: For the problem of slow decompression of directories with multiple files in `paddle.utils.download.get_path_from_url`, replace the original way (traverse directory in loop) of decompressing files in directories one by one by calling extractall on the directory, which greatly improves the decompression speed. ([#37311](https://github.com/PaddlePaddle/Paddle/pull/37311))

- Speed up the quantization training for`fake_quantize_range_abs_max`、`fake_quantize_abs_max`、`fake_quantize_dequantize_abs_max`、 `fake_quantize_moving_average_abs_max`, etc. ([#40491](https://github.com/PaddlePaddle/Paddle/pull/40491))


### **(3) Performance optimization**

#### **Distributed Training**

- Hybrid parallel optimizer `sharding_optimizer` supports `optimize_cast` optimization, which move the parameter cast during forward and backwark stage to the optimizer stage. This improves performance by 7%. ([#35878](https://github.com/PaddlePaddle/Paddle/pull/35878))

- GPUPS optimization: support for gradient fuse allreduce training. This improves training performance by 20%. ([#35131](https://github.com/PaddlePaddle/Paddle/pull/35131))

- GPUPS optimization: dump CPU optimization speed improves by 3.21x. ([#40068](https://github.com/PaddlePaddle/Paddle/pull/40068))

- CPU parameter server streaming training optimization: support for automatic statistics of sparse parameter statistics, incremental saving of sparse parameters, etc. The training performance improves by 20%. ([#36465](https://github.com/PaddlePaddle/Paddle/pull/36465), [#36601](https://github.com/PaddlePaddle/Paddle/pull/36601), [#36734](https://github.com/PaddlePaddle/Paddle/pull/36734), [#36909](https://github.com/PaddlePaddle/Paddle/pull/36909), [#36943](https://github.com/PaddlePaddle/Paddle/pull/36943), [#37181](https://github.com/PaddlePaddle/Paddle/pull/37181), [#37194](https://github.com/PaddlePaddle/Paddle/pull/37194), [#37515](https://github.com/PaddlePaddle/Paddle/pull/37515), [#37626](https://github.com/PaddlePaddle/Paddle/pull/37626), [#37995](https://github.com/PaddlePaddle/Paddle/pull/37995), [#38582](https://github.com/PaddlePaddle/Paddle/pull/38582), [#39250](https://github.com/PaddlePaddle/Paddle/pull/39250), [#40762](https://github.com/PaddlePaddle/Paddle/pull/40762), [#41234](https://github.com/PaddlePaddle/Paddle/pull/41234), [#41320](https://github.com/PaddlePaddle/Paddle/pull/41320), [#41400](https://github.com/PaddlePaddle/Paddle/pull/41400))

#### **Auto-tuning**

Add hardware-aware automatic performance tuning for the full training process, with performance improvements of about 3% to 50% or more on image classification, segmentation, detection, and image generation tasks compared to the model's default configuration. The auto-tuning status is set via the `paddle.incubate.autotune.set_config ` API. By default, it is currently disabled. Auto-tuning has three specific levels:

- Add the auto-tuning function to `paddle.io.DataLoader `, to select the best num_workers based on training data and device resources.  ([#42004](https://github.com/PaddlePaddle/Paddle/pull/42004))

- Add mixed-precision training data layout auto-tuning feature, to select the best data layout based on device type and data type, and automatically convert it at runtime. ([#41964](https://github.com/PaddlePaddle/Paddle/pull/41964))

- Add the automatic tuning of the required workspace size threshold for Conv, which is automatically set based on the GPU's currently available requested device memory resources. Add the automatic selection of Conv cuDNN algorithms based on the generic AlgorithmCache design and Kernel timing component, which supports data variation length models. ([#41833](https://github.com/PaddlePaddle/Paddle/pull/41833))

#### **Operator Optimization**

- Optimize `FasterTokenizer` performance, with a 10% performance improvement compared to pre-optimization. ([#36701](https://github.com/PaddlePaddle/Paddle/pull/36701))

- Optimize `index_select` inverse computation, with 3.7~25.2x performance improvement over pre-optimization. ([#37055](https://github.com/PaddlePaddle/Paddle/pull/37055))

- Optimize the performance of `paddle.nn.ClipByGlobalNorm`. Take 10*10 `paddle.nn.Linear` as an example. In contrast to pre-optimization, the performance improves by about 30%. ([#38209](https://github.com/PaddlePaddle/Paddle/pull/38209))

- Optimize the performance of `pnorm` with very large or very small `axis` dimensions, with 31-96x improvement in forward speed and 1.1-19x improvement in backward speed. ([#37685](https://github.com/PaddlePaddle/Paddle/pull/37685), [#38215](https://github.com/PaddlePaddle/Paddle/pull/38215), [#39011](https://github.com/PaddlePaddle/Paddle/pull/39011))

- Optimize `softmax` forward and backward performance, with a speedup ratio of about 2x for the `axis!=-1` configuration. ([#38602](https://github.com/PaddlePaddle/Paddle/pull/38602), [#38609](https://github.com/PaddlePaddle/Paddle/pull/38609), [#32387](https://github.com/PaddlePaddle/Paddle/pull/32387), [#37927](https://github.com/PaddlePaddle/Paddle/pull/37927/files))

- Optimize `log_softmax` forward and backward performance, with a speedup ratio of about 6x to 20x for `axis!=-1` configurations. ([#38992](https://github.com/PaddlePaddle/Paddle/pull/38992), [#40612](https://github.com/PaddlePaddle/Paddle/pull/40612))

- Optimize `softmax_with_cross_entropy` forward and backward performance, with a speedup ratio of about 1.3x for the `hard_label` configuration. ([#39553](https://github.com/PaddlePaddle/Paddle/pull/39553), [#40424](https://github.com/PaddlePaddle/Paddle/pull/40424), [#40643](https://github.com/PaddlePaddle/Paddle/pull/40643))

- Optimize `top_k` performance, with a speedup ratio of more than 22x for one-dimension and larger `k` (k=5000) configuration. ([#40941](https://github.com/PaddlePaddle/Paddle/pull/40941))

- Optimize `elementwise_mul` backward computation, with 1.85~12.16x performance improvement over pre-optimization. ([#37728](https://github.com/PaddlePaddle/Paddle/pull/37728))

- Optimize `elementwise_min` and `elementwise_max` backward computation, to equalize or improve performance by 1.05x to 18.75x over pre-optimization. ([#38236](https://github.com/PaddlePaddle/Paddle/pull/38236), [#37906](https://github.com/PaddlePaddle/Paddle/pull/37906))

- Optimize `nearest_interp` forward and backward computation, with forward performance improvement by 1.5x to 2.3x over pre-optimization, and backward performance improvement by 60% to 1.8x over pre-optimization. ([#38528](https://github.com/PaddlePaddle/Paddle/pull/38528), [#39067](https://github.com/PaddlePaddle/Paddle/pull/39067))

- Optimize `bilinear_interp` forward and backward computation, with forward performance improvement by 0.4x to 2.3x over pre-optimization, and backward performance improvement by 10%-30% over pre-optimization. ([#39243](https://github.com/PaddlePaddle/Paddle/pull/39243), [#39423](https://github.com/PaddlePaddle/Paddle/pull/39423))

- Optimize `dropout` forward and backward computation, with performance improvement by about 20%. ([#39795](https://github.com/PaddlePaddle/Paddle/pull/39795), [#38859](https://github.com/PaddlePaddle/Paddle/pull/38859), [#38279](https://github.com/PaddlePaddle/Paddle/pull/38279), [#40053](https://github.com/PaddlePaddle/Paddle/pull/40053))

- Optimize `grid_sampler` forward and backward computation, with forward performance improvement by 10% to 30% over pre-optimization, and backward performance improvement by 10% to 60% over pre-optimization. ([#39751](https://github.com/PaddlePaddle/Paddle/pull/39751))

- Optimize `group_norm` forward and backward computation, with the forward performance improvement by 1.04x to 2.35x, and backward performance improvement by 1.12x to 1.18x. ([#39944](https://github.com/PaddlePaddle/Paddle/pull/39944), [#40657](https://github.com/PaddlePaddle/Paddle/pull/40657), [#39596](https://github.com/PaddlePaddle/Paddle/pull/39596))

- Optimize `conv1d` forward and backward computation, with the forward performance improvement by 1.00x to 2.01x, and backward performance improvement by 1.01x to 474.56x. ([#38425](https://github.com/PaddlePaddle/Paddle/pull/38425))

- Optimize `elementwise_div` backward computation, with the backward performance improvement by 1.02x to 29.25x. ([#38044](https://github.com/PaddlePaddle/Paddle/pull/38044))

- Optimize `gelu` forward and backward computation, with the backward performance improvement by 1.13x to 1.43x, and reverse performance improvement by 1.10x to 1.55x. ([#38188](https://github.com/PaddlePaddle/Paddle/pull/38188), [#38263](https://github.com/PaddlePaddle/Paddle/pull/38263))

- Optimize `elementwise_sub` backward computation, with the backward performance improvement by 1.04x to 15.64x. ([#37754](https://github.com/PaddlePaddle/Paddle/pull/37754))

- Optimize `flip's` forward performance on one-dimensional data input, with the performance improvement by 100%. ([#37825](https://github.com/PaddlePaddle/Paddle/pull/37825))

- Optimize `layer_norm` forward and backward computation, with the forward performance improvement by 2x to 5x over pre-optimization, and backward performance improvement by 20% to 50% over pre-optimization. ([#39167](https://github.com/PaddlePaddle/Paddle/pull/39167), [#39247](https://github.com/PaddlePaddle/Paddle/pull/39247))

- Optimize `embedding` forward and backward computation, with a maximum improvement of 1.51x in forward performance and 1.03x to 7.79x in backward performance. ([#39856](https://github.com/PaddlePaddle/Paddle/pull/39856), [#39886](https://github.com/PaddlePaddle/Paddle/pull/398866))

- Optimize `gelu` FP16 forward and backward calculations, with forward performance improvement by 9% to 12% over pre-optimization, and backward performance improvement by 2% to 9% over pre-optimization. ([#38980](https://github.com/PaddlePaddle/Paddle/pull/38980))

- Remove CPU -> GPU explicit data transfer operation in `gather_nd` forward and backward operators, and remove the explicit synchronous operation in `index_select` forward and backward operators. Change GPU -> GPU data transfer in `scatter_nd` from synchronous operation to asynchronous operation. ([#40933](https://github.com/PaddlePaddle/Paddle/pull/40933))

- Optimize `Lars optimzier` computation, with the training performance improvement of Resnet50 PF16 model by 5.1% over pre-optimization. ([#35652](https://github.com/PaddlePaddle/Paddle/pull/35652), [#35476](https://github.com/PaddlePaddle/Paddle/pull/35476))

- Optimize `AvgPool2dGrad` computation, with the performance improvement by 2.6x over pre-optimization. ([#35389](https://github.com/PaddlePaddle/Paddle/pull/35389))

- Optimize `Elementwise` computation for multivariate output, improving performance by up to 15% over pre-optimization. ([#38329](https://github.com/PaddlePaddle/Paddle/pull/38329), [#38410](https://github.com/PaddlePaddle/Paddle/pull/38410))

- Optimize `Categorical`the probs computation, simplify the computation logic, and improve the performance by 4x to 5x. ([#42178](https://github.com/PaddlePaddle/Paddle/pull/42178))

- Optimize the `paddle.sum ` performance, with performance improvement by about 20%.  ([#42309](https://github.com/PaddlePaddle/Paddle/pull/42309))

- Remove CudaStreamSync operation from `paddle.nn.ClipGradByGlobalNorm ` to reduce scheduling overhead during execution, with 5% performance improvement on ptb models. ([#42170](https://github.com/PaddlePaddle/Paddle/pull/42170))

- Optimize a series of underlying data structures and detailed implementations in the original dynamic graph execution system to improve the scheduling performance of the original dynamic graph. ([#42010](https://github.com/PaddlePaddle/Paddle/pull/42010), [#42171](https://github.com/PaddlePaddle/Paddle/pull/42171), [#42224](https://github.com/PaddlePaddle/Paddle/pull/42224), [#42256](https://github.com/PaddlePaddle/Paddle/pull/42256), [#42306](https://github.com/PaddlePaddle/Paddle/pull/42306), [#42329](https://github.com/PaddlePaddle/Paddle/pull/42329)[, #42340](https://github.com/PaddlePaddle/Paddle/pull/42340), [#42368](https://github.com/PaddlePaddle/Paddle/pull/42368), [#42425](https://github.com/PaddlePaddle/Paddle/pull/42425))

- Simplify the probs calculation logics of `paddle.distribution.Categorical `, to improve performance by 4x to 5x.  ([#42178](https://github.com/PaddlePaddle/Paddle/pull/42178))

### **(4) Bug fixing**

#### API

- Fix the output type error with `paddle.sum` when the input parameter type and output parameter type do not match and the number of reduce elements on the `axis` is 1. ([#36123](https://github.com/PaddlePaddle/Paddle/pull/36123))

- Fix an `AttributeError` in `paddle.flops` when the layer output type is tuple. ([#38850](https://github.com/PaddlePaddle/Paddle/pull/38850))

- Fix the `paddle.diag` failing to propagate gradients because there is no backward kernel. ([#40447](https://github.com/PaddlePaddle/Paddle/pull/40447))

- Fix an error in sorting `paddle.sort` input with NaN values. ([#41070](https://github.com/PaddlePaddle/Paddle/pull/41070))

- Fix the error when`paddle.full_like`'s input contains INF value. ([#40232](https://github.com/PaddlePaddle/Paddle/pull/40232))

- Fix the bug in `paddle.strided_slice`: strided_slice result does not consistent with slice when the data in the input of starts is less than -rank. ([#39066](https://github.com/PaddlePaddle/Paddle/pull/39066))

- Fix the bug in the `max_pool` family of operators where infer_shape is calculated incorrectly when index is returned. This affects the APIs: `paddle.nn.functional.max_pool1d/2d/3d`, `paddle.nn.functional.adaptive_max_pool1d/2d/3d`, `paddle.nn.MaxPool1D/2D/3D`, `paddle.nn.AdaptiveMaxPool1D/2D/3D`. ([#40139](https://github.com/PaddlePaddle/Paddle/pull/40139))

- Fix an issue where the dtype of pooling_mask returned by the `max_pool` family of operators is incorrect. Now the dtype of pooling_mask is int32. The affected APIs are `paddle.nn.functional.max_pool1d/2d/3d`, `paddle.nn.functional.adaptive_max_pool1d/2d/3d`, `paddle.nn.MaxPool1D/2D/3D`, `paddle.nn.AdaptiveMaxPool1D/2D/3D`. ([#39314](https://github.com/PaddlePaddle/Paddle/pull/39314) )

- Fix the bug with `paddle.shape` where the backward gradient by default causes a computation error. ([#37340](https://github.com/PaddlePaddle/Paddle/pull/37340))

- Fix the bug in `paddle.nn.Layer's` `to` method when converting both dtype and place at the same time. ([#37007](https://github.com/PaddlePaddle/Paddle/pull/38007))

- Fix the bug that `paddle.amp.decorate` fails to rewrite the parameters of non-leaf network layers to FP16. ([#38402](https://github.com/PaddlePaddle/Paddle/pull/38402))

- Fix the bug that the `paddle.amp.decorate` rewrites the non-input parameter in `paddle.nn.BatchNorm1D`, `paddle.nn.BatchNorm2D`, and `paddle.nn.BatchNorm3D` to FP16. ([#38541](https://github.com/PaddlePaddle/Paddle/pull/38541))

- Fix the bug that the `paddle.amp.decorate` rewrites the non-input parameter in `paddle.nn.SyncBatchNorm` to FP16. ([#40943](https://github.com/PaddlePaddle/Paddle/pull/40943))

- Fix redundant warnings in `paddle.nn.Layer.to`. ([#36700](https://github.com/PaddlePaddle/Paddle/pull/36700))

- Fix the bug in `paddle.nn.RNN` when being used inside control flow. ([#41162](https://github.com/PaddlePaddle/Paddle/pull/41162))

- Fix the bug that the `paddle.to_tensor` fails to specify the CUDAPlace of the Tensor. ([#39662](https://github.com/PaddlePaddle/Paddle/pull/39662))

- Fix the issue that`paddle.nn.Identity` is not exposed. ([#39615](https://github.com/PaddlePaddle/Paddle/pull/39615))

- Fix the bug where the output values of the `fill_` and `zero_` inplace APIs are incorrect when the input is on a CUDAPinned Place after dynamic graph reconstruction. ([#41229](https://github.com/PaddlePaddle/Paddle/pull/41229))

- After refactoring the dynamic graph, fix the bug of incorrect inplace version value of the output Tensor when calling assign op using the append op. Change it to call assign op using the `_C_ops`. ([#41118](https://github.com/PaddlePaddle/Paddle/pull/41118))

- Remove unreasonable codes in the `elementwise_add` 's third-order kernel, and fix an uninitialized issue in the network creation process. ([#36618](https://github.com/PaddlePaddle/Paddle/pull/36618))

- Fix the missing attribute bug in `conv2d` execution of cuDNN Kernel. ([#38827](https://github.com/PaddlePaddle/Paddle/pull/38827))

- Fix an issue where `multiclass_nms3` output shape is incorrect. ([#40059](https://github.com/PaddlePaddle/Paddle/pull/40059))

- Fix an issue with `yolo_box` outputting incorrect shape. ([#40056](https://github.com/PaddlePaddle/Paddle/pull/40056))

- Fix an issue where the higher-order differentiation `gradients` interface does not take effect as expected when target_grad is specified. ([#40940](https://github.com/PaddlePaddle/Paddle/pull/40940/))

- Fix an issue that the network parameter type is incorrect when the default_dtype is modified in the op`_BatchNormBase` base class in the dynamic graph mode. The affected APIs are `paddle.nn.BatchNorm1D`，`paddle.nn.BatchNorm2D`，`paddle.nn.BatchNorm3D`， and `paddle.nn.SyncBatchNorm`. Specific reason: when `get_default_dtype() == 'float16'`, the default parameter data type is modified by `set_default_dtype('float32')`. The parameter type in dynamic graph mode is created by default_dtype; therefore, the change of the default parameter type causes the subsequent networking Parameter type error. ([#36376](https://github.com/PaddlePaddle/Paddle/pull/36376))

- Fix the bug of the undefined intermediate variable in the backward op in batchnorm op in case that the data type is FP32 and the data dimension is `dims = 2 and data_layout = NHWC`. ([#37020](https://github.com/PaddlePaddle/Paddle/pull/37020))

- Fix the bug that shape of weights is incorrect, when using`paddle.static.nn.prelu` in static graph mode, and input format is`NHWC`, `mode==channel`. ([#38310](https://github.com/PaddlePaddle/Paddle/pull/38310))

- Fix the bug of `paddle.nn.functional.class_center_sample`: CUDA seed setting issue in multi-machine case. ([#38815](https://github.com/PaddlePaddle/Paddle/pull/38815))

- Fix the bug of failing to report error when the input of`paddle.nn.functional.one_hot`is incorrect. ([#41335](https://github.com/PaddlePaddle/Paddle/pull/41335))

- Fix an issue where a callback to reclaim device memory on a DCU device is not triggered in time, resulting in an OOM of the device memory. ([#40445](https://github.com/PaddlePaddle/Paddle/pull/40445))

- Fix the bugs of `setitem` backward gradient abnormal and inplace logic handling abnormal in some dynamic graph scenarios. ([#37023](https://github.com/PaddlePaddle/Paddle/pull/37023), [#38298](https://github.com/PaddlePaddle/Paddle/pull/38298))

- Fix the bug of index abnormal when Tensor array uses the Slice to index in the dynamic to static scenarios. ([#39251](https://github.com/PaddlePaddle/Paddle/pull/39251))

- Fix the bug of memory or device memory leaks caused by some temporary variables not being correctly destructed when `paddle.Tensor.register_hook` interface is used. ([#40716](https://github.com/PaddlePaddle/Paddle/pull/40716))

- Fix the bug that `Tensor.getitem` cannot get the value when the index is a bool Tensor with all False. ([#41297](https://github.com/PaddlePaddle/Paddle/pull/41297))

- Fix the bug that `Tensor.getitem` cannot get the value when the index is a bool scalar Tensor. ([#40829](https://github.com/PaddlePaddle/Paddle/pull/40829))

- Fix the bug in `paddle.index_select` when index is a 0-shape Tensor. ([#41383](https://github.com/PaddlePaddle/Paddle/pull/41383))

- Fix the bug when the number of GPU threads requested by `paddle.index_select` and `paddle.index_sample` exceeds the limited machine resources. ([#41127](https://github.com/PaddlePaddle/Paddle/pull/41127), [#37816](https://github.com/PaddlePaddle/Paddle/pull/37816), [#39736](https://github.com/PaddlePaddle/Paddle/pull/39736), [#41563](https://github.com/PaddlePaddle/Paddle/pull/41563))

- Fix the bug when ReduceConfig, elemwise_grad, gather, gather_nd, and scatter ops request more GPU threads than the limited machine resources. ([#40813](https://github.com/PaddlePaddle/Paddle/pull/40813), [#41127](https://github.com/PaddlePaddle/Paddle/pull/41127))

- Fix the bug that the memory access is out of boundary when NX ! = 1 in ReadData, ReadDataBc, and ReadDataReduce in Kernel Primitive API. ([#36373](https://github.com/PaddlePaddle/Paddle/pull/36373))

- Fix the bug of the computation result abnormal due to data overflow caused by the IndexRandom data type error. ([#39867](https://github.com/PaddlePaddle/Paddle/pull/39867), [#39891](https://github.com/PaddlePaddle/Paddle/pull/39891))

- Fix the bug of the returned computing result error of reduce op when reduce_num = 1. ([#38771](https://github.com/PaddlePaddle/Paddle/pull/38771))

- Fix the bug of the memory access out-of-bound of reduce op in the middle dimension of reduce in HIP environments. ([#41273](https://github.com/PaddlePaddle/Paddle/pull/41273))

- Fix the bug of Kernel failed to properly release in the computation of two FP16 one-dimensional vectors of matmul op.

- Fix the bug caused by CUDA integer computation overflow for some operators, including: bernoulli, gaussian_random, gumbel_softmax, multinomial, truncated_gaussian_random, uniform_ random_inplace, and uniform_random ops. ([#37670](https://github.com/PaddlePaddle/Paddle/pull/37670))

- Fix the bug where `paddle.nn.Sequential` reports a KeyError error when traversing sublayers in a for loop. ([#39372](https://github.com/PaddlePaddle/Paddle/pull/39372))

- Fix the bug of the check shape error in `paddle.nn.functional.unfold` when compiling in static graphs. ([#38907](https://github.com/PaddlePaddle/Paddle/pull/38907), [#38819](https://github.com/PaddlePaddle/Paddle/pull/38819))

- Fix the bug of reporting an error if `axis` is specified when using dropout for static graphs. ([#37223](https://github.com/PaddlePaddle/Paddle/pull/37223))

- Migrate the matmul operator in the `paddle.nn.MultiHeadAttention` to the matmul_v2 operator. ([#36222](https://github.com/PaddlePaddle/Paddle/pull/36222))

- Fix the bug occurred in throwing FPE when the empty Tensor is used in `paddle.nn.functional.label_smooth`. ([#35861](https://github.com/PaddlePaddle/Paddle/pull/35861))

- Fix the deformation bug of reshape op when input is an empty Tensor. Support the empty Tensor rehape to [-1]. ([#36087](https://github.com/PaddlePaddle/Paddle/pull/36087))

- Fix the bug of the modified values will incorrectly override other rows when the `fill_diagonal` 's input parameter offset is non-zero. ([#36212](https://github.com/PaddlePaddle/Paddle/pull/36212))

- Modify stop_gradient returned by the range op bing set to True in dynamic graph mode. ([#37486](https://github.com/PaddlePaddle/Paddle/pull/37486))

- Fix the bug where Lamb optimizer is updated incorrectly when Beta1Pow and Beta2Pow are on the GPU. ([#38518](https://github.com/PaddlePaddle/Paddle/pull/38518))

- Fix the bug where the conv2d operator doesn't respect to FLAGS_cudnn_deterministic. ([#37173](https://github.com/PaddlePaddle/Paddle/pull/37173))

- Fix the bug caused by an earlier version of cufft that does not define CUFFT_VERSION. ([#37312](https://github.com/PaddlePaddle/Paddle/pull/37312))

- Fix the computing error of `paddle.ifftshit` and `paddle.fftshift`. ([#36834](https://github.com/PaddlePaddle/Paddle/pull/36834), [#36748](https://github.com/PaddlePaddle/Paddle/pull/36748))

- Fix the `axis` computation error in `paddle.fft` series of APIs. ([#36321](https://github.com/PaddlePaddle/Paddle/pull/36321))

- Fix an output data type registration bug of batch_norm_grad op in case of FP16 data type. This bug causes the compilation failure in some scenarios. There is also the impact on FP16 computational precision. ([#42461](https://github.com/PaddlePaddle/Paddle/pull/42461))

- Fix the incorrect Infershape information bug in the `paddle.nn.functional.pad ` API when the padding is Tensor in dynamic to static conversion. ([#42414](https://github.com/PaddlePaddle/Paddle/pull/42414))

- Fix an exception in `paddle.distribution.StickBreakingTransform ` when the input dimension exceeds 2. ([#41762](https://github.com/PaddlePaddle/Paddle/pull/41672))

- Fix a nan/inf bug calculated with QK^T in fused_attention op. ([#42032](https://github.com/PaddlePaddle/Paddle/pull/42032))

- Fix a nan/inf bug calculated in fused_attention op with FusedResidualDropoutBias on V100. ([#42398](https://github.com/PaddlePaddle/Paddle/pull/42398))

- Fix a redundant data transform bug introduced by the full_like op during execution. ([#41973](https://github.com/PaddlePaddle/Paddle/pull/41973))

- Fix a problem with p_norm op calculating nan on GPU environments. ([#41804](https://github.com/PaddlePaddle/Paddle/pull/41804))

- Fix a section error of split op when the sections parameter has a size of 0. ([#41755](https://github.com/PaddlePaddle/Paddle/pull/41755))

- Fix the bug of reporting not supporting Place (gpu:0) in multi-card training when broadcast is required in 6 elementwise ops (pow, complex, divide_double, multiply_double, fmax, and fmin). ([#42332](https://github.com/PaddlePaddle/Paddle/pull/42332))

- Fix the bug that the deprecated interface reports a warning in case of `import paddle` due to a PIL version update. ([#42307](https://github.com/PaddlePaddle/Paddle/pull/42307))

- Fix the bug that `paddle.linalg.matrix_rank ` does not support tol as FP64 Tensor under static graph. ([#42085](https://github.com/PaddlePaddle/Paddle/pull/42085))

#### IR(Intermediate Representation)

- Dynamic to static graphs

  - Fix a type derivation error in reverse gradient accumulation when the `tensor_array` is used with the control flow. ([#39585](https://github.com/PaddlePaddle/Paddle/pull/39585), [#39689](https://github.com/PaddlePaddle/Paddle/pull/39689))

  - Fix an issue where the parameter gradient type is not set correctly during dynamic to static AMP training. ([#40938](https://github.com/PaddlePaddle/Paddle/pull/40938))

  - Fix an issue of reporting an error in the dynamic to static transcription when there are misplaced annotations in the codes. ([#39035](https://github.com/PaddlePaddle/Paddle/pull/39035), [#38003](https://github.com/PaddlePaddle/Paddle/pull/38003))

  - Fix an issue where Tensor is not properly converted to Variable when calling a non-forward function in dynamic to static codes. ([#37296](https://github.com/PaddlePaddle/Paddle/pull/37296), [#38540](https://github.com/PaddlePaddle/Paddle/pull/38540))

  - Fix an issue where `paddle` is incorrectly passed as a variable when dynamic to static transcription. ([#37999](https://github.com/PaddlePaddle/Paddle/pull/37999))

  - Fix an issue where model parameters are incorrectly counted when calling `paddle.flops` after model dynamic to static conversion. ([#36852](https://github.com/PaddlePaddle/Paddle/pull/36852))

  - Fix an issue where GPU memory will keep growing in train mode and no_grad contexts after loading models using the `paddle.jit.save/load` interface. ([#36434](https://github.com/PaddlePaddle/Paddle/pull/36434))

  - Add warning in function of convert_call when converting the generator function. ([#35369](https://github.com/PaddlePaddle/Paddle/pull/35369))

  - Fix the run_program op dependency analysis bug. ([#38470](https://github.com/PaddlePaddle/Paddle/pull/38470))

  - Fix the code conversion bug when returning a single value in control flow For. ([#40683](https://github.com/PaddlePaddle/Paddle/pull/40683))

  - Fix the bug when generating a reverse op when the input to conditional_block op contains LoDTensorArray. ([#39585](https://github.com/PaddlePaddle/Paddle/pull/39585))

  - Fix the bug that `padddle.jit.save ` loses the forward_pre_hook and forward_post_hook of the top Layer in case of the export of a dynamic-to-static graph mode. ([#42273](https://github.com/PaddlePaddle/Paddle/pull/42273))

  - Fix the dynamic to static conversion error report where the shape parameter in `paddle.expand ` contains a Tensor. ([#41973](https://github.com/PaddlePaddle/Paddle/pull/41973))


#### **Distributed Training**

- Distributed training basic functions

  - Fix the bug of a port reporting error in the distributed multi-machine training. ([#37274](https://github.com/PaddlePaddle/Paddle/pull/37274))

  - Fix the brpc compilation dependency bug. ([#37064](https://github.com/PaddlePaddle/Paddle/pull/37064))

  - Fix an occupied port issue due to tcp self-connections when Fleet starts. ([#38174](https://github.com/PaddlePaddle/Paddle/pull/38174))

  - Fix the precision degradation bug under data parallel due to inconsistent initialization of FP16 parameters under multiple cards. ([#38838](https://github.com/PaddlePaddle/Paddle/pull/38838), [#38563](https://github.com/PaddlePaddle/Paddle/pull/38563), [#38405](https://github.com/PaddlePaddle/Paddle/pull/38405))

  - Fix the precision degradation under data parallel due to FP16 gradient synchronization without dividing by the number of cards. ([#38378](https://github.com/PaddlePaddle/Paddle/pull/38378))

- Dynamic graph mixing parallel

  - Fix the bug where parameters are not updated in FP16 mode under mixed parallel by using the new update interface. ([#36017](https://github.com/PaddlePaddle/Paddle/pull/36017))
- Static graph mixing parallel

  - Fix an issue where grad merge is not compatible with ClipGradientByGlobalNorm in distributed dp mode. ([#36334](https://github.com/PaddlePaddle/Paddle/pull/36334))

  - Fix an issue under hybrid parallelism where the non-distributed parameters of tensor model parallelism are not broadcast during the initialization phase, resulting in inconsistent non-distributed parameters across cards. ([#36186](https://github.com/PaddlePaddle/Paddle/pull/36186))

  - Fix the issue that sharding's save_persistables interface does not save FP16 parameters and offload persistent variables when sharding is enabled with offload. ([#40477](https://github.com/PaddlePaddle/Paddle/pull/40477))

  - Fix the bug where ema parameters are not saved on non-0 cards when sharding is enabled for training. ([#39860](https://github.com/PaddlePaddle/Paddle/pull/39860))

  - Fix an issue where FC incorrectly calculates gradients according to column cuts. ([#38724](https://github.com/PaddlePaddle/Paddle/pull/38724))

  - Fix the bug reported when DistributedStrategy is set to without_graph_optimizer when used with rnn. ([#36176](https://github.com/PaddlePaddle/Paddle/pull/36176))

- GPUPS Parameter Server Training

  - Fix the CPU branch compilation bug triggered by the GPUPS macro definition. ([#37248](https://github.com/PaddlePaddle/Paddle/pull/37248))

  - Fix an occasional error raised when saving delta and pullsparse concurrency during GPUPS streamline training. ([#37233](https://github.com/PaddlePaddle/Paddle/pull/37233))

  - Fix a download error issue caused by HDFSClient querying a directory without returning the full path. ([#36590](https://github.com/PaddlePaddle/Paddle/pull/36590))

  - Fix the bug with pulling old parameters in GPUPS streamline training. ([#36512](https://github.com/PaddlePaddle/Paddle/pull/36512))

  - Fix a GPUPS multi-stream allocation issue. ([#37476](https://github.com/PaddlePaddle/Paddle/pull/37476))

  - Fix the bug of the GPUPS pybind out of core. ([#37287](https://github.com/PaddlePaddle/Paddle/pull/37287))


#### **Other**

- Fix the clip_extra issue when saving models for dynamic graph quantization training. ([#38323](https://github.com/PaddlePaddle/Paddle/pull/38323))

- Fix an issue with abs_max scale initialization for dynamic graph quantization training. ([#39307](https://github.com/PaddlePaddle/Paddle/pull/39307))

- Fix an issue of exceptions in saving model in dynamic graph quantization training. ([#38102](https://github.com/PaddlePaddle/Paddle/pull/38102), [#38012](https://github.com/PaddlePaddle/Paddle/pull/38012))

- Fix the offline quantization flatten op output error. ([#37722](https://github.com/PaddlePaddle/Paddle/pull/37722))

- Fix the non-matching dimension bug in case of inverse quantization matmul op. ([#36982](https://github.com/PaddlePaddle/Paddle/pull/36982))

- Fix the bug of adding quantization op when quantizing matmul_v2 without weights. ([#36593](https://github.com/PaddlePaddle/Paddle/pull/36593))

- Fix the error of saving the quant_axis attribute in the conv op channel-wise quantization when saving the models. ([#39054](https://github.com/PaddlePaddle/Paddle/pull/39054))

- Fix the slow training of channel-wise quantization. ([#40772](https://github.com/PaddlePaddle/Paddle/pull/40772))

- Fix the bug of quantization training when dividing by tensor(initialized as 0) leads to nan. ([#36762](https://github.com/PaddlePaddle/Paddle/pull/36762))

- Fix incorrect settings of amp_level for mixed precision in multi-threaded scenarios. ([#39198](https://github.com/PaddlePaddle/Paddle/pull/39198))

- Fix an issue where PyLayer and Recompute is not set mixed precision correctly when mixed precision training is used with PyLayer and Recompute. ([#39950](https://github.com/PaddlePaddle/Paddle/pull/39950), [#40042](https://github.com/PaddlePaddle/Paddle/pull/40042))

- Fix an issue where `D_GLIBCXX_USE_CXX11_ABI` does not take effect when compiling custom operators under Mac. ([#37878](https://github.com/PaddlePaddle/Paddle/pull/37878))

- Fix the bug of inconsistent dynamic and static behaviors in case of block=None the initializer-related API. ([#37827](https://github.com/PaddlePaddle/Paddle/pull/37827))

- Fix the bug in python 3.6 where there is no fluid module. ([#35862](https://github.com/PaddlePaddle/Paddle/pull/35862))

- Fix the bug where optimizer `paddle.optimizer.Adamw` incorrectly calls adam op. ([#36028](https://github.com/PaddlePaddle/Paddle/pull/36028))

- Fix a logic error when the `paddle.optimizer.Momentum` optimizer parameter `regularizer` property is None under the multi tensor policy. ([#38344](https://github.com/PaddlePaddle/Paddle/pull/38344))

- Fix the bug that the `paddle.optimizer.Momentum` and `paddle.optimizer.Adam` optimizers modify the `multi_precision` property under the multi tensor policy. ([#38991](https://github.com/PaddlePaddle/Paddle/pull/38991))

- Fix the code compilation error when using final-state API amp in combination with optional Tensor. ([#40980](https://github.com/PaddlePaddle/Paddle/pull/40980))

- Fix the bug where paddle+lite+xpu prediction library would report an error when calling lite CPU prediction, and fix the bug where paddle+lite(without NNAdapter) would report an error when compiling. ([#37449](https://github.com/PaddlePaddle/Paddle/pull/37449))

- Fix the bug in Debug compile mode where LoDTensorArray crashes due to inconsistent Pybind11 bindings. ([#37954](https://github.com/PaddlePaddle/Paddle/pull/37954))

- Fix the bug that prevents correct construction of Tensor in the extreme case where the shape parameter is a list of Tensor mix with int. ([#38284](https://github.com/PaddlePaddle/Paddle/pull/38284))

- Fix a compatibility issue with the `paddle.optimizer.AdamW` API. ([#37905](https://github.com/PaddlePaddle/Paddle/pull/37905))

- Fix the bug in _InstanceNormBase where the returne value of extra_repr is incorrect. ([#38537](https://github.com/PaddlePaddle/Paddle/pull/38537))

- Fix the bug that the Paddle Inference lacks of the symbol `paddle::distributed::TensorTable` when the -DWITH_DISTRIBUTED is uesd. ([#41128](https://github.com/PaddlePaddle/Paddle/pull/41128))

- matmul_v2 op reports error when there is a 0 value in the shape. ([#35791](https://github.com/PaddlePaddle/Paddle/pull/35791))

- Fix the problem of the repeated printing for no gradient input hint message of the recomputed in dynamic graphs. Change it to the printing only once with using warning. ([#38293](https://github.com/PaddlePaddle/Paddle/pull/38293))

- Fix the low accuracy bug on the validation set in later epoch training in visual models in the gelu op. ([#38450](https://github.com/PaddlePaddle/Paddle/pull/38450))

- Fix adamw op error in numerical computation. ([#37746](https://github.com/PaddlePaddle/Paddle/pull/37746))

- Add the parameters in the sparse_momentum `_C_ops` interface. ([#39969](https://github.com/PaddlePaddle/Paddle/pull/39969))

- Fix the bug where there is no `distributed` module in python 3.6. ([#35848](https://github.com/PaddlePaddle/Paddle/pull/35848))

- Fix the eigh unit test data initialization problem. ([#39568](https://github.com/PaddlePaddle/Paddle/pull/39568))

- Fix the eigvalsh unit test data initialization problem. ([#39841](https://github.com/PaddlePaddle/Paddle/pull/39841))

- Fix the bug of not working properly due to excessive register usage on V100 by segment op. ([#38113](https://github.com/PaddlePaddle/Paddle/pull/38113))

- Fix the bug with conv-related op sparsification incorrectly set dimension. ([#36054](https://github.com/PaddlePaddle/Paddle/pull/36054))

- Provide Automatic SParsity training for static graph-related function Alias to `Paddle.static.sparsity`. ([#36525](https://github.com/PaddlePaddle/Paddle/pull/36525))

- Fix the bug where divide op’s integer division is still an integer. ([#40890](https://github.com/PaddlePaddle/Paddle/pull/40890))

- Fix the crash bug of`paddle.multiplex` when input Tensor value is 0. ([#34972](https://github.com/PaddlePaddle/Paddle/pull/34972))

- Fix a speed exception for set `reduction` parameter in `paddlpaddle.nn.functional.kl_div`. ([#37283](https://github.com/PaddlePaddle/Paddle/pull/37283))

- Fix the data source unsorted bug in loading the Cifar dataset. ([#37272](https://github.com/PaddlePaddle/Paddle/pull/37272))

- Fix the conversion of loss from uint16 to float in the ProgressBar class. ([#39231](https://github.com/PaddlePaddle/Paddle/pull/39231))

- Fix the ShareBufferWith shared data type problem. ([#37464](https://github.com/PaddlePaddle/Paddle/pull/37464), [#37247](https://github.com/PaddlePaddle/Paddle/pull/37247))

- Fix the performance issue when `paddle.io.DataLoader` uses IterableDataset and num_workers>0. ([#40541](https://github.com/PaddlePaddle/Paddle/pull/40541))

- Fix the bug with `paddle.vision.ops.yolo_loss` returns incomplete values in dynamic graph. ([#40185](https://github.com/PaddlePaddle/Paddle/pull/40185))

- Remove the restriction that the input parameter dataset of `paddle.io.BatchSampler` needs to be the `paddle.io.Dataset` type, to expand the support for user-defined datasets. ([#40184](https://github.com/PaddlePaddle/Paddle/pull/40184))

- Fix the bug of `paddle.summary` reporting that op_flops does not exist. ([#36489](https://github.com/PaddlePaddle/Paddle/pull/36489))

- Fix the formula error of lars_momentum op when lars_weight_decay=0. ([#40892](https://github.com/PaddlePaddle/Paddle/pull/40892))

- Fix the bug that the optimize-offload cannot save presistable var. ([#36433](https://github.com/PaddlePaddle/Paddle/pull/36433))

- Fix an issue where optimizer-offload does not support adamw op type. ([#36432](https://github.com/PaddlePaddle/Paddle/pull/36432))

- Fix an issue where enable_program_desc_tracing_data in Tracer is not safe in multi-threaded scenarios. ([#39776](https://github.com/PaddlePaddle/Paddle/pull/39776))

- Fix an issue where the model file size is not initialized when the model is read. ([#40518](https://github.com/PaddlePaddle/Paddle/pull/40518))

- Fix the logic bug of the Expand op. When the dimension of the input Tensor X is smaller than the shape to be expanded, it may result in the incorrect Out.Shape. ([#38677](https://github.com/PaddlePaddle/Paddle/pull/38677))

- Fix the dynamic to static transcription error when the Expand_As op takes only y.shape without Y variable entered. ([#38677](https://github.com/PaddlePaddle/Paddle/pull/38677))

- Fix the logic error when Expand_As op computes the output shape. ([#38677](https://github.com/PaddlePaddle/Paddle/pull/38677))

- Fix the bug that the variables of the `core.VarDesc.VarType.STRINGS` type report error when getting the `lod_level` property and setting its `lod_level` to None. ([#39077](https://github.com/PaddlePaddle/Paddle/pull/39077))

- Fix an issue where the framework function `Pylayer` does not support different dtypes. ([#37974](https://github.com/PaddlePaddle/Paddle/pull/37974))

- Fix the bug of division by zero of the learning rate decay API `paddle.optimizer.lr.PolynomialDecay`. ([#38782](https://github.com/PaddlePaddle/Paddle/pull/38782))

- Fix the issue where some logs remained after calling the DisableGlogInfo() interface. ([#36356](https://github.com/PaddlePaddle/Paddle/pull/36356))

- Fix an error in backward of multi-layer RNN (when dropout is set to 0) in the training of SimpleRNN, GRU and LSTM API CPU. ([#37080](https://github.com/PaddlePaddle/Paddle/pull/37080))

- Add cache for fft on the backend of cufft and hipfft. ([#36646](https://github.com/PaddlePaddle/Paddle/pull/36646))

- Enable the shifts parameter of `paddle.roll` to support transfer in Tensor. ([#36727](https://github.com/PaddlePaddle/Paddle/pull/36727))

- Add onemkl to fft as an optional computation backend. ([#36414](https://github.com/PaddlePaddle/Paddle/pull/36414))

- Fix the precision bug in the bfloat16 type under two mamtul_v2 and elementwise_div ops. ([#42479](https://github.com/PaddlePaddle/Paddle/pull/42479))

- Fix a possible error in the next step caused by LoDTensorArray clearing only the internal Tensor and not clearing the Array during device memory recycling. ([#42398](https://github.com/PaddlePaddle/Paddle/pull/42398))


## **4. Deployment Direction (Paddle Inference)**

### **(1) New features**

#### **New APIs**

- Add the Java API so that Java developers can implement high performance inference on the server and in the cloud through a simple and flexible interface. ([#37162](https://github.com/PaddlePaddle/Paddle/pull/37162))

- Add `GetTrtCompileVersion` and `GetTrtRuntimeVersion` interfaces for getting TensorRT version information. ([#36429](https://github.com/PaddlePaddle/Paddle/pull/36429))

- Add the `ShareExternalData` interface to avoid memory copy of input data during inference. ([#39809](https://github.com/PaddlePaddle/Paddle/pull/39809))


#### **New functions**

- Add ONNX Runtime backend support. Currently it supports only CPU in the integrated version. ([#39988](https://github.com/PaddlePaddle/Paddle/pull/39988), [#40561](https://github.com/PaddlePaddle/Paddle/pull/40561))

- Add support for Ascend 310 inference based on the Paddle Lite subgraph approach. ([#35226](https://github.com/PaddlePaddle/Paddle/pull/35226))

- Add the native GPU FP16 inference. ([#40531](https://github.com/PaddlePaddle/Paddle/pull/40531))

- For the switch_ir_debug interface, add the dump model function. ([#36581](https://github.com/PaddlePaddle/Paddle/pull/36581))

- Add the configuration interface for TensorRT config: `void UpdateConfigInterleaved(paddle_infer::Config* c, bool with_interleaved)` for special data layout in int8 quantization inference. ([#38884](https://github.com/PaddlePaddle/Paddle/pull/38884))

- Add TensorRT inspector output information to the log. It is valid only for TensorRT 8.2 or later. ([#38362](https://github.com/PaddlePaddle/Paddle/pull/38362)，[#38200](https://github.com/PaddlePaddle/Paddle/pull/38200)))

- Add the support of the TensorRT ASP sparse inference. ([#36413](https://github.com/PaddlePaddle/Paddle/pull/36413))


### **(2) Underlying optimization**

#### **CPU performance optimization**

- Optimize the caching mechanism of MKLDNN. ([#38336](https://github.com/PaddlePaddle/Paddle/pull/38336), [#36980](https://github.com/PaddlePaddle/Paddle/pull/36980), [#36695](https://github.com/PaddlePaddle/Paddle/pull/36695))

- Add matmul_scale_fuse pass. ([#37962](https://github.com/PaddlePaddle/Paddle/pull/37962))

- Add MKLDNN reshape_transpose_matmul_v2_mkldnn_fuse_pass. ([#37847](https://github.com/PaddlePaddle/Paddle/pull/37847), [#40948](https://github.com/PaddlePaddle/Paddle/pull/40948))

- Add MKLDNN conv_hard_sigmoid_mkldnn_fuse_pass. ([#36869](https://github.com/PaddlePaddle/Paddle/pull/36869))

- Add MKLDNN matmul_v2_transpose_reshape_fuse_pass. ([#36481](https://github.com/PaddlePaddle/Paddle/pull/36481))

- Add MKLDNN softplus_activation_mkldnn_fuse_pass. ([#36657](https://github.com/PaddlePaddle/Paddle/pull/36657))

- Add MKLDNN elt_act_mkldnn_fuse_pass. ([#36541](https://github.com/PaddlePaddle/Paddle/pull/36541))

- Add MKLDNN mish operator and conv_mish_mkldnn_fuse_pass. ([#38623](https://github.com/PaddlePaddle/Paddle/pull/38623))


#### **GPU performance optimization**

- Change the inference default video memory allocation policy from `naive_best_fit` to `auto_growth`, to solve the problem of some models filled up with the GPU video memory. ([#41491](https://github.com/PaddlePaddle/Paddle/pull/41491))

- Support gelu and FC+gelu ops using TensorRT inference. ([#38399](https://github.com/PaddlePaddle/Paddle/pull/38399))

- Support `deformable_conv` inference using TensorRT under static shape. ([#36612](https://github.com/PaddlePaddle/Paddle/pull/36612) [#36850](https://github.com/PaddlePaddle/Paddle/pull/36850) [#37345](https://github.com/PaddlePaddle/Paddle/pull/37345))

- Support nearest_interp_v2 op using TensorRT inference. ([#34126](https://github.com/PaddlePaddle/Paddle/pull/34126))

- Add `yolo_box` TensorRT plugin to support input parameters `iou_aware` and `iou_aware_factor` so that the IoU computed by inference is used as a factor for confidence. ([#34128](https://github.com/PaddlePaddle/Paddle/pull/34128))

- Support `elementwise_sub` and `elementwise_div` calling for TensorRT inference. ([#40806](https://github.com/PaddlePaddle/Paddle/pull/40806) [#41253](https://github.com/PaddlePaddle/Paddle/pull/41253))

- Support `multiclass_nms3` using TensorRT inference. ([#41181](https://github.com/PaddlePaddle/Paddle/pull/41181) [#41344](https://github.com/PaddlePaddle/Paddle/pull/41344))

- Support flatten_contiguous_rang op using TensorRT inference. ([#38922](https://github.com/PaddlePaddle/Paddle/pull/38922))

- Support for `pool2d` attribute `padding` using TensorRT inference when dimension is 4, and `global_pooling` and `ceil_mode` are True. ([#39545](https://github.com/PaddlePaddle/Paddle/pull/39545))

- Support batch_norm and elementwise_add using TensorRT inference when dimension is 5. ([#36446](https://github.com/PaddlePaddle/Paddle/pull/36446))

- Add pool3d to use TensorRT inference. ([#36545](https://github.com/PaddlePaddle/Paddle/pull/36545), [#36783](https://github.com/PaddlePaddle/Paddle/pull/36783))

- Add the `reduce` int32 and float types to use TensorRT inference. Add `reduce_mean` GPU operator int32 and int64 registration. ([#39088](https://github.com/PaddlePaddle/Paddle/pull/39088))

- Modify MatmulV2ToMul pass. Modify the qualifier (not support of broadcast) and op_teller mapping condition. ([#36652](https://github.com/PaddlePaddle/Paddle/pull/36652))

- Add the support for TenorRT plugin interface AddPluginV2IOExt. ([#36493](https://github.com/PaddlePaddle/Paddle/pull/36493))

- Add the aligned attribute in roi_align op and support for TensorRT inference. ([#38905](https://github.com/PaddlePaddle/Paddle/pull/38905))

- Add the support for TensorRT inference with concat attribute `axis = -1`. ([#39096](https://github.com/PaddlePaddle/Paddle/pull/39096))

- Add TensorRT plugin: preln_emb_eltwise_layernorm, preln_skip_la, and rnorm ops, for ERNIE-like model performance optimization. ([#39570](https://github.com/PaddlePaddle/Paddle/pull/39570))

- Add TensorRT fuse pass: preln_embedding_eltwise_layernorm_fuse_pass, preln_skip_layernorm_fuse_pass, for ERNIE-like model performance optimization. ([#39508](https://github.com/PaddlePaddle/Paddle/pull/39508))

- Split matmul fusion-related passes based on different backends (GPU, CPU, TensorRT), to support transpose function for FC weights. ([#39369](https://github.com/PaddlePaddle/Paddle/pull/39369))

- Add the support to TensorRT by roll, strided_slice, and slice op in case of dynamic shapes.  ([#41913](https://github.com/PaddlePaddle/Paddle/pull/41913), [#41573](https://github.com/PaddlePaddle/Paddle/pull/41573), [#41467](https://github.com/PaddlePaddle/Paddle/pull/41467))

- Add div op support for TensorRT.  ([#41243](https://github.com/PaddlePaddle/Paddle/pull/41243))

- Quantization support

  - For the `PostTrainingQuantization` API, add the support for `paddle.io.DataLoader` object or `Python Generator` input. ([#38686](https://github.com/PaddlePaddle/Paddle/pull/38686))

  - ERNIE full quantization model inference supports for interleaved data layout. ([#39424](https://github.com/PaddlePaddle/Paddle/pull/39424))

  - Support for PaddleSlim new quantile model format inference. ([#41049](https://github.com/PaddlePaddle/Paddle/pull/41049))

  - Add matmul int8 quantization inference op converter and plugin. ([#37285](https://github.com/PaddlePaddle/Paddle/pull/37285))

  - Add pass to determine if all ops in the model can support int8 quantization. ([#36042](https://github.com/PaddlePaddle/Paddle/pull/36042))

  - Support quantization inference for the FC part of the multihead attention of the non-variable-length branch. ([#39660](https://github.com/PaddlePaddle/Paddle/pull/39660))


#### **Ascend NPU Related Features**

- - Refactor shape operator forward computation logic to support execution on NPU. ([#39613](https://github.com/PaddlePaddle/Paddle/pull/39613))

  - Refactor reshape operator forward computation logic to support ShapeTensor input. ([#38748](https://github.com/PaddlePaddle/Paddle/pull/38748))

  - Uniform accuracy type when loading model weights. ([#39160](https://github.com/PaddlePaddle/Paddle/pull/39160))


### **(3) Bug fixing**

#### **Framework and API fixing**

- Fix the bug of model clipping when saving static graphs. ([#37579](https://github.com/PaddlePaddle/Paddle/pull/37579))

- For the C API, add wrapper PD_Cstr for strings, and provide construction and destructing methods to avoid users to use C runtime library to destruct strings directly. ([#38667](https://github.com/PaddlePaddle/Paddle/pull/38667))

- Fix the logic bug with memory reuse at prediction time. ([#37324](https://github.com/PaddlePaddle/Paddle/pull/37324))

- Fix memory reuse error reporting in multi-threading. ([#37894](https://github.com/PaddlePaddle/Paddle/pull/37894))

- Allow passing empty strings for inference when no weight file is available. ([#38579](https://github.com/PaddlePaddle/Paddle/pull/38579))

- Fix an issue of clone not being supported when TensorRT dynamic shape is enabled. ([#38520](https://github.com/PaddlePaddle/Paddle/pull/38520))

- Fix multi-threaded clone error after TensorRT dynamic shape is enabled. ([#40067](https://github.com/PaddlePaddle/Paddle/pull/40067))

- Fix a TensorRT engine destructing issue. ([#35842](https://github.com/PaddlePaddle/Paddle/pull/35842), [#35938](https://github.com/PaddlePaddle/Paddle/pull/35938))

- For the lite xpu interface, fix an issue where the xpu card cannot be selected. ([#36610](https://github.com/PaddlePaddle/Paddle/pull/36610))

- The TensorRT dynamic shape parameter automatically generate the interface, to add the file existence check. ([#36628](https://github.com/PaddlePaddle/Paddle/pull/36628))

- Fix the bug that the MKLDNN does not support conv3d. ([#42055](https://github.com/PaddlePaddle/Paddle/pull/42055))

#### **Backend Capability Fixing**

- Fix cuDNN default algorithm selection configuration for prediction, with using non-deterministic policies. ([#41491](https://github.com/PaddlePaddle/Paddle/pull/41491))

- Fix the bug with deformable_conv op in TensorRT plugin resource recovery handling error. ([#38374](https://github.com/PaddlePaddle/Paddle/pull/38374))

- Fix a serialization error in the TensorRT plugin for deformable_conv op. ([#38057](https://github.com/PaddlePaddle/Paddle/pull/38057))

- Adapt the new refactor engine and serialization API of TensorRT 8.0. ([#36769](https://github.com/PaddlePaddle/Paddle/pull/36769))

- Fix the bug that the Flatten2MatmulFusePass, Squeeze2MatmulFusePass, and Reshape2MatmulFusePass do not take effect. ([#37644](https://github.com/PaddlePaddle/Paddle/pull/37644))

- Fix the bug with TensorRT input data reporting errors. ([#37427](https://github.com/PaddlePaddle/Paddle/pull/37427))

- Add error message when input dimension is wrong. ([#38962](https://github.com/PaddlePaddle/Paddle/pull/38962))

- Fix the bug with EmbEltwiseLayernorm output type error. ([#40015](https://github.com/PaddlePaddle/Paddle/pull/40015))

- Remove conv_affine_channel_fuse_pass and the corresponding unit test. ([#39817](https://github.com/PaddlePaddle/Paddle/pull/39817))

- Fix an issue where the adaptive_pool2d pass incorrectly replaces the pool attribute. ([#39600](https://github.com/PaddlePaddle/Paddle/pull/39600))

- Fix the bug that shuffle_channel_detect_pass incorrectly generates shuffle_channel op. ([#39242](https://github.com/PaddlePaddle/Paddle/pull/39242))

- Fix transpose parameter error. ([#39006](https://github.com/PaddlePaddle/Paddle/pull/39006))

- Fix the crash bug when nearest_interp_v2 input scale dimension is less than 1. ([#38725](https://github.com/PaddlePaddle/Paddle/pull/38725))

- Fix the bug that the prelu does not support one-dimensional input in dynamic shape. ([#39389](https://github.com/PaddlePaddle/Paddle/pull/39389))

- Fix the bug in the kernel function of slice's special_slice_plugin. ([#39875](https://github.com/PaddlePaddle/Paddle/pull/39875))

- Temporarily disable int8 branch under skip_layernorm variable length to prevent accuracy degradation. ([#39991](https://github.com/PaddlePaddle/Paddle/pull/39991))

- Fix some bugs regarding support for preln_ernie models. ([#39733](https://github.com/PaddlePaddle/Paddle/pull/39733))

- Fix the bug that slice may exceed threads limit in ERNIE. Fix the bug that the spacial_slice is incorrectly triggered. ([#39096](https://github.com/PaddlePaddle/Paddle/pull/39096))

- Fix the bug that the elementwise does not support broadcast when the dimension is the same. ([#37908](https://github.com/PaddlePaddle/Paddle/pull/37908))

- Fix the problem that the underlying implementation is different in the nearest_interp op when align_corners is True and TensorRT layer results and native op have diff. ([#37525](https://github.com/PaddlePaddle/Paddle/pull/37525))

- Fix qkv_plugin: Kernel function computation error. ([#37096](https://github.com/PaddlePaddle/Paddle/pull/37096))

- Fix the bug with inference pass for dynamic quantization. ([#35879](https://github.com/PaddlePaddle/Paddle/pull/35879))

- Reuse directly when Tensor requests less memory than the allocated size. ([#37880](https://github.com/PaddlePaddle/Paddle/pull/37880))

- Fix the hang bug when ERNIE fixed-length model is enabled with TensorRT. ([#37839](https://github.com/PaddlePaddle/Paddle/pull/37839))

- Fix the crash bug when TensorRT int8 lacks of dynamic range information. ([#36900](https://github.com/PaddlePaddle/Paddle/pull/36900))

- Fix the bug with slice deserialization code. ([#36588](https://github.com/PaddlePaddle/Paddle/pull/36588))

- Fix yolo box calculation formula error. ([#36240](https://github.com/PaddlePaddle/Paddle/pull/36240))

- Fix the crash bug when the earlier version model uses a later version of roi_align. ([#38788](https://github.com/PaddlePaddle/Paddle/pull/38788)) External Developers

- Fix the bug of a large performance difference of softmax between python and C++. ([#37130](https://github.com/PaddlePaddle/Paddle/pull/37130))

- Fix matmul inference failure on static shape 2-dimensional input and dynamic shape 3-dimensional input. ([#36849](https://github.com/PaddlePaddle/Paddle/pull/36849))

- Fix reshape_transpose_matmul_mkldnn_fuse_pass mishandling of shapes. ([#36731](https://github.com/PaddlePaddle/Paddle/pull/36731))

- Fix an issue where TensorRT gets 4 dimensions when the input is 2 dimensions. ([#36614](https://github.com/PaddlePaddle/Paddle/pull/36614))

- Fix the bug report when the interpolate_v2 MKLDNN operator is null in the scale attribute. ([#36623](https://github.com/PaddlePaddle/Paddle/pull/36623))

- Fix poor performance of the recurrent operator in multi-threaded scenarios. ([#36052](https://github.com/PaddlePaddle/Paddle/pull/36052))

- Remove restrictions of relu, sigmoid, tanh, relu6, batch_norm, clip, concat, gelu, hard_sigmoid, prelu, softmax, split, and swish on TensorRT 2-dimensional inputs. ([#37097](https://github.com/PaddlePaddle/Paddle/pull/37097))

- Fix reshape op to use TensorRT inference. ([#41090](https://github.com/PaddlePaddle/Paddle/pull/41090))

- Fix matmul related pass, which is compatible with matmul_v2. ([#36424](https://github.com/PaddlePaddle/Paddle/pull/36424))

- Support VALID and SAME attributes in the padding method of the conv2d operator when TensorRT is enabled. ([#38999](https://github.com/PaddlePaddle/Paddle/pull/38999))

- Fix MKLDNN multi-input operator quantization problem. ([#39593](https://github.com/PaddlePaddle/Paddle/pull/39593), [#39346](https://github.com/PaddlePaddle/Paddle/pull/39346), [#40717](https://github.com/PaddlePaddle/Paddle/pull/40717))

- Fix scale error of conv+activation in MKLDNN quantization scenarios. ([#38331](https://github.com/PaddlePaddle/Paddle/pull/38331))

- Fix the bug in MKLDNN quantization without parameters where the quantization of subsequent operators is handled differently. ([#39342](https://github.com/PaddlePaddle/Paddle/pull/39342))

- Fix a data type related issue in MKLDNN cpu_bfloat16_placement_pass. ([#38702](https://github.com/PaddlePaddle/Paddle/pull/38702))

- Fix a split operator execution issue in MKLDNN bfloat16 inference. ([#39548](https://github.com/PaddlePaddle/Paddle/pull/39548))

- Fix the bug with MKLDNN matmul_v2 operator not supporting 6 dimensions. ([#36342](https://github.com/PaddlePaddle/Paddle/pull/36342), [#38665](https://github.com/PaddlePaddle/Paddle/pull/38665))

- Fix MKLDNN DeviceContext error in MKLDNN matmul_v2_transpose_reshape. ([#38554](https://github.com/PaddlePaddle/Paddle/pull/38554))

- Fix incorrectly calculated results for segmentation models in MKLDNN inference scenarios. ([#37310](https://github.com/PaddlePaddle/Paddle/pull/37310))

- Fix MKLDNN bfloat16 placement operator list and add the missing operator. ([#36291](https://github.com/PaddlePaddle/Paddle/pull/36291))

- Fix the format bug of MKLDNN operators, including: FC, conv_transpose, 6-dimensional Tensor error reporting, and wrong output format of conv to NHWC input. ([#38890](https://github.com/PaddlePaddle/Paddle/pull/38890), [#37344](https://github.com/PaddlePaddle/Paddle/pull/37344), [#37175](https://github.com/PaddlePaddle/Paddle/pull/37175), [#38553](https://github.com/PaddlePaddle/Paddle/pull/38553), [#40049](https://github.com/PaddlePaddle/Paddle/pull/40049), [#39097](https://github.com/PaddlePaddle/Paddle/pull/39097))

- Fix MKLDNN multi-threaded reasoning scenario error due to cache mechanism. ([#36290](https://github.com/PaddlePaddle/Paddle/pull/36290), [#35884](https://github.com/PaddlePaddle/Paddle/pull/35884))

- Fix MKLDNN quantization model accuracy anomaly caused by matmul and FC. ([#38023](https://github.com/PaddlePaddle/Paddle/pull/38023), [#37618](https://github.com/PaddlePaddle/Paddle/pull/37618))

- Fix the abnormal quantization model accuracy issue in MKLDNN quantization conversion scripts caused by missing passes. ([#37619](https://github.com/PaddlePaddle/Paddle/pull/37619), [#40542](https://github.com/PaddlePaddle/Paddle/pull/40542),[#38912](https://github.com/PaddlePaddle/Paddle/pull/38912))

- Fix the crash bug in MKLDNN enabling volume op due to data type mismatch. ([#38133](https://github.com/PaddlePaddle/Paddle/pull/38133))

- Fix an issue where some MKLDNN ops need to change back to the original layout after modifying the layout. ([#39422](https://github.com/PaddlePaddle/Paddle/pull/39422))

- Fix the bug of Python API error report due to conflict with Ascend software stack, because the GIL lock is not released in the Ascend 910 inference scenario. ([#38605](https://github.com/PaddlePaddle/Paddle/pull/38605))


## **5. Environment Adaptation**

### **Compile and Install**

- From version 2.3.0, PaddlePaddle has adjusted and upgraded the types of GPU architectures supported by the framework. (For more information, please refer to: [GPU architectures supported by PaddlePaddle](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.3rc/install/Tables.html#gpu))


Notes:

- PIP source installation means downloading the installation package and dependency libraries from PIP official website with using `pip install paddlepaddle` or `pip install paddlepaddle-gpu`. This supports less architecture types, and lighter installation package,and only one CUDA version of the installation package is provided(compared with BOS source).

  - Prior to version 2.3, the PIP source installer (CUDA10.2) supports the following GPU architectures: 3.5, 5.0, 5.2, 6.0, 6.1, 7.0, and 7.5.

  - Later than version 2.3, the PIP source installer (CUDA11.0) supports the following GPU architectures: 6.0, 6.1, 7.0, 7.5, 8.0

- The BOS source is a way to download the installation package and dependency libraries from the official website of PaddlePaddle, which supports more GPU architectures. The download source is from China and it is much faster. (compared with PIP source, it supports more kinds of architectures and provides multiple CUDA versions of installation packages).

  - Prior to version 2.3, the GPU architectures supported by the bos source installer on the PaddlePaddle website:

    - CUDA10: 3.5, 5.0, 5.2, 6.0, 6.1, 7.0, 7.5；

    - CUDA11: 5.2，6.0，6.1，7.0，7.5，8.0。

  - Later than version 2.3, the GPU architectures supported by the bos source installer on the PaddlePaddle website:

    - CUDA10: 3.5, 5.0, 5.2, 6.0, 6.1, 7.0, 7.5；

    - CUDA11: 3.5, 5.0, 6.0, 6.1, 7.0, 7.5, 8.0。

- Support Python 3.10. Fix compilation bugs caused by some PythonC API changes on Windows. ([#41180](https://github.com/PaddlePaddle/Paddle/pull/42180))

- The Windows platform supports the compilation through Visual Studio 2019. ([#38719](https://github.com/PaddlePaddle/Paddle/pull/38719))

- Eliminate various warnings when compiling on the Windows platform. ([#38034](https://github.com/PaddlePaddle/Paddle/pull/38034), [#37890](https://github.com/PaddlePaddle/Paddle/pull/37890), [#37442](https://github.com/PaddlePaddle/Paddle/pull/37442), [#37439](https://github.com/PaddlePaddle/Paddle/pull/37439), [#36857](https://github.com/PaddlePaddle/Paddle/pull/36857))

- Fix jetson compilation issues introduced by the underlying data structure upgrade. ([#39669](https://github.com/PaddlePaddle/Paddle/pull/39669), [#39441](https://github.com/PaddlePaddle/Paddle/pull/39441))


### **New Hardware Backend Extention**

- Custom device support: provide a plug-in way to extend PaddlePaddle hardware backend. With this function, developers do not need to modify PaddlePaddle codes for specific hardware, but simply implement the standard interface and compile it into a dynamic link library that can be called by PaddlePaddle as a plug-in.This reduces the development effort of adding a new hardware backend to PaddlePaddle. Currently it supports custom Runtime and custom Kernel.

- Support Huawei NPU chip (Ascend910) training/inference. Support ResNet50, YoloV3, BERT, Transformer and many other models. Support static + dynamic graph and auto-mixed precision training. Support single card, and distribute training across multiple cards, multiple machines.

- Support Graphcore IPU chip (including IPU Mk2 GC200 and Bow IPU) training/inference. Support ResNet50, BERT and other models. Support static graph training. Support single card, and distribute training across multiple cards, multiple machines.

- Support cambricon MLU chip (MLU370x4) training/inference. Support models such as ResNet50. Support static graph + dynamic graph training. Support auto-mixed precision training. Support single card, and distribute training across multiple cards, multiple machines.

- Support KUNLUNXIN 2 chips (KUNLUNXIN AI acceleration cards R200, R300) training/inference. Support ResNet50, YoloV3, OCR-DB, SSD, MobilnetV3, UNet, BERT, Transformer, GPT-2, Wide&Deep, and DeepFM. Support static graph + dynamic graph training. Support auto-mixed precision training. Support single card, and distribute training across multiple cards, multiple machines.


## Thanks to our Contributors

This release contains contributions from the project core team as well as:

Adam Osewski, Allen Guo, arlesniak, chenenquan, chenyanlann, fengkuangxiaxia, fuqianya, fwenguang, guguguzi, helen88, houj04, Jacek Czaja, jakpiase, jianghaicheng, joanna.wozna.intel, joeqiao12, Leo Chen, Leo Guo, Li-fAngyU, lidanqing, Liyulingyue, Matsumoto GAO, maxhuiy, Ming-Xu Huang, Nyakku Shigure, piotrekobi, piotrekobiIntel, QingshuChen, qipengh, Skr Bang, Sylwester Fraczek, Sławomir Siwek, taixiurong, tanzhipeng, Tomasz Socha, TTerror, Webbley, yaozhixin, ykkk2333, yujun, Zhangjingyu06, zhangxiaoci, zhangyikun02, zhangyk0314, zlsh80826, zn, Zuza.
