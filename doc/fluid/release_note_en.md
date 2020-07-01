# Release Note

## Important Statements

- This version is a beta version. It is still in iteration and is not stable at present. Incompatible upgrade may be subsequently performed on APIs based on the feedback. For developers who want to experience the latest features of Paddle, welcome to this version. For industrial application scenarios requiring high stability, the stable Paddle Version 1.8 is recommended.
- This version mainly popularizes the imperative programming development method and provides the encapsulation of high-level APIs. The imperative programming(dynamic graph) mode has great flexibility and high-level APIs can greatly reduces duplicated codes. For beginners or basic task scenarios, the high-level API development method is recommended because it is simple and easy to use. For senior developers who want to implement complex functions, the imperative programming API is commended because it is flexible and efficient.
- This version also optimizes the Paddle API directory system. The APIs in the original directory can create an alias and are still available, but it is recommended that new programs use the new directory structure.

## Basic Framework

### Basic APIs

- Networking APIs achieve dynamic and static unity and support operation in imperative programming and declarative programming modes(static graph)

- The API directory structure is adjusted. In the Paddle Version 1.x, the APIs are mainly located in the paddle.fluid directory. This version adjusts the API directory structure so that the classification is more reasonable. The specific adjustment rules are as follows:

  - Moves the APIs related to the tensor operations in the original fluid.layers directory to the paddle.tensor directory
  - Moves the networking-related operations in the original fluid.layers directory to the paddle.nn directory. Puts the types with parameters in the paddle.nn.layers directory and the functional APIs in the paddle.nn.functional directory
  - Moves the special API for imperative programming in the original fluid.dygraph directory to the paddle.imperative directory
  - Creates a paddle.framework directory that is used to store framework-related program, executor, and other APIs
  - Creates a paddle.distributed directory that is used to store distributed related APIs
  - Creates a paddle.optimizer directory that is used to store APIs related to optimization algorithms
  - Creates a paddle.metric directory that is used to create APIs related to evaluation index calculation
  - Creates a paddle.incubate directory that is used to store incubating codes. APIs may be adjusted. This directory stores codes related to complex number computation and high-level APIs
  - Creates an alias in the paddle directory for all APIs in the paddle.tensor and paddle.framework directories. For example, paddle.tensor.creation.ones can use paddle.ones as an alias

- The added APIs are as follows:

  - Adds eight networking APIs in the paddle.nn directory: interpolate, LogSoftmax, ReLU, Sigmoid, loss.BCELoss, loss.L1Loss, loss.MSELoss, and loss.NLLLoss
  - Adds 59 tensor-related APIs in the paddle.tensor directory: add, addcmul, addmm, allclose, arange, argmax, atan, bmm, cholesky, clamp, cross, diag\_embed, dist, div, dot, elementwise\_equal, elementwise\_sum, equal, eye, flip, full, full\_like, gather, index\_sample, index\_select, linspace, log1p, logsumexp, matmul, max, meshgrid, min, mm, mul, nonzero, norm, ones, ones\_like, pow, randint, randn, randperm, roll, sin, sort, split, sqrt, squeeze, stack, std, sum, t, tanh, tril, triu, unsqueeze, where, zeros, and zeros\_like
  - Adds device\_guard that is used to specify a device. Adds manual\_seed that is used to initialize a random number seed

- Some of the APIs in the original fluid directory have not been migrated to the paddle directory
  - The following API under fluid.contrib directory are kept in the original location, not migrated：BasicGRUUnit, BasicLSTMUnit, BeamSearchDecoder, Compressor, HDFSClient, InitState, QuantizeTranspiler, StateCell, TrainingDecoder, basic_gru, basic_lstm, convert_dist_to_sparse_program, ctr_metric_bundle, extend_with_decoupled_weight_decay, fused_elemwise_activation, fused_embedding_seq_pool, load_persistables_for_increment, load_persistables_for_inference, match_matrix_tensor, memory_usage, mixed_precision.AutoMixedPrecisionLists, mixed_precision.decorate, multi_download, multi_upload, multiclass_nms2, op_freq_statistic, search_pyramid_hash, sequence_topk_avg_pooling, shuffle_batch, tree_conv, var_conv_2d
  - The following APIs related to LodTensor are still under development and have not been migrated yet：LoDTensor, LoDTensorArray, create_lod_tensor, create_random_int_lodtensor, DynamicRNN, array_length, array_read, array_write, create_array, ctc_greedy_decoder, dynamic_gru, dynamic_lstm, dynamic_lstmp, im2sequence, linear_chain_crf, lod_append, lod_reset, sequence_concat, sequence_conv, sequence_enumerate, sequence_expand, sequence_expand_as, sequence_first_step, sequence_last_step, sequence_mask, sequence_pad, sequence_pool, sequence_reshape, sequence_reverse, sequence_scatter, sequence_slice, sequence_softmax, sequence_unpad, tensor_array_to_tensor
  - The following APIs related to distributed training are still under development, not migrated yet
  - The following APIs in fluid.nets directory will be implemented with high level API， not migrated：nets.glu, nets.img_conv_group, nets.scaled_dot_product_attention, nets.sequence_conv_pool, nets.simple_img_conv_pool
  - The following APIs are to be improved, not migrated：dygraph.GRUUnit, layers.DecodeHelper, layers.GreedyEmbeddingHelper, layers.SampleEmbeddingHelper, layers.TrainingHelper, layers.autoincreased_step_counter, profiler.cuda_profiler, profiler.profiler, profiler.reset_profiler, profiler.start_profiler, profiler.stop_profiler
  - The following APIs are no longer recommended and are not migrated：DataFeedDesc, DataFeeder, clip.ErrorClipByValue, clip.set_gradient_clip, dygraph_grad_clip.GradClipByGlobalNorm, dygraph_grad_clip.GradClipByNorm, dygraph_grad_clip.GradClipByValue, initializer.force_init_on_cpu, initializer.init_on_cpu, io.ComposeNotAligned.with_traceback, io.PyReader, io.load_params, io.load_persistables, io.load_vars, io.map_readers, io.multiprocess_reader, io.save_params, io.save_persistables, io.save_vars, io.xmap_readers, layers.BasicDecoder, layers.BeamSearchDecoder, layers.Decoder, layers.GRUCell, layers.IfElse, layers.LSTMCell, layers.RNNCell, layers.StaticRNN, layers.Switch, layers.While, layers.create_py_reader_by_data, layers.crop, layers.data, layers.double_buffer, layers.embedding, layers.fill_constant_batch_size_like, layers.gaussian_random_batch_size_like, layers.get_tensor_from_selected_rows, layers.load, layers.merge_selected_rows, layers.one_hot, layers.py_reader, layers.read_file, layers.reorder_lod_tensor_by_rank, layers.rnn, layers.uniform_random_batch_size_like, memory_optimize, release_memory, transpiler.memory_optimize, transpiler.release_memory

### High-level APIs

- Adds a paddle.incubate.hapi directory. Encapsulates common operations such as networking, training, evaluation, inference, and access during the model development process. Implements low-code development. Uses the imperative programming implementation mode of MNIST task comparison. High-level APIs can reduce 80% of executable codes.
- Adds model-type encapsulation. Inherits the layer type. Encapsulates common basic functions during the model development process, including:
  - Provides a prepare API that is used to specify a loss function and an optimization algorithm
  - Provides a fit API to implement training and evaluation. Implements the execution of model storage and other user-defined functions during the training process by means of callback
  - Provides an evaluate interface to implement the inference and evaluation index calculation on the evaluation set
  - Provides a predict interface to implement specific test data inference
  - Provides a train\_batch interface to implement the training of single-batch data
- Adds a dataset interface to encapsulate commonly-used data sets and supports random access to data
- Adds encapsulation of common Loss and Metric types
- Adds 16 common data processing interfaces including Resize and Normalize in the CV field
- Adds lenet, vgg, resnet, mobilenetv1, and mobilenetv2 image classification backbone networks in the CV field
- Adds MultiHeadAttention, BeamSearchDecoder, TransformerEncoder, TransformerDecoder, and DynamicDecode APIs in the NLP field
- Releases 12 models based on high-level API implementation, including Transformer, Seq2seq, LAC, BMN, ResNet, YOLOv3, VGG, MobileNet, TSM, CycleGAN, Bert, and OCR

### Performance Optimization

- Adds a `reshape+transpose+matmul` fuse so that the performance of the INT8 model is improved by about 4% (on the 6271 machine) after Ernie quantization. After the quantization, the speed of the INT8 model is increased by about 6.58 times compared with the FP32 model on which DNNL optimization (including fuses) and quantization are not performed

### Debugging Analysis

- To solve the problem of program printing contents being too lengthy and low utilization efficiency during debugging, considerably simplifies the printing strings of objects such as programs, blocks, operators, and variables, thus improving the debugging efficiency without losing effective information
- To solve the problem of insecure third-party library APIs `boost::get` and difficulty in debugging due to exceptions during running, adds the `BOOST_GET` series of macros to replace over 600 risky `boost::get` in Paddle. Richens error message during `boost::bad_get` exceptions. Specifically, adds the C++ error message stack, error file and line No., expected output type, and actual type, thus improving the debugging experience

## Bug Fixes

- Fix the bug of wrong computation results when any slice operation exists in the while loop
- Fix the problem of degradation of the transformer model caused by inplace ops
- Fix the problem of running failure of the last batch in the Ernie precision test
- Fix the problem of failure to correctly exit when exceptions occur in context of fluid.dygraph.guard
