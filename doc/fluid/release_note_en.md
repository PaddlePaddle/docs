# Release Notes

## Framework

* new pip installation package is available, which can be run on Windows CPU environment.
* support of python3.6、python3.7
* Reconstruction of memory allocator modular :Allocator. Improvement on memory allocating strategy in CPU environment.
  Increase in utility ratio of video memory (disabled by default, use ``FLAGS_allocator_strategy`` to enable it).
* Restriction to the usage of SelectedRows, and fix made to bugs on sparse regulation and sparse optimization.
* Tensor supports DLPack，to facilitate integration of other frameworks or into them.
* OP
	* Issues on inference of expand op ``shape`` have been resolved.
	* Activation function ``Selu`` is included.


## Inference Engine
* Server Prediction
	* GPU supports image fusion, and cooperation with TensorRT to realize image modifying. In common image processing models like Resnet50 and Googlenet, with bs=1, the performance has reached a level 50~100% higher.
	* GPU supports DDPG Deep Explore prediction.
	* Paddle-TRT supports more models, including Resnet， SE-Resnet， DPN，GoogleNet.
	* CPU, GPU, TensorRT and other accelerators are merged into AnalysisPredictor，collectively controlled by AnalysisConfig.
	* Add interfaces to call multi-thread mathematic library.
	* Support for TensorRT plugins，including `split operator` , `prelu operator` ,  `avg_pool operator` , `elementwise_mul operator` .
	* This version has included JIT CPU Kernel， which is able to perform basic vector operations, partial implementation of common algorithms including ReLU，LSTM and GRU, and automatic runtime switch between AVX and AVX2 instruction set.
	* FDSFDF optimized CRF decoding and implementation of LayerNorm on AVX and AVX2 instruction set.
	* Issue fixed: AnalysisPredictor on GPU or in the transition from CPU to GPU cannot delete transfer data.
	* Issue fixed: Variable has consistent increase of occupied memory of container.
	* Issue fixed: `fc_op` cannot process 3-D Tensor
	* Issue fixed: on GPU, when running pass, issues happened to Analysis predictor
	* Issue fixed: GoogleNet problems on TensorRT
	* Promotion of prediction performance
		* Max Sequence pool optimization，with single op performance 10% higher.
		* `Softmax operator` optimization，with single op performance 14% higher.
		* `Layer Norm operator` optimization， inclusive of AVX2 instruction set, with single op performance 5 times higher.
		* `Stack operator` optimization，with single op performance 3.6 times higher.
		* add depthwise_conv_mkldnn_pass to accelerate MobileNet prediction.
		* reduce image analysis time in analysis mode， and the velocity is 70 times quicker.
		* DAM open-source model，reached 118.8% of previous version.
* Mobile Endpoint Prediction
	* This version has realized winograd algorithm， with the help of which the performance of GoogleNet v1 enjoys a dramatic promotion of 35%.
	* improvement on GoogleNet 8bit，14% quicker compared with float.
	* support for MobileNet v1 8bit， 20% faster than float.
	* support for MobileNet v2 8bit， 19% faster than float.
	* FPGA V1 has developed Deconv operator
	* Android gpu supports mainstream network models like MobileNet、MobileNetSSD、GoogleNet、SqueezeNet、YOLO、ResNet.


## Model

* CV image classifying tasks publish pre-trained models: MobileNet V1, ResNet101, ResNet152，VGG11
* CV Metric Learning models are extended with loss function arcmargin, and the training method is altered. The new method is to adopt element-wise as pre-trained model, and use pair-wise to make further slight adjustment to improve precision.
* NLP model tasks are newly equipped with LSTM implementation based on cudnn. Compared with the implementation based on PaddingRNN, the cudnn method is 3~5 times quicker under diverse argument settings.
* Distributed word2vec model is included，including the new tree-based softmax operator，negative sampling，in line with classic word2vec algorithms.
* Distributed settings of GRU4Rec、Tag-Space algorithms are added.
* Multi-view Simnet model is optimized， with an additional inference setting.
* Reinforcement learning algorithm DQN is supported.
* Currently compatible python3.x models: Semantic model DAM, reading comprehension BiDAF, machine translation Transformer, language model, reinforcement learning DQN, DoubleDQN model, DuelingDQN model, video classification TSN, Metric Learning, character recognition in natural scenes CRNN-CTC 、OCR Attention，Generative Adversarial Networks ConditionalGAN, DCGAN, CycleGAN, Semantic segmentation ICNET, DeepLab v3+, object detection Faster-RCNN, MobileNet-SSD, PyramidBox, iSE-ResNeXt, ResNet, customized recommendation TagSpace、GRU4Rec、SequenceSemanticRetrieval、DeepCTR、Multiview-Simnet.


## Distributed training

* multi-CPU asynchronous training
	* Asynchronous concurrent workers： `AsyncExecutor` is added. With a executive granularity of single training file, it supports lock-less asynchronous worker-end computation in distributed training, and single machine training. Take CTR task as an example, general throughput from single machine training is 14 times larger.
	* IO optimization：This version has added compatibility with `AsyncExecutor` to DataFeed; enabled customized universal classification task formats; incorporated CTRReader for CTR tasks to linearly elevate speed of reading data. In PaddleRec/ctr tasks，the general throughput increases by 2 times.
	* Better data communication： As for sparse access Dense arguments, like Embedding, the sparse data communication mechanism is adopted. Take tasks of semantic matching for instance, the amount of fetched arguments can be compressed to 1% and below. In searching groundtruth data, the general output reached 15 times more.
* multi-GPU synchronous training
	* Issue fixed: In Transformer、Bert models, P2P training mode may be hung.

## Documentation

* API
	* Add 13 api guides
	* Add 300 entries of Chinese API Reference
	* Improve 77 entries of English API Reference, including Examples and argument explanation and other adjustable sections.
* Documentation about installation
	* Add installing guide on python3.6、python3.7.
	* Add installing guide on windows pip install.
* Book Documentation
	* Code examples in Book documentation are substituted with Low level API.
