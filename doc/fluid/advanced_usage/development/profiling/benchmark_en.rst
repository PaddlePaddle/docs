#########################
How to do benchmark test
#########################

The document introduces how to do benchmark test for deep learning framework. The work for benchmark test mainly covers accuracy and performance of the model. This article also involves contents like building test environment, choosing benchmark test model and verifying test results.

The verification of deep learning framework can be divided into two stages, namely training and testing. The verification metrics are slightly different in the two phases. This article only introduces the verification of metrics in the training phase. 

The training phase focuses on the accuracy of the model training set. The training set is complete. Therefore, it emphasizes the training speed for the big batch\_size and pays attention to the throughput. For example, the batch\_size=128 is commonly used in the image model and will increase in the case of multi-card. 

The inference phase focuses on the accuracy on the test set. Online service test data can not be collected in advance, so it emphasizes the prediction speed for small batch\_size as well as latency, such as batch\_size=1, 4 which is commonly used by inference service.


`Fluid <https://github.com/PaddlePaddle/Paddle>`__ is the design that PaddlePaddle introduced since version 0.11.0. The benchmark test in this article was completed on this version.


Build Environment
""""""""""""""""""

The accuracy of the model in benchmark test is independent of hardware and framework, but is determined by the model structure and data; the performance is determined by the test hardware and framework performance. Framework benchmark is to compare the differences between the frameworks, so it keeps versions of hardware environment, the system library consistent. The comparative experiment below is performed under the same hardware conditions and system environment conditions.

The performance of GPU cards of different architectures is vastly different. When verifying the training performance of the model on GPU, you can use the tool `nvidia-smi` provided by NVIDIA to check the GPU model in use. For testing the performance of multi-card training, you need to confirm the hardware connection is `nvlink <https://zh.wikipedia.org/en/NVLink>`__ or `PCIe <https://zh.wikipedia.org/zh-hans/PCI_Express>`__ . Similarly, the CPU model can greatly affect the training performance of the model on the CPU. The parameters in `/proc/cpuinfo` can be read to confirm the CPU model currently in use.

Download the GPU-compatible Cuda Tool Kit and Cudnn, or use NVIDIA's official nvidia-docker image `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`__, which contains Cuda and Cudnn. This article uses this approach. The Cuda Tool Kit contains the base library used by the GPU code which affects the performance of the binary Fluid compiled on this basis.

After preparing the Cuda environment, download Paddle from github and compile the source code to generate the corresponding sm\_arch binary \ `sm\_arch <https://docs.nvidia.com/cuda/cuda-compiler -driver-nvcc/index.html>`__\ . In addition, cudnn has a huge impact on convolution tasks, and requires versions in the benchmark test consistent. For example, Cudnn7.0.2 and Cudnn7.1.4 have more than 5% difference on Resnet.

Choose Benchmark Model
""""""""""""""""""""""""""""""""

To do benchmark test on the framework, it is necessary to cover different training tasks and different sizes of models. Here the five most commonly used models of images and NLP are selected.

=====================   ==============  =================  ============
Type of task            Model name       Network Structure  Dataset   
=====================   ==============  =================  ============
Image Classification      mnist         Lenet              mnist
Image Classification      VGG           VGG-16             Flowers102
Image Classification      Resnet        Resnet-50          Flowers102
Text  Classification     Stacked-LSTM   Stacked-LSTM       IMDB 
Machine Translation      seq-seq        Stacked-LSTM       wmt14 
=====================  ==============  =================  ============

mnist, VGG, Resnet are CNN model, while stacked-lstm, seq2seq represent RNN model.

In the `benchmark <https://github.com/PaddlePaddle/Paddle/tree/develop/benchmark/fluid>`__ model test script, the training process of the first several batch is skipped. The reason is that the loading data and the allocated memory are affected by the current condition of the system, which may result in inaccurate statistical performance. After running several rounds, the corresponding metrics are counted.


For the selection of data for the benchmark model, a public dataset with a large amount of data and a large number of verification effects is preferred, such as Image model VGG and resnet. Here we choose `flowers102 <http://www.robots.ox.ac.uk/~vgg/data/flowers/102/>`__ , the image size is preprocessed to the same size as Imagenet, so Performance can be directly compared.
The NLP model has fewer public and influential large datasets. The seq2seq model selects wmt14 data, and the stacked-lstm model selects `imdb <https://www.imdb.com/interfaces/>`__ data.


Note that each sample of the image model has the same size, and the image has the same size after transformation. Therefore, the calculation paths are basically the same, the fluctuation of calculation speed and the memory usage is relatively small, and the current training performance data can be sampled from the data of a number of batches. 

However, NLP models have varied sample lengths, the calculation path and memory usage are different. Therefore, the speed and memory consumption can only be calculated after several rounds of complete operation.

(Video) Memory allocation is a particularly time-consuming operation, so Fluid by default occupies all available memory space to form a memory pool to speed up the allocation of memory during the calculation process. If you need to calculate the actual memory consumption of the model, you can set the environment variable `FLAGS_fraction_of_gpu_memory_to_use=0.0` to observe the maximum memory consumption.

Test Process
"""""""""""""

-  Test on single node CPU with single thread

To test the performance of a single thread on the CPU, first set the CUDA environment variable to NULL, ``CUDA_VISIBLE_DEVICES=``, and close the OpenMP and MKL multithreading through setting ``OMP_NUM_THREADS=1`` , ``MKL_NUM_THREADS=1;`` .
Then the code is set to use CPUPlace, if you use the script in the Paddle repository, you only need to pass use_gpu=False to the command line argument.

.. code-block:: python

    import paddle.fluid as fluid
    place = fluid.CPUPlace() 

.. code:: bash

    docker run -it --name CASE_NAME --security-opt seccomp=unconfined -v $PWD/benchmark:/benchmark paddlepaddle/paddle:latest-dev /bin/bash


-  Test on single node and single card GPU

This tutorial uses Cuda8, Cudnn7.0.1. The source is ``nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04`` .

.. code:: bash

    nvidia-docker run -it --name CASE_NAME --security-opt seccomp=unconfined -v $PWD/benchmark:/benchmark -v /usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu paddlepaddle/paddle:latest-dev /bin/bash

Test on a single card, set the CUDA environment variable to use one GPU, ``CUDA_VISIBLE_DEVICES=0``
Then the code is set to use CUDAPlace. If you use the script in the Paddle repository, you only need to pass use_gpu=True to the command line argument.

.. code-block:: python

    import paddle.fluid as fluid
    place = fluid.CUDAPlace(0) // 0 represent the zero place


Test Result
""""""""""""
This tutorial compares the performance of Fluid0.12.0 and TensorFlow1.4.0 in the same environment.
The hardware environment is CPU: Intel(R) Xeon(R) CPU E5-2660 v4 @ 2.00GHz, GPU: TITAN X(Pascal) 12G x 1, Nvidia-Driver 384.90.
The system environment is Ubuntu 16.04.3 LTS. The docker environment is used in here. The system version is nvidia-docker17.05.0-ce.
The Fluid version used in the test is \ `v.0.12.0 <https://github.com/PaddlePaddle/Paddle/releases/tag/v.0.12.0>`__ .
The version of TensorFlow is \ `v.1.4.0-rc1 <https://github.com/tensorflow/tensorflow/tree/v1.4.0-rc1>`__ .
Check the script `benchmark <https://github.com/PaddlePaddle/Paddle/tree/develop/benchmark/fluid>`__ for the scripts and configuration used.
The unit in the chart is samples/second.


- Test result for single node CPU with single thread

  ================  ====================  ===================
   Speed            Fluid CPU              TensorFlow CPU    
  ================  ====================  ===================
  mnist             1298.75 samples/s     637.57 samples/s  
  VGG-16            0.4147 images/s       0.1229 images/s   
  Resnet-50         1.6935 images/s       0.3657 images/s   
  Stacked-LSTM      472.3225 words/s      48.2293words/s    
  Seq2Seq           217.1655 words/s      28.6164 words/s   
  ================  ====================  ===================

- Test result for single card GPU

  =============== =====================  =================
   Speed           Fluid GPU              TensorFlow GPU      
  =============== =====================  =================
   mnist           19710.90 samples/s    15576.3 samples/s        
   VGG-16          59.83327 images/s     40.9967 images/s    
   Resnet-50       105.84412             97.8923 images/s    
   Stacked-LSTM    1319.99315            1608.2526 words/s   
   Seq2Seq         7147.89081            6845.1161 words/s   
  =============== =====================  =================
