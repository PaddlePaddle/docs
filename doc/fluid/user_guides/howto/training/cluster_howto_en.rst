.. _cluster_howto_en:

Manual for Distributed Training with Fluid
==========================================

Basic Idea Of Distributed Training
-------------------------------------

Distributed deep learning training is usually divided into two parallelization methods: data parallelism, model parallelism. Refer to the following figure:

.. image:: src/parallelism.png

In the model parallelism mode, the layers and parameters of the model will be distributed on multiple nodes. The model will go through multiple communications across nodes in the feeding forward and back propagation training of a mini-batch. Each node only saves a part of the entire model; 

In data parallelism mode, each node holds the complete layers and parameters of the model, each node performs feeding forward and back propagation calculations on its own, and then conducts the aggregation of the gradients and updates the parameters on all nodes synchronously. 

Current version of Fluid only provides data parallelism mode. In addition, implementations of special cases in model parallelism mode (e.g. large sparse model training ) will be explained in subsequent documents.

In the training of data parallelism mode, Fluid uses two communication modes to deal with the requirements of distributed training for different training tasks, namely RPC Communication and Collective Communication. The RPC communication method uses `gRPC <https://github.com/grpc/grpc/>`_ , Collective communication method uses `NCCL2 <https://developer.nvidia.com/nccl>`_ . 

.. csv-table:: The table above is a horizontal comparison of RPC communication and Collective communication
	:header: "Feature", "Collective", "RPC"

	"Ring-Based Communication", "Yes", "No"
	"Asynchronous Training", "Yes", "Yes"
	"Distributed Model", "No", "Yes"
	"Fault-tolerant Training", "No", "Yes"
	"Performance", "Faster", "Fast"

- Structure of RPC Communication Method:

  .. image:: src/dist_train_pserver.png

  Data-parallelised distributed training in RPC communication mode will start multiple pserver processes and multiple trainer processes, each pserver process will save a part of the model parameters and be responsible for receiving the gradients sent from the trainers and updating these model parameters; Each trainer process will save a copy of the complete model, and use a part of the data to train, then send the gradients to the pservers, finally pull the updated parameters from the pserver.

  The pserver process can be on a compute node that is completely different from the trainer, or it can share the same node with a trainer. The number of pserver processes required for a distributed task usually needs to be adjusted according to the actual situation to achieve the best performance. However, usually pserver processes are no more than trainer processes.

  When using GPU training, the pserver can choose to use the GPU or only use the CPU. If the pserver also uses the GPU, it will result in the extra overhead of copying the gradient data received from the CPU to the GPU. In some cases, the overall training performance will be degraded.

- Structure of NCCL2 communication method:

  .. image:: src/dist_train_nccl2.png

NCCL2 (Collective communication method) for distributed training avoids the need of pserver processes. Each trainer process holds a complete set of model parameters. After the calculation of the gradient, the trainer, through mutual communications, "Reduce" the gradient data to all devices of all nodes and then each node completes parameter updates of its own.

Training in the Parameter Server Manner 
----------------------------------------------

Use the :code:`transpiler` API to quickly convert a program that can be executed on a single machine into a program that can be executed in a distributed manner. On different server nodes, pass values to corresponding arguments at :code:`transpiler` to get the :code:`Program` which current node is to execute:


.. csv-table:: required configuration parameters
   :header: "parameter", "description"

   "role", "\ **required**\  distinguishes whether to start as pserver or trainer, this arugument is not passed into ``transpile`` , you can also use other variable names or environment variables"
   "trainer_id", "\ **required**\  If it is a trainer process, it is used to specify the unique id of the current trainer in the task, starting from 0, and must be guaranteed not to be repeated in one task"
   "pservers", "\ **required**\ ip:port list string of all pservers in current task, for example: 127.0.0.1:6170,127.0.0.1:6171"
   "trainers", "\ **required**\  the number of trainer nodes"
   "sync_mode", "\ **optional**\  True for synchronous mode, False for asynchronous mode"
   "startup_program", "\ **optional**\  If startup_program is not the default fluid.default_startup_program(), this parameter needs to be passed in"
   "current_endpoint", "\ **optional**\  This parameter is only required for NCCL2 mode"

For example, suppose there are two nodes, namely :code:`192.168.1.1` and :code:`192.168.1.2`, use port 6170 to start 4 trainers.
Then the code can be written as:

.. code-block:: python

	role = "PSERVER"
	trainer_id = 0 # get actual trainer id from cluster
	pserver_endpoints = "192.168.1.1:6170,192.168.1.2:6170"
	current_endpoint = "192.168.1.1:6170" # get actual current endpoint
	trainers = 4
	t = fluid.DistributeTranspiler()
	t.transpile(trainer_id, pservers=pserver_endpoints, trainers=trainers)
	if role == "PSERVER":
		    pserver_prog = t.get_pserver_program(current_endpoint)
		    pserver_startup = t.get_startup_program(current_endpoint,Pserver_prog)
		    exe.run(pserver_startup)
		    exe.run(pserver_prog)
	elif role == "TRAINER":
		train_loop(t.get_trainer_program())


Choose Synchronous Or Asynchronous Training
+++++++++++++++++++++++++++++++++++++++++++++

Fluid distributed tasks support synchronous training or asynchronous training. 

In the synchronous training mode, all trainer nodes will merge the gradient data of all nodes synchronously per mini-batch and send them to the parameter server to complete the update. 

In the asynchronous mode, each trainer does not wait for each other, and independently update the parameters on the parameter server. 

In general, using the asynchronous training method can have a higher overall throughput than the synchronous training mode when there are more trainer nodes.

When the :code:`transpile` function is called, the distributed training program is generated by default. The asynchronous training program can be generated by specifying the :code:`sync_mode=False` parameter:

.. code-block:: python

	t.transpile(trainer_id, pservers=pserver_endpoints, trainers=trainers, sync_mode=False)



Whether To Use The Distributed Embedding Table For Training
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Embedding is widely used in various network structures, especially text processing related models.
In some scenarios, such as recommendation systems or search engines, the number of feature ids of embedding may be very large. When it reaches a certain number, the embedding parameter will become very large.
On the one hand, the memory of the single machine may not be competent, resulting in the inability to train.
On the other hand, the normal training mode needs to synchronize the complete set of parameters for each iteration. If the parameter is too large, the communication will become very slow, which will affect the training speed.

Fluid supports the training of very large scale sparse features embedding at hundred billion level. The embedding parameter is only saved on the parameter server. The parameter prefetch and gradient sparse update method greatly reduce the traffic and improve the communication speed.

This feature is only valid for distributed training and cannot be used on a single machine. Need to be used with sparse updates.

Usage: When configuring embedding, add the parameters :code:`is_distributed=True` and :code:`is_sparse=True`.
Parameters :code:`dict_size` Defines the total number of ids in the data. The id can be any value in the int64 range. As long as the total number of ids is less than or equal to dict_size, it can be supported.
So before you configure, you need to estimate the total number of feature ids in the data.

.. code-block:: python

	emb = fluid.layers.embedding(
		is_distributed=True,
		input=input,
		size=[dict_size, embedding_width],
		is_sparse=True)


Select Parameter Distribution Method
++++++++++++++++++++++++++++++++++++++

Parameter :code:`split_method` can specify how the parameters are distributed on the parameter servers.

Fluid uses `RoundRobin <https://en.wikipedia.org/wiki/Round-robin_scheduling>`_ by default to scatter parameters to multiple parameter servers. 
In this case, the parameters are evenly distributed on all parameter servers in the case where the parameter segmentation is not turned off by default. 
If you need to use something else, you can pass in other methods. The currently available methods are: :code:`RoundRobin` and :code:`HashName` . You can also use a customized distribution method, just refer to `here <https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/transpiler/ps_dispatcher.py#L44>`_
to write customized distribution function


Turn Off the slice-up of Parameters 
++++++++++++++++++++++++++++++++++++++

Parameter :code:`slice_var_up` specifies whether to split large (more than 8192 elements) parameters into multiple parameter servers to balance the computational load. The default is on.

When the sizes of the trainable parameters in the model are relatively uniform or a customized parameter distribution method is used, which evenly distributes the parameters on multiple parameter servers, you can choose to turn off the slice-up function, which reduces the computational and copying overhead of slicing and reorganization:

.. code-block:: python

	t.transpile(trainer_id, pservers=pserver_endpoints, trainers=trainers, slice_var_up=False)


Turn On Memory Optimization
++++++++++++++++++++++++++++++

In the parameter server distributed training mode, to enable memory optimization :code:`memory_optimize` , compared with a single machine, you need to pay attention to the following rules:

- On the pserver side, **don't** execute :code:`memory_optimize`
- On the trainer side, execute :code:`fluid.memory_optimize` and then execute :code:`t.transpile()`
- On the trainer side, calling :code:`memory_optimize` needs to add :code:`skip_grads=True` to ensure the gradient sent is not renamed : :code:`fluid.memory_optimize(input_program, skip_grads=True)`

Example:

.. code-block:: python

	if role == "TRAINER":
		fluid.memory_optimize(fluid.default_main_program(), skip_grads=True)
	t = fluid.DistributeTranspiler()
	t.transpile(trainer_id, pservers=pserver_endpoints, trainers=trainers)
	if role == "PSERVER":
		# start pserver here
	elif role == "TRAINER":
		# start trainer here


Training Using NCCL2 Communication
--------------------

Distributed training in NCCL2 mode, because there is no parameter server role, the trainers directly communicate with each other. Pay attention to the following tips:

* Configure :code:`mode="nccl2"` in :code:`fluid.DistributeTranspilerConfig` .
* When calling :code:`transpile`, :code:`trainers` is fed with the endpoints of all trainer nodes, and passed with the argument :code:`current_endpoint` .
* Initialize :code:`ParallelExecutor` with :code:`num_trainers` and :code:`trainer_id` .

For example:

.. code-block:: python

	trainer_id = 0 # get actual trainer id here
	trainers = "192.168.1.1:6170,192.168.1.2:6170"
	current_endpoint = "192.168.1.1:6170"
	config = fluid.DistributeTranspilerConfig()
	config.mode = "nccl2"
	t = fluid.DistributeTranspiler(config=config)
	t.transpile(trainer_id, trainers=trainers, current_endpoint=current_endpoint)
	txe = fluid.ParallelExecutor(use_cuda,
		loss_name=loss_name, num_trainers=len(trainers.split(",")), trainer_id=trainer_id)
	...

.. csv-table:: Description of the necessary parameters for NCCL2 mode
	:header: "parameter", "description"

	"trainer_id", "The unique ID of each trainer node in the task, starting at 0, there cannot be any duplication"
	"trainers", "endpoints of all trainer nodes in the task, used to broadcast NCCL IDs when NCCL2 is initialized"
	"current_endpoint", "endpoint of current node"

Currently, distributed training using NCCL2 only supports synchronous training. The distributed training using NCCL2 mode is more suitable for the model which is relatively large and needs \
synchronous training and GPU training. If the hardware device supports RDMA and GPU Direct, this can achieve high distributed training performance.

Start Up NCCL2 Distributed Training in Muti-Process Mode
++++++++++++++++++++++++++++++++++++++++++++++

 Usually you can get better multi-training performance by using multi-process mode to start up NCCL2 distributed training assignment.Paddle provides :code:`paddle.distributed.launch` module to start up multi-process assignment,after which each training process will use an independent GPU device.

Attention during usage:

 * set the number of nodes:set the number of nodes of an assignment by the environment variable :code:`PADDLE_NUM_TRAINERS` ,and this variable will also be set in every training process.
 * set the number of devices of each node:by activating the parameter :code:`--gpus` ,you can set the number of GPU devices of each node,and the sequence number of each process will be set in the environment variable :code:`PADDLE_TRAINER_ID` automatically.
 * data segment:mult-process mode means one process in each device.Generally,each process manages a part of training data,in order to make sure that all processes can manage the whole data set.
 * entrance file：entrance file is the training script for actual startup.
 * journal：for each training process,the joural is saved in the default :code:`./mylog` directory,and you can assign by the parameter :code:`--log_dir` .

  startup example:

  .. code-block:: bash

     > PADDLE_NUM_TRAINERS=<TRAINER_COUNT> python -m paddle.distributed.launch train.py --gpus <NUM_GPUS_ON_HOSTS> <ENTRYPOINT_SCRIPT> --arg1 --arg2 ...

Important Notes on NCCL2 Distributed Training
++++++++++++++++++++++++++++++++++++++++++++++

**Note** : Please ensure each node has the same amount of data to train in NCCL2 mode distributed training, which prevents
exit at the final iteration. There are two common ways:

- Randomly sample some data to complement nodes where less data are distributed. (We recommend this method for sake of a complete dataset to be trained)
- Each node only trains fixed number of batches per pass, which is controlled by python codes. If a node has more data than this fixed amount, then these 
  marginal data will not be trained.

**Note** : If there are multiple network devices in the system, you need to manually specify the devices used by NCCL2.

Assuming you need to use :code:`eth2` as the communication device, you need to set the following environment variables:

.. code-block:: bash

    export NCCL_SOCKET_IFNAME=eth2

In addition, NCCL2 provides other switch environment variables, such as whether to enable GPU Direct, whether to use RDMA, etc. For details, please refer to
`ncclknobs <https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html#ncclknobs>`_ .
