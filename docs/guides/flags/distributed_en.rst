
distributed
==================

FLAGS_communicator_fake_rpc
**************************************
(since 1.5.0)

When set true, communicator will not really do rpc call, so the speed will not be affected by network communication. This flag is used for debugging purpose.

Values accepted
---------------
Bool. The default value is false.

Example
-------
FLAGS_communicator_fake_rpc=True will enable communicator fake mode.

Note
-------
This flag is only for developer of paddlepaddle, user should not set it.


FLAGS_communicator_independent_recv_thread
**************************************
(since 1.5.0)

use an independent thread to receive parameter from parameter server

Values accepted
---------------
Bool. The default value is True.

Example
-------
FLAGS_communicator_independent_recv_thread=True will use an independent thread to receive parameter from parameter server.

Note
-------
This flag is for developer to debug and optimize the framework. User should not set it.


FLAGS_communicator_max_merge_var_num
**************************************
(since 1.5.0)

max gradient number to merge and send as one gradient by communicator. Trainer will put all gradients into a queue, then communicator will take the gradients out from the queue and merge them before send.

Values accepted
---------------
Int32. The default value is 20.

Example
-------
FLAGS_communicator_max_merge_var_num=16 will set the max gradient number to merge and send as one gradient to 16.

Note
-------
This flag has strong relationship with trainer thread num. The default value should be the same with thread num.


FLAGS_communicator_merge_sparse_grad
*******************************
(since 1.5.0)

merge sparse gradient before sending.

Values accepted
---------------
Bool. The default value is True.

Example
-------
FLAGS_communicator_merge_sparse_grad=True will merge sparse gradient before sending.

Note
-------
Merging sparse gradient would be time-consuming. If the sparse gradient has many duplicated ids, it will save memory and communication could be much faster. Otherwise it will not save memory.


FLAGS_communicator_min_send_grad_num_before_recv
*******************************************
(since 1.5.0)

In communicator, there is one send thread that send gradient to parameter server and one receive thread that receive parameter from parameter server. They work independently. This flag is used to control the frequency of receive thread. Only when the send thread send at least FLAGS_communicator_min_send_grad_num_before_recv gradients will the receive thread receive parameter from parameter server.

Values accepted
---------------
Int32. The default value is 20.

Example
-------
FLAGS_communicator_min_send_grad_num_before_recv=10 will set the number of gradients sent by the send thread to 10 before the receive thread receive parameter from parameter server.

Note
-------
This flag has strong relation with the training threads of trainer. because each training thread will send it's grad. So the default value should be training thread num.


FLAGS_communicator_send_queue_size
*******************************************
(since 1.5.0)

The queue size for each gradient. Trainer will put gradient into a queue, and communicator will take gradient out from the queue and then send them out. When communicator is slow, the queue may be full and then the trainer will be blocked until the queue has space. It's used to avoid the situation that training is much more faster than communication. There will be too much gradients that is not sent out in time.

Values accepted
---------------
Int32. The default value is 20.

Example
-------
FLAGS_communicator_send_queue_size=10 will set the queue size for each gradient to 10.

Note
-------
This flag will affect the training speed, if the queue size is larger, the speed may be faster, but may make the result worse.


FLAGS_communicator_send_wait_times
*******************************************
(since 1.5.0)

times that send thread will wait if merge number does not reach max_merge_var_num.

Values accepted
---------------
Int32. The default value is 5.

Example
-------
FLAGS_communicator_send_wait_times=5 set the times that send thread will wait if merge number does not reach max_merge_var_num to 5.


FLAGS_communicator_thread_pool_size
*******************************************
(since 1.5.0)

Set the thread pool size that used to do gradient send and parameter receive.

Values accepted
---------------
Int32. The default value is 5.

Example
-------
FLAGS_communicator_thread_pool_size=10 set the thread pool size to 10.

Note
-------
Most of time user does not need to set this flag.


FLAGS_dist_threadpool_size
*******************************************
(Since 1.0.0)

Control the number of thread used for distributed module. If it's not set, it will be set to hardware threads.

Values accepted
---------------
Int32. The default value is 0.

Example
-------
FLAGS_dist_threadpool_size=10 will enable 10 threads as max number of thread used for distributed module.


FLAGS_rpc_deadline
*******************************************
(Since 1.0.0)

It controls the deadline timeout of the rpc communication.

Values accepted
---------------
Int32. The default value is 180000 in ms.

Example
-------
FLAGS_rpc_deadline=180000 will set deadline timeout to 3 minute.


FLAGS_rpc_disable_reuse_port
*******************************************
(since 1.2.0)

When FLAGS_rpc_disable_reuse_port is true, the flag of grpc GRPC_ARG_ALLOW_REUSEPORT will be set to false to
disable the use of SO_REUSEPORT if it's available.

Values accepted
---------------
Bool. The default value is False.

Example
-------
FLAGS_rpc_disable_reuse_port=True will disable the use of SO_REUSEPORT.


FLAGS_rpc_get_thread_num
*******************************************
(Since 1.0.0)

It controls the number of threads used to get parameter from parameter server.

Values accepted
---------------
Int32. The default value is 12.

Example
-------
FLAGS_rpc_get_thread_num=6 will use 6 threads to get parameter from parameter server.


FLAGS_rpc_send_thread_num
*******************************************
(Since 1.0.0)

It controls the number of threads used for send rpc.

Values accepted
---------------
Int32. The default value is 12.

Example
-------
FLAGS_rpc_send_thread_num=6 will set number thread used for send to 6.


FLAGS_rpc_server_profile_path
*******************************************
since(v0.15.0)

Set the profiler output log file path prefix. The complete path will be FLAGS_rpc_server_profile_path_listener_id, listener_id is a random number.

Values accepted
---------------
String. The default value is "./profile_ps".

Example
-------
FLAGS_rpc_server_profile_path="/tmp/pserver_profile_log" generate profile log file at "/tmp/pserver_profile_log_listener_id".


FLAGS_apply_pass_to_program
*******************************************
since(v2.2.0)

It controls whether to apply IR pass to program when using Fleet APIs.

Values accepted
---------------
Bool. The default value is false.

Example
-------
FLAGS_apply_pass_to_program=true would apply IR pass to program when using Fleet APIs.


FLAGS_allreduce_record_one_event
*******************************************
since(v2.2.0)

Make the allreduce operations would only wait one event instead of multiple events. Currently, only fuse allreduce supports this. Otherwise, the precision may be wrong.

Values accepted
---------------
Bool. The default value is false.

Example
-------
FLAGS_allreduce_record_one_event=true would make the allreduce operations would only wait one event instead of multiple events.
