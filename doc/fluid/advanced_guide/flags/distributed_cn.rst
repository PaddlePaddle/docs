
分布式
==================


FLAGS_communicator_fake_rpc
**********************
(始于1.5.0)

当设为True时，通信器不会实际进行rpc调用，因此速度不会受到网络通信的影响。该flag用于调试。

取值范围
---------------
Bool型，缺省值为False。

示例
-------
FLAGS_communicator_fake_rpc=True - 启用通信器fake模式。

注意
-------
该flag仅用于paddlepaddle的开发者，普通用户不应对其设置。


FLAGS_communicator_independent_recv_thread
**************************************
(始于1.5.0)

使用独立线程以从参数服务器接收参数。

取值范围
---------------
Bool型，缺省值为True。

示例
-------
FLAGS_communicator_independent_recv_thread=True - 使用独立线程以从参数服务器接收参数。

注意
-------
开发者使用该flag进行框架的调试与优化，普通用户不应对其设置。


FLAGS_communicator_max_merge_var_num
**************************************
(始于1.5.0)

要通过通信器合并为一个梯度并发送的最大梯度数。训练器将所有梯度放入队列，然后通信器将从队列中取出梯度并在合并后发送。

取值范围
---------------
Int32型，缺省值为20。

示例
-------
FLAGS_communicator_max_merge_var_num=16 - 将要通过通信器合并为一个梯度并发送的最大梯度数设为16。

注意
-------
该flag和训练器线程数有着密切关联，缺省值应和线程数一致。


FLAGS_communicator_merge_sparse_grad
*******************************************
(始于1.5.0)

在发送之前，合并稀疏梯度。

取值范围
---------------
Bool型，缺省值true。

示例
-------
FLAGS_communicator_merge_sparse_grad=true - 设置合并稀疏梯度。

注意
-------
合并稀疏梯度会耗费时间。如果重复ID较多，内存占用会变少，通信会变快；如果重复ID较少，则并不会节约内存。


FLAGS_communicator_min_send_grad_num_before_recv
*******************************************
(始于1.5.0)

在通信器中，有一个发送线程向参数服务器发送梯度，一个接收线程从参数服务器接收参数，且它们之间彼此独立。该flag用于控制接收线程的频率。 仅当发送线程至少发送FLAGS_communicator_min_send_grad_num_before_recv数量的梯度时，接收线程才会从参数服务器接收参数。

取值范围
---------------
Int32型，缺省值为20。

示例
-------
FLAGS_communicator_min_send_grad_num_before_recv=10 - 在接收线程从参数服务器接收参数之前，发送线程发送的梯度数为10。

注意
-------
由于该flag和训练器的训练线程数强相关，而每个训练线程都会发送其梯度，所以缺省值应和线程数一致。


FLAGS_communicator_send_queue_size
*******************************************
(始于1.5.0)

每个梯度的队列大小。训练器将梯度放入队列，然后通信器将其从队列中取出并发送出去。 当通信器很慢时，队列可能会满，训练器在队列有空间之前被持续阻塞。它用于避免训练比通信快得多，以致太多的梯度没有及时发出的情况。

取值范围
---------------
Int32型，缺省值为20。

示例
-------
FLAGS_communicator_send_queue_size=10 - 设置每个梯度的队列大小为10。

注意
-------
该flag会影响训练速度，若队列大小过大，速度会变快但结果可能会变差。


FLAGS_communicator_send_wait_times
*******************************************
(始于1.5.0)

合并数没有达到max_merge_var_num的情况下发送线程等待的次数。

取值范围
---------------
Int32型，缺省值为5。

示例
-------
FLAGS_communicator_send_wait_times=5 - 将合并数没有达到max_merge_var_num的情况下发送线程等待的次数设为5。


FLAGS_communicator_thread_pool_size
*******************************************
(始于1.5.0)

设置用于发送梯度和接收参数的线程池大小。

取值范围
---------------
Int32型，缺省值为5。

示例
-------
FLAGS_communicator_thread_pool_size=10 - 设置线程池大小为10。

注意
-------
大部分情况下，用户不需要设置该flag。


FLAGS_dist_threadpool_size
*******************************************
(始于1.0.0)

控制用于分布式模块的线程数。如果未设置，则将其设置为硬线程。

取值范围
---------------
Int32型，缺省值为0。

示例
-------
FLAGS_dist_threadpool_size=10 - 将用于分布式模块的最大线程数设为10。


FLAGS_rpc_deadline
*******************************************
(始于1.0.0)

它控制rpc通信的deadline超时。

取值范围
---------------
Int32型，缺省值为180000，单位为ms。

示例
-------
FLAGS_rpc_deadline=180000 - 将deadline超时设为3分钟。


FLAGS_rpc_disable_reuse_port
*******************************************
(始于1.2.0)

FLAGS_rpc_disable_reuse_port为True时，grpc的 GRPC_ARG_ALLOW_REUSEPORT会被设置为False以禁用SO_REUSEPORT。

取值范围
---------------
Bool型，缺省值为False。

示例
-------
FLAGS_rpc_disable_reuse_port=True - 禁用SO_REUSEPORT。


FLAGS_rpc_get_thread_num
*******************************************
(始于1.0.0)

它控制用于从参数服务器获取参数的线程数。

取值范围
---------------
Int32型，缺省值为12。

示例
-------
FLAGS_rpc_get_thread_num=6 - 将从参数服务器获取参数的线程数设为6。


FLAGS_rpc_send_thread_num
*******************************************
(始于1.0.0)

它控制用于发送rpc的线程数。

取值范围
---------------
Int32型，缺省值为12。

示例
-------
FLAGS_rpc_send_thread_num=6 - 将用于发送的线程数设为6。


FLAGS_rpc_server_profile_path
*******************************************
since(v0.15.0)

设置分析器输出日志文件路径前缀。完整路径为FLAGS_rpc_server_profile_path_listener_id，其中listener_id为随机数。 

取值范围
---------------
String型，缺省值为"./profile_ps"。

示例
-------
FLAGS_rpc_server_profile_path="/tmp/pserver_profile_log" - 在"/tmp/pserver_profile_log_listener_id"中生成配置日志文件。