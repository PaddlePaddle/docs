#############################
Custom Runtime
#############################

Custom Runtime offers a new method to register the runtime of new devices via plug-ins. Responsible for the management of PaddlePaddle devices and Runtime/Driver API, DeviceManager provides a uniform API for the framework to invoke device capabilities, offers a series of APIs to register Custom Runtime, and ensure that the binary system is compatible through C API. The APIs can be found in  `device_ext.h <https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/backends/device_ext.h>`_ . Developers can add custom runtime for PaddlePaddle only by implementing these APIs.

- `Data type <./runtime_data_type_en.html>`_ : to introduce definitions of data types of custom runtime.
- `Device API <./device_api_en.html>`_ : to introduce definitions and functions of Device APIs.
- `Memory API <./memory_api_en.html>`_ : to introduce definitions and functions of Memory APIs.
- `Stream API <./stream_api_en.html>`_ : to introduce definitions and functions of Stream APIs.
- `Event API <./event_api_en.html>`_ : to introduce definitions and functions of Event APIs.


Device APIs
############

+------------------------+----------------------------------------+----------+
|          API           |                Function                | Required |
+========================+========================================+==========+
| initialize             | To initialize the device backend       | N        |
+------------------------+----------------------------------------+----------+
| finalize               | To de-initialize the device backend    | N        |
+------------------------+----------------------------------------+----------+
| init_device            | To initialize the designated device    | N        |
+------------------------+----------------------------------------+----------+
| deinit_device          | To de-initialize the designated device | N        |
+------------------------+----------------------------------------+----------+
| set_device             | To set the current device              | Y        |
+------------------------+----------------------------------------+----------+
| get_device             | To get the current device              | Y        |
+------------------------+----------------------------------------+----------+
| synchronize_device     | To synchronize the desginated device   | Y        |
+------------------------+----------------------------------------+----------+
| get_device_count       | To count available devices             | Y        |
+------------------------+----------------------------------------+----------+
| get_device_list        | To get the list of available devices   | Y        |
+------------------------+----------------------------------------+----------+
| get_compute_capability | To get computing capability of devices | Y        |
+------------------------+----------------------------------------+----------+
| get_runtime_version    | To get the runtime version             | Y        |
+------------------------+----------------------------------------+----------+
| get_driver_version     | To get the driver version              | Y        |
+------------------------+----------------------------------------+----------+


Memory APIs
############

+---------------------------+-------------------------------------------------------------------+----------+
|            API            |                             Function                              | Required |
+===========================+===================================================================+==========+
| device_memory_allocate    | To allocate the device memory                                     | Y        |
+---------------------------+-------------------------------------------------------------------+----------+
| device_memory_deallocate  | To deallocate the device memory                                   | Y        |
+---------------------------+-------------------------------------------------------------------+----------+
| host_memory_allocate      | To allocate pinned host memory                                    | N        |
+---------------------------+-------------------------------------------------------------------+----------+
| host_memory_deallocate    | To deallocate pinned host memory                                  | N        |
+---------------------------+-------------------------------------------------------------------+----------+
| unified_memory_allocate   | To allocated unified memory                                       | N        |
+---------------------------+-------------------------------------------------------------------+----------+
| unified_memory_deallocate | To deallocate unified memory                                      | N        |
+---------------------------+-------------------------------------------------------------------+----------+
| memory_copy_h2d           | To copy synchronous memory from host to device                    | N        |
+---------------------------+-------------------------------------------------------------------+----------+
| memory_copy_d2h           | To copy synchronous memory from device to host                    | N        |
+---------------------------+-------------------------------------------------------------------+----------+
| memory_copy_d2d           | To copy synchronous memory in the device                          | N        |
+---------------------------+-------------------------------------------------------------------+----------+
| memory_copy_p2d           | To copy synchronous memory between devices                        | N        |
+---------------------------+-------------------------------------------------------------------+----------+
| async_memory_copy_h2d     | To copy asynchronous memory from host to device                   | N        |
+---------------------------+-------------------------------------------------------------------+----------+
| async_memory_copy_d2h     | To copy asynchronous memory from device to host                   | N        |
+---------------------------+-------------------------------------------------------------------+----------+
| async_memory_copy_d2d     | To copy asynchronous memory in the device                         | N        |
+---------------------------+-------------------------------------------------------------------+----------+
| async_memory_copy_p2d     | To copy asynchronous memory between devices                       | N        |
+---------------------------+-------------------------------------------------------------------+----------+
| device_memory_set         | To fill the device memory                                         | N        |
+---------------------------+-------------------------------------------------------------------+----------+
| device_memory_stats       | To measure device memory utilization                              | N        |
+---------------------------+-------------------------------------------------------------------+----------+
| device_min_chunk_size     | To check the minimum size of device memory chunks                 | N        |
+---------------------------+-------------------------------------------------------------------+----------+
| device_max_chunk_size     | To check the maximum size of device memory chunks                 | N        |
+---------------------------+-------------------------------------------------------------------+----------+
| device_max_alloc_size     | To check the maximum size of allocatable device memory            | N        |
+---------------------------+-------------------------------------------------------------------+----------+
| device_extra_padding_size | To check the extra padding size of device memory                  | N        |
+---------------------------+-------------------------------------------------------------------+----------+
| device_init_alloc_size    | To check the size of allocated device memory after initialization | N        |
+---------------------------+-------------------------------------------------------------------+----------+
| device_realloc_size       | To check the size of reallocated device memory                    | N        |
+---------------------------+-------------------------------------------------------------------+----------+


Stream APIs
############

+---------------------+--------------------------------------------------------------------+----------+
|         API         |                              Function                              | Required |
+=====================+====================================================================+==========+
| create_stream       | To create a stream object                                          | N        |
+---------------------+--------------------------------------------------------------------+----------+
| destroy_stream      | To destroy a stream object                                         | N        |
+---------------------+--------------------------------------------------------------------+----------+
| query_stream        | To query whether all the tasks on the stream are done              | N        |
+---------------------+--------------------------------------------------------------------+----------+
| synchronize_stream  | To synchronize the stream and wait for the completion of all tasks | N        |
+---------------------+--------------------------------------------------------------------+----------+
| stream_add_callback | To add a host and call it back on the stream                       | N        |
+---------------------+--------------------------------------------------------------------+----------+
| stream_wait_event   | To wait for the completion of an event on the stream               | N        |
+---------------------+--------------------------------------------------------------------+----------+


Event APIs
############

+-------------------+------------------------------------------------------+----------+
|        API        |                       Function                       | Required |
+===================+======================================================+==========+
| create_event      | To create an event                                   | Y        |
+-------------------+------------------------------------------------------+----------+
| destroy_event     | To destroy an event                                  | Y        |
+-------------------+------------------------------------------------------+----------+
| record_event      | To record an event on the stream                     | Y        |
+-------------------+------------------------------------------------------+----------+
| query_event       | To query whether the event is done                   | N        |
+-------------------+------------------------------------------------------+----------+
| synchronize_event | To synchronize the event and wait for its completion | Y        |
+-------------------+------------------------------------------------------+----------+

Collective communication APIs
############

+-------------------------+---------------------------------------------------------+----------+
|           API           |                        Function                         | Required |
+=========================+=========================================================+==========+
| xccl_get_unique_id_size | Get the size of unique_id object                        | N        |
+-------------------------+---------------------------------------------------------+----------+
| xccl_get_unique_id      | Get unique_id object                                    | N        |
+-------------------------+---------------------------------------------------------+----------+
| xccl_comm_init_rank     | To initialize  communicator。                           | N        |
+-------------------------+---------------------------------------------------------+----------+
| xccl_destroy_comm       | To destroy  communicator。                              | N        |
+-------------------------+---------------------------------------------------------+----------+
| xccl_all_reduce         | Collective communication AllReduce operation            | N        |
+-------------------------+---------------------------------------------------------+----------+
| xccl_broadcast          | Collective communication Broadcast operation            | N        |
+-------------------------+---------------------------------------------------------+----------+
| xccl_reduce             | Collective communication Reduce operation               | N        |
+-------------------------+---------------------------------------------------------+----------+
| xccl_all_gather         | Collective communication AllGather operation            | N        |
+-------------------------+---------------------------------------------------------+----------+
| xccl_reduce_scatter     | Collective communication ReduceScatter operation        | N        |
+-------------------------+---------------------------------------------------------+----------+
| xccl_group_start        | Begin aggregation of collection communication operation | N        |
+-------------------------+---------------------------------------------------------+----------+
| xccl_group_end          | Stop aggregation of collection communication operation  | N        |
+-------------------------+---------------------------------------------------------+----------+
| xccl_send               | Collective communication Send operation                 | N        |
+-------------------------+---------------------------------------------------------+----------+
| xccl_recv               | Collective communication Recv operation                 | N        |
+-------------------------+---------------------------------------------------------+----------+


Profiler APIs
############

+-----------------------------+-----------------------------------+----------+
|             API             |             Function              | Required |
+=============================+===================================+==========+
| profiler_initialize         | To initialize profiler            | N        |
+-----------------------------+-----------------------------------+----------+
| profiler_finalize           | To de-initialize profiler         | N        |
+-----------------------------+-----------------------------------+----------+
| profiler_prepare_tracing    | Prepare to collect profiling data | N        |
+-----------------------------+-----------------------------------+----------+
| profiler_start_tracing      | Start collecting profiling data   | N        |
+-----------------------------+-----------------------------------+----------+
| profiler_stop_tracing       | Stop collecting profiling data    | N        |
+-----------------------------+-----------------------------------+----------+
| profiler_collect_trace_data | Profiler data conversion          | N        |
+-----------------------------+-----------------------------------+----------+

..  toctree::
    :hidden:

    runtime_data_type_en.md
    device_api_en.md
    memory_api_en.md
    stream_api_en.md
    event_api_en.md
