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

+------------------------+----------------------------------------+
|        API             |               Function                 |
+========================+========================================+
| initialize             | To initialize the device backend       |
+------------------------+----------------------------------------+
| finalize               | To de-initialize the device backend    |
+------------------------+----------------------------------------+
| init_device            | To initialize the designated device    |
+------------------------+----------------------------------------+
| deinit_device          | To de-initialize the designated device |
+------------------------+----------------------------------------+
| set_device             | To set the current device              |
+------------------------+----------------------------------------+
| get_device             | To get the current device              |
+------------------------+----------------------------------------+
| synchronize_device     | To synchronize the desginated device   |
+------------------------+----------------------------------------+
| get_device_count       | To count available devices             |
+------------------------+----------------------------------------+
| get_device_list        | To get the list of available devices   |
+------------------------+----------------------------------------+
| get_compute_capability | To get computing capability of devices |
+------------------------+----------------------------------------+
| get_runtime_version    | To get the runtime version             |
+------------------------+----------------------------------------+
| get_driver_version     | To get the driver version              |
+------------------------+----------------------------------------+


Memory APIs
############

+---------------------------+-------------------------------------------------------------------+
|         API               |              Function                                             |
+===========================+===================================================================+
| device_memory_allocate    | To allocate the device memory                                     |
+---------------------------+-------------------------------------------------------------------+
| device_memory_deallocate  | To deallocate the device memory                                   |
+---------------------------+-------------------------------------------------------------------+
| host_memory_allocate      | To allocate pinned host memory                                    |
+---------------------------+-------------------------------------------------------------------+
| host_memory_deallocate    | To deallocate pinned host memory                                  |
+---------------------------+-------------------------------------------------------------------+
| unified_memory_allocate   | To allocated unified memory                                       |
+---------------------------+-------------------------------------------------------------------+
| unified_memory_deallocate | To deallocate unified memory                                      |
+---------------------------+-------------------------------------------------------------------+
| memory_copy_h2d           | To copy synchronous memory from host to device                    |
+---------------------------+-------------------------------------------------------------------+
| memory_copy_d2h           | To copy synchronous memory from device to host                    |
+---------------------------+-------------------------------------------------------------------+
| memory_copy_d2d           | To copy synchronous memory in the device                          |
+---------------------------+-------------------------------------------------------------------+
| memory_copy_p2d           | To copy synchronous memory between devices                        |
+---------------------------+-------------------------------------------------------------------+
| async_memory_copy_h2d     | To copy asynchronous memory from host to device                   |
+---------------------------+-------------------------------------------------------------------+
| async_memory_copy_d2h     | To copy asynchronous memory from device to host                   |
+---------------------------+-------------------------------------------------------------------+
| async_memory_copy_d2d     | To copy asynchronous memory in the device                         |
+---------------------------+-------------------------------------------------------------------+
| async_memory_copy_p2d     | To copy asynchronous memory between devices                       |
+---------------------------+-------------------------------------------------------------------+
| device_memory_set         | To fill the device memory                                         |
+---------------------------+-------------------------------------------------------------------+
| device_memory_stats       | To measure device memory utilization                              |
+---------------------------+-------------------------------------------------------------------+
| device_min_chunk_size     | To check the minimum size of device memory chunks                 |
+---------------------------+-------------------------------------------------------------------+
| device_max_chunk_size     | To check the maximum size of device memory chunks                 |
+---------------------------+-------------------------------------------------------------------+
| device_max_alloc_size     | To check the maximum size of allocatable device memory            |
+---------------------------+-------------------------------------------------------------------+
| device_extra_padding_size | To check the extra padding size of device memory                  |
+---------------------------+-------------------------------------------------------------------+
| device_init_alloc_size    | To check the size of allocated device memory after initialization |
+---------------------------+-------------------------------------------------------------------+
| device_realloc_size       | To check the size of reallocated device memory                    |
+---------------------------+-------------------------------------------------------------------+


Stream APIs
############

+---------------------+-------------------------------------------------------------------+
|      API            |                Function                                           |
+=====================+===================================================================+
| create_stream       | To create a stream object                                         |
+---------------------+-------------------------------------------------------------------+
| destroy_stream      | To destroy a stream object                                        |
+---------------------+-------------------------------------------------------------------+
| query_stream        | To query whether all the tasks on the stream are done             |
+---------------------+-------------------------------------------------------------------+
| synchronize_stream  | To synchronize the stream and wait for the completion of all tasks|
+---------------------+-------------------------------------------------------------------+
| stream_add_callback | To add a host and call it back on the stream                      |
+---------------------+-------------------------------------------------------------------+
| stream_wait_event   | To wait for the completion of an event on the stream              |
+---------------------+-------------------------------------------------------------------+


Event APIs
############

+-------------------+---------------------------------------------------------+
|     API           |          Function                                       |
+===================+=========================================================+
| create_event      | To create an event                                      |
+-------------------+---------------------------------------------------------+
| destroy_event     | To destroy an event                                     |
+-------------------+---------------------------------------------------------+
| record_event      | To record an event on the stream                        |
+-------------------+---------------------------------------------------------+
| query_event       | To query whether the event is done                      |
+-------------------+---------------------------------------------------------+
| synchronize_event | To synchronize the event and wait for its completion    |
+-------------------+---------------------------------------------------------+


..  toctree::
    :hidden:

    runtime_data_type_en.md
    device_api_en.md
    memory_api_en.md
    stream_api_en.md
    event_api_en.md
