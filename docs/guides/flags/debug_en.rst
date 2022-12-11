
debug
==================

FLAGS_check_nan_inf
**************************************
(since 0.13.0)

This Flag is used for debugging. It is used to check whether the result of the Operator has Nan or Inf.

Values accepted
---------------
Bool. The default value is False.

Example
-------
FLAGS_check_nan_inf=True will check the result of Operator whether the result has Nan or Inf.


FLAGS_cpu_deterministic
*******************************************
(since 0.15.0)

This Flag is used for debugging. It indicates whether to make the result of computation deterministic in CPU side. In some case, the result of the different order of summing maybe different，for example, the result of `a+b+c+d` may be different with the result of `c+a+b+d`.

Values accepted
---------------
Bool. The default value is False.

Example
-------
FLAGS_cpu_deterministic=True will make the result of computation deterministic in CPU side.


FLAGS_enable_rpc_profiler
*******************************************
(Since 1.0.0)

Enable RPC profiler or not.

Values accepted
----------------
Bool. The default value is False.

Example
-------
FLAGS_enable_rpc_profiler=True will enable rpc profiler and record the timeline to profiler file.


FLAGS_multiple_of_cupti_buffer_size
*******************************************
(since 1.4.0)

This Flag is used for profiling. It indicates the multiple of the CUPTI device buffer size. When you are profiling, if the program breaks down or bugs rise when loading timeline file in chrome://traxing, try increasing this value.

Values accepted
---------------
Int32. The default value is 1.

Example
-------
FLAGS_multiple_of_cupti_buffer_size=1 set the multiple of the CUPTI device buffer size to 1.


FLAGS_reader_queue_speed_test_mode
*******************************************
(since 1.1.0)

Set the pyreader data queue to test mode. In test mode, pyreader will cache some data, executor will then read the cached data, so reader will not be the bottleneck.

Values accepted
---------------
Bool. The default value is False.

Example
-------
FLAGS_reader_queue_speed_test_mode=True will enable the pyreader test mode.

Note
-------
This flag will work only when you are using py_reader.

..  toctree::
    :hidden:

    check_nan_inf_en.md
