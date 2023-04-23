
device management
==================


FLAGS_paddle_num_threads
*******************************************
(since 0.15.0)

Control the number of threads of each paddle instance.

Values accepted
---------------
Int32. The default value is 1.

Example
-------
FLAGS_paddle_num_threads=2 will enable 2 threads as max number of threads for each instance.


FLAGS_selected_gpus
*******************************************
(since 1.3)

Set the GPU devices used for training or inference.

Values accepted
---------------
A comma-separated list of device IDs, where each device ID is a nonnegative integer less than the number of GPU devices your machine have.

Example
-------
FLAGS_selected_gpus=0,1,2,3,4,5,6,7 makes GPU devices 0-7 to be used for training or inference.

Note
-------
The reason for using this flag is that we want to use collective communication between GPU devices, but with CUDA_VISIBLE_DEVICES can only use share-memory.
