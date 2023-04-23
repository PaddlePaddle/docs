
executor
==================


FLAGS_enable_parallel_graph
*******************************************
(since 1.2.0)

This Flag is used for ParallelExecutor to disable parallel graph execution mode.

Values accepted
---------------
Bool. The default value is False.

Example
-------
FLAGS_enable_parallel_graph=False will force disable parallel graph execution mode by ParallelExecutor.


FLAGS_pe_profile_fname
*******************************************
(since 1.3.0)

This Flag is used for debugging for ParallelExecutor. The ParallelExecutor will generate the profile result by gperftools, and the profile result will be stored in the file which is specified by FLAGS_pe_profile_fname. Only valid when compiled `WITH_PRIFILER=ON`. Empty if disable.

Values accepted
---------------
String. The default value is empty ("").

Example
-------
FLAGS_pe_profile_fname="./parallel_executor.perf" will store the profile result to parallel_executor.perf.


FLAGS_print_sub_graph_dir
*******************************************
(since 1.2.0)

This Flag is used for debugging. If some subgraphs of the transformed graph from the program are disconnected, the result may be problematic. We can print these disconnected subgraphs to a file specified by the flag. Empty if disable.

Values accepted
---------------
String. The default value is empty ("").

Example
-------
FLAGS_print_sub_graph_dir="./sub_graphs.txt" will print the disconnected subgraphs to "./sub_graphs.txt".


FLAGS_use_ngraph
*******************************************
(since 1.4.0)

Give a choice to run with Intel nGraph(https://github.com/NervanaSystems/ngraph) engine on inference or training. This will obtain much performance boost on Intel Xeon CPU.

Values accepted
---------------
Bool. The default value is False.

Example
-------
FLAGS_use_ngraph=True will enable running with nGraph support.

Note
-------
Intel nGraph is only supported in few models yet. We have only verified [ResNet-50](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/README_ngraph.md) training and inference.
