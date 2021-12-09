#######################
Dygraph to Static Graph
#######################

The imperative-style coding of PaddlePaddle takes advantage of flexibility, Pythonic coding, and easy-to-debug interface. In dygraph mode, code immediately executes kernels and gets numerical results, which allows users to enjoy traditional Pythonic code order. Therefore it is efficient to transform idea into real code and simple to debug. However, Python code is usually slower than C++ thus lots of industrial systems (such as large recommend system, mobile devices) prefer to deploy with C++ implementation.

Static graph is better at speed and portability. Static graph builds the network structure during compiling time and then does computation. The built network intermediate representation can be executed in C++ and gets rids of Python dependency.

While dygraph has usability and debug benefits and static graph yields performance and deployment advantage, we adds functionality to convert dygraph to static graph. Users use imperative mode to write dygraph code and PaddlePaddle will analyze the Python syntax and turn it into network structure of static graph mode. Our approach retains both the usability of dygraph and portability of static graph.

We introduce the transformation of dygraph to static graph in the following links:

- `Basic Usage <basic_usage_en.html>`_ : Introduce the basic usage for @to_static.

- `Supported Grammars <grammar_list_en.html>`_ : Introduce supported grammars and unsupported grammars .

- `Predictive Model Export Tutorial <export_model_en.html>`_ : Introduce the tutorial for exporting predictive model.

- `Case analysis of InputSpec <export_model_en.html>`_ : Introduce the common case studies of @to_static.

- `Error Debugging Experience <debugging_en.html>`_ : Introduce the debugging methods when using @to_static.


..  toctree::
    :hidden:

    basic_usage_en.rst
    grammar_list_en.md
    export_model_en.md
    case_analysis_en.md
    debugging_en.md

