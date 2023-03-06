#######################
Dynamic to Static Graph
#######################

The static graph mode of PaddlePaddle takes advantage of flexibility, Pythonic coding, and easy-to-debug interface. In dynamic graph mode, code immediately executes kernels and gets numerical results, which allows users to enjoy traditional Pythonic code order. Therefore it is efficient to transform idea into real code and simple to debug. However, Python code is usually slower than C++ thus lots of industrial systems (such as large recommend system, mobile devices) prefer to deploy with C++ implementation.

Static graph is better at speed and portability. Static graph builds the network structure during compiling time and then does computation. The built network intermediate representation can be executed in C++ and gets rids of Python dependency.

While dynamic graph has usability and debug benefits and static graph yields performance and deployment advantage, we adds functionality to convert dynamic graph to static graph. Users use dynamic graph mode to write dynamic graph code and PaddlePaddle will analyze the Python syntax and turn it into network structure of static graph mode. Our approach retains both the usability of dynamic graph and portability of static graph.

We introduce the transformation of dynamic graph to static graph in the following links:


- `Supported Grammars <grammar_list_en.html>`_ : Introduce supported grammars and unsupported grammars .

- `Error Debugging Experience <debugging_en.html>`_ : Introduce the debugging methods when using @to_static.


..  toctree::
    :hidden:

    grammar_list_en.md
    debugging_en.md
