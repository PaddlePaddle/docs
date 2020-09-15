Architecture
==============

The basic idea of TracedLayer is tracing, it is relatively simple so we won't expend here. This section will talk about the source code transformation of ProgramTranslator.

The transformation is implemented in the decorator so transformation happens when user calls the decorated function, the procedure includes these steps:

Function and cache
--------------------

The entity for transforming dygraph to static graph is the decorated function. For the PaddlePaddle APIs in the function, since they are same code under dygraph mode and static mode, we don't have to transform those code. However, those APIs are computation in dygraph model while they are building network in static graph mode, if the transformed functions are called multiple times, those APIs will build network multiple times in static graph, which can cause problem. To solve it as well as speed up the transformation, we maintain a cache that maps from function, input shapes, input data types to the Program built by the transformed function. If the function hits cache, we run the stored Program in static graph mode to get result, else we do the code transformation on the function and store the transformed Program into the cache.

From dygraph source code to AST (Abstract Syntax Tree)
--------------------------------------------------------

The core of transforming dygraph to static graph is similar to a compiler, we parse the dygraph code into AST, change AST, then turn it back into static graph code. We use Python ``inspect.getsource`` to get the source code string of the function. Python provides `ast <https://docs.python.org/3/library/ast.html>`_ library to parse string code into AST, but Python2, Python3 have slight grammar difference. To avoid the work to handle different grammars, we used an open source AST library `gast <https://github.com/serge-sans-paille/gast>`_ that provides compatibility AST among various Python versions. There is no essential difficulty to turn function into AST with these library.

Transform AST and turn it to static graph code
------------------------------------------------

This part is the key part in ProgramTranslator, we modify AST for supported grammars. Those important Python control flows, such as ``if-elif-else, while, for`` loop are converted to PaddlePaddle static graph API ``cond, while_loop`` and so on. We created a Transformer (AST-to-AST Transformer in Python, not the Transformer in Natural Language Process) to transform each grammar. Every Transformer scans AST and modify it. Lastly, we turn AST back to source code string by ``gast`` library.

Running static graph code as part of dygraph
----------------------------------------------

In order to increase usability and re-use the transformed static graph code in dygraph, we wrap the generated Program as an dygraph op, the op can run the forward and backward computation of transformed Program. Then we can not only speed up dygraph code or save it for deployment, but also enable user to run part of their dygraph code in static graph mode so that they can continue training or other dygraph computation in their dygraph code.

Error handling and Debug
--------------------------

Compiler usually supports debug functionality like breakpoint, throwing exception, print some mid-level codes. ProgramTranslator is similar to a compiler, users may would like to set breakpoints for debugging, or see whether the transformed static graph code is expected. So we also implemented those error handling and debug functionality. Here we list those functions and their implementation.

A. Report errors/exceptions on dygraph code line. Because the transformed static graph code is different to original dygraph code, when Python executes the static graph code, the exceptions will be reported at static graph code. To locate the corresponding dygraph code, we attach some informations such as line number on AST nodes when we transform AST, then we can re-write the static graph exception to the corresponding dygraph code exception.

B. We support ``pdb.set_trace()`` when running ProgramTranslator, user can add this line to set breakpoints.

C. Check the transformed static graph code. Our transformed output is a Python class named ``StaticLayer``, this class can be called, but it also stores the transformed code string. Users could call ``StaticLayer.code`` to get the converted code.

D. Print mid-level transformed code, such as what's the code after transforming ``for`` loop. We provide APIs to set log level to let user check the mid-level code.


