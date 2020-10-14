Supported Grammars
====================

The key part of ProgramTranslator is transforming Python grammar into PaddlePaddle static graph code, but there exists difference between Python and PaddlePaddle static graph which causes some limitation of the code transformation.

In this section we will talk about the supported grammars and unsupported grammars, also give some suggestions when the grammar is unsupported. 

There are several kinds of supported grammars:

Control flow keywords
---------------------

Control flow means those keywords that controls the execution order of program statements, for example ``if-elif-else, while`` . Conditional operation and loop were implemented as ``cond, while_loop`` APIs in PaddlePaddle static graph. If the condition of a Python dygraph control flow depends on PaddlePaddle Tensor, the ProgramTranslator will convert the control flow into equivalent PaddlePaddle control flow APIs, else it will still be executed as Python control flow. The transformations of those control flow keywords are listed below:

1. ``if-elif-else`` statements

If the condition of ``if <condition>`` is Tensor, ProgramTranslator will turn this ``if-elif-else`` statement to equivalent PaddlePaddle static graph ``cond`` statements, otherwise the ``if-elif-else`` statement is executed as normal Python conditional statement. Note that ``cond`` API only accepts input conditional Tensor with numel equals to 1, so please use this kind of Tensor to write dygraph conditional statement, other Tensors will cause error.

2. ``while`` loop

If the condition of ``while`` is Tensor, ProgramTranslator will turn this ``while`` statement to equivalent PaddlePaddle static graph ``while_loop`` statements, otherwise the ``while`` statement is executed as normal Python ``while`` loop statement. Note that ``while_loop`` API only accepts input conditional Tensor with numel equals to 1, so please use this kind of Tensor to write dygraph loop condition statement, other Tensors will cause error.

3. ``for`` loop

3.1 ``for _ in range(__)`` loop

Firstly, ProgramTranslator will transform it into equivalent Python while loop, then convert dygraph to static graph by same logic of ``while`` loop.

3.2 ``for _ in x`` loop

If ``x`` is a Python container, iterator, or generator, it will be executed as original Python statement. Otherwise ``x`` is a Tensor, ProgramTranslator will transform the loop into PaddlePaddle static graph loop and fetches ``x[0], x[1], ...`` as loop iteration variable in each loop iteration.

3.3 ``for idx, val in enumerate(x)`` loop

If ``x`` is a Python container, iterator, or generator, it will be executed as original Python statement. Otherwise ``x`` is a Tensor, Program
Translator will transform the  loop into PaddlePaddle static graph loop. The ``idx`` will be transformed to 1-D tensor with value ``0, 1, ...`` and the ``val`` will be transformed to ``x[0], x[1], ...`` in each loop iteration.

4. ``break, continue``

ProgramTranslator supports ``break, continue`` statements in loop. ProgramTranslator will add some PaddlePaddle static graph ``cond`` statements to skip execution of corresponding part when ``break, continue`` condition is meet.

5. ``return``

ProgramTranslator supports ``return`` in a conditonal block or loop body, not necessary to be at the end of a function. It also supports returning tuple with various length of Tensors with different dtype. The implementation is adding some PaddlePaddle static graph ``cond`` statement to skipparts of code when ``return`` is triggered.


Some Python basic operators
---------------------------

1. ``+, -, *, /, **, >, <, >= , <=, ==`` etc. 

Because PaddlePaddle static graph overrides those Python basic arithmetic operators and comparison operators, ProgramTranslator can support those operators.

2. ``and, or, not`` logical operators

Python has ``and, or, not`` keywards as basic logical operators, ProgramTranslator will check whether the variables of the logical operators are Tensors, if they are Tensors, ProgramTranslator replaces the ``and, or, not`` statements into corresponding PaddlePaddle static graph logical operator and run it.

3. Type casting

In dygraph mode, users can use Python type casting grammar. For instance, if ``x`` is a Tensor, ``float(x)`` casts the data type of ``x`` to float. ProgramTranslator will check whether ``x`` is a Tensor during run time, if it is, the casting sentence will be modified to PaddlePaddle static graph ``cast`` API so that its dtype can be changed in the dygraph to static transformation.

Python functions
------------------------------

1. ``print``

In dygraph mode, ``print(x)`` will print Tensor value if ``x`` is a Tensor. ProgramTranslator converts the built-in ``print`` to PaddlePaddle static graph ``Print`` API during dygraph to static graph transformation if the arguments are Tensors, otherwise ProgramTranslator won't convert the ``print``. 

2. ``len``

If ``x`` is a Tensor, ``len(x)`` can get the length at 0-dimension of ``x`` in dygraph mode. ProgramTranslator turns it to PaddlePaddle static graph ``shape`` API and returns the 0-dimension of the ``shape``, else if ``x`` is a TensorArray, then ``len(x)`` will be transformed to static graph API ``control_flow.array_length`` to return the length of TensorArray. In other cases, the ``len`` function will be executed as Python built-in ``len``

3. lambda expression

ProgramTranslator supports Python lambda expression and it modifies code to return the expected result.


4. Calling function

If the transformed function calls another function, ProgramTranslator also transform the called function. The benefit is that users can add one decorator at the outside function to do transformation, no need to add the decorator for each function. Note that ProgramTranslator doesn't support 
that a function calls itself recursively, the details is in the unsupported grammars section below.


Errors and Exceptions
---------------------

1. ``assert``

If ``x`` is a Tensor, ``assert x`` statement can assert ``x`` to be ``True`` or non-zero value in dygraph mode. ProgramTranslator converts the statement into PaddlePaddle static graph ``Assert`` API to support this grammar.


Python containers
-----------------

1. ``list``: if all elements in a list are Tensors, then ProgramTranslator converts it to TensorArray. PaddlePaddle static graph TensorArray supports append, pop, and modify, other list operations such as sort cannot be supported. When not all elements in a list are Tensors, ProgramTranslator will treat it as normal Python list.

2. ``dict``: ProgramTranslator will add the Tensors in a dict into PaddlePaddle static graph ``Program``, so ``dict`` is supported by ProgramTranslator.

Unsupported grammars
--------------------

1. Use the shape of output tensor of ``reshape``

For example, ``x = reshape(x, shape=shape_tensor)`` , then use ``x.shape[0]`` to do other operation. Due to the difference between dygraph and static graph, it is okay in dygraph but it will fail in static graph. The reason is that APIs return computation result in dygraph mode, so ``x.shape`` has deterministic value after calling ``reshape`` . However, static graph doesn't have the value ``shape_tensor`` during building network, so PaddlePaddle doesn't know the value of ``x.shape`` after calling ``reshape``. PaddlePaddle static graph will set -1 to represent unknown shape value for each dimension of ``x.shape`` in this case, not the expected value.

We suggest to set fixed shape value as much as possible, reduce the reshape operation.

2. List of list of Tensor

For example: ``l = [[tensor1, tensor2], [tensor3, tensor4]]``, because ProgramTranslator transformed a list whose elements are all Tensors into PaddlePaddle static graph TensorArray, but TensorArray doesn't support multi-dimensions, ProgramTranslator cannot run this case.

We suggest to use 1-D list at most time, or use PaddlePaddle API ``create_array, array_read, array_write`` to control TensorArray.

3. Convert Tensor to numpy array and do operation

For example, user doesn't return Tensor in the decorated function but call ``numpy.array(tensor)`` to convert Tensor to numpy array and then use numpy API to compute on it. In dygraph mode, it is okey because Tensor has value, but Tensor is variable for building network in static graph mode, it doesn't contain value if not in static graph running time, so we cannot do numpy calculation on it.

We suggest to use PaddlePaddle APIs to replace numpy API in this case.

4. A function calls itself recursively

ProgramTranslator doesn't support a function calls itself recursively, the reason is that recursive function usually uses ``if-else`` for a condition to stop the recursion, the stop condition will be transformed to a ``cond`` in static graph mode. Since ``cond`` just builds network, it cannot determine how many times it recursively builds network during network built stage, so the function will recursively call itself and build network until stack overflow. Due to above reason, ProgramTranslator cannot support a function calls itself recursively now.

We suggest to write non-recursive function in this case.
