.. _api_paddle_static_Program:

Program
-------------------------------

.. py:class:: paddle.static.Program

.. note::
    By default, Paddle internally contains :ref:`default_startup_program <api_paddle_static_default_startup_program>` and :ref:`default_main_program <api_paddle_static_default_main_program>`, which share parameters. :ref:`default_startup_program <api_paddle_static_default_startup_program>` runs only once to initialize parameters, and :ref:`default_main_program <api_paddle_static_default_main_program>` runs in each mini batch and updates weights.

Program is Paddle's static description of the computational graph. Using the Program constructor, you can create a Program. Program includes at least one :ref:`Block <api_guide_Block>`. When there are control flow OPs with conditional selection (such as :ref:`while_loop <api_paddle_static_nn_while_loop>`) in the :ref:`Block <api_guide_Block>`, the Program will contain nested :ref:`Blocks <api_guide_Block>`, that is, the :ref:`Block <api_guide_Block>` outside the control flow will contain the :ref:`Block <api_guide_Block>` inside the control flow, and the element access control of the nested :ref:`Block <api_guide_Block>` will be determined by the specific control flow OP. For the specific structure and included types of Program, please refer to `framework.proto <https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/framework.proto>`_.

A collection of Programs usually includes an initialization program (startup_program) and a main program (main_program). The initialization program is a Program that contains some initialization work. The main program will contain the network structure and variables used for training. When using the same :ref:`Executor <api_guide_executor>` to execute, they will share the results of initialization work, such as initialized parameters. A collection of Programs can be used for testing or training. When used for training, ``Paddle`` will use all the OPs and variables used by the user to build a training network. When used for testing, you can call Program-related interfaces such as ``clone`` to cut out OPs and variables that are not related to testing, such as backpropagation OPs and variables.

**Returns**

Program, the created empty Program.

**Examples**

.. code-block:: python

    import paddle
    
    # Create a new Program
    prog = paddle.static.Program()
    
    # Check basic information of Program
    print("Number of blocks in Program:", prog.num_blocks)
    print("Random seed of Program:", prog.random_seed)

**Methods**

to_string(throw_on_error, with_details=False)
'''''''''

Convert Program to string.

**Parameters**

 - **throw_on_error** (bool) - Whether to throw an exception when required fields are not set.
 - **with_details** (bool) - When true, print more information about variables and parameters, such as ``trainable``, ``optimize_attr``, etc.

**Returns**

str, string converted from Program.

**Examples**

.. code-block:: python

    import paddle
    
    # Create a Program
    prog = paddle.static.Program()
    
    # Convert Program to string
    prog_str = prog.to_string(throw_on_error=True, with_details=True)
    print(prog_str)

clone(for_test=False)
'''''''''

.. note::
    1. The ``Program.clone()`` method will not clone data reading related parts such as :ref:`DataLoader <api_paddle_io_DataLoader>`, which may cause data reading parts to be lost after cloning;
    2. This API will trim some OPs and variables when ``for_test=True``. To prevent incorrect trimming, it is recommended to use ``clone(for_test=True)`` before :ref:`append_backward <api_paddle_static_append_backward>` and optimizer execution.

When ``for_test=True``, create a new Program that contains only the forward content of the current Program. Otherwise, create a new Program that is exactly the same as the current Program.

Some OPs behave differently between training and testing, such as :ref:`batch_norm <api_paddle_static_nn_batch_norm>`. They have an ``is_test`` attribute to control behavior. When ``for_test=True``, this method will change their ``is_test`` attribute to True.

- When cloning Program for training, set ``for_test`` to False.
- When cloning Program for testing, set ``for_test`` to True. Although in this case, if ``clone`` is called after using the optimizer, we will still automatically trim the backpropagation and optimizer-related content in the Program, but we strongly recommend using ``clone`` before using the optimizer. For example, if you are using :ref:`Momentum <api_paddle_optimizer_Momentum>`, you can use it like this:

**Examples**

.. code-block:: python

    import paddle
    
    def print_prog(prog):
        """Print basic information of Program"""
        print(f"Number of blocks: {prog.num_blocks}")
        print(f"Random seed: {prog.random_seed}")
        
    # Create and print original Program
    prog = paddle.static.Program()
    print("Original Program:")
    print_prog(prog)
    
    # Clone Program
    cloned_prog = prog.clone(for_test=False)
    print("\nCloned Program:")
    print_prog(cloned_prog)

**Parameters**

    - **for_test** (bool, optional) – When set to True, the clone method will internally set the ``is_test`` attribute of operators to True and trim backpropagation OPs and parameter optimization OPs. Default value: False.

**Returns**

Program, when ``for_test=True``, returns a new Program that contains only the forward content of the current Program. Otherwise, returns a new Program that is exactly the same as the current Program.

**static** parse_from_string(binary_str)
'''''''''

Convert to Program through deserialization of `protobuf <https://en.wikipedia.org/wiki/Protocol_Buffers>`_.

**Parameters**

 - **binary_str** (str) – `protobuf <https://en.wikipedia.org/wiki/Protocol_Buffers>`_ binary string.

**Returns**

Program, deserialized Program.

**Examples**

.. code-block:: python

    import paddle
    
    # Create a Program
    prog = paddle.static.Program()
    with paddle.static.program_guard(prog):
        x = paddle.static.data(name='x', shape=[None, 10], dtype='float32')
        y = paddle.static.nn.fc(x, 1)
    
    # Serialize Program to binary string
    binary_str = prog.to_string(throw_on_error=True)
    
    # Deserialize Program from binary string
    restored_prog = paddle.static.Program.parse_from_string(binary_str)
    print("Program deserialization successful")

**Attributes**

num_blocks
'''''''''

The number of :ref:`Blocks <api_guide_Block>` in this Program.

**Returns**

int, the number of :ref:`Blocks <api_guide_Block>` in this Program.

**Examples**

.. code-block:: python

    import paddle
    
    # Create a Program
    prog = paddle.static.Program()
    
    # Get number of blocks
    block_count = prog.num_blocks
    print(f"Number of blocks in Program: {block_count}")  # Output: 1

random_seed
'''''''''

.. note::
    Must be set before related OPs are added.

Default random seed for random operators in the program. 0 means randomly generate random seed.

**Returns**

int64, the random seed currently being used in this Program.

**Examples**

.. code-block:: python

    import paddle
    
    # Create a Program
    prog = paddle.static.Program()
    
    # Get random seed
    seed = prog.random_seed
    print(f"Random seed of Program: {seed}")  # Output: 0

global_block()
'''''''''

Get the first :ref:`Block <api_guide_Block>` of this Program.

**Returns**

:ref:`Block <api_guide_Block>`, the first :ref:`Block <api_guide_Block>` of this Program.

**Examples**

.. code-block:: python

    import paddle
    
    # Create a Program
    prog = paddle.static.Program()
    
    # Get global block
    global_block = prog.global_block()
    print(f"Global block: {global_block}")
    print(f"ID of global block: {global_block.idx}")  # Output: 0

block(index)
'''''''''

Return the :ref:`Block <api_guide_Block>` specified by ``index`` in this Program. ``index`` type is ``int``.

**Parameters**

    - **index** (int) - The index of the :ref:`Block <api_guide_Block>` to get.

**Returns**

:ref:`Block <api_guide_Block>`, the :ref:`Block <api_guide_Block>` corresponding to index in this Program.

**Examples**

.. code-block:: python

    import paddle
    
    # Create a Program
    prog = paddle.static.Program()
    
    # Get block with specified index
    block = prog.block(0)  # Get the first block
    print(f"Block 0: {block}")
    print(f"ID of Block 0: {block.idx}")  # Output: 0

current_block()
'''''''''

Get the current :ref:`Block <api_guide_Block>`. The current :ref:`Block <api_guide_Block>` is used to add OPs.

**Returns**

:ref:`Block <api_guide_Block>`, the :ref:`Block <api_guide_Block>` where the user is currently located in this Program.

**Examples**

.. code-block:: python

    import paddle
    
    # Create a Program
    prog = paddle.static.Program()
    
    # Get current block
    current_block = prog.current_block()
    print(f"Current block: {current_block}")
    print(f"ID of current block: {current_block.idx}")  # Output: 0

list_vars()
'''''''''

Get all variables in the current Program. The return value is an iterable object.

**Returns**

Generator, will yield each variable in the Program.

**Examples**

.. code-block:: python

    import paddle
    
    # Create a Program and add variables
    prog = paddle.static.Program()
    with paddle.static.program_guard(prog):
        x = paddle.static.data(name='x', shape=[None, 10], dtype='float32')
        y = paddle.static.nn.fc(x, 1)
    
    # List all variables in Program
    print("Variables in Program:")
    for var in prog.list_vars():
        print(f"Variable name: {var.name}, shape: {var.shape}, dtype: {var.dtype}")

all_parameters()
'''''''''

Get all :ref:`parameters <api_guide_parameter>` in the current Program. The return value is a list.

**Returns**

list[:ref:`parameter <api_guide_parameter>`], a list containing all parameters in the current Program.

**Examples**

.. code-block:: python

    import paddle
    
    # Create a Program and add parameters
    prog = paddle.static.Program()
    with paddle.static.program_guard(prog):
        x = paddle.static.data(name='x', shape=[None, 10], dtype='float32')
        y = paddle.static.nn.fc(x, 1)
    
    # Get all parameters
    params = prog.all_parameters()
    print(f"Number of parameters in Program: {len(params)}")
    for param in params:
        print(f"Parameter name: {param.name}, shape: {param.shape}")

state_dict(mode='all', scope=None)
'''''''''

Get persistent variables of the current ``Program`` and store all persistent variables in a dict structure.

**Parameters**

    - **mode** (str, optional) - What persistent variables to get. Currently supports the following options: (1) ``opt``: Get persistent variables of optimizer in ``dict`` structure; (2) ``param``: Get persistent variables of network in ``dict`` structure, not including persistent variables in optimizer; (3) ``all``: Get persistent variables of network and optimizer in dict structure; Default value: ``all``.
    - **scope** (Scope, optional) - If scope is ``None``, get global/default scope instance through `paddle.static.global_scope()` and get ``state_dict`` from it; otherwise get ``state_dict`` from specified ``scope``. Default value: ``None``.

**Returns**

dict, dict containing persistent variables, key is the name of persistent variable, value is persistent variable.

**Examples**

.. code-block:: python

    import paddle
    import numpy as np
    
    # Create a Program
    prog = paddle.static.Program()
    with paddle.static.program_guard(prog):
        x = paddle.static.data(name='x', shape=[None, 10], dtype='float32')
        y = paddle.static.nn.fc(x, 1)
    
    # Get state_dict
    state_dict = prog.state_dict()
    print("state_dict of Program:")
    for name, var in state_dict.items():
        print(f"{name}: {var}")

set_state_dict(state_dict, scope=None)
'''''''''

Set persistent variables in ``state_dict`` to ``Program``.

**Parameters**

    - **state_dict** (dict) - Dictionary containing persistent variables. Key is the name of persistent variable, value is persistent variable.
    - **scope** (Scope, optional) - If scope is ``None``, get global/default scope instance through `paddle.static.global_scope()` and set persistent variables in ``state_dict`` to this scope; otherwise set ``state_dict`` to specified ``scope``. Default value: ``None``.

**Returns**

None.

**Examples**

.. code-block:: python

    import paddle
    import numpy as np
    
    # Create two Programs
    prog1 = paddle.static.Program()
    with paddle.static.program_guard(prog1):
        x = paddle.static.data(name='x', shape=[None, 10], dtype='float32')
        y = paddle.static.nn.fc(x, 1)
    
    prog2 = paddle.static.Program()
    with paddle.static.program_guard(prog2):
        x = paddle.static.data(name='x', shape=[None, 10], dtype='float32')
        y = paddle.static.nn.fc(x, 1)
    
    # Get state_dict from prog1
    state_dict = prog1.state_dict()
    
    # Set state_dict to prog2
    prog2.set_state_dict(state_dict)
    print("state_dict set successfully")
