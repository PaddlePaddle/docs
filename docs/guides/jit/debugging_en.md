# Error Debugging Experience

## 1、Dynamic Graph to Static Graph Error Log
### 1.1 How to Read the Error Log
The following is an example code of Dynamic-to-Static error reporting:
```python
import paddle
import numpy as np

@paddle.jit.to_static
def func(x):
    two = paddle.full(shape=[1], fill_value=2, dtype="int32")
    x = paddle.reshape(x, shape=[1, two])
    return x

def train():
    x = paddle.to_tensor(np.ones([3]).astype("int32"))
    func(x)

if __name__ == '__main__':
    train()
```
After execution, the error log is shown below:

<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/04_dygraph_to_static/images/dy2stat_error_log.png" style="zoom:45%" />

The error log can be divided into 4 parts from top to bottom:

- **Native Python error stack**: As shown in the first two lines, it represents a series of subsequent errors caused by the function `train()` called on line 145 of the `/workspace/Paddle/run_dy2stat_error.py` file.

- **The start flag of Dynamic-to-Static error stack**: `In transformed code`, represents the dynamic-to-static error message stack, and refers to the error message when the transformed code is running. In the actual scene, you can directly search for the `In transformed code` keyword, and start from this line and read the error log.

- **User code error stack**: It hides the useless error message at the framework level, and reports the error stack of the user code. We add a wavy line and HERE indicator under the error code to indicate the specific error location. We also expanded the error line code context to help you quickly locate the error location. As shown in the third part of the above figure, it can be seen that the user code that made the last error is `x = paddle.reshape(x, shape=[1, two])`.

- **Error message at the frame level**: Provides static graph networking error information. Generally, you can directly locate the error reported in which OpDesc was generated directly based on the information in the last three lines, which is usually the error reported by the infershape logic that executed this Op. The error message in the above figure indicates that the reshape Op error occurred. The cause of the error is that the shape of tensor x is [3], and it is not allowed to reshape it to [1, 2].

**NOTE**: In some scenarios, the error type will be identified and suggestions for modification will be given, as shown in the figure below. `Revise suggestion` The following are troubleshooting suggestions for errors. You can check and modify the code according to the suggestions.

<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/04_dygraph_to_static/images/revise_suggestion.png" style="zoom:45%" />

### 1.2 Customized Display of Error Information
#### 1.2.1 Native error message without being processed by the Dynamic-to-Static error reporting module
If you want to view Paddle's native error message stack, that is, the error message stack that has not been processed by the Dynamic-to-Static error reporting module, you can set the environment variable `TRANSLATOR_DISABLE_NEW_ERROR=1` to turn off the dynamic-to-static error module. The default value of this environment variable is 0, which means that the module is enabled by default.
Add the following code to the code in section 1.1 to view the native error message:
```python
import os
os.environ["TRANSLATOR_DISABLE_NEW_ERROR"] = '1'
```
You can get the following error message:

<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/04_dygraph_to_static/images/original_error_log.png" style="zoom:45%" />

#### 1.2.2 C++ error stack
The C++ error stack is hidden by default. You can set the C++ environment variable `FLAGS_call_stack_level=2` to display the C++ error stack information. For example, you can enter `export FLAGS_call_stack_level=2` in the terminal to set it, and then you can see the error stack on the C++ side:

<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/04_dygraph_to_static/images/c%2B%2B_error_log.png" style="zoom:45%" />

## 2、Debugging Method
Before debugging, **please ensure that the dynamic graph code before conversion can run successfully**. The following introduces several debugging methods recommended in Dynamic-to-Static.
### 2.1 Pdb Debugging
pdb is a module in Python that defines an interactive Pyhton source code debugger. It supports setting breakpoints and single stepping between source lines, listing source code and variables, running Python code, etc.
#### 2.1.1 Debugging steps

- step1: Insert `import pdb; pdb.set_trace()` before the code where you want to enable pdb debugging.
    ```python
    import paddle
    import numpy as np

    @paddle.jit.to_static
    def func(x):
        x = paddle.to_tensor(x)
        import pdb; pdb.set_trace()       # <------ enable pdb debugging
        two = paddle.full(shape=[1], fill_value=2, dtype="int32")
        x = paddle.reshape(x, shape=[1, two])
        return x

    func(np.ones([3]).astype("int32"))
    ```

- Step2: Run the .py file normally, the following similar result will appear in the terminal, enter the corresponding pdb command after the `(Pdb)` position for debugging.
    ```
    > /tmp/tmpm0iw5b5d.py(9)func()
    -> two = paddle.full(shape=[1], fill_value=2, dtype='int32')
    (Pdb)
    ```

- step3: Enter l, p and other commands in the pdb interactive mode to view the corresponding code and variables of the static graph after Dynamic-to-Static, and then troubleshoot related problems.
    ```
    > /tmp/tmpm0iw5b5d.py(9)func()
    -> two = paddle.full(shape=[1], fill_value=2, dtype='int32')
    (Pdb) l
      4     import numpy as np
      5     def func(x):
      6         x = paddle.assign(x)
      7         import pdb
      8         pdb.set_trace()
      9  ->     two = paddle.full(shape=[1], fill_value=2, dtype='int32')
     10         x = paddle.reshape(x, shape=[1, two])
     11         return x
    [EOF]
    (Pdb) p x
    var assign_0.tmp_0 : LOD_TENSOR.shape(3,).dtype(int32).stop_gradient(False)
    (Pdb)
    ```

#### 2.1.2 Common commands

<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/04_dygraph_to_static/images/pdb_cmd_en.png" style="zoom:45%" />

For more pdb usage methods, please check the [official document](https://docs.python.org/zh-cn/3/library/pdb.html)

### 2.2 Print the Converted Static Graph Code
You can print the converted static graph code in two ways:


#### 2.2.1 set_code_level() or TRANSLATOR_CODE_LEVEL
By calling `set_code_level()` or setting the environment variable `TRANSLATOR_CODE_LEVEL`, you can view the converted code in the log:
```python
import paddle
import numpy as np

@paddle.jit.to_static
def func(x):
    x = paddle.to_tensor(x)
    if x > 3:
        x = x - 1
    return x

paddle.jit.set_code_level() # You can also set os.environ["TRANSLATOR_CODE_LEVEL"] = '100', the effect is the same
func(np.ones([1]))
```
In addition, if you want to also output the converted code to `sys.stdout`, you can set the parameter `also_to_stdout` to True, otherwise it will only be output to `sys.stderr`. The `set_code_level` function can be set to view the converted codes of different AST Transformers. For details, please see [set_code_level](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/jit/set_code_level_cn.html)。

#### 2.2.2 The code attribute of the decorated function
In the following code, the decorator @to_static converts the function func into a class object StaticFunction. You can use the code attribute of StaticFunction to get the converted code.
```python
import paddle
import numpy as np

@paddle.jit.to_static
def func(x):
    x = paddle.to_tensor(x)
    if x > 3:
        x = x - 1
    return x

func(np.ones([1]))
print(func.code)
```
After running, you can see the static graph code after Dynamic-to-Static:
```python
def func(x):
    x = paddle.assign(x)

    def true_fn_0(x):
        x = x - 1
        return x

    def false_fn_0(x):
        return x
    x = paddle.jit.dy2static.convert_ifelse(x > 3, true_fn_0, false_fn_0, (
        x,), (x,), (x,))
    return x
```

### 2.3 Use Print to View Variables
The print function can be used to view variables, and the function will be transformed. When only the Paddle Tensor is printed, it will be converted to the Paddle operator Print, otherwise Python print will be run.

```python
import paddle
import numpy as np

@paddle.jit.to_static
def func(x):
    x = paddle.to_tensor(x)

    # Print x, x is Paddle Tensor, and Paddle Print(x) will be run
    print(x)
    # Print comments, non-Paddle Tensor, Python print will be run
    print("Here call print function.")

    if len(x) > 3:
        x = x - 1
    else:
        x = paddle.ones(shape=[1])
    return x

func(np.ones([1]))
```
After running, you can see the value of x:
```
Variable: assign_0.tmp_0
  - lod: {}
  - place: CUDAPlace(0)
  - shape: [1]
  - layout: NCHW
  - dtype: double
  - data: [1]
```
### 2.4 Print Log
Dynamic-to-Static module records additional debugging information in the log to help you understand whether the function is successfully converted during the Dynamic-to-Static. You can call `paddle.jit.set_verbosity(level=0, also_to_stdout=False)` or set the environment variable `TRANSLATOR_VERBOSITY=level` to set the log detail level and view the log information of different levels. Currently, `level` can take values 0-3:

- 0: No log
- 1: Including the information of the dynamic-to-static conversion process, such as the source code before conversion and the callable object of conversion
- 2: Including the above information and more detailed function conversion logs
- 3: Including the above information, as well as a more detailed Dynamic-to-Static log

> **WARNING:**
> The log includes source code and other information. Please make sure that it does not contain sensitive information before sharing the log.
Sample code for printing log:
```python
import paddle
import numpy as np
import os

@paddle.jit.to_static
def func(x):
    x = paddle.to_tensor(x)
    if len(x) > 3:
        x = x - 1
    else:
        x = paddle.ones(shape=[1])
    return x

paddle.jit.set_verbosity(3)
# Or set os.environ["TRANSLATOR_VERBOSITY"] = '3'
func(np.ones([1]))
```

The results:
```
Sun Sep 26 08:50:20 Dynamic-to-Static INFO: (Level 1) Source code:
@paddle.jit.to_static
def func(x):
    x = paddle.to_tensor(x)
    if len(x) > 3:
        x = x - 1
    else:
        x = paddle.ones(shape=[1])
    return x

Sun Sep 26 08:50:20 Dynamic-to-Static INFO: (Level 1) Convert callable object: convert <built-in function len>.
```
In addition, if you want to output the log to sys.stdout, you can set the parameter also_to_stdout to True, otherwise it will only be output to sys.stderr, please see [set_verbosity](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/jit/set_verbosity_cn.html)。


## 3、Quickly determine the cause of the problem
After summarizing the types of error messages, the Dynamic-to-Static problems can be roughly divided into the following categories:
### 3.1 (NotFound) Input("X")
**The error message is roughly as follows:**
```
RuntimeError: (NotFound) Input("Filter") of ConvOp should not be null.
    [Hint: Expected ctx->HasInputs("Filter") == true, but received ctx->HasInputs("Filter"):0 != true:1.]
    [operator < conv2d > error]
```
The reasons for such problems are generally:
> When the execution reaches the error line, the type of some input or weight is still the Tensor of the dynamic graph, rather than the Variable or Parameter of the static graph.

**Troubleshooting suggestions:**

- First confirm whether the sublayer where the code is located inherits nn.Layer
- Whether the function of this line of code bypasses the forward function and is called separately (before version 2.1)
- How to check whether it is of Tensor or Variable type, which can be debugged through pdb

### 3.2 Expected input_dims[i] == input_dims[0]
**The error message is roughly as follows:**
```
[Hint: Expected input_dims[i] == input_dims[0], but received input_dims[i]:-1, -1 != input_dims[0]:16, -1.]
    [operator < xxx_op > error]
```
The reasons for such problems are generally:
> When append_op generates static graph Program one by one, when a certain Paddle API is executed, infershape does not meet the requirements during compilation.

**Troubleshooting suggestions:**

- At the code level, determine whether the upstream uses reshape to cause the -1 pollution spread
> Since the shape of the dynamic graph is known during execution, reshape(x, [-1, 0, 128]) is no problem. However, when static graphs are networked, they are all shapes at compile time (may be -1), so when using the reshape interface, try to reduce the use of -1.

- It can be combined with debugging skills to determine whether it is the output shape of an API that has diff behavior under the dynamic and static graph
> For example, some Paddle API dynamic graph return 1-D Tensor, but the static graph is always consistent with the input, such as ctx->SetOutputDim("Out", ctx->GetInputDim("X"));

### 3.3 desc->CheckGuards() == true
**The error message is roughly as follows:**
```
[Hint: Expected desc->CheckGuards() == true, but received desc->CheckGuards():0 != true: 1.]
```
The reasons for such problems are generally:
> When the execution reaches the error line, the type of some input or weight is still the Tensor of the dynamic graph, rather than the Variable or Parameter of the static graph.

The following is a summary of the slice syntax functions of the current dynamic and static graph:

<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/04_dygraph_to_static/images/slice.png" style="zoom:45%" />

**Troubleshooting suggestions:**

- Does the model code have the above-mentioned complex Tensor slice operation?
- It is recommended to use the paddle.slice interface to replace complex Tensor slice operations

### 3.4 Segment Fault
When a segfault occurs in the dynamic-to-static module, there will be very little error stack information, but the cause of such problems is generally clear.
The general causes of such problems are:
> A certain sublayer does not inherit nn.Layer, and there is a call to the paddle.to_tensor interface in the \__init__.py function. As a result, the Tensor data of the dynamic graph is accessed in the static graph mode when the Program is generated or the model parameters are saved.

**Troubleshooting suggestions:**

- Ensure that each sublayer inherits nn.Layer

### 3.5 Recommendations for Using Container
Under the dynamic graph, the following container classes of container are provided:

- ParameterList
    ```python
    class MyLayer(paddle.nn.Layer):
        def __init__(self, num_stacked_param):
            super().__init__()

            w1 = paddle.create_parameter(shape=[2, 2], dtype='float32')
            w2 = paddle.create_parameter(shape=[2], dtype='float32')

            # In this usage, MyLayer.parameters() returns empty
            self.params = [w1, w2]                            # <----- Wrong usage

            self.params = paddle.nn.ParameterList([w1, w2])   # <----- Correct usage
    ```

- LayerList
    ```python
    class MyLayer(paddle.nn.Layer):
        def __init__(self):
            super().__init__()

            layer1 = paddle.nn.Linear(10, 10)
            layer2 = paddle.nn.Linear(10, 16)

            # In this usage, MyLayer.parameters() returns empty
            self.linears = [layer1, layer2]                        # <----- Wrong usage

            self.linears = paddle.nn.LayerList([layer1, layer2])   # <----- Correct usage
    ```
