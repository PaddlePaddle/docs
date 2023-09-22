# Supported Grammars

## Abstract
In this section we will talk about the supported grammars and unsupported grammars, also give some suggestions when the grammar is unsupported.

This article is mainly for the following scene :

1. Not sure whether the dynamic graph model can be converted into a static graph correctly.

2. There was a problem in the conversion process but don’t know how to troubleshoot.

3. How to modify the source code to adapt dynamic-to-static grammar when there is an unsupported grammar

If you are new to the dynamic-to-static module, or are not familiar with this function, you are recommended to read [Introduction to dynamic and static documents](https://www.paddlepaddle.org.cn/documentation/docs/en/guides/04_dygraph_to_static/basic_usage_en.html) ;

If you want to export model for prediction, or want to learn more about this module, you are recommended to read: [Predictive Model Export Tutorial](https://www.paddlepaddle.org.cn/documentation/docs/en/guides/04_dygraph_to_static/export_model_en.html) ;

If you encounter problems with @to_static, or want to learn about debugging skills, you are recommended to read [Error Debugging Experience](https://www.paddlepaddle.org.cn/documentation/docs/en/guides/debugging_en.html).

## Supported Grammars

| category | python grammar | is it supported ? | summary |
|:--:|:--:|:--:|:--:|
|Control flow keywords | [if-else](#1)  |  Yes | Adaptively recognize and convert to cond Op, or keep python if-else |
||[while](#2)|Yes|Adaptively recognize and convert to While Op, or keep python if-else|
||[for](#3)|Yes|Support iterative access to Tensor by  `for _ in x` |
||[break<br>continue](#4)|Yes|Support putting `break` and `continue` at any position inside a loop |
||[return](#4)|Yes|Support putting `return` inside a loop |
|operators|` + - * / ** > < >= <= ==`|Yes|Support by operator overloading|
||[and, or, not](#5)|Yes|1.if both are tensors, convert to static graph. <br>2.if neither is tensor, degenerate to python operator. <br>3. If only one is tensor, degenerate to python operator, but the results is correct. |
||[Type conversion operator](#6)|Yes|Convert to `paddle.cast` if operand is tensor|
|Paddle shape|[Tensor.shape()](#9)|Partial support|Support obtaining shape information, but there may be -1|
|python functions | [print(x)](#7) | Yes | Convert to `PrintOp` if operand is tensor |
| | [len()](#7) | Yes | Return the `shape[0]` if operand is tensor. |
| | [lambda expression](#7) | Yes | Use `convert_call` to convert lambda function|
| | [function call](#7) | Yes | Use `convert_call` to convert function recursively |
| | [recursively call](#7) | No | While converting, converted function can't get value from static tensor |
| | [list sort](#8) | No | Lists will be converted to `TensorArray` and `TensorArray` don't support sort|
| Errors and Exceptions |  assert | Yes | Convert to `Assert` if operand is tensor |
| Python containers | [list](#8) | Partial support | Lists will be converted to `TensorArray`. Only `append`, `pop`, `index` is supported |
| | [dict](#8) | Yes | @to_static will add the Tensors in a dict into PaddlePaddle static graph `Program`, so dict is supported by @to_static. |
| Third party library | numpy | Partial support | We suggest to use PaddlePaddle APIs to replace numpy API in this case. |

## Details

### if-else
<span id='1'></span>

#### principle :

While in the dynamic graph, the code is interpreted and executed line by line,  so the value of condition variables used by `if` are determined,  which means that the logic branch of False will not be executed.

However,  in the static graph, the control flow is realized through the `cond` operators. Each branch is represented by `true_fn` and `false_fn` respectively .  Under this circumstance, the `false_fn` will be executed to build the computation graph.

When the condition variables in `If` are `Tensor`,  `if-else` will be transformed to a `cond` operators.

When the condition variables in `If` aren't `Tensor`,  `if-else` will be executed as a python `if-else` code.

> Note:  When the condition variables in `If` are `Tensor`,  the `tensor.numel()` must equal to 1.


#### Error correction guide:

- When using` if `statement, please determine whether the type of condition variable is  `Paddle.Tensor`. If it is not a Tensor type, it will be executed according to the normal python logic and will not be converted into a static graph.

- Check the `numel()` of the condition variable is 1 .

### while
<span id='2'></span>


#### principle :

When the condition variable in the `while` loop is `Tensor`, the while statement will be converted into the `while_loop` API statement in the static graph, otherwise it will run as a normal Python while.

> Note:  When the condition variables in `If` are `Tensor`,  the `tensor.numel()` must equal to 1.


#### Error correction guide:

The same as `if-elif-else`

### for loops
<span id='3'></span>


#### principle :

`For loops` have different semantics according to different usage. Normally, the use of `for loops `can be divided into the following categories:

- `for _ in range(len)`  :  The statement will be converted into an equivalent Python while loop, and then the dynamic and static conversion will be performed according to the logic of the while loop.

- `for _ in x ` :  When x is a python container or iterator, it will run with normal Python logic. When x is `Tensor`, it will be converted to obtain `x[0]`, `x[1]`,… in sequence.

- `for idx, val in enumerate(x) ` :  When x is a Python container or iterator, it will run with normal python logic.  When x is a `Tensor`, idx will be converted into a 1-D Tensor with value 0, 1, ... in sequence. val will be converted into `x[0]`, `x[1]`, ... every time in the loop.

In views of implementation, the `for-loop` will eventually be transformed into the corresponding while statement, and then use `WhileOp` for static graph.

#### examples :
``` python
def ForTensor(x):
    """Fetch element in x and print the square of each x element"""
    for i in x :
        print (i * i)
#usage: ForTensor(paddle.to_tensor(x))
```

### return / break / continue
<span id='4'></span>


#### principle :

The current dynamic-to-static supports adding break and continue statements in for, while loops to change the control flow. It also supports adding return statements at any position inside the loop, and supports return tuples with different lengths and different `dtypes` of `Tensor`.

#### examples :
```python
# break usage example :
def break_usage(x):
    tensor_idx = -1
    for idx, val in enumerate(x) :
        if val == 2.0 :
            tensor_idx = idx
            break  # <------- jump out of while loop when break ;
    return tensor_idx
```

When you execute
```python
paddle.jit.to_static(break_usage)(paddle.to_tensor([1.0, 2.0, 3.0])
```
the tensor\_idx is `Tensor([1])`

> Note : Although idx is integer here, the return value is still Tensor. Because tensor_idx is converted into Tensor in the while loop.

### and / or / not
<span id='5'></span>


#### principle :

The dynamic-to-static module supports the conversion of and, or, and non-operators . According to the types of the two operands `x` and `y`, there will be different semantics:

- If both x, y are tensors, this statement will be converted to static graph.

- If neither x, y is tensor, this statement will degenerate to python operator.

- If only one of them is tensor, this statement will degenerate to python operator, but the results are always correct.

> Note :  If executed according to the semantics of paddle, `and` `or`, and `not` no longer support the lazy mode, it means that both expressions will be evaluated, instead of evaluating y according to the value of x.

#### examples :

```python
def and(x, y):
    z = y and x
    return z
```

### Type conversion operator
<span id='6'></span>

#### principle :

In dynamic graphs, you can directly use python's type conversion to convert Tensor types. For example, if x is a Tensor, float(x) can convert the data type of x to float.

Dynamic-to-static module will judge whether x is a `Tensor` at runtime. If so, use the static graph `paddle.cast` interface to convert to the target data type.

#### examples :
```python
def float_convert(x):
    z = float(x)
    return z
# if the  x = Tensor([True]) ，then z = Tensor([1.0])
```

### python function calls
<span id='7'></span>


Most circumstances of python function call is supported. The function calls will be converted into the form of `convert_xxx()`, and the data type of arguments will be determined during running the function. If some of arguments is `Tensor`, it will be transformed into a static computation graph; otherwise, it will be executed according to the original python semantics.

Some common functions are illustrated:

- `print (x)`
If the parameter is Tensor, `print(x)` can print the value of x in dynamic graph mode.While in dynamic-to-static graph model, It will be converted into a `Print` call. If the parameter is not Tensor, it will be executed according to python's print statement.

- `len (x)`
If the parameter is Tensor, `len(x)` can get the length of the 0th dimension of tensor x. While in dynamic-to-static graph model, It will be converted into a `control_flow.array_length` call. If the parameter is not Tensor, it will be executed according to python's print statement.

- `lambda`
The `to_static` function will call `convert_call` to convert lambda function as it's a normal function.

- function call (not recursively)
The `to_static` function will call `convert_call` to convert called function as it's a normal function.  Just put `@to_static` in the top-level function once.

#### examples :

```
def lambda_call(x):
    t = lambda x : x * x
    z = t(x)
    return z
# if the x is Tensor([2.0]) ，then z equals to Tensor([4.0]).
```

#### unsupported usage

- recursively call
While converting, converted function can’t get value from static tensor.

```
def recur_call(x):
    if x > 10:
        return x
    return recur_call(x * x) # < ------ If x = Tensor([2.0]) ，in dygraph mode the output is Tensor([16])，while in dygraph-to-static graph mode call stack overflows
```

### list / dict
<span id='8'></span>


#### principle :

1. list: if all elements in a list are Tensors, then @to_static converts it to TensorArray. PaddlePaddle static graph TensorArray supports append, pop, and modify, other list operations such as sort cannot be supported. When not all elements in a list are Tensors, @to_static will treat it as normal Python list.

2. dict: @to_static will add the Tensors in a dict into PaddlePaddle static graph Program, so dict is supported by @to_static.

```
def list_example(x, y):
     a = [ x ]   # < ------ supported
     a.append(x) # < ------ supported
     a[1] = y    # < ------ supported
     return a[0] # < ------ supported
```

> Note: List does not support multiple nesting and other operations. For specific error cases, see below.

#### unsupported usage

- multiple nesting

For example: l = [[tensor1, tensor2], [tensor3, tensor4]], because @to_static transformed a list whose elements are all Tensors into PaddlePaddle static graph TensorArray, but TensorArray doesn’t support multi-dimensions, @to_static cannot run this case.

We suggest to use 1-D list at most time, or use PaddlePaddle API create_array, array_read, array_write to control TensorArray.

- complex operators, such as sort

```
def sort_list(x, y):
    a = [x, y]
    sort(a)   # < -----  unsupported
    return a
```

### paddle.shape
<span id='9'></span>


#### principle :

> partial supported

- Support simple usage of `shape`, such as get the shape of a tensor.

- Don't support get shape after a reshape operators. You may get a -1 in shape value.

For example, `x = reshape(x, shape=shape_tensor)` , then use `x.shape[0]` to do other operation. Due to the difference between dynamic and static graph, it is okay in dynamic but it will fail in static graph. The reason is that APIs return computation result in dynamic graph mode, so x.shape has deterministic value after calling reshape . However, static graph doesn’t have the value shape_tensor during building network, so PaddlePaddle doesn’t know the value of x.shape after calling reshape. PaddlePaddle static graph will set -1 to represent unknown shape value for each dimension of x.shape in this case, not the expected value. Similarily, calling the shape of the output tensor of those APIs which change the shape, such as expend, cannot be converted into static graph properly.

#### examples :

```
def get_shape(x):
    return x.shape[0] # <---- supported
```

```
def error_shape(x, y):
    y = y.cast('int32')
    t = x.reshape(y)
    return t.shape[0] # <------- don't supported ; if x = Tensor([2.0, 1.0])，y = Tensor([2])，in dygraph mode the output is 2，while in dygraph-to-static graph mode the output is -1.
```
