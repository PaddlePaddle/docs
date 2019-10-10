
# Guide to Fluid Programming

This document will instruct you to program and create a simple nueral network with Fluid API. From this guide, you will get the hang of:

- Core concepts of Fluid
- How to define computing process in Fluid
- How to run fluid operators with executor
- How to model practical problems logically
- How to call API（layers, datasets, loss functions, optimization methods and so on)

Before building model, you need to figure out several core concepts of Fluid at first:

## Express data with Tensor

Like other mainstream frameworks, Fluid uses Tensor to hold data.

All data transferred in neural network are Tensor which can simply be regarded as a multi-dimensional array. In general, the number of dimensions can be any. Tensor features its own data type and shape. Data type of each element in single Tensor is the same. And **the shape of Tensor** refers to the dimensions of Tensor.

Picture below visually shows Tensor with dimension from one to six:
<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/beginners_guide/image/tensor.jpg" width="400">
</p>


There are three special kinds of Tensor in Fluid:

**1. Learnable parameters of models**

The lifetime of learnable parameters (such as network weight, bias and so on) of model is equal to the time of training task. The parameters will be updated by optimization algorithms. We use Parameter, the derived class of Variable, to express parameters.

We can create learnable parameters with `fluid.layers.create_parameter` in Fluid:

```python
w = fluid.layers.create_parameter(name="w",shape=[1],dtype='float32')
```


In general, you don't need to explicitly create learnable parameters of network. Fluid encapsulates most fundamental computing modules in common networks. Take the fully connected model as a simplest example, The codes below create connection weight(W) and bias(bias) for fully connected layer with no need to explicitly call associated APIs of Parameter.

```python
import paddle.fluid as fluid
y = fluid.layers.fc(input=x, size=128, bias_attr=True)
```


**2. Input and Output Tensor**

The input data of the whole neural network is also a special Tensor in which the sizes of some dimensions can not be decided at the definition time of models. Such dimensions usually includes batch size, or width and height of image when such data formats in a mini-batch are not constant. Placeholders for these uncertain dimension are necessary at the definition phase of model.


`fluid.layers.data` is used to receive input data in Fluid, and it needs to be provided with the shape of input Tensor. When the shape is not certain, the correspondent dimension is defined as None.

The code below exemplifies the usage of `fluid.layers.data` :

```python
import paddle.fluid as fluid

#Define the dimension of x : [3,None]. What we could make sure is that the first dimension of x is 3.
#The second dimension is unknown and can only be known at runtime.
x = fluid.layers.data(name="x", shape=[3,None], dtype="int64")

#batch size doesn't have to be defined explicitly.
#Fluid will automatically assign zeroth dimension as batch size dimension and fill right number at runtime.
a = fluid.layers.data(name="a",shape=[3,4],dtype='int64')

#If the width and height of image are variable, we can define the width and height as None.
#The meaning of three dimensions of shape is channel, width of image, height of image respectively.
b = fluid.layers.data(name="image",shape=[3,None,None],dtype="float32")
```

dtype=“int64” indicates signed int 64 bits data. For more data types supported by Fluid, please refer to [Data types currently supported by Fluid](../../user_guides/howto/prepare_data/feeding_data_en.html#fluid).

**3. Constant Tensor**

`fluid.layers.fill_constant` is used to define constant Tensor in Fluid. You can define the shape, data type and value of Constant Tensor. Code is as follows:

```python
import paddle.fluid as fluid
data = fluid.layers.fill_constant(shape=[1], value=0, dtype='int64')
```

Notice that the tensor defined above is not assigned with values. It merely represents the operation to perform. If you print data directly, you will get information about the description of this data:

```python
print data
```
Output:

```
name: "fill_constant_0.tmp_0"
type {
    type: LOD_TENSOR
    lod_tensor {
        tensor {
            data_type: INT64
            dims: 1
        }
    }
}
persistable: false
```

Specific output value will be shown at the runtime of Executor. There are two ways to get runtime Variable value. The first way is to  use `paddle.fluid.layers.Print` to create a print op that will print the tensor being accessed. The second way is to add Variable to Fetch_list. 

Code of the first way is as follows:

```python
import paddle.fluid as fluid
data = fluid.layers.fill_constant(shape=[1], value=0, dtype='int64')
data = fluid.layers.Print(data, message="Print data: ")
```

Output at  the runtime of Executor:

```
1563874307	Print data: 	The place is:CPUPlace
Tensor[fill_constant_0.tmp_0]
	shape: [1,]
	dtype: x
	data: 0,
```

For more information on how to use the Print API, please refer to [Print operator](https://www.paddlepaddle.org.cn/documentation/docs/en/1.5/api/layers/control_flow.html#print).

Detailed process of the second way Fetch_list will be explained later.

## Feed data

The method to feed data in Fluid:

You need to use `fluid.layers.data` to configure data input layer and use ``executor.run(feed=...)`` to feed training data into `fluid.Executor` or `fluid.ParallelExecutor` .

For specific preparation for data, please refer to [Preparation for data](../../user_guides/howto/prepare_data/index_en.html).


## Operators -- operations on data

All operations on data are achieved by Operators in Fluid.

To facilitate development, on Python end, Operators in Fluid are further encapsulated into `paddle.fluid.layers` , `paddle.fluid.nets` and other modules.

It is because some common operations for Tensor may be composed of many fundamental operations. To make it more convenient, fundamental Operators are encapsulated in Fluid to reduce repeated coding, including the creation of learnable parameters which Operator relies on, details about initialization of learnable parameters and so on.

For example, you can use `paddle.fluid.layers.elementwise_add()` to add up two input Tensor:

```python
#Define network
import paddle.fluid as fluid
a = fluid.layers.data(name="a",shape=[1],dtype='float32')
b = fluid.layers.data(name="b",shape=[1],dtype='float32')

result = fluid.layers.elementwise_add(a,b)

#Define Exector
cpu = fluid.core.CPUPlace() #define computing place. Here we choose to train on CPU
exe = fluid.Executor(cpu) #create executor
exe.run(fluid.default_startup_program()) #initialize network parameters

#Prepare data
import numpy
data_1 = int(input("Please enter an integer: a="))
data_2 = int(input("Please enter an integer: b="))
x = numpy.array([[data_1]])
y = numpy.array([[data_2]])

#Run computing
outs = exe.run(
feed={'a':x,'b':y},
fetch_list=[result.name])

#Verify result
print "%d+%d=%d" % (data_1,data_2,outs[0][0])
```

Output:
```
a=7
b=3
7+3=10
```

At runtime, input a=7,b=3, and you will get output=10.

You can copy the code, run it locally, input different numbers following the prompt instructions and check the computed result.

If you want to get the specific value of a,b at the runtime of network, you can add variables you want to check into ``fetch_list`` .

```python
...
#Run computing
outs = exe.run(
    feed={'a':x,'b':y},
    fetch_list=[a,b,result.name]
#Check output
print outs
```

Output:
```
[array([[7]]), array([[3]]), array([[10]])]
```

## Use Program to describe neural network model

Fluid is different from most other deep learning frameworks. In Fluid, static computing map is replaced by Program to dynamically describe the network. This dynamic method delivers both flexible modifications to network structure and convenience to build model. Moreover, the capability of expressing a model is enhanced significantly while the performance is guaranteed.

All Operators will be written into Program, which will be automatically transformed into a descriptive language named ProgramDesc in Fluid. It's like to write a general program to define Program. If you are an experienced developer, you can naturally apply the knowledge you have acquired on Fluid programming.

You can describe any complex model by combining sequential processes, branches and loops supported by Fluid.

**Sequential Process**

You can use sequential structure to build network:

```python
x = fluid.layers.data(name='x',shape=[13], dtype='float32')
y_predict = fluid.layers.fc(input=x, size=1, act=None)
y = fluid.layers.data(name='y', shape=[1], dtype='float32')
cost = fluid.layers.square_error_cost(input=y_predict, label=y)
```

**Conditional branch——switch,if else：**

Switch and if-else class are used to implement conditional branch in Fluid. You can use the structure to adjust learning rate in learning rate adapter or perform other operations :

```python
lr = fluid.layers.tensor.create_global_var(
        shape=[1],
        value=0.0,
        dtype='float32',
        persistable=True,
        name="learning_rate")

one_var = fluid.layers.fill_constant(
        shape=[1], dtype='float32', value=1.0)
two_var = fluid.layers.fill_constant(
        shape=[1], dtype='float32', value=2.0)

with fluid.layers.control_flow.Switch() as switch:
    with switch.case(global_step == zero_var):
        fluid.layers.tensor.assign(input=one_var, output=lr)
    with switch.default():
        fluid.layers.tensor.assign(input=two_var, output=lr)
```


For detailed design principles of Program, please refer to [Design principle of Fluid](../../advanced_usage/design_idea/fluid_design_idea_en.html).

For more about control flow in Fluid, please refer to [Control Flow](../../api/layers.html#control-flow).


## Use Executor to run Program

The design principle of Fluid is similar to C++, JAVA and other advanced programming language. The execution of program is divided into two steps: compile and run.

Executor accepts the defined Program and transforms it to a real executable Fluid Program at the back-end of C++. This process performed automatically is the compilation.

After compilation, it needs Executor to run the compiled Fluid Program.

Take add operator above as an example, you need to create an Executor to initialize and train Program after the construction of Program:

```python
#define Executor
cpu = fluid.core.CPUPlace() #define computing place. Here we choose training on CPU
exe = fluid.Executor(cpu) #create executor
exe.run(fluid.default_startup_program()) #initialize Program

#train Program and start computing
#feed defines the order of data transferred to network in the form of dict
#fetch_list defines the output of network
outs = exe.run(
    feed={'a':x,'b':y},
    fetch_list=[result.name])
```

## Code example

So far, you have got a primary knowledge of core concepts in Fluid. Why not try to configure a simple network ? You can finish a very simple data prediction under the guide of the part if you are interested. If you have learned this part, you can skip this section and read [What's next](#what_next).

Firstly, define input data format, model structure,loss function and optimized algorithm logically. Then you need to use PaddlePaddle APIs and operators to implement the logic of model. A typical model mainly contains four parts. They are: definition of input data format; forward computing logic; loss function; optimization algorithm.

1. Problem

    Given a pair of data $<X,Y>$，construct a function $f$ so that $y=f(x)$ . $X$ , $Y$ are both one dimensional Tensor. Network finally can predict $y_{\_predict}$ accurately according to input $x$.

2. Define data

    Supposing input data X=[1 2 3 4]，Y=[2,4,6,8], make a definition in network:

    ```python
    #define X
    train_data=numpy.array([[1.0],[2.0],[3.0],[4.0]]).astype('float32')
    #define ground-truth y_true expected to get from the model prediction
    y_true = numpy.array([[2.0],[4.0],[6.0],[8.0]]).astype('float32')
    ```

3. Create network (define forward computing logic)

    Next you need to define the relationship between the predicted value and the input. Take a simple linear regression function for example:

    ```python
    #define input data type
    x = fluid.layers.data(name="x",shape=[1],dtype='float32')
    #create fully connected network
    y_predict = fluid.layers.fc(input=x,size=1,act=None)
    ```

    Now the network can predict output. Although the output is just a group of random numbers, which is far from expected results:

    ```python
    #load library
    import paddle.fluid as fluid
    import numpy
    #define data
    train_data=numpy.array([[1.0],[2.0],[3.0],[4.0]]).astype('float32')
    y_true = numpy.array([[2.0],[4.0],[6.0],[8.0]]).astype('float32')
    #define predict function
    x = fluid.layers.data(name="x",shape=[1],dtype='float32')
    y_predict = fluid.layers.fc(input=x,size=1,act=None)
    #initialize parameters
    cpu = fluid.core.CPUPlace()
    exe = fluid.Executor(cpu)
    exe.run(fluid.default_startup_program())
    #start training
    outs = exe.run(
        feed={'x':train_data},
        fetch_list=[y_predict.name])
    #observe result
    print outs
    ```

    Output:

    ```
    [array([[0.74079144],
               [1.4815829 ],
               [2.2223744 ],
               [2.9631658 ]], dtype=float32)]
    ```

4. Add loss function

    After the construction of model, we need to evaluate the output result in order to make accurate predictions. How do we evaluate the result of prediction? We usually add loss function to network to compute the *distance* between ground-truth value and predict value.

    In this example, we adopt [mean-square function](https://en.wikipedia.org/wiki/Mean_squared_error) as our loss function ：

    ```python
    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_cost = fluid.layers.mean(cost)
    ```
    Output predicted value and loss function after a process of computing:

    ```python
    #load library
    import paddle.fluid as fluid
    import numpy
    #define data
    train_data=numpy.array([[1.0],[2.0],[3.0],[4.0]]).astype('float32')
    y_true = numpy.array([[2.0],[4.0],[6.0],[8.0]]).astype('float32')
    #define network
    x = fluid.layers.data(name="x",shape=[1],dtype='float32')
    y = fluid.layers.data(name="y",shape=[1],dtype='float32')
    y_predict = fluid.layers.fc(input=x,size=1,act=None)
    #define loss function
    cost = fluid.layers.square_error_cost(input=y_predict,label=y)
    avg_cost = fluid.layers.mean(cost)
    #initialize parameters
    cpu = fluid.core.CPUPlace()
    exe = fluid.Executor(cpu)
    exe.run(fluid.default_startup_program())
    #start training
    outs = exe.run(
        feed={'x':train_data,'y':y_true},
        fetch_list=[y_predict.name,avg_cost.name])
    #observe output
    print outs
    ```
    Output:

    ```
    [array([[0.9010564],
        [1.8021128],
        [2.7031693],
        [3.6042256]], dtype=float32), array([9.057577], dtype=float32)]
    ```

    We discover that the loss function after the first iteration of computing is 9.0, which shows there is a great improve space.

5. Optimization of network

    After the definition of loss function,you can get loss value by forward computing and then get gradients of parameters with chain derivative method.

    Parameters should be updated after you have obtained gradients. The simplest algorithm is random gradient algorithm: w=w−η⋅g,which is implemented by `fluid.optimizer.SGD`:
    ```python
    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.01)
    ```
    Let's train the network for 100 times and check the results:

    ```python
    #load library
    import paddle.fluid as fluid
    import numpy
    #define data
    train_data=numpy.array([[1.0],[2.0],[3.0],[4.0]]).astype('float32')
    y_true = numpy.array([[2.0],[4.0],[6.0],[8.0]]).astype('float32')
    #define network
    x = fluid.layers.data(name="x",shape=[1],dtype='float32')
    y = fluid.layers.data(name="y",shape=[1],dtype='float32')
    y_predict = fluid.layers.fc(input=x,size=1,act=None)
    #define loss function
    cost = fluid.layers.square_error_cost(input=y_predict,label=y)
    avg_cost = fluid.layers.mean(cost)
    #define optimization algorithm
    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.01)
    sgd_optimizer.minimize(avg_cost)
    #initialize parameters
    cpu = fluid.core.CPUPlace()
    exe = fluid.Executor(cpu)
    exe.run(fluid.default_startup_program())
    ##start training and iterate for 100 times
    for i in range(100):
        outs = exe.run(
            feed={'x':train_data,'y':y_true},
            fetch_list=[y_predict.name,avg_cost.name])
    #observe result
    print outs
    ```

    Output:
    ```
    [array([[2.2075021],
            [4.1005487],
            [5.9935956],
            [7.8866425]], dtype=float32), array([0.01651453], dtype=float32)]
    ```
    Now we discover that predicted value is nearly close to real value and the loss value descends from original value 9.05 to 0.01 after iteration for 100 times.

    Congratulations! You have succeed to create a simple network. If you want to try advanced linear regression —— predict model of housing price, please read [linear regression](../../beginners_guide/basics/fit_a_line/README.en.html). More examples of model can be found in [models](../../user_guides/models/index_en.html).

<a name="what_next"></a>
## What's next

If you have been familiar with fundamental operations, you can start your next journey to learn fluid:

You will learn how to build model for practical problem with fluid: [The configuration of simple network](../../user_guides/howto/configure_simple_model/index_en.html).

After the construction of network, you can start training your network in single node or multiple nodes. For detailed procedures, please refer to [train neural network](../../user_guides/howto/training/index_en.html).

In addition, there are three learning levels in documentation according to developer's background and experience: [Beginner's Guide](../../beginners_guide/index_en.html) , [User Guides](../../user_guides/index_en.html) and [Advanced User Guides](../../advanced_usage/index_en.html).

If you want to read examples in more application scenarios, you can go to [basic knowledge of deep learning](../../beginners_guide/basics/index_en.html) .If you have learned basic knowledge of deep learning, you can read from [user guide](../../user_guides/index_en.html).
