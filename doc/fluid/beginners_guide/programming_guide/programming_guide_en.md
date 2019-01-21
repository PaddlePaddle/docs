
# Guide of Fluid Program

This document will guide you how to use Fluid API to program and create a simple nueral network. After finishing the document,you will know:

- Core conception of Fluid
- How to define computing process in fluid
- How to run fluid with executor
- How to build model for real problem logically
- How to call API（layer,data set,lost function,optimized methods and so on)

Before building model,you need to figure out several core conception of fluid at first:

## Express data with Tensor 

Like other main frames, in Fluid,Tensor is used to carry data.

Data transferring in nueral network is Tensor which can simply be regarded as a multi-dimension array.In general,the dimension is unlimit.Tensor features its own data type and shape. Data type of each element in the same Tensor is the same. And the shape of Tensor is dimension of Tensor.

Picture below directly shows Tensor with dimension from one to six: 
<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/beginners_guide/image/tensor.jpg" width="400">
</p>


There are three special Tensor in Fluid:

**1. Learnable parameters of model**

The life expectancy of Learnable parameters(such as network weight,bias and so on) of model is the same as time of train task.The parameters will be updated by optimized algorithms. We use Parameter,child class of Variable, to express parameters.

We can create learnable parameters with `fluid.layers.create_parameter` in Fluid:

```python
w = fluid.layers.create_parameter(name="w",shape=[1],dtype='float32')
```


In general,you don't need to create learnable parameters of network. There are packages for fudamental computing modules in most common networks. Take the simplest full connection model as an example, you can create connection weight(W) and bias(bias) these two learnable parameters for full connection layer with no need to call associated APIs of Parameter.

```python
import paddle.fluid as fluid
y = fluid.layers.fc(input=x, size=128, bias_attr=True)
```


**2. Input and Output Tensor**

The input data of the whole neural network is also a special Tensor in which some dimensions' size  at the definition of model can not be defined,usually including batch size,width and height of image supposing data between mini-batch is variable.Definition of model results in occupation of storage.


`fluid.layers.data` is used to receieve input data in Fluid, `fluid.layers.data` needs to provide shape of input Tensor. When the shape is not certain,the correspondent dimension is defined as None. Code is as follows:

```python
import paddle.fluid as fluid

#Define the dimension of x is [3,None]. What we could make sure is that the first shape of x is 3 and the second shape is unknown and known in the running process of program.
x = fluid.layers.data(name="x", shape=[3,None], dtype="int64")

#batch size doesn't have to be defined explicitly.The frame will automatically assign zero dimension as batch size and fill right number at runtime.
a = fluid.layers.data(name="a",shape=[3,4],dtype='int64')

#If the width and height of image are variable,we can define the width and height as None.
#The meaing of three dimensions of shape is channel,width of image,height of image respectively.
b = fluid.layers.data(name="image",shape=[3,None,None],dtype="float32")
```

dtype=“int64” indicates sign int 64 bits data. About more data types supported by Fluid now,please refer to [Data types currently supported by Fluid](../../user_guides/howto/prepare_data/feeding_data.html#fluid).

**3. Constant variable Tensor**

`fluid.layers.fill_constant` is used to implement constant variable Tensor in Fluid,you can define the shape,data type and value of constant variable.Code is as follows:

```python
import paddle.fluid as fluid
data = fluid.layers.fill_constant(shape=[1], value=0, dtype='int64')
```

Attention needs to be paid that tensor defined above is not entitle with value which merely represents the operation to be take.If you print data directly,you will get information about the description of this data:

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
    
Specific output value will be shown at the runtime of Executor.Detailed implemention will be expanded later.

## Feed data

There is typical method to feed data in Fluid:

You need to use `fluid.layers.data` to configure data input layer and use executor.run(feed=...) to feed train data in `fluid.Executor` or `fluid.ParallelExecutor` .

About specific preparation for data,please refer to [Preparation for data](../../user_guides/howto/prepare_data/index.html).


## Use Operator to indicate operations for data

All operation for data are indicated by Operator in Fluid.You can use intrisic command to describe their networks.

For the convenience of you,at the terminal of Python,Operator in Fluid is further packaged into `paddle.fluid.layers`,`paddle.fluid.nets` and other modules.

It is because some common operations for Tensor may be composed by many fundamental operations.To make it more convenient,fundamental Operator is packaged in frame,including the creation of learnable parameters relied by Operator,details about initialization of learnable parameters and so on,to reduce the cost of repeated development.

For example,you can use `paddle.fluid.layers.elementwise_add()` to implement the add of two input Tensor:

```python
#Define network
import paddle.fluid as fluid
a = fluid.layers.data(name="a",shape=[1],dtype='float32')
b = fluid.layers.data(name="b",shape=[1],dtype='float32') 

result = fluid.layers.elementwise_add(a,b)

#Define Exector
cpu = fluid.core.CPUPlace() #define computing space.Here we choose train in CPU
exe = fluid.Executor(cpu) #create executor
exe.run(fluid.default_startup_program()) #initialize network parameters

#Prepare data
import numpy
data_1 = input("a=")
data_2 = input("b=")
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

At this runtime,inputa=7,b=3,and you will get outs=10.

You can copy the code,run it locally,input different number according to indication to checkout computing result.

If you want to get the value of a,b at the runtime of network,you can add variable you want to check into fetch_list.

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

Fluid is different from other most deep learning frames, in which static computing map is replaced by Program to dynamically describe the computation.The dynamic method of computing desciption features flexible modification of network structure and convenience to build model to largely stengthen the ability of expression frame makes for model under good performance.

All Operator of developers will be written into Program,which will be automatically transformed as a description language named ProgramDesc in Fluid.It's like to write a general program to define Program.If you are an experinced developer,you can naturally combine the knowledge you have acquired with the process naturally.

You can describe any complex model by combining sequential,branch and loop structure supported by Fluid.

**Run in order**

You can use sequential structure to build network:

```python
x = fluid.layers.data(name='x',shape=[13], dtype='float32')
y_predict = fluid.layers.fc(input=x, size=1, act=None)
y = fluid.layers.data(name='y', shape=[1], dtype='float32')
cost = fluid.layers.square_error_cost(input=y_predict, label=y)
```

**Conditional branch——switch,if else：**

switch and if-else class are used to implement conditional branch in Fluid.You can use the structure to adjust learning rate or other perform other operations in learning rate adjustor:

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

    
About detailed design principle of Program in ,please refer to [Design principle of Fluid](../../user_guides/design_idea/fluid_design_idea.html).

About more control flow in Fluid,please refer to [API Document](http://www.paddlepaddle.org/documentation/api/zh/1.0.0/layers.html#permalink-1-control_flow).


## Use Executor to run Program

The design principle of Fluid is similiar to C++,JAVA and other advanced programming language. The execution of program is divided into two steps:build and run.

With definition of Program finished,Executor receieves the Program and transforms it to real executable FluidProgram at the backend of C++.The automatic execution is called build.

It needs Executor to run the built FluidProgram after building.

Take add operator above as an example,you need to create Executor to initialize and train Program after the contruction of Program:

```python
#defineExecutor
cpu = fluid.core.CPUPlace() #define computing space.Here we choose train in CPU
exe = fluid.Executor(cpu) #create executor
exe.run(fluid.default_startup_program()) #initialize Program

#train Program and start computing
#feed defines the order of data transferring to network in the form of dict
#fetch_list defines the output of network
outs = exe.run(
    feed={'a':x,'b':y},
    fetch_list=[result.name])
```

## Code examaple

You have a primary knowledge of core conception of Fluid.Then you can try to configure a simple network and finish a very simple data prediction under the guide of the part if you are insterested.If you have learned the part,you can skip this section and read [What's next](#what_next).

After finishing defined input data format,model structure,loss function and optimized algorithm logically,you need to use PaddlePaddle APIs and operators to implement the logic of model.A typical model mainly contains four parts which are definition of input data format,forward computing logic,loss function and optimized algorithm.

1. Problem

    Given a pair of data $<X,Y>$，figure out function $f$, $y=f(x)$.$X$,$Y$ are both one dimensional Tensor.Network finally can predict $y_{\_predict}$ accurately according to input $x$.

2. Define data

    Supposing input data X=[1 2 3 4]，Y=[2,4,6,8],make a definition in network:
    
    ```python
    #define X
    train_data=numpy.array([[1.0],[2.0],[3.0],[4.0]]).astype('float32')
    #define real y_true expected to predict
    y_true = numpy.array([[2.0],[4.0],[6.0],[8.0]]).astype('float32')
    ```
        
3. Create network(define forward computing logic)

   Next you need to define the relationship between prediced value and input.Take a simple linear regression function to predict this time:
    
    ```python
    #define input data type
    x = fluid.layers.data(name="x",shape=[1],dtype='float32')
    #create full connected network
    y_predict = fluid.layers.fc(input=x,size=1,act=None)
    ```
    
    Now the network can be predicted.Although the output is just a group of random numbers,which is far from expected results:
    
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
    
    After the construction of model,how do we evaluate the result of prediction?We usually add loss fuction to network to compute the gap between real value and predict value.

    In the example,we adopt loss function [mean-square function](https://en.wikipedia.org/wiki/Mean_squared_error)：
    ```python
    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_cost = fluid.layers.mean(cost)
    ```
    Output predict value and loss function after a process of computing:
    
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

    It can be refered that the loss function after the first process of computing is 9.0,which means a great potential descent space.
    
5. Optimization of network
    
    After the definition of loss function,you can get loss value by forward computing and then get gradient value of parameters with chain derivative method.
    
    Parameters should be updated after you geting gradient value.The simplest algorithm is random gradient algorithm:w=w−η⋅g,which is implemented by `fluid.optimizer.SGD`:
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
    #define optimized algorithm
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
    It can be refered that predict value is nearly close to real value and the loss value descents from original value 9.05 to 0.01 after iteration for 100 times.
    
    Congratulation! You have succeed to create a simple network.If you want to try advanced linear regression——predict model of housing price,please read [linear regression](../../beginners_guide/quick_start/fit_a_line/README.cn.html).More examples of model can be found in [models](../../user_guides/models/index_cn.html).

<a name="what_next"></a>
## What's next

If you have acquired fundamental operations,you can start your next journey to learn fluid:

Following the lesson,you will learn how to build model for real problem with fluid: [The configuration of simple network](../../user_guides/howto/configure_simple_model/index.html).

After the construction of network,you can start training your network in single node or multiple nodes.About detailed procedures,please refer to [train neural network](../../user_guides/howto/training/index.html).

In addition,there are three learning stage in documentation according to developer's learning background and experience: [beginner guide](../../beginners_guide/index.html),[user guide](../../user_guides/index.html) and [advanced usage](../../advanced_usage/index.html).

If you want to read examples in more application scenarios,you can follow navigation entering into [quick start](../../beginners_guide/quick_start/index.html) and[basic knowledge of deep learning](../../beginners_guide/basics/index.html).If you have learned basic knowledge of deep learning,you can read from [user guide](../../user_guides/index.html).
