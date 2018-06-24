前言
########

在解决实际问题时,可以先从逻辑层面对问题进行建模,明确模型所需要的**输入数据类型**,**计算逻辑**,**求解目标**以及**优化算法**.模型定义清晰后,可以使用PaddlePaddle提供的丰富算子来搭建模型并使用训练数据进行训练.完整流程如下图所示: 

下面举例说明如何使用PaddlePaddle解决一个简单的问题. 该例子完整代码参见*ref fit_a_line*.

问题描述及定义
##############

问题描述: 给定一组数据<X, Y>,求解出函数f,使得y=f(x),其中X属于R13,每维代表房屋的一个特征,y属于R表示该房屋的房价.考虑到输入样本为一个13维的实数向量,输出为一个实数值, 可以尝试用回归模型来对问题建模.回归模型的求解目标函数有很多,这里使用均方误差作为损失函数,确定了损失函数后,需要对f的复杂度进行判断,这里假定f为简单的线性变换函数. 除了明确模型的输入格式,求解目标以及模型结构外,还需要选择合适的优化方法,这里选用随机梯度下降算法来求解模型.

使用PaddlePadle建模
###################

上一节从逻辑层面明确了输入数据格式,模型结构,损失函数以及优化算法,下面需要使用PaddlePaddle提供的算子来实现模型.模型实现后,需要使用训练数据对模型参数求解,训练完成后,需要提供模型推断功能,即加载训练好的模型,预测样本x的输出值.

数据层
-------

PaddlePaddle提供了data算子来描述输入数据的格式,data算子的输出是一个tensor,tensor可以表示具有多个维度的数据,比较重要的参数包括shape和type.训练神经网络时一般会使用mini-batch的方式喂入数据,而batch size在训练过程中可能不固定,data算子会依据喂入的数据来推断batch size,所以这里定义shape时不用关心batch size的大小,只需关心一条样本的shape即可.从上可知,x为13维的实数向量,y为实数,可使用下面代码定义数据层:

.. code-block:: python

    x = fluid.layers.data(name='x', shape=[13], dtype='float32')
    y = fluid.layers.data(name='y', shape=[1], dtype='float32')

PaddlePaddle还提供了效率更高的方式来喂入数据,不需要先转成python numpy的中间表示,直接从文件中读取,这种方式可以减少数据i/o的时间消耗,具体可以参考*ref data provider*

计算逻辑
---------

模型结构里面最重要的部分是计算逻辑的定义,比如图像相关任务中会使用较多的卷积算子,序列任务中会使用LSTM/GRU等算子.一个算子通常对应一种或一组变换逻辑,算子输出即为对输入数据执行变换后的结果.复杂模型通常会组合多种算子,以完成复杂的变换,PaddlePaddle提供了非常自然的方式来组合算子,一般地可以使用下面的方式:

.. code-block:: python

    op_1_out = fluid.layers.op_1(input=op_1_in, ...)
    op_2_out = fluid.layers.op_2(input=op_1_out, ...)
    ...

其中op_1和op_2表示算子,可以是fc来执行线性变化,可以是conv来执行卷积变换.更复杂的模型可能需要使用控制流算子,依据输入数据来动态执行相关逻辑,针对这种情况,PaddlePaddle提供了IfElseOp和WhileOp,有关算子的文档可参考*ref op doc*. 具体到这个任务, 我们使用一个fc算子:

.. code-block:: python

    y_predict = fluid.layers.fc(input=x, size=1, act=None)

损失函数
----------

损失函数对应求解目标,模型通过最小化损失来求解模型参数.大多数模型会使用平均损失作为最终的loss,所以在PaddlePaddle中一般会在损失算子后面接mean算子来求平均.模型在一次前向迭代后会得到一个损失值,框架会依据该值计算梯度,然后自动执行链式求导法则计算模型里面每个参数对应的梯度值.这里使用均方误差损失:

.. code-block:: python

    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_cost = fluid.layers.mean(cost)

更多损失函数算子可以参考*ref loss op*.

优化方法
----------

确定损失函数后,可以通过前向计算得到损失值,然后通过链式求导法则得到参数的梯度值.获取梯度值后需要更新参数,最简单的sgd,*sgd 公式*,但是sgd有一些缺点,比如收敛不稳定等,为了改善模型的训练速度以及效果,先后提出了很多优化算子,包括momentum/autograd/rmsprop/adam等,这些优化算法采用不同的策略来更新模型参数,可以针对不同任务尝试不同的优化方法,同时需要对一些超参数进行实验调参.一般来讲learning rate是优化算子中的一个重要参数,需要仔细调整,这里采用sgd优化算法:

.. code-block:: python

    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)

更多优化算子可以参考*ref opt op*.

模型训练
----------

至此,模型已经实现完成,下面需要在训练数据上训练模型的参数.首先需要实现一个reader以batch的方式提供数据,PaddlePaddle提供了易用的封装,可以使用`paddle.batch`来实现一个batch data迭代器,通常在训练阶段对数据打乱可以增强模型的效果,在paddle.batch里面嵌套'paddle.reader.shuffle'即可.接下来需要定义执行器,通过`fluid.Executor`来定义一个执行器,执行器文档可参考'executor doc'. 最后可以循环来不断迭代训练模型,通过执行器中的`run`函数来完成一次迭代.在训练过程中,可以通过调用'fluid.io.save_inference_model'来保存中间的模型.完整代码:

.. code-block:: python

    BATCH_SIZE = 20

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.uci_housing.train(), buf_size=500),
        batch_size=BATCH_SIZE)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    def train_loop(main_program):
        feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
        exe.run(fluid.default_startup_program())

        PASS_NUM = 100
        for pass_id in range(PASS_NUM):
            for data in train_reader():
                avg_loss_value, = exe.run(main_program,
                                          feed=feeder.feed(data),
                                          fetch_list=[avg_cost])
                print(avg_loss_value)
                if avg_loss_value[0] < 10.0:
                    if save_dirname is not None:
                        fluid.io.save_inference_model(save_dirname, ['x'],
                                                      [y_predict], exe)
                    return
                if math.isnan(float(avg_loss_value)):
                    sys.exit("got NaN loss, training failed.")
        raise AssertionError("Fit a line cost is too large, {0:2.2}".format(
            avg_loss_value[0]))

模型推断
----------

模型训练完成后,需要提供预测功能,给定x后,可以预测对应的y值.与训练阶段一样,需要首先定义一个执行器,然后调用'fluid.io.load_inference_model'来加载保存的模型,最后调用'exe.run'来执行预测. 完整代码见:

.. code-block:: python

    def infer(use_cuda, save_dirname=None):
        if save_dirname is None:
            return
    
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
    
        inference_scope = fluid.core.Scope()
        with fluid.scope_guard(inference_scope):
            # Use fluid.io.load_inference_model to obtain the inference program desc,
            # the feed_target_names (the names of variables that will be feeded
            # data using feed operators), and the fetch_targets (variables that
            # we want to obtain data from using fetch operators).
            [inference_program, feed_target_names,
             fetch_targets] = fluid.io.load_inference_model(save_dirname, exe)
    
            # The input's dimension should be 2-D and the second dim is 13
            # The input data should be >= 0
            batch_size = 10
            tensor_x = numpy.random.uniform(0, 10, 
                                            [batch_size, 13]).astype("float32")
            assert feed_target_names[0] == 'x' 
            results = exe.run(inference_program,
                              feed={feed_target_names[0]: tensor_x},
                              fetch_list=fetch_targets)
        print("infer shape: ", results[0].shape)
        print("infer results: ", results[0])

总结
#####

