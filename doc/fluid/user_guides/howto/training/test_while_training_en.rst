.. _user_guide_test_while_training_en:

##############################
Evaluate model while training
##############################

:code:`fluid.Program` for model test and evaluation is different from the one for training. In the test and evalution phase:

1. There is no back propagation and no process of optimizing and updating parameters in evaluation and test.
2. Operations in model evaluation can be different.

   * Take the operator BatchNorm for example, algorithms are different in train and test.

   * Evaluation model and training model can be totally different.

Generate :code:`fluid.Program` for test
#######################################

Generate test :code:`fluid.Program` by cloning training :code:`fluid.Program` 
============================================================================

:code:`Program.clone()` can generate a copied new :code:`fluid.Program` . You can generate a copy of Program with operations applied for test by setting :code:`Program.clone(for_test=True)` . Simple usage is as follows:

.. code-block:: python

   import paddle.fluid as fluid

   img = fluid.layers.data(name="image", shape=[784])
   prediction = fluid.layers.fc(
     input=fluid.layers.fc(input=img, size=100, act='relu'),
     size=10,
     act='softmax'
   )
   label = fluid.layers.data(name="label", shape=[1], dtype="int64")
   loss = fluid.layers.mean(fluid.layers.cross_entropy(input=prediction, label=label))
   acc = fluid.layers.accuracy(input=prediction, label=label)

   test_program = fluid.default_main_program().clone(for_test=True)

   adam = fluid.optimizer.Adam(learning_rate=0.001)
   adam.minimize(loss)

Before using :code:`Optimizer` , please copy :code:`fluid.default_main_program()` into a :code:`test_program` . Then you can pass test data to :code:`test_program` so that you can run test program without influencing training result.

Configure training :code:`fluid.Program` and test :code:`fluid.Program` individually
=====================================================================================

If the training program is largely different from test program, you can define two totally different :code:`fluid.Program` , and perform training and test individually. In PaddlePaddle Fluid, all parameters are named. If two different operations or even two different networks use parameters with the same name, the value and memory space of these parameters are shared.

Fluid adopts :code:`fluid.unique_name` package to randomly initialize the names of unnamed parameters. :code:`fluid.unique_name.guard` can keep the initialized names consistent across multiple times of calling some function.

For example:

.. code-block:: python

   import paddle.fluid as fluid

   def network(is_test):
       file_obj = fluid.layers.open_files(filenames=["test.recordio"] if is_test else ["train.recordio"], ...)
       img, label = fluid.layers.read_file(file_obj)
       hidden = fluid.layers.fc(input=img, size=100, act="relu")
       hidden = fluid.layers.batch_norm(input=hidden, is_test=is_test)
       ...
       return loss

   with fluid.unique_name.guard():
       train_loss = network(is_test=False)
       sgd = fluid.optimizer.SGD(0.001)
       sgd.minimize(train_loss)

   test_program = fluid.Program()
   with fluid.unique_name.guard():
       with fluid.program_gurad(test_program, fluid.Program()):
           test_loss = network(is_test=True)

   # fluid.default_main_program() is the train program
   # fluid.test_program is the test program

Perform test :code:`fluid.Program`
###################################

Run test :code:`fluid.Program` with :code:`Executor` 
=======================================================

You can run test :code:`fluid.Program` with :code:`Executor.run(program=...)` .

For example:

.. code-block:: python

   exe = fluid.Executor(fluid.CPUPlace())
   test_acc = exe.run(program=test_program, feed=test_data_batch, fetch_list=[acc])
   print 'Test accuracy is ', test_acc

Run test :code:`fluid.Program` with :code:`ParallelExecutor` 
=====================================================================

You can use :code:`ParallelExecutor` for training and :code:`fluid.Program` for test to create a new test :code:`ParallelExecutor` ; then use test :code:`ParallelExecutor.run` to run test process.

For example:

.. code-block:: python

   train_exec = fluid.ParallelExecutor(use_cuda=True, loss_name=loss.name)

   test_exec = fluid.ParallelExecutor(use_cuda=True, share_vars_from=train_exec,
                                      main_program=test_program)
   test_acc = test_exec.run(fetch_list=[acc], ...)

