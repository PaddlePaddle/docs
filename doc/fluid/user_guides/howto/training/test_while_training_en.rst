.. _user_guide_test_while_training:

##############################
Evaluate model during train
##############################

Model testing evaluation is different from :code:`fluid.Program` .In testing evalution:

1. Don't broadcast backward and optimize and update parameters while evaluating test.
2. Operations in evaluating test can be different.

   * Take operation BatchNorm for example,algorithms are different in train and test.

   * Test model and train model can be totally different.

Generate test :code:`fluid.Program`
#################################

Generate test :code:`fluid.Program` by cloning train :code:`fluid.Program` 
============================================================================

:code:`Program.clone()` can copy new :code:`fluid.Program` . You can copy Program with operations applied for test by setting :code:`Program.clone(for_test=True)` . Simple usage is as follows:

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

Before the usage of :code:`Optimizer` ,please copy :code:`fluid.default_main_program()` as a :code:`test_program` .Then you can use test data to run :code:`test_program` so that you can run testing program without influencing training result.

Configure train :code:`fluid.Program` and test :code:`fluid.Program` respectively
==================================================================================

If train program is largely different from test program, you can define two different :code:`fluid.Program`, performing train and test respectively.In PaddlePaddle Fluid, all parameters are entitled with name. If two different operations and even two different networks use parameters with same name, the value and memory space of these parameters are shared.

You can use :code:`fluid.unique_name` package to randomly initialize your undefined names of parameters in PaddlePaddle Fluid. When you call the parameter initialization of certain function for several times, :code:`fluid.unique_name.guard` can keep names of initialization consistant.

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
#################################

Run test :code:`fluid.Program` with :code:`Executor` 
=======================================================

You can run test :code:`fluid.Program` with :code:`Executor.run(program=...)` .

For example:

.. code-block:: python

   exe = fluid.Executor(fluid.CPUPlace())
   test_acc = exe.run(program=test_program, feed=test_data_batch, fetch_list=[acc])
   print 'Test accuracy is ', test_acc

You can run test :code:`fluid.Program` with :code:`ParallelExecutor` 
=====================================================================

You can use :code:`ParallelExecutor` for train and :code:`fluid.Program` for test to create a new test :code:`ParallelExecutor` ;then use test :code:`ParallelExecutor.run` to run test.

For example:

.. code-block:: python

   train_exec = fluid.ParallelExecutor(use_cuda=True, loss_name=loss.name)

   test_exec = fluid.ParallelExecutor(use_cuda=True, share_vars_from=train_exec,
                                      main_program=test_program)
   test_acc = test_exec.run(fetch_list=[acc], ...)

