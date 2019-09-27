.. _cn_api_fluid_clip_GradientClipByNorm:

GradientClipByNorm
-------------------------------

.. py:class:: paddle.fluid.clip.GradientClipByNorm(clip_norm)

将输入多维Tensor :math:`X` 转换为L2范数不超过给定的二范数最大值（ ``clip_norm`` ）的多维Tensor。（多维Tensor不是从该类传入， 而是通过 ``fluid.program_guard`` 的 ``main_program`` 参数传入）。

该类限制了输入多维Tensor :math:`X` 的L2范数不会超过 ``clip_norm`` 。

.. math::

  Out=\left\{
  \begin{aligned}
   X & & if (norm(X) <= clip\_norm)\\
  \frac{clip\_norm∗X}{norm(X)}  & & if (norm(X) > clip\_norm) \\
  \end{aligned}
  \right.


其中 :math:`norm（X）` 代表 :math:`X` 的L2范数

.. math::
  \\norm(X) = (\sum_{i=1}^{n}|x_i|^2)^{\frac{1}{2}}\\

参数:
 - **clip_norm** (float) - 二范数最大值


**代码示例**

.. code-block:: python

  import paddle.fluid as fluid
  import paddle.fluid.core as core
  import paddle
  place = core.CPUPlace()
  prog = fluid.framework.Program()
  startup_program = fluid.framework.Program()
  with fluid.program_guard(
              main_program=prog, startup_program=startup_program):
      image = fluid.layers.data(name='x', shape=[784], dtype='float32')
      label = fluid.layers.data(name='y', shape=[1], dtype='int64')
      hidden1 = fluid.layers.fc(input=image, size=128, act='relu')
      hidden2 = fluid.layers.fc(input=hidden1, size=64, act='relu')
      predict = fluid.layers.fc(input=hidden2, size=10, act='softmax')
      cost = fluid.layers.cross_entropy(input=predict, label=label)
      avg_cost = fluid.layers.mean(cost)
  prog_clip = prog.clone()
  avg_cost_clip = prog_clip.block(0).var(avg_cost.name)
  p_g = fluid.backward.append_backward(loss=avg_cost)
  p_g_clip = fluid.backward.append_backward(loss=avg_cost_clip)
  with fluid.program_guard(main_program=prog_clip, startup_program=startup_program):
      fluid.clip.set_gradient_clip(
          fluid.clip.GradientClipByNorm(clip_norm=2.0))
      p_g_clip = fluid.clip.append_gradient_clip_ops(p_g_clip)
  grad_list = [elem[1] for elem in p_g]
  grad_clip_list = [elem[1] for elem in p_g_clip]
  train_reader = paddle.batch(
      paddle.reader.shuffle(
          paddle.dataset.mnist.train(), buf_size=8192),
      batch_size=128)

  exe = fluid.Executor(place)
  feeder = fluid.DataFeeder(feed_list=[image, label], place=place)
  exe.run(startup_program)

  count = 0
  for data in train_reader():
      count += 1
      print("count:%s" % count)
      if count > 5:
         break
      out = exe.run(prog, feed=feeder.feed(data), fetch_list=grad_list)
      out_clip = exe.run(prog_clip,
                         feed=feeder.feed(data),
                         fetch_list=grad_clip_list)
