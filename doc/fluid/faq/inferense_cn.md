# 预测引擎

## Q：VGG模型，训练时候使用`fluid.io.save_inference_model`保存模型，预测的时候使用`fluid.io.load_inference_model`加载模型文件。保存的是我自己训练的 VGG 模型。保存没问题，加载的时候报错`paddle.fluid.core.EnforceNotMet: Cannot read more from file` ？
+ 版本、环境信息
1）PaddlePaddle版本：Paddle Fluid 1.2.0。
2）系统环境：Linux，Python2.7。
+ 错误信息
```
Traceback (most recent call last):
  File "classify-infer.py", line 43, in <module>
    executor=exe)
  File "/home/work/xiangyubo/paddle_release_home/python/lib/python2.7/site-packages/paddle/fluid/io.py", line 784, in load_inference_model
    load_persistables(executor, dirname, program, params_filename)
  File "/home/work/xiangyubo/paddle_release_home/python/lib/python2.7/site-packages/paddle/fluid/io.py", line 529, in load_persistables
    filename=filename)
  File "/home/work/xiangyubo/paddle_release_home/python/lib/python2.7/site-packages/paddle/fluid/io.py", line 395, in load_vars
    filename=filename)
  File "/home/work/xiangyubo/paddle_release_home/python/lib/python2.7/site-packages/paddle/fluid/io.py", line 436, in load_vars
    executor.run(load_prog)
  File "/home/work/xiangyubo/paddle_release_home/python/lib/python2.7/site-packages/paddle/fluid/executor.py", line 472, in run
    self.executor.run(program.desc, scope, 0, True, True)
paddle.fluid.core.EnforceNotMet: Cannot read more from file ./classify-model/vgg-classify-params at [/paddle/paddle/fluid/operators/load_combine_op.cc:58]
PaddlePaddle Call Stacks: 
0       0x7f4bdb4b2226p paddle::platform::EnforceNotMet::EnforceNotMet(std::__exception_ptr::exception_ptr, char const*, int) + 486
1       0x7f4bdbb97ab4p paddle::operators::LoadCombineOp::RunImpl(paddle::framework::Scope const&, boost::variant<paddle::platform::CUDAPlace, paddle::platform::CPUPlace, paddle::platform::CUDAPinnedPlace, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_> const&) const + 2004
2       0x7f4bdcda56afp paddle::framework::OperatorBase::Run(paddle::framework::Scope const&, boost::variant<paddle::platform::CUDAPlace, paddle::platform::CPUPlace, paddle::platform::CUDAPinnedPlace, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_> const&) + 463
3       0x7f4bdb588ef3p paddle::framework::Executor::RunPreparedContext(paddle::framework::ExecutorPrepareContext*, paddle::framework::Scope*, bool, bool, bool) + 227
4       0x7f4bdb589920p paddle::framework::Executor::Run(paddle::framework::ProgramDesc const&, paddle::framework::Scope*, int, bool, bool) + 128
5       0x7f4bdb49e0dbp
6       0x7f4bdb4da18ep
7       0x7f4c3efcabb8p PyEval_EvalFrameEx + 25016
8       0x7f4c3efce0bdp PyEval_EvalCodeEx + 2061
9       0x7f4c3efcb345p PyEval_EvalFrameEx + 26949
10      0x7f4c3efce0bdp PyEval_EvalCodeEx + 2061
11      0x7f4c3efcb345p PyEval_EvalFrameEx + 26949
12      0x7f4c3efce0bdp PyEval_EvalCodeEx + 2061
13      0x7f4c3efcb345p PyEval_EvalFrameEx + 26949
14      0x7f4c3efce0bdp PyEval_EvalCodeEx + 2061
15      0x7f4c3efcb345p PyEval_EvalFrameEx + 26949
16      0x7f4c3efce0bdp PyEval_EvalCodeEx + 2061
17      0x7f4c3efcb345p PyEval_EvalFrameEx + 26949
18      0x7f4c3efce0bdp PyEval_EvalCodeEx + 2061
19      0x7f4c3efce1f2p PyEval_EvalCode + 50
20      0x7f4c3eff6f42p PyRun_FileExFlags + 146
21      0x7f4c3eff82d9p PyRun_SimpleFileExFlags + 217
22      0x7f4c3f00e00dp Py_Main + 3149
23      0x7f4c3e20bbd5p __libc_start_main + 245
24            0x4007a1p
```
+ 问题解答
错误提示可能的原因如下，请检查。
1、 模型文件有损坏或缺失。
2、 模型参数和模型结构不匹配。

## Q：infer时，当先后加载检测和分类两个网络时，分类网络的参数为什么未被load进去？
+ 问题解答
尝试两个模型在不同的scope里面infer，使用`with fluid.scope_guard(new_scope)`，另外定义模型前加上`with fluid.unique_name.guard()`解决。

## Q：c++调用paddlepaddle多线程预测出core？
+ 问题解答
Paddle predict 库里没有多线程的实现，当上游服务并发时，需要用户起多个预测服务，[参考示例](http://paddlepaddle.org/documentation/docs/zh/1.3/advanced_usage/deploy/inference/native_infer.html)。


## Q：两个模型都load之后，用第一个模型的时候会报错？
+ 问题解答
由于用`load_inference_model`的时候会修改一些用户不可见的环境变量，所以执行后一个`load_inference_model`的时候会把前一个模型的环境变量覆盖，导致前一个模型不能用，或者说再用的时候就需要再加载一次。此时需要用如下代码保护一下，[参考详情](https://github.com/PaddlePaddle/Paddle/issues/16661)。
```
xxx_scope = fluid.core.Scope()
with fluid.scope_guard(xxx_scope):
    [...] = fluid.load_inference_model(...)
```