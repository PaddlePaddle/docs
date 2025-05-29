## [torch 参数更多]torch.jit.script

### [torch.jit.script](https://pytorch.org/docs/stable/generated/torch.jit.script.html#torch-jit-script)

```python
torch.jit.script(obj, optimize=None, _frames_up=0, _rcb=None, example_inputs=None)
```

### [paddle.jit.to_static](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/jit/to_static_cn.html#paddle.jit.to_static)

```python
paddle.jit.to_static(function, input_spec=None, build_strategy=None, backend=None, **kwargs)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                                |
| ------------- | ------------ | ------------------------------------------------------------------- |
| obj             | function         | 待转换的函数，仅参数名不一致。                       |
| example_inputs  | input_spec           | 用于指定被装饰函数中输入 Tensor 信息，仅参数名不一致。    |
| optimize | -            | 控制是否对转换后的模块进行优化，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。    |
| _rcb             | -      | 控制编译行为，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| _frames_up             | -      | 控制编译时的堆栈帧深度，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| -    | build_strategy      | 对转换后的计算图进行优化方法，PyTorch 无此参数，Paddle 保持默认即可。 |
| -             | backend      | 指定后端编译器，PyTorch 无此参数，Paddle 保持默认即可。 |
