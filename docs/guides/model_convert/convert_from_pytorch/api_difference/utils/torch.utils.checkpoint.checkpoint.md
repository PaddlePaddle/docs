## [ torch 参数更多 ]torch.utils.checkpoint.checkpoint
### [torch.utils.checkpoint.checkpoint](https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint)

```python
torch.utils.checkpoint.checkpoint(function, *args, use_reentrant=True, **kwargs)
```

### [paddle.distributed.fleet.utils.recompute](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/utils/cpp_extension/CppExtension_cn.html)

```python
paddle.distributed.fleet.utils.recompute(function, *args, **kwargs)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| function          | function            | 模型前向传播的部分连续的层函数组成的序列。  |
| preserve_rng_state         | preserve_rng_state         | 是否保存前向的 rng。   |
| use_reentrant         | use_reentrant         |  recompute 的实现方式。   |
| determinism_check         | -         | 控制是否在反向传播时检查操作的确定性, Paddle 无此参数，暂无转写方式。   |
|*args         | *args          |   function 的输入。 |
| **kwargs      | **kwargs        |   用于指定 Extension 的其他参数，支持的参数与 setuptools.Extension 一致。 |
