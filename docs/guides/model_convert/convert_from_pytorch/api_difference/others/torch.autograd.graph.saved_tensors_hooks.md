## [参数完全一致]torch.autograd.graph.saved_tensors_hooks

### [torch.autograd.graph.saved_tensors_hooks](https://pytorch.org/docs/stable/autograd.html?highlight=saved_tensors_hooks#torch.autograd.graph.saved_tensors_hooks)

```python
torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook)
```

### [paddle.autograd.saved_tensors_hooks](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/autograd/saved_tensors_hooks_cn.html)

```python
paddle.autograd.saved_tensors_hooks(pack_hook, unpack_hook)
```

两者功能一致，参数完全一致，具体如下：

### 参数映射

| PyTorch     | PaddlePaddle | 备注                                                                                      |
| ----------- | ------------ | ----------------------------------------------------------------------------------------- |
| pack_hook   | pack_hook    | 当某个算子的前向执行时，存在 Tensor 需要保留给反向计算梯度使用时， pack_hook 将会被调用。 |
| unpack_hook | unpack_hook  | 当反向执行，需要用到前向保留的 Tensor 时， unpack_hook 会被调用。                         |
