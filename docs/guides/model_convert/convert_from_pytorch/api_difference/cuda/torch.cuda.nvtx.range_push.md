## [仅参数名不一致]torch.cuda.nvtx.range_push

### [torch.cuda.nvtx.range_push](https://pytorch.org/docs/1.13/generated/torch.cuda.nvtx.range_push.html#torch.cuda.nvtx.range_push)

```python
torch.cuda.nvtx.range_push(msg)
```

### [paddle.fluid.core.nvprof_nvtx_push](https://github.com/PaddlePaddle/Paddle/blob/f00a06d817b97bde23e013c2fb0cd1a6c9c1076b/paddle/fluid/pybind/pybind.cc#L2261)

```python
paddle.fluid.core.nvprof_nvtx_push(name)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                       |
| ------- | ------------ | ------------------------------------------ |
| msg     | name         | 关联 range 的 ASCII 消息，仅参数名不一致。 |
