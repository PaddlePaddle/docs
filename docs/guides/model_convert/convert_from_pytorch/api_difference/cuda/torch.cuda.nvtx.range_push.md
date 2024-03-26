## [ 仅参数名不一致 ]torch.cuda.nvtx.range_push

### [torch.cuda.nvtx.range_push](https://pytorch.org/docs/stable/generated/torch.cuda.nvtx.range_push.html?highlight=range_push#torch.cuda.nvtx.range_push)

```python
torch.cuda.nvtx.range_push(msg)
```

### [paddle.framework.core.nvprof_nvtx_push](https://github.com/PaddlePaddle/Paddle/blob/645dfb4040a15712cea9ccfed4dcb0655aeeb0ea/paddle/fluid/pybind/pybind.cc#L2465)

```python
paddle.framework.core.nvprof_nvtx_push(name)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                       |
| ------- | ------------ | ------------------------------------------ |
| msg     | name         | 关联 range 的 ASCII 消息，仅参数名不一致。 |
