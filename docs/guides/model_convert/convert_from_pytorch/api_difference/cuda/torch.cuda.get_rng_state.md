## [ torch 参数更多 ]torch.cuda.get_rng_state
### [torch.cuda.get_rng_state](https://pytorch.org/docs/stable/generated/torch.cuda.get_rng_state.html#torch-cuda-get-rng-state)

```python
torch.cuda.get_rng_state(device='cuda')
```

### [paddle.get_cuda_rng_state](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/get_cuda_rng_state_cn.html#get-cuda-rng-state)

```python
paddle.get_cuda_rng_state()
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| device | - |  返回 RNG 状态的设备，Paddle 无此参数，暂无转写方式。 |
