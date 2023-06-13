## [torch 参数更多]torch.cuda.get_rng_state

### [torch.cuda.get_rng_state](https://pytorch.org/docs/1.13/generated/torch.cuda.get_rng_state.html#torch.cuda.get_rng_state)

```python
torch.cuda.get_rng_state(device='cuda')
```

### [paddle.get_cuda_rng_state](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/get_cuda_rng_state_cn.html)

```python
paddle.get_cuda_rng_state()
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle   | 备注                                                                                               |
| ------- | -------------- | -------------------------------------------------------------------------------------------------- |
| device  | -              | 返回随机数生成器状态的设备，Paddle 无此参数，暂无转写方式。                                        |
| Tensor  | GeneratorState | 返回参数类型不一致, Pytorch 返回 torch.ByteTensor，Paddle 返回 GeneratorState 列表，需要进行转写。 |

### 转写示例

#### 返回参数类型不同

```python
# PyTorch 写法
x = torch.cuda.get_rng_state(device='cuda')

# Paddle 写法
x = paddle.get_cuda_rng_state()
```
