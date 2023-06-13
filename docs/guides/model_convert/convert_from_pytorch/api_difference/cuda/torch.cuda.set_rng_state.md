## [torch 参数更多]torch.cuda.set_rng_state

### [torch.cuda.set_rng_state](https://pytorch.org/docs/1.13/generated/torch.cuda.set_rng_state.html#torch.cuda.set_rng_state)

```python
torch.cuda.set_rng_state(new_state, device='cuda')
```

### [paddle.set_cuda_rng_state](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/set_cuda_rng_state_cn.html)

```python
paddle.set_cuda_rng_state(state_list)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch   | PaddlePaddle | 备注                                                                                             |
| --------- | ------------ | ------------------------------------------------------------------------------------------------ |
| new_state | -            | 类型为 torch.ByteTensor，表示设备需要的状态，Paddle 无此参数，需要进行转写。                     |
| device    | -            | 指定随机数生成器状态的设备，Paddle 无此参数，暂无转写方式。                                      |
| -         | state_list   | 类型为 GeneratorState 列表，需要设置的随机数生成器状态信息列表，PyTorch 无此参数，需要进行转写。 |

### 转写示例

#### 参数类型不同

```python
# PyTorch 写法
x = torch.cuda.get_rng_state(device='cuda')
torch.cuda.set_rng_state(x, device='cuda')

# Paddle 写法
x = paddle.get_cuda_rng_state()
paddle.set_cuda_rng_state(x)
```
