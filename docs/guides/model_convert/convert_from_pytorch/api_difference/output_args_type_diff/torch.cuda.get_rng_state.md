## [ 返回参数类型不一致 ]torch.cuda.get_rng_state
### [torch.cuda.get_rng_state](https://pytorch.org/docs/stable/generated/torch.cuda.get_rng_state.html#torch-cuda-get-rng-state)

```python
torch.cuda.get_rng_state(device='cuda')
```

### [paddle.get_cuda_rng_state](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/get_cuda_rng_state_cn.html#get-cuda-rng-state)

```python
paddle.get_cuda_rng_state()
```

torch 参数更多，并且 torch 与 paddle 的返回参数类型不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| device | - |  返回 RNG 状态的设备，Paddle 无此参数，需要转写。 |
| 返回值  | 返回值       | 返回参数类型不一致, PyTorch 返回 torch.ByteTensor，Paddle 返回 GeneratorState 对象列表。 |

### 转写示例

#### 返回参数类型不同

```python
# PyTorch 写法，返回 torch.ByteTensor
x = torch.cuda.get_rng_state(device='cuda：0')

# Paddle 写法，返回 GeneratorState 对象
x = paddle.get_cuda_rng_state()[0]
```

```python
# PyTorch 写法，返回 torch.ByteTensor
x = torch.cuda.get_rng_state()

# Paddle 写法，返回 GeneratorState 对象
x = paddle.get_cuda_rng_state()[paddle.framework._current_expected_place().get_device_id()]
```
