## [参数不一致]torch.cuda.get_rng_state_all

### [torch.cuda.get_rng_state_all](https://pytorch.org/docs/stable/generated/torch.cuda.get_rng_state_all.html#torch.cuda.get_rng_state_all)

```python
torch.cuda.get_rng_state_all()
```

### [paddle.get_rng_state]()

```python
paddle.get_rng_state(device='gpu')
```

paddle 参数更多，并且 torch 与 paddle 的返回参数类型不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                               |
| ------- | ------------ | -------------------------------------------------------------------------------------------------- |
| -       | device       | 返回随机数生成器状态的设备，PyTorch 无此参数，Paddle 需设置为'gpu' 。           |
| 返回值  | 返回值       | 返回参数类型不一致, PyTorch 返回 torch.ByteTensor，Paddle 返回 GeneratorState 对象。 |

### 转写示例

#### 返回参数类型不同

```python
# PyTorch 写法，返回 torch.ByteTensor
x = torch.cuda.get_rng_state_all()

# Paddle 写法，返回 GeneratorState 对象
x = paddle.get_rng_state(device='gpu')
```
