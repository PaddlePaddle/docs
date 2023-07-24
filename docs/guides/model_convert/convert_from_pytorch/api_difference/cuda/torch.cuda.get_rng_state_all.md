## [仅 paddle 参数更多]torch.cuda.get_rng_state_all

### [torch.cuda.get_rng_state_all](https://pytorch.org/docs/stable/generated/torch.cuda.get_rng_state_all.html#torch.cuda.get_rng_state_all)

```python
torch.cuda.get_rng_state_all()
```

### [paddle.get_rng_state]()

```python
paddle.get_rng_state(device=None)
```

其中 paddle 参数更多，Pytorch 与 Paddle 的返回参数类型不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                               |
| ------- | ------------ | -------------------------------------------------------------------------------------------------- |
| -       | device       | 返回随机数生成器状态的设备，Paddle 取值 gpu。                                              |
| 返回值  | 返回值       | 返回参数类型不一致, Pytorch 返回 torch.ByteTensor，Paddle 返回 GeneratorState 对象，暂无转写方式。 |

### 转写示例

#### 返回参数类型不同

```python
# PyTorch 写法
x = torch.cuda.get_rng_state_all()

# Paddle 写法
x = paddle.get_rng_state(device='gpu')
```
