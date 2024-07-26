## [ 返回参数类型不一致 ] torch.random.get_rng_state

### [torch.random.get_rng_state](https://pytorch.org/docs/stable/random.html#torch.random.get_rng_state)

```python
torch.random.get_rng_state()
```

### [paddle.get_rng_state]()

```python
paddle.get_rng_state()
```

其中 PyTorch 与 Paddle 的返回参数类型不一致。

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> Tensor </font>         | <font color='red'> GeneratorState </font>            | 返回类型不一致, PyTorch 返回 torch.ByteTensor，Paddle 返回 GeneratorState 对象。需要转写。                                     |



### 转写示例
#### 返回参数类型不同
```python
# PyTorch 写法
x = torch.random.get_rng_state()

# Paddle 写法
x = paddle.get_rng_state()
```
