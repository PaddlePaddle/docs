## [ 输入参数用法不一致 ] torch.set_rng_state

### [torch.set_rng_state](https://pytorch.org/docs/stable/generated/torch.set_rng_state.html#torch.set_rng_state)

```python
torch.set_rng_state(new_state)
```

### [paddle.set_rng_state]()

```python
paddle.set_rng_state(state_list)
```

其中 PyTorch 与 Paddle 的输入参数类型不一致

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> new_state </font>         | <font color='red'> state_list </font>            | 表示需要设置的新状态，PyTorch 输入类型为 torch.ByteTensor, Paddle 为 list[GeneratorState]                               |



### 转写示例

#### new_state: 指定输入
```python
# PyTorch 写法
torch.set_rng_state(x)

# Paddle 写法
paddle.set_rng_state(x)
```
