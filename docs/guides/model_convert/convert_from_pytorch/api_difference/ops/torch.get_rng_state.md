## [参数不一致] torch.get_rng_state

### [torch.get_rng_state](https://pytorch.org/docs/stable/generated/torch.get_rng_state.html#torch.get_rng_state)

```python
torch.get_rng_state()
```

### [paddle.get_rng_state]()

```python
paddle.get_rng_state()
```

其中 Pytorch 与 Paddle 的返回参数类型不一致

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> Tensor </font>         | <font color='red'> GeneratorState </font>            | 返回类型不一致                                     |



### 转写示例
#### 返回参数类型不同
```python
# Pytorch 写法
x = torch.get_rng_state()

# Paddle 写法
x = paddle.get_rng_state()
```
