## [ 组合替代实现 ]torch.random.initial_seed

### [torch.initial_seed](https://pytorch.org/docs/stable/random.html#torch.random.initial_seed)

```python
torch.random.initial_seed()
```

获取当前随机数种子。返回一个用于播种 RNG 的 64 位数字。

PaddlePaddle 目前无对应 API，可使用如下代码组合实现该 API。

### 转写示例

```python
# Pytorch 写法
torch.random.initial_seed()

# Paddle 写法
paddle.get_rng_state()[0].current_seed()
```
