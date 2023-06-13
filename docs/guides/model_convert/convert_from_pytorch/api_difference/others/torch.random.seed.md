## [ 组合替代实现 ]torch.random.seed

### [torch.random.seed](https://pytorch.org/docs/stable/random.html#torch.random.seed)
```python
torch.random.seed()
```

将生成随机数的种子设置为非确定性随机数。返回一个用于播种 RNG 的 64 位数字。

PaddlePaddle 目前无对应 API，可使用如下代码组合实现该 API。

###  转写示例
```python
# Pytorch 写法
torch.random.seed()

# Paddle 写法
paddle.get_cuda_rng_state()[0].current_seed()
```
