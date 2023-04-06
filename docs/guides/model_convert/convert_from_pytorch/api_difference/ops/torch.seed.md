## [ 组合替代实现 ]torch.seed

### [torch.seed](https://pytorch.org/docs/master/generated/torch.seed.html#torch.seed)
```python
torch.seed()
```

###  功能介绍
将生成随机数的种子设置为非确定性随机数。返回一个用于播种 RNG 的 64 位数字。

PaddlePaddle 目前无对应 API，可使用如下代码组合实现该 API。

```python
import paddle

paddle.get_cuda_rng_state()[0].current_seed()
```
