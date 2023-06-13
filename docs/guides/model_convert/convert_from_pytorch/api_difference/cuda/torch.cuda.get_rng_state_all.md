## [参数不一致]torch.cuda.get_rng_state_all

### [torch.cuda.get_rng_state_all](https://pytorch.org/docs/1.13/generated/torch.cuda.get_rng_state_all.html#torch.cuda.get_rng_state_all)

```python
torch.cuda.get_rng_state_all()
```

### [paddle.get_cuda_rng_state](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/get_cuda_rng_state_cn.html)

```python
paddle.get_cuda_rng_state()
```

其中 Pytorch 与 Paddle 的返回参数类型不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle   | 备注                                                                                           |
| ------- | -------------- | ---------------------------------------------------------------------------------------------- |
| Tensor  | GeneratorState | 返回参数类型不一致, Pytorch 返回 torch.ByteTensor，Paddle 返回 GeneratorState 对象，需要进行转写。 |

### 转写示例

#### 返回参数类型不同

```python
# PyTorch 写法
x = torch.cuda.get_rng_state_all()

# Paddle 写法
x = paddle.get_cuda_rng_state()
```
