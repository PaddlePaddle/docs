## torch.as_tensor
### [torch.as_tensor](https://pytorch.org/docs/stable/generated/torch.as_tensor.html#torch.as_tensor)

```python
torch.as_tensor(data,
                dtype=None,
                device=None)
```

### [paddle.to_tensor](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/to_tensor_cn.html#to-tensor)

```python
paddle.to_tensor(data,
                dtype=None,
                place=None,
                stop_gradient=True)
```

两者功能不完全一致，Paddle 相比 Pytorch 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| data          | data         | 表示输入的 Tensor 。                                     |
| dtype           | dtype            | 表示 Tensor 的数据类型。               |
| device           | place            | 表示 Tensor 的存放位置。               |
| -           | stop_gradient            | 表示是否阻断梯度传导， PyTorch 无此参数， Paddle 保持默认即可。             |


### 转写示例
#### device：指定 Tensor 存放位置
```python
# Pytorch 写法
torch.as_tensor([1, 2, 3], device=torch.device('cuda:0'))

# Paddle 写法
paddle.to_tensor([1, 2, 3], place=paddle.CUDAPlace(0))
```
