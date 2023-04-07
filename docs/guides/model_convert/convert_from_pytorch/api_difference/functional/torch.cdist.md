## [ 组合替代实现 ] torch.cdist

### [torch.dist](https://pytorch.org/docs/stable/generated/torch.cdist.html?highlight=cdist#torch-cdist)

```python
torch.Tensor.cdist(x1, x2, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
```

Pytorch 中 torch.dist 计算两组行向量集合中每一对之间的 p 范数, PaddlePaddle 目前无对应 API，可通过 paddle.dist 单独计算两个向量 p 范数来实现，使用如下代码组合实现该 API 转写。

### 转写示例
```python
# torch 写法
a = torch.tensor([[0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059]])
b = torch.tensor([[-2.1763, -0.4713], [-0.6986,  1.3702]])
torch.cdist(a, b, p=2)

# paddle 写法
a = paddle.to_tensor(data=[[0.9041, 0.0196], [-0.3108, -2.4423], [-0.4821,
    1.059]])
b = paddle.to_tensor(data=[[-2.1763, -0.4713], [-0.6986, 1.3702]])
def cdist(x1, x2, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
    s1, s2 = x1.shape, x2.shape
    for i in range(s1[0]):
        for j in range(s1[1]):
            for k in range(s2[0]):
                x[i, j, k] = paddle.dist(x1[i: j], x2[k], p=2)
    return x
```
