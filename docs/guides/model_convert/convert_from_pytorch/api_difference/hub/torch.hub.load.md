## [torch 参数更多] torch.hub.load

### [torch.hub.load](https://pytorch.org/docs/1.13/hub.html?highlight=hub+load#torch.hub.load)

```python
torch.hub.load(repo_or_dir,
                model,
                *args,
                source='github',
                trust_repo=None,
                force_reload=False,
                verbose=True,
                skip_validation=False,
                **kwargs)
```

### [paddle.hub.load](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/hub/load_cn.html)

```python
paddle.hub.load(repo_dir,
                model,
                source='github',
                force_reload=False,
                **kwargs)
```

其中，Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
|repo_or_dir    |repo_dir      |repo 地址，支持 git 地址形式和 local 地址，参数名不同。|
|model          | model        |模型的名字。|
|source          |source        |指定 repo 托管的位置，支持 github、gitee 和 local，默认值：github。|
| trust_repo    | -            |在 v1.14 中被移除。Paddle 无此参数，可直接删除。|
| force_reload   | force_reload |指定是否强制拉取，默认值: False。参数名相同。         |
|verbose         | -          |如果设置为 False，将不会显示关于命中本地缓存的消息，默认为 True，Paddle 无此参数，直接删除即可。|
| skip_validation| -       |检查由 github 参数指定的分支或提交是否属于存储库所有者,默认为 False，Paddle 无此参数，直接删除即可。|
|-              |source        |指定 repo 托管的位置，支持 github、gitee 和 local，默认值：github。Pytorch 无此参数， Paddle 保持默认即可。|
