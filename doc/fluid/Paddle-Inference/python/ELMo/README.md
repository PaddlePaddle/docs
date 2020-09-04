## 基于ELMo的LAC分词预测样例

### 一：准备环境

请您在环境中安装1.7或以上版本的Paddle，具体的安装方式请参照[飞桨官方页面](https://www.paddlepaddle.org.cn/)的指示方式。

### 二：下载模型以及测试数据


1) **获取预测模型**

点击[链接](https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/elmo/elmo.tgz)下载模型，如果你想获取更多的**模型训练信息**，请访问[链接](https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleNLP/pretrain_language_models/ELMo)。解压后存储到该工程的根目录。

2) **获取相关数据**

点击[链接](https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/elmo/elmo_data.tgz)下载相关数据，解压后存储到该工程的根目录。

### 三：运行预测

`reader.py` 包含了数据读取等功能。
`infer.py` 包含了创建predictor，读取输入，预测，获取输出的等功能。

运行：
```
python infer.py
```

分词结果为：

```
1 sample's result: <UNK>/n 电脑/vn 对/v-I 胎儿/v-I 影响/vn-B 大/v-I 吗/a
2 sample's result: 这个/r 跟/p 我们/ns 一直/p 传承/n 《/p 易经/n 》/n 的/u 精神/n 是/v-I 分/v 不/d 开/v 的/u
3 sample's result: 他们/p 不/r 但/ad 上/v-I 名医/v-I 门诊/n ,/w 还/n 兼/ns-I 作/ns-I 门诊/n 医生/v-I 的/n 顾问/v-I 团/nt
4 sample's result: 负责/n 外商/v-I 投资/v-I 企业/n 和/v-I 外国/v-I 企业/n 的/u 税务/nr-I 登记/v-I ,/w 纳税/n 申报/vn 和/n 税收/vn 资料/n 的/u 管理/n ,/w 全面/c 掌握/n 税收/vn 信息/n
5 sample's result: 采用/ns-I 弹性/ns-I 密封/ns-I 结构/n ,/w 实现/n 零/v-B 间隙/v-I
6 sample's result: 要/r 做/n 好/p 这/n 三/p 件/vn 事/n ,/w 支行/q 从/q 风险/n 管理/p 到/a 市场/q 营销/n 策划/c 都/p 必须/vn 专业/n 到位/vn
7 sample's result: 那么/nz-B ,/r 请/v-I 你/v-I 一定/nz-B 要/d-I 幸福/ad ./v-I
8 sample's result: 叉车/ns-I 在/ns-I 企业/n 的/u 物流/n 系统/vn 中/ns-I 扮演/ns-I 着/v-I 非常/q 重要/n 的/u 角色/n ,/w 是/u 物料/vn 搬运/ns-I 设备/n 中/vn 的/u 主力/ns-I 军/v-I
9 sample's result: 我/r 真/t 的/u 能够/vn 有/ns-I 机会/ns-I 拍摄/v-I 这部/vn 电视/ns-I 剧/v-I 么/vn
10 sample's result: 这种/r 情况/n 应该/v-I 是/v-I 没/n 有/p 危害/n 的/u
```

### 相关链接
- [Paddle Inference使用Quick Start！]()
- [Paddle Inference Python Api使用]()
