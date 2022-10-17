# [Call for Contribution] Tutorials for PaddlePaddle 2.4

## 基于飞桨2.4的应用案例：教程建设 

### 第一期任务 🎺

## 1. 建设目标
飞桨官网的应用实践栏目是许多人学习及使用飞桨的重要的、基础的材料。
<br />   <br />
本期活动的目标是：

- 1）基于 **飞桨框架 2.4** 版本，建设 **5** ~ **10** 个 **tutorial**，呈现在官网的 [应用实践](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.4rc/practices/index_cn.html) 里。
<br />   <br />
- 2）新增的应用实践案例，需完美适配于 **PaddlePaddle 2.0** 至 **PaddlePaddle 2.4** 版本。
<br />   <br />
- 3）😊 呼吁社区开发者，参与共建飞桨框架的应用案例教程，了解、学习如何使用 **2.4** 开发相关任务。

## 2. 教程清单
我们从现有及待补充的内容进行了初步评估，现梳理如下：
### 【应用场景】
* <big>**CV (image)**</big>：       使用深度学习网络处理图片类数据。
* <big>**CV (Point)**</big>：        使用深度学习网络处理点云类数据。
* <big>**Video （视频）**</big>：使用经典的深度学习方法处理视频类数据。
* <big>**可解释性**</big> ：          神经网络性能的可解释性。
* <big>**时间序列**</big> ：          使用经典的深度学习方法处理时间序列数据。
* <big>**图数据**</big> ：             使用经典的深度学习方法处理图数据。
* <big>**NLP**</big> ：                 自然语言处理任务。
* <big>**Audio （语音）**</big> ：使用深度学习方法处理语音文件。
* <big>**推荐**</big>：                 使用深度学习方法完成推荐任务。
* <big>**强化学习**</big>:             使用深度学习中的强化学习。 
* <big>**GAN**</big> ：                使用深度学习生成对抗网络。
* <big>**多模态**</big> ：             使用深度学习方法处理多种模态的数据。（图片、音频、文本）

### 【参与者-选题方式】
- <u>**方式一**</u>：**列表选题**，见招募列表（提供了选题方向、题目、优秀的对标项目、文章和代码，以供学习）。
<br />   <br />
- <u>**方式二**</u>：**自选题目**，对于非参考列表内的题目，可自主命题。

## 招募列表（10月20日公开）

| 序号  | 领域  | 题目 | 新增点  | 参考范例  | 认领人  |
|:----| :----- |  :----- |  :----- |  :-----| :-----|
| 01  | CV (image)| cifar100数据集上基于Vision Transformer 实现图片分类|Vision Transformer (ViT)|https://keras.io/examples/vision/image_classification_with_vision_transformer/| ---- | 
| 02  |CV (image)|Crowd Instance-level Human Parsing Dataset 数据集上使用DeepLabV3+实现多分类语义分割 |DeepLabV3|https://keras.io/examples/vision/deeplabv3_plus/| --   |
| 03  |CV (image)|Caltech 101 数据集上基于Vision Transformer 实现飞机目标识别|Vision Transformer (ViT)|https://keras.io/examples/vision/object_detection_using_vision_transformer/| ---- |
| 04  |CV (image)|MosMedData: 新冠肺炎胸部 CT扫描数据集上基于3D-CNN实现二分类任务。|3D-CNN（医学图像）|https://keras.io/examples/vision/3D_image_classification/| ---- |
| 05  |CV (Point)|PASCAL 3D+数据集上使用 PointNet 实现5类点云的分割|PointNet （点云数据）|https://keras.io/examples/vision/pointnet_segmentation/| ---- |
| 06  |Video （视频）|UCF101数据集上实现基于TimeSformer模型的视频分类|transformer|https://aistudio.baidu.com/aistudio/projectdetail/2162025?channelType=0&channel=0| ---- |
| 07  |Video （视频）|COCO数据集上实现基于Mask-RCNN的视频分割|Mask-RCNN|https://github.com/facebookresearch/maskrcnn-benchmark| ---- |
| 08  |Video （视频）|Moving MNIST 数据集上基于卷积LSTMs的下一帧视频预测|卷积LSTMs|https://keras.io/examples/vision/conv_lstm/| ---- |
| 09  |Video （视频）|UCF101数据集上基于TSM的视频理解|TSM|https://aistudio.baidu.com/aistudio/projectdetail/4114499?channelType=0&channel=0| ---- |
| 10  |Video （视频）|MedMNIST数据集上基于Video Vision Transformer的医学轻量视频分类任务。|Video ViT（医学视频）|https://keras.io/examples/vision/vivit/| ---- |
| 11  |可解释性（神经网络性能的可解释性）|使用Grad-CAM 类激活可视化深度学习网络性能的可解释性|Grad-CAM|https://keras.io/examples/vision/grad_cam/| ---- |
| 12  |时间序列|PeMSD7 真实交通速度数据集上利用 图神经网络 和 LSTM 进行交通预测|GNN & LSTM|https://keras.io/examples/timeseries/timeseries_traffic_forecasting/| ---- |
| 13  |时间序列|Jena Climate时间序列数据集上使用LSTM进行温度的预报|LSTM|https://keras.io/examples/timeseries/timeseries_weather_forecasting/| ---- |
| 14  |时间序列|FordA噪声测量数据集上使用FCN进行发动机噪声分类|FCN|https://keras.io/examples/timeseries/timeseries_classification_from_scratch/| ---- |
| 15  |时间序列|baostock 证券数据集下使用LSTM 模型预测 A 股走势|LSTM|https://colab.research.google.com/github/invisprints/blog/blob/master/_notebooks/2020-04-17-LSTM-stock.ipynb| ---- |
| 16  |图数据|Cora数据集上使用图关注网络GAT进行科学论文的预测|GAT|https://keras.io/examples/graph/gat_node_classification/| ---- |
| 17  |图数据|BBBP分子药数据集上使用MPNN模型进行分子的性质预测|药物性质预测|https://keras.io/examples/graph/mpnn-molecular-graphs/| ---- |
| 18  |NLP|IMDB数据集上基于双边 LSTM的文本分类任务|双边 LSTM|https://keras.io/examples/nlp/bidirectional_lstm_imdb/| ---- |
| 19  |NLP|English-to-Spani数据集下基于sequence-to-sequence Transformer的机器翻译任务|Transformer|https://keras.io/examples/nlp/neural_machine_translation_with_transformer/| --   |
| 20  |NLP|文献检索数据集下基于Pair-wise模型的语义检索系统|Pair-wise|https://aistudio.baidu.com/aistudio/projectdetail/3351784?channelType=0&channel=0| --   |
| 21  |NLP|RSC15数据集下构造序列语义检索SSR模型|SSR模型|https://aistudio.baidu.com/aistudio/projectdetail/205028?channelType=0&channel=0| --   |
| 22  |NLP|NLPCC2018数据集下基于word2vec构造知识库问答系统系统|word2vec|https://aistudio.baidu.com/aistudio/projectdetail/3206157?channelType=0&channel=0| --   |
| 23  |NLP|NLPCC14-SC和ChnSentiCorp数据集下基于SKEP构造句子级的文本情感分析|SKEP|https://aistudio.baidu.com/aistudio/projectdetail/2097355?channelType=0&channel=0| --   |
| 24  |Audio （语音）|LJSpeech 数据集下使用CTC进行自动语音识别|CTC|https://keras.io/examples/audio/ctc_asr/| --   |
| 25  |Audio （语音）|LJSpeech 数据集下使用Transformer进行音频语音片段的自动语音识别(ASR)|Transformer|https://keras.io/examples/audio/transformer_asr/| --   |
| 26  |推荐|Goodbooks-10k数据集下基于隐式推荐算法模型完成图书推荐系统|隐式推荐算法|https://aistudio.baidu.com/aistudio/projectdetail/2556840?channelType=0&channel=0| --   |
| 27  |推荐|基于PaddlePaddle的SR-GNN推荐算法解决Session-based 会话推荐模型|SR-GNN推荐|https://aistudio.baidu.com/aistudio/projectdetail/124382?channelType=0&channel=0| --   |
| 28  |强化学习|用DQN强化学习算法玩“合成大西瓜”！|DQN|https://aistudio.baidu.com/aistudio/projectdetail/1556392?channelType=0&channel=0| --   |
| 29  |强化学习|用飞桨框架基于AlphaZero算法造一个会下五子棋的AI模型|AlphaZero算法|https://aistudio.baidu.com/aistudio/projectdetail/1403398| --   |
| 30  |GAN|selfie2anime数据集下基于CycleGAN的将照片变为二次元卡通风格|CycleGAN|https://aistudio.baidu.com/aistudio/projectdetail/1153303?channelType=0&channel=0| --   |
| 31  |GAN|MNIST数据集下用Paddle框架的动态图模式玩耍经典对抗生成网络（GAN）|GAN|https://aistudio.baidu.com/aistudio/projectdetail/551962?channelType=0&channel=0| --   |
| 32  |GAN|cifar10数据集上复现大规模的生成对抗网络模型BigGAN|BigGAN|https://aistudio.baidu.com/aistudio/projectdetail/860925?channelType=0&channel=0| --   |
| 33  |GAN|城市街景分割数据集下使用对抗网络Pix2Pix生成分割掩码，并使用掩码生成街景|对抗网络Pix2Pix|https://aistudio.baidu.com/aistudio/projectdetail/1119048?channelType=0&channel=0| --   |
| 34  |多模态|基于UGC用户制作的真实短视频业务数据，融合文本、视频图像、音频三种模态进行视频多模标签分类|多模融合|https://aistudio.baidu.com/aistudio/projectdetail/3469740?channelType=0&channel=0| --   |
| 35  |多模态|XFUND数据集下基于多模态(OCR+文档视觉问答)技术实现金融表单识别|(OCR+文档视觉问答)|https://aistudio.baidu.com/aistudio/projectdetail/3884375?channelType=0&channel=0| --   |
| 36  |多模态|GAMMA比赛多模态眼底图像数据集下基于EfficientNet和resnet构造fundus_img和oct_img的分类模型|EfficientNetB3+resnet34|https://aistudio.baidu.com/aistudio/projectdetail/2334029?channelType=0&channel=0| --   |


 <mark>【注意】招募列表外的，欢迎开发者主动贡献👏 <mark> 

## 3. 贡献指南
### 3.1 飞桨框架2.4 版本安装和使用
- 1. 飞桨（PaddlePaddle）版本统一使用2.4 最新版，安装说明：https://www.paddlepaddle.org.cn/install/quick。
- 2. 2.4 版本已有应用实践教程：https://www.paddlepaddle.org.cn/documentation/docs/zh/2.4rc/practices/index_cn.html。

### 3.2 题目认领
为避免重复选题、知晓任务状态、方便统计管理，请根据如下操作认领您的题目。
* 1）方式一（招募列表）：在“招募列表”中选择题目，并在本 Issue 中，回复下列信息：
```
【列表选题】
编号：XX
题目：XXXX
认领人：XX

```

* 2）方式二（自选题目）：自主命题，直接在本 Issue 中，回复下列信息：
```
【自选题目】
题目：XXXX
认领人：XX

```

### 3.3 原则及注意事项 （极其重要）！！
1. <u>必须</u>使用 **paddle** 的框架, 必须避免使用 **paddle的套件**。
 <br />   <br />
2. 示例<u>必须适合</u> **2.0~2.4** 版本.
<br />   <br />
3. <u>数据集</u>：**公开的**、**大小合适** 的示例数据集。（建议用小数据集，测试即可）
<br />   <br />
4. <u>单个任务运行时长</u>：**20-30** 分钟可以运行完成
<br />   <br />
5. <u>文字描述</u>：统一使用 **中文** 编写，注意概念和描述的清晰度，尽量让大家通俗易懂，如果实在难以解释，可以给出一些能够详细介绍的页面链接。
<br />   <br />
6. <u>代码相关</u>：
- *  使用API编写代码：编写过程中优先使用高层API，无法使用高层API时使用基础API。[高层API使用指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/practices/quick_start/high_level_api.html) ： （注意：避免使用套件）
<br />   <br />
- * 易读性好：代码封装得当，易读性好，不用一些随意的变量/类/函数命名。
<br />   <br />
- * 注释清晰：不仅说明做了什么，也要说明为什么这么做。
<br />   <br />
- * 代码运行自测：做好代码的自测工作，每段代码块需要有运行结果。
<br />   <br />
7. <u>任务难度</u>：初级、中级


### 3.4 教程编写
1. 应用案例教程统一使用Notebook格式（.ipynb）来进行编写，可以在本地安装使用Jupyter开发，或使用 [AIStudio](https://aistudio.baidu.com) 。
<br />   <br />
2. 为了方便大家进行教程的编写，并统一阅读体验，下面为大家提供了一个简单的撰写框架，大家根据实际任务结合实际场景进行微调。

如果对模板有一些建议，也可以回复评论。

```
# 题目

作者信息：Github ID (github个人主页URL)
更新日期：** 年 ** 月 ** 日
--

## 1. 简要介绍
简单的一段文字介绍本案例场景和用到的一些知识点，不用太复杂的讲述知识细节，

## 2. 环境设置
导入包，运行一些初始化方法

## 3. 数据集
讲述数据集的一些基础信息，描述数据集组成等。进行数据集的下载、抽样查看、数据集定义等。

## 4. 模型组网
基于Layer定义模型网络结构，模型的可视化展示。可以概要讲述一些网络结构代码设计的原因。

## 5. 模型训练
使用模型网络结构和数据集进行模型训练。需要讲述一些实践中的知识点。

## 6. 模型评估
使用评估数据评估训练好的模型。

## 7. 模型预测
对模型进行预测，展示效果。

```


### 3.5 教程上传
1. 写好的文档通过向[https://github.com/PaddlePaddle/docs)仓库提交Pull Request的方式来进行教程文档的上传。
<br />   <br />
2. 对提交好的PR可以 指定Reviewer 进行内容和代码的评审，通过后会由具有Merge权限的同学进行最终的合入。


## 4. 已合入仓库的教程
目前已经有 **23** 篇基于 **飞桨2.3** 的教程贡献，查看方式：
1. Repo目录查看已经Merge的Notebook源文件：docs/practices。
<br />   <br />
2. 查看官网渲染后的页面：应用实践。

## 5. 还有不清楚的问题？

欢迎大家随时在此Issue下提问，飞桨会有专门的管理员进行疑问解答。

有任何问题，请联系：飞桨的PM-莫琰 [momozi1996](https://github.com/momozi1996) ! 

非常感谢大家为飞桨贡献！共建飞桨繁荣社区！


