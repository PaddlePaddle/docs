# 热身打卡活动

各位参加飞桨黑客松的开发者，大家好，欢迎大家参加飞桨第三期黑客马拉松！本次黑客马拉松共设置了十个方向、两个难度等级，供大家选择。注意到一些小伙伴加群后还没有行动起来，飞桨团队特地推出了本次热身打卡活动，参与热身打卡活动并将截图发送至 paddle-hack@baidu.com，即可获得精美的飞桨黑客松周边礼品！数量有限，赶快行动起来吧！

## 热身打卡活动简介

在飞桨框架二次开发中，编译 paddle 是一个重要环节，也是很多任务（如API开发、算子性能优化、数据类型扩展等）的前置条件。本次热身打卡活动，要求参与者通过 github 拉取 [PaddlePaddle/Paddle](https://github.com/PaddlePaddle/Paddle) 仓库代码，并参考 [源码编译教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/compile/fromsource.html) 完成 paddle 编译，截图编译成功的界面后，参考如下格式向 paddle-hack@baidu.com 发送邮件，打卡成功后即可获得精美的飞桨黑客松周边礼品！

注：打卡任务包含基础任务和附加任务，每完成一个任务都可以活动一个礼品。本次热身打卡活动对硬件没有要求！CPU/GPU 均可，赶快行动起来吧~

## 编译流程

### 1. 增加时间戳

即在命令最开始加上time，以[【Linux下从源码编译】文档的【使用Docker编译】](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/compile/linux-compile.html#span-id-compile-from-docker-docker-span)为例，

【9.执行cmake】和【10.执行编译】增加时间戳的命令为：

- CPU版本的cmake：`time cmake .. -DPY_VERSION=3.7 -DWITH_GPU=OFF`
- GPU版本的cmake：`time cmake .. -DPY_VERSION=3.7 -DWITH_GPU=ON`
- 执行编译：`time make -j$(nproc)`，举例有4核，即`time make -j4`

### 2. 初次编译/二次编译

初次编译时间较长，二次编译因为有编译缓存的存在，时间会缩短，对日常开发来说，二次编译时间才是影响开发效率的。让我们来感受下修改不同文件的二次编译时间。

- 修改底层的头文件：paddle/fluid/platform/enforce.h
- 修改Op的cc文件：paddle/fluid/operators/center_loss_op.cc
- 修改python文件：python/paddle/tensor/math.py

二次编译方式：对应文件加一个空行/空格保存退出后，然后执行编译命令`time make -j$(nproc)`，二次编译不再需要执行cmake。

### 3. 安装whl包

参考[【Linux下从源码编译】文档的【使用Docker编译】](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/compile/linux-compile.html#span-id-compile-from-docker-docker-span)中[【11. 寻找whl包】](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/compile/linux-compile.html#paddle-build-python-dist-whl)和[【12. 安装whl包】](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/compile/linux-compile.html#whl)。

### 4. 运行单元测试

不同的编译选项，能编译出不同的功能，对应的编译时间也各不相同。可以参考[编译选项表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#bianyixuanxiangbiao)，尝试打开`WITH_TESTING=ON`编译出单元测试，并正确运行一个单测。

- 重新运行cmake命令：`cmake .. -DPY_VERSION=3.7 -DWITH_GPU=OFF -DWITH_TESTING=ON`（在原来的cmake命令后加入`-DWITH_TESTING=ON`）
- 执行编译命令`make -j$(nproc)`
- 进入build目录，运行单元测试：参考[【飞桨API python端开发指南】之【运行单元测试】](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/new_python_api_cn.html#yunxingdanyuanceshi)，执行`ctest -R test_logsumexp`运行logsumexp的单测。

## 邮件格式

标题： [Hackathon 热身打卡]

内容：

飞桨团队你好，

【GitHub ID】：XXX

【打卡内容】：初次编译/二次编译/安装whl包/运行单元测试

【打卡截图】：

如：

标题： [Hackathon 热身打卡]

内容：

飞桨团队你好，

【GitHub ID】：paddle-hack

【打卡内容】：初次编译&二次编译&安装whl包&运行单元测试

【打卡截图】：

| 硬件               | ![](https://github.com/PaddlePaddle/docs/blob/release/2.3/docs/guides/10_contribution/img/hackathon%233_warming_up_1.png?raw=text) |
| ------------------ | ------------------------------------------------------------ |
| 编译方式           | 参考【Linux下从源码编译】文档的【使用Docker编译】（[源码编译文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/compile/fromsource.html) 有多种编译方式，请大家填写本次编译参考的文档） |
| cmake命令和时间    | 命令：`time cmake .. -DPY_VERSION=3.7 -DWITH_GPU=OFF`</br>时间：注意要将commit号截图进来哦</br>![](https://github.com/PaddlePaddle/docs/blob/release/2.3/docs/guides/10_contribution/img/hackathon%233_warming_up_2.png?raw=text) |
| 初次编译命令和时间 | 命令：`time make -j20` （写一下大家用几核哦）</br>时间：以下时间仅作为示例，不代表真实的初次编译时间</br>![](https://github.com/PaddlePaddle/docs/blob/release/2.3/docs/guides/10_contribution/img/hackathon%233_warming_up_3.png?raw=text) |
| 二次编译时间       | 时间：以下时间仅作为示例，不代表真实的二次编译时间</br>paddle/fluid/platform/enforce.h</br>![](https://github.com/PaddlePaddle/docs/blob/release/2.3/docs/guides/10_contribution/img/hackathon%233_warming_up_4.png?raw=text)</br>paddle/fluid/operators/center_loss_op.cc</br>![](https://github.com/PaddlePaddle/docs/blob/release/2.3/docs/guides/10_contribution/img/hackathon%233_warming_up_5.png?raw=text)</br>python/paddle/tensor/math.py</br>![](https://github.com/PaddlePaddle/docs/blob/release/2.3/docs/guides/10_contribution/img/hackathon%233_warming_up_6.png?raw=text) |
| 安装whl包          | ![](https://github.com/PaddlePaddle/docs/blob/release/2.3/docs/guides/10_contribution/img/hackathon%233_warming_up_7.png?raw=text) |
| 运行单元测试       | ![](https://github.com/PaddlePaddle/docs/blob/release/2.3/docs/guides/10_contribution/img/hackathon%233_warming_up_8.png?raw=text) |




## 礼品发放

- 打卡确认后，你会收到一封回复邮件，收集你的邮寄地址，请提供准确的邮件地址，以便小礼品可以飞速送到你的手上！
- 每一次成功都可以获得一个小礼品，如打卡 初次编译&二次编译&安装whl包&运行单元测试 可以获得4个黑客松周边布贴，多打卡多得哦。
- 在编译过程中有任何的问题或建议，都可以提交 issue，注明 PaddlePaddle Hackathon，经飞桨团队确认是 bug 或有效建议后，可额外获得一个小礼品。
- 将本次活动或活动奖品分享到社交平台（如B站、微博、小红书、朋友圈等），截图发在QQ/微信群或邮件中，也有机会获得更多周边礼品哦~
