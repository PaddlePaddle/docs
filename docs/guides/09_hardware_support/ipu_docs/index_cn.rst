.. _cn_ipu_information:

#########################
Graphcore IPU芯片运行飞桨
########################

Colossus™ GC200是Graphcore第二代IPU芯片，拥有594亿个晶体管，并采用最新的台积电 7纳米工艺制造，是世界上最先进的处理器。每个MK2 IPU具有1472个强大的处理器内核，可运行近9,000个独立的并行程序线程。每个IPU拥有前所未有的900MB处理器内存储器和250 TeraFlops AI计算，片内访存带宽达到47.5TB/s。片内IPU-Core之间有8TB/s的all-to-all的IPU-Exchange总线；另外我们设计了单独的IPU-Link来做IPU之间的高速通信总线。 更多IPU产品信息请 `点击这里 <https://www.graphcore.ai/products>`_ 。

参考以下内容体验IPU芯片：

- `飞桨框架IPU版安装说明 <./paddle_install_cn.html>`_ : 飞桨框架IPU版安装说明
- `飞桨框架IPU版训练示例 <./train_example_cn.md>`_ : 飞桨框架IPU版训练示例
- `飞桨预测库IPU使用示例 <./inference_example_cn.md>`_ : 飞桨预测库IPU使用示例

..  toctree::
    :hidden:
    
    paddle_install_cn.md
    train_example_cn.md
    inference_example_cn.md

