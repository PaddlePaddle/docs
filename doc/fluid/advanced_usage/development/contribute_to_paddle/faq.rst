.. _contribute_to_paddle_faq:

###################
FAQ
###################

..  contents::

1. CLA签署不成功，怎么办？
---------------------------

由于 `CLA <https://github.com/cla-assistant/cla-assistant>`_ 是第三方开源库，有时候会不稳定。如果确定自己已成功签署CLA，可尝试：

* 关闭并重新开启本PR，来重新触发CLA。点击 :code:`Close pull request` ，再点击 :code:`Reopen pull request` ，并等待几分钟。
* 如果上述操作重复2次仍未生效，请重新提一个PR或评论区留言。

2. CI没有触发，怎么办？
------------------------

* 请在commit信息中添加正确的CI触发规则：

  * develop分支请添加 :code:`test=develop`
  * release分支请添加如 :code:`test=release/1.4` 来触发release/1.4分支
  * 文档预览请添加 :code:`test=document_preview`
      
* 该CI触发规则以commit为单位，即对同一个PR来说，不管前面的commit是否已经添加，如果新commit想继续触发CI，那么仍然需要添加。
* 添加CI触发规则后，仍有部分CI没有触发：请关闭并重新开启本PR，来重新触发CI。


3. CI随机挂，即错误信息与本PR无关，怎么办？
--------------------------------------

由于develop分支代码的不稳定性，CI可能会随机挂。
如果确定CI错误和本PR无关，请在评论区贴上错误截图和错误链接。

4. 如何修改API.spec？
-----------------------

为了保证API接口/文档的稳定性，我们对API进行了监控，即API.spec文件。
修改方法请参考 `diff_api.py <https://github.com/PaddlePaddle/Paddle/blob/ddfc823c73934d483df36fa9a8b96e67b19b67b4/tools/diff_api.py#L29-L34>`_ 。

**注意**：提交PR后请查看下diff，不要改到非本PR修改的API上。
