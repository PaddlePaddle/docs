.. _contribute_to_paddle_faq:

###################
FAQ
###################

..  contents::

1. CLA 签署不成功，怎么办？
---------------------------

由于 `CLA <https://github.com/cla-assistant/cla-assistant>`_ 是第三方开源库，有时候会不稳定。如果确定自己已签署 CLA，但 CLA 没触发成功，可尝试：

* 关闭并重新开启本 PR，来重新触发 CLA。点击 :code:`Close pull request` ，再点击 :code:`Reopen pull request` ，并等待几分钟。
* 如果上述操作重复 2 次仍未生效，请重新提一个 PR 或评论区留言。

2. CI 没有触发，怎么办？
------------------------

* 请在 commit 信息中添加正确的 CI 触发规则：

  * develop 分支请添加 :code:`test=develop`
  * release 分支请添加如 :code:`test=release/1.4` 来触发 release/1.4 分支
  * 文档预览请添加 :code:`test=document_preview`

* 该 CI 触发规则以 commit 为单位，即对同一个 PR 来说，不管前面的 commit 是否已经添加，如果新 commit 想继续触发 CI，那么仍然需要添加。
* 添加 CI 触发规则后，仍有部分 CI 没有触发：请关闭并重新开启本 PR，来重新触发 CI。


3. CI 随机挂，即错误信息与本 PR 无关，怎么办？
--------------------------------------

由于 develop 分支代码的不稳定性，CI 可能会随机挂。
如果确定 CI 错误和本 PR 无关，请在评论区贴上错误截图和错误链接。

4. 如何修改 API.spec？
-----------------------

为了保证 API 接口/文档的稳定性，我们对 API 进行了监控，即 API.spec 文件。
修改方法请参考 `diff_api.py <https://github.com/PaddlePaddle/Paddle/blob/ddfc823c73934d483df36fa9a8b96e67b19b67b4/tools/diff_api.py#L29-L34>`_ 。

**注意**：提交 PR 后请查看下 diff，不要改到非本 PR 修改的 API 上。
