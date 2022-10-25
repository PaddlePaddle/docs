# Code Review 约定

为了使评审人在评审代码时更好地专注于代码本身，请你每次提交代码时，遵守以下约定：

1）请保证 CI 中测试任务能顺利通过。如果没过，说明提交的代码存在问题，评审人一般不做评审。

2）如果解决了某个 Issue 的问题，请在该 Pull Request 的**第一个**评论框中加上：`fix #issue_number`，这样当该 Pull Request 被合并后，会自动关闭对应的 Issue。关键词包括：close, closes, closed, fix, fixes, fixed, resolve, resolves, resolved，请选择合适的词汇。详细可参考[Closing issues via commit messages](https://help.github.com/articles/closing-issues-via-commit-messages)。

此外，在回复评审人意见时，请你遵守以下约定：

1）评审人的每个意见都必须回复（这是开源社区的基本礼貌，别人帮了忙，应该说谢谢）：

   - 对评审意见同意且按其修改完的，给个简单的`Done`即可；

   - 对评审意见不同意的，请给出你自己的反驳理由。

2）如果评审意见比较多：

   - 请给出总体的修改情况。

   - 请采用[start a review](https://help.github.com/articles/reviewing-proposed-changes-in-a-pull-request/)进行回复，而非直接回复的方式。原因是每个回复都会发送一封邮件，会造成邮件灾难
