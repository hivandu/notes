---
title: Auto-save-photo-to-qqmail
date: 2012-4-10 12:19:22
categories:
- web
tags: 
- Mail
- ifttt
---

### 前言

为什么是QQ Mail?

因为它大,而且不断自动扩容,你想把它装满暂时是不太现实. 而且来说,QQ邮箱的体验还是非常不错的!过滤规则也很能满足要求,归档搜索查找都不错!一些不涉及隐私而又想保存的文件或者照片或者其他什么东东,存在QQ邮箱里还是不错的!比如:XXX

<!-- more -->

**如果想同步到Google+请看完文章后看最后部分的更新说明!**

### 如何实现

通过众所周知的[ifttt](http://ifttt.com/)

其实,这主要是一个我为了保存自己照片的方式!(爱信不信,不相信拉倒!)

建立一个task, if Instagram 设定条件:New Liked photo, then Gmail 设定条件:Send an email.

好了,填上 To address: xxx@qq.com 就OK了! 简单吧?

如果你熟悉某个联系人,那么建立规则by Username,然后包含此用户名关键词的主题都标上相应的关键词.比如[by ladiiprang](https://twitter.com/#!/ladiiprang)在邮箱规则里就可以加上"妹子"的tags.

![妹子](http://farm6.staticflickr.com/5454/7062008689_141e7cde9a_z.jpg)

### 注意点

如果你不想后期被众多的新邮件搞得头昏脑胀的话,那么你一开始就要设定好过滤条件!

在ifttt中,Send an email的时候Subject记得填写上一些关键词,比如From Instagram,这样,对于主题内有关键词的邮件就好管理的多了.添加过滤规则就好了!将来自己发送邮箱Gmail的邮件主题包含From和Instagram的都自动移动到一个新建的文件夹内,Ex:Photo DB,完工!

### 后记

同理,我们也可以建立来自Flickr的发送规则,原理是一样的!注意Gmail邮箱里的过滤条件要建好!否则Gmail爆满是迟早的事情!运用这样的规则,我们还可以发送[Dropbox](http://db.tt/46tx8aDd)里的文档到QQ邮箱内保存,不过规则限定发送的只能是Public内的文档!

其实一开始我不确定是发送文件还是只有地址!所以我开始的方法很绕,就是将相片想Save到Dropbox,然后再通过Dropbox建立if.then.Gmail.不过试验下来既然能直接发送文件,建立task就简单多了!

还等什么,快去你的Instagram和Flickr上收藏妹子到邮箱内吧!

### 更新

本来因为标题的原因,这点是不加在这里的!但是想着再写一篇一样意义的文章很没意思,所以就在这里说明一下好了!

在我这篇文章发布之后,G+上看到了[电脑玩物](http://playpcesor.blogspot.com/)的作者+[esor huang](https://plus.google.com/u/0/100105166562504132677/) 的一篇讲解[Instagram同步到Dropbox和Google+的说明](http://playpcesor.blogspot.com/2012/04/instagram-dropbox-google.html)!以及[这篇E文](http://www.androidcentral.com/sync-your-instagram-photos-google-how)!说起来,这样的同步方式确实很笨拙.你打算电脑24小时开着picasa来为你同步么?那么,我从我这篇文的基础上考虑可行方案!记得之前Picasaweb给每个人都有一个邮箱推送地址!就是类似username.password@picasaweb.com这样的地址!好吧,有邮箱地址就简单了.不过这个地址需要你再登录picasaweb.com去找,在Google+页面上是找不到的!同理,Flickr也有类似的推送地址! 该怎么做我想你已经清楚了吧!

最后,我想到了是否同样可以传送到QQ相册!毕竟和邮箱最大的不同就是相册是用来分享的,而邮箱是用来保存的!可惜,QQ没有针对相册的推送,而推送到QQ空间的XXX@qzone.qq.com这个地址也是必须QQ邮箱内部发才行!是的,和你们一样,我又想到了邮箱转发规则.用QQ邮箱收到邮件后转发到QQ空间邮箱去不就好了!测试后,果然.

...

果然没那么简单,这次我失误了,特么的QQ小气到不允许自己的QQ邮箱转发邮件到QQzone的邮箱来自动推送文章!提示这是一个无效地址!不过也无所谓了,毕竟是推送文章的邮件地址,你也不想自己的QQ空间全是大片的文章,而且每篇文章里只有一张相片吧 ?

这个说明本来是在G+上有提出的,但是有基友测试成功后给的是这里的url,所以我就想,补上这个说明!谁说是一样的到底,但是如果不提Picasaweb有邮件推送地址,估计很多人都已经忘记了!为了找同步到G+上的朋友会看得糊里糊涂!