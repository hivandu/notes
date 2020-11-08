---
title: ruby1.9.3 for Lion
date: 2012-7-17 12:24:15
categories:
- develop
tags:
- Ruby
- OSX
---
最近败了个MACBOOK PRO，才发现以前在PC上装的黑苹果和正宗的Lion简直是天差地别！好用到近期都不再碰Windows。

其实说起来，网上大牛们都曾经说过开发Ruby的话，在Windows下是一种折磨，所以我一开始用过Ubuntu。其实我开始用Ruby的原因仅仅是因为Octopress，并且在Win8上使用过一段时间。看过我之前博客的人应该都知道，那真是段折磨人的日子。

说回Mac的Ruby开发，当然首先都知道在Mac上开发要去下载一个xcode，在无法忍受的网络环境下将1G多的Xcode下载安装后以为就万事大吉，相反，这才是开始！在Xcode4.3中，GCC编译环境似乎并没有集成，当然，不只是GCC而已！所以，我们不得不去又去折腾一番才行。如图： 

<!-- more -->

![](http://farm9.staticflickr.com/8422/7591061306_704f304704_b.jpg)

需要下载Command Line Tools才行。说起来是很简单啦，只是我在找原因的时候花费了不少时间，并向N多大牛询问学习。

如果是使用Octopress的话，又遇到一个新问题：FSSM is dead,好吧，也只是preview 无法使用，不影响写博客并上传！

在安装完RVM后一直卡在Ruby安装这块的朋友，不妨也试试现在Xcode中Download Command Line Tools 后`rvm install 1.9.3 --with-gcc=clang`,余下的，应该就一切顺利了

**2013-12-15 update: OSX 10.9 已经无需下载gcc了**