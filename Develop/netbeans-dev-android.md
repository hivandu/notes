---
title: NetBeans开发Android
date: 2011-12-2 11:45:10
categories:
- develop
tags:
- Android
- Netbeans
---

一般我们谈论的Android开发环境就是SDK+Eclipse+JDK

其实,除了Eclipse以外,我们还可以利用NetBeans来进行Android开发,只是Google官方并没有为其配备如ADT一样的插件,所以我们需要NBAdroid这个项目作为NetBeans上的ADT来使用.

我们需要先在NetBeans上注册NBAdroid的Update center

Tools -> Plugins -> setting

add后添加 http://kenai.com/downloads/nbandroid/updatecenter/updates.xml

然后再Available plugins中找到Android TestRunner for NetBeans X.x.x安装

然后添加Android SDK位置到NetBeans里就可以了!

其实,使用下来,NetBeans做Android开发并不太顺利,似乎没有Eclipse好用,毕竟没有Google官方支持.最突出的就是import都需要自己手动添加.所以,不想折腾的还是用Eclipse就好了.这里只是做一个介绍和记录,多一个方法总归是好的.