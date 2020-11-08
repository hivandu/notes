---
layout: post
title: "用comodo 限制TM"
date: 2010-03-25 15:01
categories: 
- software
tags:
- comodo
- TM
- 安全
- 限制
published: true
comments: true
---
![image](http://farm3.static.flickr.com/2561/4462263640_d7626800ec_o.png)

最近都在研究HIPS,说实话,这是一个比较麻烦的事情!

而我用Comodo的主要目的就在于自己的NOD32到期后可以单奔,但是暂时,我仅仅想对腾讯对我的侵犯进行限制!

说明一下,为了安全和避免麻烦,最好选用TM.如果选用QQ,对应规则还是一样的,只是会不断的弹出selfupdate.exe对话框,不嫌麻烦的就尽量去点吧!以下规则在TM中测试完全没问题,正常使用,但是在QQ中就无法访问自己的文件夹,就算添加了允许文件夹列表也不行,必须完全允许QQ访问文件夹,而在Comodo日志中也找不到相应的访问记录!所以,用QQ的可以考虑将受保护文件/目录设为允许,可是安全性就大打折扣!

<!--more-->

以下是我的总规则:

![comodo限制TM](http://farm5.static.flickr.com/4050/4462242570_336feb0f14_o.jpg)

详细说明:

<span style="color: #ff0000;">“运行一个可执行”程序，询问。</span>

点击右侧“修改”，在“允许的程序列表”中，添加常用浏览器路径，比如我是chrome. //这是为了方便大部分人能在TM中调用浏览器访问QQ空间或者直接打开好友给的地址.<br />
在“禁止的程序列表”中，添加禁用的调用程序中，添加 `*\selfupdate.exe` //TM的自升级程序</p>

<p><span style="color: #ff0000;">“进程间内存访问”，阻止。</span>

“允许的程序列表”中，可以添加TM本身，即`*\tm.exe`//对正常程序来说，允许访问程序自身内存空间就足够了。</p>

<p><span style="color: #ff0000;">“窗口或钩子事件”，阻止。</span>

“允许的程序列表”中，添加`C:\WINDOWS\system32\MSCTF.DLL`(处于“疯狂模式”下的COMODO，添加该项是确保聊天窗口中输入法切换正常；) 及 TM本身</p>

<p><span style="color: #ff0000;">“进程终止”和“设备驱动程序安装”，两项全部选为阻止；<span style="color: #000000;">//如果在语音或视频聊天时提示需要安装驱动，可临时将其选为询问，并记录为规则，然后再切换到阻止；</span></span></p>

<p><span style="color: #ff0000;">“窗口消息钩子”，阻止。</span></p>

<p>“允许的程序列表”中，添加`c:\windows\explorer.exe`</p>

<p><span style="color: #ff0000;">“受保护的COM口”和“受保护的注册表项”，毫不客气，全部阻止；</span></p>

<p><span style="color: #ff0000;">“受保护的文件/目录”，阻止。</span>
“允许的文件/夹列表”中，添加\Device\Afd\Endpoint</p>

<p>“屏幕监视器”设为允许，保证截图功能可用；“键盘设为允许”，确保聊天对话可输入；</p>

<p>以上诸项设置完成之后，还需要在“我的被拦截文件夹”中添加两项：<br />
```
*\Application Data\Tencent\TM\SafeBasetseh.dat<br />
*\Application Data\Tencent\TM\SafeBaseselfupdate.exe<br />
```

亦可用通配符添加：<br />

```
*\safebase\tseh.dat<br />
*\safebase\selfupdate.exe</p>
```
<p>本人系统,window 7,软件为:TM2009最新,以及comodo V4,QQ2011以及其他QQ版本一样适用!</p>
