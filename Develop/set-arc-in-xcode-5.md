---
title: set ARC in Xcode 5
date: 2013-9-28 12:29:12
categories:
- develop
tags:
- xcode
- ARC
---

在Xcode5中遇到一个问题，从老师那里继承的习惯，就是自己写release, 而到了xcode5中，自己写release会显示错误，其中的原因就是xcode5默认是使用自动ARC机制的。

![][image-1]

<!--more-->

在Xcode 4中，我们在建立一个Project的时候下边是三个选项的，其中一项就是勾选是否ARC的，如果不打勾，那么就可是自己手动写release，按照我们老师的话，虽然自己写稍微麻烦一点，但是对于理解iOS内部的内存管理机制是有利的，并且鼓励我们都自己写。于是，也养成了此样的习惯。

![][image-2]

是不是我们就没办法了呢？其实，在Xcode5中还是可以设置回手动进行内存管理的。需要进行一些设置:

首先我们要点击项目进入Project设置:

![][image-3]

然后我在进入build setting:

![][image-4]

在然后我们找到OC的ARC设置，将YES改为NO就可以了.

![][image-5]

更改完毕后再回去将alloc的函数release掉，现在不会报错了！

[image-1]:	https://farm3.staticflickr.com/2877/9979865694_1d3430c275_o.png
[image-2]:	http://farm4.staticflickr.com/3697/9979901576_a4f354c04c_o.png
[image-3]:	http://farm3.staticflickr.com/2818/9979858514_9ccebcb3a0_o.png
[image-4]:	http://farm3.staticflickr.com/2891/9979953893_3a9da1381e_o.png
[image-5]:	http://farm3.staticflickr.com/2880/9979958443_9d40de37cc_o.png