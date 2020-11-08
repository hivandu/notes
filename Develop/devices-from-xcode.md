---
title: Devices from xcode
date: 2013-8-25 17:27:08
categories:
- develop
tags:
- xcode
- ios
- iPhone
---
在不用开发者帐号升级iPhone的时候就只有下载网上放出的固件进行升级，可是总有不小心的是没有进行升级，而是进行了重置！

OK，如果没有开发者帐号而使用了iOS7固件进行重置，那么你的机子就根本无法通过Apple.inc的验证，也就是说只能卡在一开始的设置界面，无法开始使用你的iPhone。

<!--more-->

网上有解决办法，就是电源键+Home，然后再用iTunes重置到iOS6，再升级！好吧，如果有Mac的话，基本就不用那么麻烦了，直接在Xcode Help中找到Devices，然后直接在里边找到已经连接到电脑上的iPhone ,iption+左键，选择6.1.3，降级后先验证，然后从新升级就OK了。

![xcode](http://farm6.staticflickr.com/5346/9597147864_7cf8e82cda_z.jpg)