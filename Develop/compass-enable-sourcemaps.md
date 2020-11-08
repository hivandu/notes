---
layout: compass
title: compass 开启 sourcemaps
date: 2014-10-29 15:28:41
tags:
- compass
- sourcemaps
- sass
---

sass 3.0 开始包含了sourcemaps.... （妈蛋，都是屁话，大家都知道了）

直接进入正题，sass开启sourcemaps的话，命令行如下:

<!-- more -->

`sass --sourcemaps --watch`

可是在compass中我们如果这么用的话:

`compass --sourcemaps --watch`
是绝对行不通的。

如果compass下开启，首先还是需要下载一个扩展包
（假设你已经下载安装好ruby,sass, compass等一众开发环境)

`gem install compass-sourcemaps --pre`

然后在项目内打开设置文件`config.rb`,如果你是`compass creat`的话，这个文件是自动生成的

在内添加如下代码开启

```
# enable sourcemaps
enable_sourcemaps = true
sass_options = {:sourcemap => true}
```

然后再到Chrome中看吧，OK了。

也推荐使用Koala: [http://koala-app.com/](http://koala-app.com/)

记得设置调用系统组件


