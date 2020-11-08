---
title: white-space:nowrap
date: 2013-9-2 14:19:50
categories:
- develop
tags:
- HTML
---

奇葩的需求就有奇葩的设计，虽然这样做会超出父类，制式表格超过边线，不过既然有这样的需求也没办法。

在Table中强制为一行显示，我也是今天才知道原来有这样的方式！

```
table
  tr
    td white-space:nowrap;
```

以上是sass写法，css应该为:

```
table tr td{white-space:nowrap;}
```
