---
title: Ruby Range
date: 2012-11-6 12:19:48
categories:
- develop
tags:
- Ruby
---

Range在概念上看是非常直观的。不过在实际的使用中，我们可能会遇到一些令人混淆的东西。
看如下代码：
```
digits = 0..9     #0到9
scale1 = 0..10    #0到10
scale2 = 0...10   #0到9,不包含10
```
<!-- more -->

`..`操作符将包含上限，而`...`不包含上限。

不过，Range不只是作用于数字类型，基本上对于任何的对象都有用，但结果是否有实际意义要看实际的情况了。

```
a = 'A'..'Z'
a.to_a.each{|c| puts c}
```

我们称..这样的Range为"关闭"的Range,而`...`的Range为"开放"的Range。
使用first和last方法(或同义方法begin和end)，可以获取一个Range的开始和结束元素：

```
r1 = 3..6
r2 = 3...6
r1a, r1b = r1.first, r1.last    # 3, 6
r1c, r1d = r1.begin, r1.end     # 3, 6
r2a, r2b = r2.begin, r2.end     # 3, 6 (注意：不是3和5)
```

`exclude_end?`方法可以得到Range是否排除上限项(即是否是...的Range)

```
r1.exclude_end?   # false
r2.exclude_end?   # true
```

### 对Range进行迭代

Range是可迭代的，不过，为了更加实际有用，确认你的Range包含的对象已经有一个有意义的succ方法。

```
(3..6).each {|x| puts x }
```

我们看一个有趣的例子：

```
r1 = "7".."9"
r2 = "7".."10"
r1.each {|x| puts x }   # 打印出7,8,9
r2.each {|x| puts x }   # 未打印任何东西
```

为什么会出现这样的情况？这是因为这里都是字符串，由于r1中，"7"比"9"小，所以，它是个合理的Range；而表达式r2中， "7"比"10"大，下限大于了上限，就不合理了。

浮点数的Range可以进行迭代么？我们来看一下：

```
fr = 2.0..2.2
fr.each {|x| puts x }   # 错误!
```

为什么浮点数不可以迭代呢？因为浮点数对象没有succ方法。是因为不能实现么？理论上，这是没有问题的，但是，实际上，如果浮点数 Range迭代，这有可能出现：很小的一个范围，将产生非常庞大的迭代量。这对语言的实现有非常高的要求。况且，这样的功能，极少有用到。

### 测试范围关系

include?方法（同义方法member?）可以判断一个值是否处在当前的Range中：

```
r1 = 23456..34567
x = 14142
y = 31416
r1.include?(x)      # false
r1.include?(y)      # true
```

它内部是怎么实现的呢？其实，它只是把给出的值和该Range的上限做比较得出的（所以，它依赖与一个有意义的<=>）。

### 转化为数组

很简单，to_a方法搞定：

```
r = 3..12
arr = r.to_a     # [3,4,5,6,7,8,9,10,11,12]
```

### 反向的Range

我们前面讨论过了下限大于上限的Range，如：

```
r = 6..3
x = r.begin              # 6
y = r.end                # 3
flag = r.end_excluded?   # false
```

它确实是个合法的Range，但是，它包含的内容缺并不是我们想像的那样：

```
arr = r.to_a       # 得到空的数组[]
r.each {|x| p x}   # 无结果
r.include?(5)      # false
```

那么说反向Range是没有什么用处的咯？那倒不是，我们可以在字符串和数组中使用反向Range：

```
string = "flowery"
str1   = string[0..-2]   # "flower"
str2   = string[1..-2]   # "lower"
str3   = string[-5..-3]  # "owe" (其实这是个正向的Range)
```