---
title: to_s；to_str；inspect
date: 2013-4-19 14:13:50
categories:
- develop
tags:
- Ruby
---

这篇文是转载自「Rubylution」，似乎那上边也是转载自「Ruby-china」。

不过作为记录，在自己的博客上记录一下总归是好的，便于查阅！ 原文地址：

[1](http://rubylution.herokuapp.com/topics/19), [2](http://rubylution.herokuapp.com/topics/17)

<!-- more -->

### 1. to_s 和 to_str的区别

在Github上面看到tenderlove大神的[一个回复](https://github.com/rails/rails/commit/188cc90af9b29d5520564af7bd7bbcdc647953ca#actionpack-lib-action_view-template-resolver-rb-P24)，解释为什么定义to_str方法，和如何强制转换成字符串。

```
irb(main):001:0> class Foo; def to_str; 'hello'; end end
=> nil
irb(main):002:0> class Bar; def to_s; 'hello'; end end
=> nil
irb(main):003:0> File.join('/', Foo.new)
=> "/hello"
irb(main):004:0> File.join('/', Bar.new)
TypeError: can't convert Bar into String
    from (irb):4:in `join'
    from (irb):4
    from /Users/aaron/.local/bin/irb:12:in `<main>'
irb(main):005:0>
```

to_s和to_str的行为在大部分时候是相同的。几乎每个对象都有to_s方法，因为都继承自Object对象。但是不是每个对象都有to_str方法，一般只有这个对象具有某些String-like的行为时才去定义to_str方法。

就像[上面的链接](https://github.com/rails/rails/commit/188cc90af9b29d5520564af7bd7bbcdc647953ca#actionpack-lib-action_view-template-resolver-rb-P24)提到的，原本Path是继承自String类的，修改之后只只定义了一个to_str方法就可以和其他String进行join了...

像这样表示的XX-like的方式还有很多，比如to_i和to_int，to_a和to_ary等。

但是，并不是所有和字符串交互的方法都会调用to_str，下面演示一下常用的：

```
class Hooopo
  def to_str
    "to_str"
  end

  def to_s
    "to_s"
  end
end

ruby-1.9.3-p0 :075 > hooopo = Hooopo.new
 => to_s 
ruby-1.9.3-p0 :076 > "hello #{hooopo}"
 => "hello to_s" 
ruby-1.9.3-p0 :078 > ["hello", hooopo].join(" ")
 => "hello to_str" #这里也说明在join的时候数组里不一定要都是string啊，只要能响应to_str或to_s方法就OK！
ruby-1.9.3-p0 :079 > "hello " + hooopo
 => "hello to_str" 
ruby-1.9.3-p0 :080 > File.join("hello", hooopo)
 => "hello/to_str" 
ruby-1.9.3-p0 :081 >
```

上面的演示可以得出结论，在字符串内插和inspect的时候调用的是to_s，在Array#join, File#join, String#+的时候都是优先调用的to_str。

### 2. to_s和inspect的区别

```
[1] pry(main)> class Hooopo
[1] pry(main)*   def to_s
[1] pry(main)*     "to_s"
[1] pry(main)*   end  
[1] pry(main)*   def inspect
[1] pry(main)*     "inspect"
[1] pry(main)*   end  
[1] pry(main)* end  
=> nil
[2] pry(main)> hooopo = Hooopo.new
=> inspect
[3] pry(main)> puts hooopo
to_s
=> nil
[4] pry(main)> print hooopo
to_s=> nil
[5] pry(main)> p hooopo
inspect
=> inspect
```

结论就是

1.puts obj ==> puts obj.to_s

2.p obj ==> puts obj.inspect

其实其他的debug工具像[awesome_print](https://github.com/michaeldv/awesome_print)和[pretty_print](http://www.ruby-doc.org/stdlib-1.9.3/libdoc/pp/rdoc/Kernel.html#method-c-pp)（就是[pp](http://www.ruby-doc.org/stdlib-1.9.3/libdoc/pp/rdoc/Kernel.html#method-c-pp)）也都是p的同类，他们的行为更接近inspect而不是to_s：

```
[6] pry(main)> require 'pp'
=> false
[7] pry(main)> require 'awesome_print'
=> true
[8] pry(main)> pp hooopo
inspect
=> inspect
[9] pry(main)> ap hooopo
inspect
```

还有就是ap和pp都会带各自的inspect方法，awesome_inspect和pretty_inspect,如果你不想只是输出结果，而是想对结果当作字符串处理可以用这两个方法：

```
[10] pry(main)> hooopo.pretty_inspect
=> "inspect\n"
[11] pry(main)> hooopo.awesome_inspect
=> "inspect"
[13] pry(main)> [1,2,3].awesome_inspect
=> "[\n    \e[1;37m[0] \e[0m\e[1;34m1\e[0m,\n    \e[1;37m[1] \e[0m\e[1;34m2\e[0m,\n    \e[1;37m[2] \e[0m\e[1;34m3\e[0m\n]"
```