---
title: ruby-「file」
date: 2012-11-5 12:14:41
categories:
- develop
tags:
- Ruby
---
读取一行

```
File.open('test.txt','r') do |f|
  while line = f.gets
      #puts line
  end
end
```
<!-- more -->

读取整个文件

```
lines = File.readlines('test.txt') #存入数组中
#puts lines
```

写入数据

```
File.open('test.txt','w') do |f|
  f.puts 'test99999';
  f.puts "ok come on" #这样会插入两行数据
end
```

遍历目录 使用 Find模块

```
require 'find'
Find.find("~/Box\ Documents/Ruby") do |f| 
  type = case
      when File.file?(f) then 'File' #是不是文件
      when File.directory?(f) then 'Dir' #是不是目录
      else '?'
      end
  #puts "#{type}: #{f}"
end
```

文件访问

```
f = File.new('test.txt')
f.seek(2,IO::SEEK_SET) #访问位置
puts f.readline
=begin

IO::SEEK_CUR -  从当前位置加上第一个参数的位置开始（相对位置）。  
IO::SEEK_END -  从文件尾开始反向读取，位置是第一个参数的绝对值。
IO::SEEK_SET -  从第一个参数给定的位置开始（绝对位置）
```

`IO::SEEK_CUR` ，`IO::SEEK_END` ，`IO::SEEK_SET` 这三个参数是相对于第一个参数的。 如果使用 IO::SEEK_CUR，那就说明第一个参数给的位置是相对位置，真正的 seek 起始点应 该是第一个参数值+当前位置。如果使用 IO::SEEK_SET，那就说明第一个参数给的位置是 绝对位置，真正的 seek 起始点就是第一个参数值的位置。如果使用 IO::SEEK_END，说明要 从文件尾开始方向读取，seek 的起始点就是第一个参数值的绝对值，为什么要说是绝对值， 是因为在这种情况下，第一个参数要给定一个负整数。还有一个算法是用这个负整数+文件 的长度，结果值就是 seek 的起始点位置。
