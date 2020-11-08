# Ruby 中文错误-ASCII

才开始学习Ruby,当然免不了诸如此类的错误!未免忘记,还是记录一下!好记性不如烂笔头...

编译含有中文字符的.rb文件会出现这种错误,原因是因为Ruby1.9是用ASCII编码来读源码的. 解决办法就是再源文件的第一行加上 `# encoding:utf-8`

或者再执行文件的时候执行如下代码: `ruby -Ku test.rb (Mac下) ruby -Ks test.rb` (Win下)