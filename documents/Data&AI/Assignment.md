# Finish the search problem

> The code address of this article is: [example_01_Assignment](https://github.com/hivandu/practise/blob/master/MicsoftAINine/example_01_Assignment.ipynb)

### Please read the answer below after thinking for yourself

Please using the search policy to implement an agent. 
This agent receives two input, one is @param start station and the other is @param destination. 
Your agent should give the optimal route based on Beijing Subway system. 

<img src="http://jtapi.bendibao.com/ditie/inc/bj/xianluda.gif" alt="图片替换文本" width="900" height="900" align="bottom" />

#### Dataflow: 

##### 1.	Get data from web page.

> a.	Get web page source from: https://baike.baidu.com/item/%E5%8C%97%E4%BA%AC%E5%9C%B0%E9%93%81/408485

> b.	You may need @package **requests** https://2.python-requests.org/en/master/ page to get the response via url

> c.	You may need save the page source to file system.

> d.	The target of this step is to get station information of all the subway lines;

> e.	You may need install @package beautiful soup https://www.crummy.com/software/BeautifulSoup/bs4/doc/  to get the url information, or just use > Regular Expression to get the url.  Our recommendation is that using the Regular Expression and BeautiflSoup both. 

> f.	You may need BFS to get all the related page url from one url. 
Question: Why do we use BFS to traverse web page (or someone said, build a web spider)?  Can DFS do this job? which is better? 

##### 2.	Preprocessing data from page source.

> a.	Based on the page source gotten from url. You may need some more preprocessing of the page. 

> b.	the Regular Expression you may need to process the text information.

> c.	You may need @package networkx, @package matplotlib to visualize data. 

> d.	You should build a dictionary or graph which could represent the connection information of Beijing subway routes. 

> e.	You may need the defaultdict, set data structures to implement this procedure. 

##### 3. Build the search agent

> Build the search agent based on the graph we build.

for example, when you run: 

```python3
>>> search('奥体中心', '天安门') 
```
you need get the result: 

奥体中心-> A -> B -> C -> ... -> 天安门

#### HTTP协议
超文本传输协议（HTTP，HyperText Transfer Protocol）是互联网上应用最为广泛的一种网络协议。所有的www文件都必须遵守这个标准。  

HTTP用于客户端和服务器之间的通信。协议中规定了客户端应该按照什么格式给服务器发送请求，同时也约定了服务端返回的响应结果应该是什么格式。    

请求访问文本或图像等信息的一端称为客户端，而提供信息响应的一端称为服务器端。 

客户端告诉服务器请求访问信息的方法：
- Get 获得内容
- Post 提交表单来爬取需要登录才能获得数据的网站
- put 传输文件  

更多参考：
[HTTP请求状态](https://www.runoob.com/http/http-status-codes.html)  
了解200 404 503
 - 200 OK      //客户端请求成功
 - 404 Not Found  //请求资源不存在，eg：输入了错误的URL
 - 503 Server Unavailable  //服务器当前不能处理客户端的请求，一段时间后可能恢复正常。

 #### Requests
纯粹HTML格式的网页通常被称为静态网页，静态网页的数据比较容易获取。   
在静态网页抓取中，有一个强大的Requests库能够让你轻易地发送HTTP请求。  

#### 在终端上安装 Requests

pip install requents

```python
# 获取响应内容

import requests

# get（输入你想要抓去的网页地址）
r = requests.get('https://www.baidu.com/')

print('文本编码：（服务器使用的文本编码）', r.encoding)

print('响应状态码：（200表示成功）', r.status_code)

print('字符串方式的响应体：（服务器响应的内容）', r.text)
```

#### 拓展知识：
- [Unicode和UTF-8有什么区别?(盛世唐朝回答)](https://www.zhihu.com/question/23374078)

###  正则表达式
正则表达式的思想是你在人群中寻找你的男朋友/女朋友，他/她在你心里非常有特点。   
同样，从一堆文本中找到需要的内容，我们可以采用正则表达式。

正经点说，是以一定的模式来进行字符串的匹配。   
掌握正则表达式需要非常多的时间，我们可以先入门，在以后的工作中遇到，可更加深入研究。

使用正则表达式有如下步骤：

- 寻找【待找到的信息】特点
- 使用符号找到特点
- 获得信息

```python
# 请先运行一下、看一下有什么参数？
# 请思考，找到会返回什么？没找到会返回什么？

import re
help(re.match)

# 请运行之后、思考 match 与 search 的区别?

m = re.search('foo', 'seafood')
print(m)
print(m.group())

print('-------------------------')

m = re.match('foo', 'seafood')
print(m)

#### `search`是搜索字符串中首次出现的位置

# 匹配多个字符串 |
m = re.match('bat|bet|bit', 'bat')
print(m.group()) if m is not None else print('None')


# 匹配任意单个字符 .
m = re.match('.end', 'kend')
print(m.group()) if m is not None else print('None')

m = re.match('.end', 'end')
print(m.group()) if m is not None else print('None')


# 字符串集合 []
m = re.match('[cr][23][dp][o2]', 'c3p2')
print(m.group()) if m is not None else print('None')


# []   与 |是不同的
m = re.match('c3po|r2d2', 'c3p2')
print(m.group()) if m is not None else print('None')
```

#### 给大家提供一个字典，供大家查询～

<table class="wikitable">
  <tbody>
    <tr>
      <th style="text-align:center;" width="20%">字符</th>
      <th style="text-align:center;" width="90%">描述</th>
    </tr>
    <tr>
      <th style="text-align:center;">\</th>
      <td style="text-align:left;"> 将下一个字符标记为一个特殊字符、或一个原义字符、或一个向后引用、或一个八进制转义符。例如，“<code>n</code>”匹配字符“<code>n</code>”。“<code>\n</code>”匹配一个换行符。串行“<code>\\</code>”匹配“<code>\</code>”而“<code>\(</code>”则匹配“<code>(</code>”。</td>
    </tr>
    <tr>
      <th style="text-align:center;">^</th>
      <td style="text-align:left;">匹配输入字符串的开始位置。如果设置了RegExp对象的Multiline属性，^也匹配“<code>\n</code>”或“<code>\r</code>”之后的位置。</td>
    </tr>
    <tr>
      <th style="text-align:center;"> \* </th>
      <td style="text-align:left;">匹配前面的子表达式零次或多次。例如，zo\*能匹配“<code>z</code>”以及“<code>zoo</code>”。\* 等价于{0,}。</td>
    </tr>
    <tr>
      <th style="text-align:center;">+</th>
      <td style="text-align:left;">匹配前面的子表达式一次或多次。例如，“<code>zo+</code>”能匹配“<code>zo</code>”以及“<code>zoo</code>”，但不能匹配“<code>z</code>”。+等价于{1,}。</td>
    </tr>
    <tr>
      <th style="text-align:center;">?</th>
      <td style="text-align:left;">匹配前面的子表达式零次或一次。例如，“<code>do(es)?</code>”可以匹配“<code>does</code>”或“<code>does</code>”中的“<code>do</code>”。?等价于{0,1}。</td>
    </tr>
    <tr>
      <th style="text-align:center;">{<span style="font-family:Times New Roman; font-style:italic;">n</span>}</th>
      <td style="text-align:left;"><span style="font-family:Times New Roman; font-style:italic;">n</span>是一个非负整数。匹配确定的<span style="font-family:Times New Roman; font-style:italic;">n</span>次。例如，“<code>o{2}</code>”不能匹配“<code>Bob</code>”中的“<code>o</code>”，但是能匹配“<code>food</code>”中的两个o。</td>
    </tr>
    <tr>
      <th style="text-align:center;">{<span style="font-family:Times New Roman; font-style:italic;">n</span>,}</th>
      <td style="text-align:left;"><span style="font-family:Times New Roman; font-style:italic;">n</span>是一个非负整数。至少匹配<span style="font-family:Times New Roman; font-style:italic;">n</span>次。例如，“<code>o{2,}</code>”不能匹配“<code>Bob</code>”中的“<code>o</code>”，但能匹配“<code>foooood</code>”中的所有o。“<code>o{1,}</code>”等价于“<code>o+</code>”。“<code>o{0,}</code>”则等价于“<code>o*</code>”。</td>
    </tr>
    <tr>
      <th style="text-align:center;">{<span style="font-family:Times New Roman; font-style:italic;">n</span>,<span style="font-family:Times New Roman; font-style:italic;">m</span>}</th>
      <td style="text-align:left;"><span style="font-family:Times New Roman; font-style:italic;">m</span>和<span style="font-family:Times New Roman; font-style:italic;">n</span>均为非负整数，其中<span style="font-family:Times New Roman; font-style:italic;">n</span>&lt;=<span style="font-family:Times New Roman; font-style:italic;">m</span>。最少匹配<span style="font-family:Times New Roman; font-style:italic;">n</span>次且最多匹配<span style="font-family:Times New Roman; font-style:italic;">m</span>次。例如，“<code>o{1,3}</code>”将匹配“<code>fooooood</code>”中的前三个o。“<code>o{0,1}</code>”等价于“<code>o?</code>”。请注意在逗号和两个数之间不能有空格。</td>
    </tr>
    <tr>
      <th style="text-align:center;">?</th>
      <td style="text-align:left;">当该字符紧跟在任何一个其他限制符（*,+,?，{<span style="font-family:Times New Roman; font-style:italic;">n</span>}，{<span style="font-family:Times New Roman; font-style:italic;">n</span>,}，{<span style="font-family:Times New Roman; font-style:italic;">n</span>,<span style="font-family:Times New Roman; font-style:italic;">m</span>}）后面时，匹配模式是非贪婪的。非贪婪模式尽可能少的匹配所搜索的字符串，而默认的贪婪模式则尽可能多的匹配所搜索的字符串。例如，对于字符串“<code>oooo</code>”，“<code>o+?</code>”将匹配单个“<code>o</code>”，而“<code>o+</code>”将匹配所有“<code>o</code>”。</td>
    </tr>
    <tr>
      <th style="text-align:center;">.</th>
      <td style="text-align:left;">匹配除“<code>\</code><span style="font-family:Times New Roman; font-style:italic;"><code>n</code></span>”之外的任何单个字符。要匹配包括“<code>\</code><span style="font-family:Times New Roman; font-style:italic;"><code>n</code></span>”在内的任何字符，请使用像“<code>(.|\n)</code>”的模式。</td>
    </tr>
    <tr>
      <th style="text-align:center;">(pattern)</th>
      <td style="text-align:left;">匹配pattern并获取这一匹配。所获取的匹配可以从产生的Matches集合得到，在VBScript中使用SubMatches集合，在JScript中则使用$0…$9属性。要匹配圆括号字符，请使用“<code>\(</code>”或“<code>\)</code>”。</td>
    </tr>
    <tr>
      <th style="text-align:center;">(?:pattern)</th>
      <td style="text-align:left;">匹配pattern但不获取匹配结果，也就是说这是一个非获取匹配，不进行存储供以后使用。这在使用或字符“<code>(|)</code>”来组合一个模式的各个部分是很有用。例如“<code>industr(?:y|ies)</code>”就是一个比“<code>industry|industries</code>”更简略的表达式。</td>
    </tr>
    <tr>
      <th style="text-align:center;">(?=pattern)</th>
      <td style="text-align:left;">正向肯定预查，在任何匹配pattern的字符串开始处匹配查找字符串。这是一个非获取匹配，也就是说，该匹配不需要获取供以后使用。例如，“<code>Windows(?=95|98|NT|2000)</code>”能匹配“<code>Windows2000</code>”中的“<code>Windows</code>”，但不能匹配“<code>Windows3.1</code>”中的“<code>Windows</code>”。预查不消耗字符，也就是说，在一个匹配发生后，在最后一次匹配之后立即开始下一次匹配的搜索，而不是从包含预查的字符之后开始。</td>
    </tr>
    <tr>
      <th style="text-align:center;">(?!pattern)</th>
      <td style="text-align:left;">正向否定预查，在任何不匹配pattern的字符串开始处匹配查找字符串。这是一个非获取匹配，也就是说，该匹配不需要获取供以后使用。例如“<code>Windows(?!95|98|NT|2000)</code>”能匹配“<code>Windows3.1</code>”中的“<code>Windows</code>”，但不能匹配“<code>Windows2000</code>”中的“<code>Windows</code>”。预查不消耗字符，也就是说，在一个匹配发生后，在最后一次匹配之后立即开始下一次匹配的搜索，而不是从包含预查的字符之后开始</td>
    </tr>
    <tr>
      <th style="text-align:center;">(?&lt;=pattern)</th>
      <td style="text-align:left;">反向肯定预查，与正向肯定预查类拟，只是方向相反。例如，“<code>(?&lt;=95|98|NT|2000)Windows</code>”能匹配“<code>2000Windows</code>”中的“<code>Windows</code>”，但不能匹配“<code>3.1Windows</code>”中的“<code>Windows</code>”。</td>
    </tr>
    <tr>
      <th style="text-align:center;">(?&lt;!pattern)</th>
      <td style="text-align:left;">反向否定预查，与正向否定预查类拟，只是方向相反。例如“<code>(?&lt;!95|98|NT|2000)Windows</code>”能匹配“<code>3.1Windows</code>”中的“<code>Windows</code>”，但不能匹配“<code>2000Windows</code>”中的“<code>Windows</code>”。</td>
    </tr>
    <tr>
      <th style="text-align:center;">x|y</th>
      <td style="text-align:left;">匹配x或y。例如，“<code>z|food</code>”能匹配“<code>z</code>”或“<code>food</code>”。“<code>(z|f)ood</code>”则匹配“<code>zood</code>”或“<code>food</code>”。</td>
    </tr>
    <tr>
      <th style="text-align:center;">[xyz]</th>
      <td style="text-align:left;">字符集合。匹配所包含的任意一个字符。例如，“<code>[abc]</code>”可以匹配“<code>plain</code>”中的“<code>a</code>”。</td>
    </tr>
    <tr>
      <th style="text-align:center;">[^xyz]</th>
      <td style="text-align:left;">负值字符集合。匹配未包含的任意字符。例如，“<code>[^abc]</code>”可以匹配“<code>plain</code>”中的“<code>p</code>”。</td>
    </tr>
    <tr>
      <th style="text-align:center;">[a-z]</th>
      <td style="text-align:left;">字符范围。匹配指定范围内的任意字符。例如，“<code>[a-z]</code>”可以匹配“<code>a</code>”到“<code>z</code>”范围内的任意小写字母字符。</td>
    </tr>
    <tr>
      <th style="text-align:center;">[^a-z]</th>
      <td style="text-align:left;">负值字符范围。匹配任何不在指定范围内的任意字符。例如，“<code>[^a-z]</code>”可以匹配任何不在“<code>a</code>”到“<code>z</code>”范围内的任意字符。</td>
    </tr>
    <tr>
      <th style="text-align:center;">\b</th>
      <td style="text-align:left;">匹配一个单词边界，也就是指单词和空格间的位置。例如，“<code>er\b</code>”可以匹配“<code>never</code>”中的“<code>er</code>”，但不能匹配“<code>verb</code>”中的“<code>er</code>”。</td>
    </tr>
    <tr>
      <th style="text-align:center;">\B</th>
      <td style="text-align:left;">匹配非单词边界。“<code>er\B</code>”能匹配“<code>verb</code>”中的“<code>er</code>”，但不能匹配“<code>never</code>”中的“<code>er</code>”。</td>
    </tr>
    <tr>
      <th style="text-align:center;">\cx</th>
      <td style="text-align:left;">匹配由x指明的控制字符。例如，\cM匹配一个Control-M或回车符。x的值必须为A-Z或a-z之一。否则，将c视为一个原义的“<code>c</code>”字符。</td>
    </tr>
    <tr>
      <th style="text-align:center;">\d</th>
      <td style="text-align:left;">匹配一个数字字符。等价于[0-9]。</td>
    </tr>
    <tr>
      <th style="text-align:center;">\D</th>
      <td style="text-align:left;">匹配一个非数字字符。等价于[^0-9]。</td>
    </tr>
    <tr>
      <th style="text-align:center;">\f</th>
      <td style="text-align:left;">匹配一个换页符。等价于\x0c和\cL。</td>
    </tr>
    <tr>
      <th style="text-align:center;">\n</th>
      <td style="text-align:left;">匹配一个换行符。等价于\x0a和\cJ。</td>
    </tr>
    <tr>
      <th style="text-align:center;">\r</th>
      <td style="text-align:left;">匹配一个回车符。等价于\x0d和\cM。</td>
    </tr>
    <tr>
      <th style="text-align:center;">\s</th>
      <td style="text-align:left;">匹配任何空白字符，包括空格、制表符、换页符等等。等价于[ \f\n\r\t\v]。</td>
    </tr>
    <tr>
      <th style="text-align:center;">\S</th>
      <td style="text-align:left;">匹配任何非空白字符。等价于[^ \f\n\r\t\v]。</td>
    </tr>
    <tr>
      <th style="text-align:center;">\t</th>
      <td style="text-align:left;">匹配一个制表符。等价于\x09和\cI。</td>
    </tr>
    <tr>
      <th style="text-align:center;">\v</th>
      <td style="text-align:left;">匹配一个垂直制表符。等价于\x0b和\cK。</td>
    </tr>
    <tr>
      <th style="text-align:center;">\w</th>
      <td style="text-align:left;">匹配包括下划线的任何单词字符。等价于“<code>[A-Za-z0-9_]</code>”。</td>
    </tr>
    <tr>
      <th style="text-align:center;">\W</th>
      <td style="text-align:left;">匹配任何非单词字符。等价于“<code>[^A-Za-z0-9_]</code>”。</td>
    </tr>
    <tr>
      <th style="text-align:center;">\x<span style="font-family:Times New Roman; font-style:italic;">n</span></th>
      <td style="text-align:left;">匹配<span style="font-family:Times New Roman; font-style:italic;">n</span>，其中<span style="font-family:Times New Roman; font-style:italic;">n</span>为十六进制转义值。十六进制转义值必须为确定的两个数字长。例如，“<code>\x41</code>”匹配“<code>A</code>”。“<code>\x041</code>”则等价于“<code>\x04&amp;1</code>”。正则表达式中可以使用ASCII编码。.</td>
    </tr>
    <tr>
      <th style="text-align:center;">\<span style="font-family:Times New Roman; font-style:italic;">num</span></th>
      <td style="text-align:left;">匹配<span style="font-family:Times New Roman; font-style:italic;">num</span>，其中<span style="font-family:Times New Roman; font-style:italic;">num</span>是一个正整数。对所获取的匹配的引用。例如，“<code>(.)\1</code>”匹配两个连续的相同字符。</td>
    </tr>
    <tr>
      <th style="text-align:center;">\<span style="font-family:Times New Roman; font-style:italic;">n</span></th>
      <td style="text-align:left;">标识一个八进制转义值或一个向后引用。如果\<span style="font-family:Times New Roman; font-style:italic;">n</span>之前至少<span style="font-family:Times New Roman; font-style:italic;">n</span>个获取的子表达式，则<span style="font-family:Times New Roman; font-style:italic;">n</span>为向后引用。否则，如果<span style="font-family:Times New Roman; font-style:italic;">n</span>为八进制数字（0-7），则<span style="font-family:Times New Roman; font-style:italic;">n</span>为一个八进制转义值。</td>
    </tr>
    <tr>
      <th style="text-align:center;">\<span style="font-family:Times New Roman; font-style:italic;">nm</span></th>
      <td style="text-align:left;">标识一个八进制转义值或一个向后引用。如果\<span style="font-family:Times New Roman; font-style:italic;">nm</span>之前至少有<span style="font-family:Times New Roman; font-style:italic;">nm</span>个获得子表达式，则<span style="font-family:Times New Roman; font-style:italic;">nm</span>为向后引用。如果\<span style="font-family:Times New Roman; font-style:italic;">nm</span>之前至少有<span style="font-family:Times New Roman; font-style:italic;">n</span>个获取，则<span style="font-family:Times New Roman; font-style:italic;">n</span>为一个后跟文字<span style="font-family:Times New Roman; font-style:italic;">m</span>的向后引用。如果前面的条件都不满足，若<span style="font-family:Times New Roman; font-style:italic;">n</span>和<span style="font-family:Times New Roman; font-style:italic;">m</span>均为八进制数字（0-7），则\<span style="font-family:Times New Roman; font-style:italic;">nm</span>将匹配八进制转义值<span style="font-family:Times New Roman; font-style:italic;">nm</span>。</td>
    </tr>
    <tr>
      <th style="text-align:center;">\<span style="font-family:Times New Roman; font-style:italic;">nml</span></th>
      <td style="text-align:left;">如果<span style="font-family:Times New Roman; font-style:italic;">n</span>为八进制数字（0-3），且<span style="font-family:Times New Roman; font-style:italic;">m和l</span>均为八进制数字（0-7），则匹配八进制转义值<span style="font-family:Times New Roman; font-style:italic;">nm</span>l。</td>
    </tr>
    <tr>
      <th style="text-align:center;">\u<span style="font-family:Times New Roman; font-style:italic;">n</span></th>
      <td style="text-align:left;">匹配<span style="font-family:Times New Roman; font-style:italic;">n</span>，其中<span style="font-family:Times New Roman; font-style:italic;">n</span>是一个用四个十六进制数字表示的Unicode字符。例如，\u00A9匹配版权符号（©）。</td>
    </tr>
  </tbody>
</table>

```python
# 匹配电子邮件地址
patt = '\w+@(\w+\.)?\w+\.com'
m = re.match(patt, 'nobody@xxx.com')
print(m.group()) if m is not None else print('None')

# 匹配QQ
m = re.search('[1-9][0-9]{4,}', '这是我的QQ号781504542,第二个qq号：10054422288')
print(m.group()) if m is not None else print('None')

# findall() 是search的升级版，可以找到所有匹配的字符串

m = re.findall('[1-9][0-9]{4,}', '这是我的QQ号781504542,第二个qq号：10054422288')
print(m) if m is not None else print('None')
```

了解了怎么使用，下面进入实现

```python
# get the data (subway for beijing ,from amap)
# 你需要用到以下的包

import requests
import re
import numpy as np
r = requests.get('http://map.amap.com/service/subway?_1469083453978&srhdata=1100_drw_beijing.json')
r.text

def get_lines_stations_info(text):
    # Please write your code here
    pass

    # Traverse the text format data to form the location data structure
    # Dict of all line information: key: line name; value: list of site names
    lines_info = {}
    
    # A dict of all site information: key: site name; value: site coordinates (x, y)
    stations_info = {}
    
    for i in range(len(lines_list)):
        # Several questions you may need to think about, get "Metro line name, station information list, station name, coordinates (x, y), add data to the information dict of the station, add data to the subway line dict"
        pass

lines_info, stations_info = get_lines_stations_info(r.text)

# According to the route information, establish the site adjacency table dict
def get_neighbor_info(lines_info):
    pass

    # Add str2 to the adjacency list of site str1
    def add_neighbor_dict(info, str1, str2):
        # Please write code here
        pass
        
    return neighbor_info
        
neighbor_info = get_neighbor_info(lines_info)
neighbor_info

# Draw subway map
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt

# If Chinese characters cannot be displayed, please refer to
matplotlib.rcParams['font.sans-serif'] = ['SimHei']

# matplotlib.rcParams['font.family']='sans-serif'


# You can use recursion to find all paths
def get_path_DFS_ALL(lines_info, neighbor_info, from_station, to_station):
    # Recursive algorithm, essentially depth first
    # Traverse all paths
    # In this case, the coordinate distance between the sites is difficult to transform into a reliable heuristic function, so only a simple BFS algorithm is used
    # Check input site name
    pass

def get_next_station_DFS_ALL(node, neighbor_info, to_station):
    pass

# You can also use the second algorithm: simple breadth first without heuristic function

def get_path_BFS(lines_info, neighbor_info, from_station, to_station):
    # Search strategy: take the number of stations as the cost (because the ticket price is calculated by station)
    # In this case, the coordinate distance between the sites is difficult to transform into a reliable heuristic function, so only a simple BFS algorithm is used
    # Since the cost of each layer is increased by 1, the cost of each layer is the same, and it does not matter whether it is calculated or not, so it is omitted
    # Check input site name
    pass

# You can also use the third algorithm: heuristic search with path distance as the cost

import pandas as pd
def get_path_Astar(lines_info, neighbor_info, stations_info, from_station, to_station):
    # Search strategy: the straight-line distance between the stations of the route is accumulated as the cost, and the straight-line distance from the current station to the target is used as the heuristic function
    # Check input site name
    pass

```

As much as you can to use the already implemented search agent. You just need to define the **is_goal()**, **get_successor()**, **strategy()** three functions. 

> a.	Define different policies for transfer system. 

> b.	Such as Shortest Path Priority（路程最短优先）, Minimum Transfer Priority(最少换乘优先), Comprehensive Priority(综合优先)

> c.	Implement Continuous transfer. Based on the Agent you implemented, please add this feature: Besides the @param start and @param destination two stations, add some more stations, we called @param by_way, it means, our path should from the start and end, but also include the  @param by_way stations. 

e.g 
```
1. Input:  start=A,  destination=B, by_way=[C] 
    Output: [A, … .., C, …. B]
2. Input: start=A, destination=B, by_way=[C, D, E]
    Output: [A … C … E … D … B]  
    # based on your policy, the E station could be reached firstly. 
```

## The Answer

```python


# get the data (subway for beijing ,from amap)
import requests
import re
import numpy as np
r = requests.request('GET', url = 'http://map.amap.com/service/subway?_1469083453978&srhdata=1100_drw_beijing.json')

def get_lines_stations_info(text):

    lines_info = {}
    stations_info = {}

    pattern = re.compile('"st".*?"kn"')
    lines_list = pattern.findall(text)

    for i in range(len(lines_list)):
        pattern = re.compile('"ln":".*?"')
        line_name = pattern.findall(lines_list[i])[0][6:-1]

        pattern = re.compile('"rs".*?"sp"')
        temp_list = pattern.findall(lines_list[i])
        station_name_list = []

        for j in range(len(temp_list)):

            pattern = re.compile('"n":".*?"')
            station_name = pattern.findall(temp_list[j])[0][5:-1]
            station_name_list.append(station_name)

            pattern = re.compile('"sl":".*?"')
            position = tuple(map(float, pattern.findall(temp_list[j])[0][6:-1].split(',')))
            
            stations_info[station_name] = position

        lines_info[line_name]  = station_name_list

    return lines_info, stations_info

lines_info, stations_info = get_lines_stations_info(r.text)
# print(stations_info)
# print(lines_info)

len(lines_info)

def get_neighbor_info(lines_info):
    def add_neighbor_dict(info, str1, str2):
        list1 = info.get(str1)
        if not list1:
            list1 = []
        list1.append(str2)
        info[str1] = list1
        return info

    neighbor_info = {}

    for line_name, station_list in lines_info.items():
        for i in range(len(station_list) -1):
            sta1 = station_list[i]
            sta2 = station_list[i+1]

            neighbor_info = add_neighbor_dict(neighbor_info, sta1, sta2)
            neighbor_info = add_neighbor_dict(neighbor_info, sta2, sta1)
    return neighbor_info

neighbor_info = get_neighbor_info(lines_info)
print(neighbor_info)

import networkx as nx
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif']  = ['Arial Unicode MS']
matplotlib.rcParams['font.size'] = 2

plt.figure(figsize = (20, 20))
stations_graph = nx.Graph()
stations_graph.add_nodes_from(list(stations_info.keys()))
nx.draw(stations_graph, stations_info, with_labels = True, node_size = 5)


stations_connection_graph = nx.Graph(neighbor_info)
plt.figure(figsize = (30, 30))
nx.draw(stations_connection_graph, stations_info, with_labels = True, node_size = 5)


# The first algorithm: recursively find all paths
def get_path_DFS_ALL(lines_info, neighbor_info, from_station, to_station):
    # # Recursive algorithm, essentially depth first
    # Traverse all paths
    # In this case, the coordinate distance between the sites is difficult to transform into a reliable heuristic function, so only a simple BFS algorithm is used
    # Check input site name
    if not neighbor_info.get(from_station):
        print('起始站点“%s”不存在。请正确输入！'%from_station)
        return None
    if not neighbor_info.get(to_station):
        print('目的站点“%s”不存在。请正确输入！'%to_station)
        return None
    path = []
    this_station = from_station
    path.append(this_station)
    neighbors = neighbor_info.get(this_station)
    node = {'pre_station':'',
            'this_station':this_station,
            'neighbors':neighbors,
            'path':path}
    
    return get_next_station_DFS_ALL(node, neighbor_info, to_station)

def get_next_station_DFS_ALL(node, neighbor_info, to_station):
    neighbors = node.get('neighbors')
    pre_station = node.get('this_station')
    path = node.get('path')
    paths = []
    for i in range(len(neighbors)):
        this_station = neighbors[i]
        if (this_station in path):
            
            # If this station is already in the path, it means a loop, and this road is unreachable
            return None
        if neighbors[i] == to_station:
            
            # Find the end, return to the path
            path.append(to_station)
            paths.append(path)
            return paths
        else:
            neighbors_ = neighbor_info.get(this_station).copy()
            neighbors_.remove(pre_station)
            path_ = path.copy()
            path_.append(this_station)
            new_node = {'pre_station':pre_station,
                        'this_station':this_station,
                        'neighbors':neighbors_,
                        'path':path_}
            paths_ =  get_next_station_DFS_ALL(new_node, neighbor_info, to_station)
            if paths_:
                paths.extend(paths_)

    return paths

paths = get_path_DFS_ALL(lines_info, neighbor_info, '回龙观', '西二旗')
print('共有%d种路径。'%len(paths))
for item in paths:
    print("此路径总计%d站:"%(len(item)-1))
    print('-'.join(item))


# The second algorithm: simple breadth first without heuristic function
def get_path_BFS(lines_info, neighbor_info, from_station, to_station):
    
    # Search strategy: take the number of stations as the cost (because the ticket price is calculated by station)
    # In this case, the coordinate distance between the sites is difficult to transform into a reliable heuristic function, so only a simple BFS algorithm is used
    # Since the cost of each layer is increased by 1, the cost of each layer is the same, and it does not matter whether it is calculated or not, so it is omitted
    # Check input site name
    if not neighbor_info.get(from_station):
        print('起始站点“%s”不存在。请正确输入！'%from_station)
        return None
    if not neighbor_info.get(to_station):
        print('目的站点“%s”不存在。请正确输入！'%to_station)
        return None
    
    # The search node is a dict, key=site name, value is a list of sites that contain passing
    nodes = {}
    nodes[from_station] = [from_station]
    
    while True:
        new_nodes = {}
        for (k,v) in nodes.items():
            neighbor = neighbor_info.get(k).copy()
            if (len(v) >= 2):
                
                # Do not go to the previous stop
                pre_station = v[-2]
                neighbor.remove(pre_station)
            for station in neighbor:
                
                # Traverse neighbors
                if station in nodes:

                    # Skip the nodes that have been searched
                    continue
                path = v.copy()
                path.append(station)
                new_nodes[station] = path
                if station == to_station:

                    # Find the path, end
                    return path
        nodes = new_nodes
        
    print('未能找到路径')
    return None

paths = get_path_BFS(lines_info, neighbor_info, '回龙观', '西二旗')
print("路径总计%d站。"%(len(paths)-1))
print("-".join(paths))

# Gaode Navigation is 31 stations, only 1 transfer
# The result of the code is 28 stations, but there are 5 transfers
# Guess Gaode's path cost is mainly time


# The third algorithm: heuristic search with path distance as the cost
import pandas as pd
def get_path_Astar(lines_info, neighbor_info, stations_info, from_station, to_station):
    
    # Search strategy: the straight-line distance between the stations of the route is accumulated as the cost, and the straight-line distance from the current station to the target is used as the heuristic function
    # Check input site name
    if not neighbor_info.get(from_station):
        print('起始站点“%s”不存在。请正确输入！'%from_station)
        return None
    if not neighbor_info.get(to_station):
        print('目的站点“%s”不存在。请正确输入！'%to_station)
        return None
    
    # Calculate the straight-line distance from all nodes to the target node, spare
    distances = {}
    x,y = stations_info.get(to_station)
    for (k,v) in stations_info.items():
        x0,y0 = stations_info.get(k)
        l = ((x-x0)**2 + (y-y0)**2)**0.5
        distances[k] = l
        
    # Nodes that have been searched, dict
    # key=site name, value is the minimum cost from a known starting point to this site    # 已搜索过的节点，dict
    searched = {}
    searched[from_station] = 0
    
    # The data structure is pandas dataframe
    # index is the site name
    # g is the path taken, h is the heuristic function value (the current straight-line distance to the target)
    nodes = pd.DataFrame([[[from_station], 0, 0, distances.get(from_station)]],
                         index=[from_station], columns=['path', 'cost', 'g', 'h']) 
    
    count = 0
    while True:
        if count > 1000:
            break
        nodes.sort_values('cost', inplace=True)
        for index, node in nodes.iterrows():
            count += 1
            
            # Search for the site that is the shortest from the destination among the neighbors
            neighbors = neighbor_info.get(index).copy()
            if len(node['path']) >= 2:
                
                # Do not search in the reverse direction of this path
                neighbors.remove(node['path'][-2])
            for i in range(len(neighbors)):
                count += 1
                neighbor = neighbors[i]
                g = node['g'] + get_distance(stations_info, index, neighbor)
                h = distances[neighbor]
                cost = g + h
                path = node['path'].copy()
                path.append(neighbor)
                if neighbor == to_station:
                    # Find the goal, end
                    print('共检索%d次。'%count)
                    return path
                if neighbor in searched:
                    if g >= searched[neighbor]:
                        # Explain that the search path is not optimal, ignore it
                        continue
                    else:
                        searched[neighbor] = g
                        # Modify the node information corresponding to this site
#                         nodes.loc[neighbor, 'path'] = path # 这行总是报错
#                         nodes.loc[neighbor, 'cost'] = cost
#                         nodes.loc[neighbor, 'g'] = g
#                         nodes.loc[neighbor, 'h'] = h
                        # I don’t know how to modify the list element in df, I can only delete and add new rows
                        nodes.drop(neighbor, axis=0, inplace=True)
                        row = pd.DataFrame([[path, cost, g, h]],
                                       index=[neighbor], columns=['path', 'cost', 'g', 'h'])
                        nodes = nodes.append(row)
                        
                else:
                    searched[neighbor] = g
                    row = pd.DataFrame([[path, cost, g, h]],
                                       index=[neighbor], columns=['path', 'cost', 'g', 'h'])
                    nodes = nodes.append(row)
            # All neighbors of this site have been searched, delete this node
            nodes.drop(index, axis=0, inplace=True)

        # The outer for loop only runs the first row of data, and then re-sort and then calculate
        continue         
        
    print('未能找到路径')
    return None

def get_distance(stations_info, str1, str2):
    x1,y1 = stations_info.get(str1)
    x2,y2 = stations_info.get(str2)
    return ((x1-x2)**2 + (y1-y2)**2)** 0.5

paths = get_path_Astar(lines_info, neighbor_info, stations_info, '回龙观', '西二旗')
if paths:
    print("路径总计%d站。"%(len(paths)-1))
    print("-".join(paths))

# Gaode Navigation is 31 stations, only 1 transfer
# The code result is 28 stations, which is the same as the result with the number of subway stations as the cost, but the path is different (from the first traversal algorithm, you can see that there are 3 paths for 28 stations to reach the destination)
# Guess Gaode's path cost is mainly time
```