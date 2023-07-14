# 使用WAMP5搭建Apache+MySQL+PHP环境

目前有不少AMP（Apache\MySQL\PHP）的集成软件，可以让我们一次安装并设置好。这对于不熟悉AMP的用户来说，好处多多。


**一、使用AMP集成软件的优点**：
1、可避免由于缺乏AMP的知识，而无法正确设置环境；
2、可快速安装并设置好AMP环境，让我们直接开始真正感兴趣的软件，如xoops；
3、可方便的搭建测试环境，对于测试“是AMP环境问题，还是XOOPS造成的问题”很有帮助，采用排除法即可。

**二、常用的AMP集成软件**：
1、AppServ：[http://www.appservnetwork.com/](http://www.appservnetwork.com/)
这个软件在台湾很流行。看到不少书籍也极力推荐，估计都是受台湾用户的影响。

2、XAMPP：[http://www.apachefriends.org/en](http://www.apachefriends.org/en)
这个软件支持多个平台,Win\Linux\Solaris\Mac OS X，目前也有不少人使用。

**3、WAMP5：[http://www.wampserver.com/en/](http://www.wampserver.com/en/)**
这是我今天极力推荐的，绝对五星级。注意它的名字是带个5的哦，意思就是WAMP5使用最新的PHP5版本，正如官方网站上的口号：Discover PHP5 with WAMP5 !

**三、根据我的经验，WAMP5有如下优点**：
1、XOOPS在WAMP5中使用，没有任何问题。
2、WAMP5专注于Windows平台，安装设置及其简单。
3、PHP默认的是5.x版本，如果需要php4.x，只要安装php4.x插件，就可以在两者之间自由的切换，非常方便。
4、MySQL默认的是5.x版本，但可以通过选择老版本的WAMP5，从而使用4.x的MySQL。
5、可视化的菜单管理，极其方便。如，打开关闭php extention、Apache module等，直接通过菜单选择就可以。
6、还有各种插件，如ZEND OPTIMIZER ADD-ON等。
7、如果有疑问，官方还有论坛可以求助。
……更多优点：谁用谁知道，早用早知道^_^

**四、使用WAMP5的经验、技巧**
1、安装时的设置：可以自定义WWW根目录的存放位置哦，强烈建议放到D盘等安全的分区中，以避免万一系统崩溃，造成数据丢失。

2、对于中文用户来说，安装结束后，首先要设置的是，把MySQL的数据库默认编码改为UTF-8，这样可以排除很多中文乱码问题：在WAMP5菜单中选择打开`my(wamp).ini`，设置其中的`default-character-set=utf8`, 然后重启WAMP5。

3、数据库默认的密码是空的，可以在phpMyAdmin中设置root帐号的密码为123456；当然修改之后，就要跟着修改phpMyAdmin的配置文件config.inc.php，否则phpMyAdmin就进不了数据库啦：
>$cfg[’Servers’][$i][’user’] = ‘root’;
>$cfg[’Servers’][$i][’password’] = ‘123456′;

4、如果需要mysql4.x + php4.x，可选用WAMP5 1.44版本以及插件PHP4.3.11；

5、由于MySQL4.1之后版本对密码验证的方法发生了改动，如果在WAMP5中使用php4.x，那么就需要启用MySQL的old password功能，否则无法登陆phpMyAdmin。在WAMP5菜单中选择`MySQL/Mysql console`，然后输入下列命令：
>mysql&gt; SET PASSWORD FOR
>-&gt; ‘root’@'localhost’ = OLD_PASSWORD(’123456′);

6、XOOPS用户关心的时区问题：WAMP5默认的时区是格林威治标准时间(GMT)，这就意味着WAMP5默认的服务器时区是GMT，但我们可以更改服务器默认时区，以对应北京时间。打开WAMP5菜单中的php.ini，在文档最后添加如下代码即可：
>[Date]
>; Defines the default timezone used by the date functions
>date.timezone = “Asia/Shanghai”

如果你不在北京，那么就改动上述的设置即可，具体设置值可参考：[http://us2.php.net/manual/en/timezones.php](http://us2.php.net/manual/en/timezones.php)

通过上述WAMP5的设置之后，我们在XOOPS中的时区设置就可以这样：服务器时区、默认时区、个人帐号的时区这三者都设置为上述的时区就可以了，如：北京时间（GMT+8）

OK，这就是所有的AMP设置秘籍！！
