---
layout: post
title: "Apache+PHP+MySQL+WordPress 本地架设的方法（转载）"
categories: 
- develop
tags: 
- apache 
- mysql 
- php 
- wordpress 
- 架设
date: 2006-07-28 17:21 
comments: true
---
转载自[google搜索](http://www.google.com/search?hl=zh-CN&amp;newwindow=1&amp;client=firefox&amp;rls=org.mozilla%3Azh-CN%3Aofficial&amp;q=%E6%9C%AC%E5%9C%B0%E6%9E%B6%E8%AE%BEphp+mysql+%E8%BD%AF%E4%BB%B6&amp;btnG=%E6%90%9C%E7%B4%A2&amp;lr=)

###系统环境:###
		
<!--more-->
**硬件:**

C1.7GHz + 256MB DDR266 + ST 40GB + 845GL

**软件:**

"Microsoft Windows XP Pro CN with SP2", "Apache 2.053 For Win32 (x86)","MySQL 4.1.10 For Win32 (x86)","PHP 5.0.3 For Win32  ", "MyAdmin 2.6.1-pl3 (patch level 3)"

**步骤详述：** 

注：以下操作假设WinXP操作系统安装于 `C:\windows` ；

将把 apache + php + sql + blog 安装在 `d:\website` 中。

* 一：安装 Apache 并进行配置使其支持 Php

	从 Apache官方网站下载的 Apache 2.053 For Win32 有两种格式，一种是 MSI 的安装文件；一种是 ZIP 压缩包。我选的是 MSI 格式的安装文件。
	运行 apache_2.0.53-win32-x86-no_ssl.msi 文件，然后根据安装向导，将 Apache 安装在 `d:\website\apache` 目录中。
	Apache 安装过程需要输入网站域名，本地调试使用localhost即可，安装过程很简单，全图形化界面，不再赘述。
	PHP官方网站提供两种格式的 Php 5.03 For Win32 下载，一种是压缩成 EXE 的文件；另一种是 ZIP 压缩包。我选择的是 ZIP 压缩包。
	首先将 php-5.0.3-Win32.zip 内的文件解压缩到 `d:\website\php` 目录中。
	然后找到 `d:\website\php\php.ini-dist` 文件，将其重命名为 `php.ini`，并复制到 `c:\windows` 目录里。
	再将 `d:\website\php\ `目录中的 `php5ts.dll` 和 `libMySQL.dll `两个文件，一起复制到 `c:\windows\system` 或 `c:\windows\system32` 目录中。
	用文本编辑软件打开 `d:\apache\apache2\conf\httpd.conf` 文件，首先找到 `DocumentRoot` 一行，将其后的路径修改为 web 服务的主目录，例如：`DocumentRoot &quot;D:/website/public_html&quot;`；
	然后找到 `DirectoryIndex` 一行，在行末加上 `index.htm index.php`，例如：`DirectoryIndex index.html index.html.var index.htm index.php`
	为 Apache 安装 Php 可以从下列两种安装模式中任选其一，建议使用模块化安装。
	仍然是编辑 `d:\apache\conf\httpd.conf` 文件：
	* 模块化安装配置：
	找到 `#LoadModule ssl_module modules/mod_ssl.so` 这行，在此行后增加一行：   `LoadModule php5_module d:/website/php/php5apache2.dll`找到 `AddType application/x-gzip .gz .tgz`，在此行后增加一行：`AddType application/x-httpd-php .php`
	* CGI安装配置：
	找到 `AddType application/x-gzip .gz .tgz`，在此行后增加三行：`ScriptAlias /php/ &quot;d:/website/php/&quot;`, `AddType application/x-httpd-php .php;`,` Action application/x-httpd-php &quot;/php/php-cgi.exe&quot;`
	注：以上两种安装模式中的 `d:/website/php/ `是指 php 5.03 的安装目录路径，请视具体情况更改。</p>  <p> 重新启动 Apache 服务。</p>  <p> 到这里，Apache + Php 环境基本已经配置完成，在 web 根目录（以上例即 d:\website\public_html\ 目录）中，用记事本创建一个 `phpinfo.php` 文件，其内容如下：`<?echo phpinfo();?> `
	然后，在浏览器中打开 <a href="http://localhost/phpinfo.php">http://localhost/phpinfo.php</a> ，如果看到 Php 配置输出信息，就说明配置正常。

* 二：安装并配置 MySQL</p>  
	
	从 MySQL 官方站下载 MySQL 4.1.10 压缩包，解压缩后会有一个EXE安装文件，运行以安装，将 MySQL 安装到 d:\website\mysql\，安装完成后可以直接启动配置向导，完成 MySQL 的配置。</p>  <p>

* 三、配置 php.ini 并测试 MySQL
	
	用文本编辑软件打开 c:\windows\php.ini 文件，然后修改以下内容：   <br /> 将 extension_dir = &quot;./&quot; 改为 extension_dir = &quot;d:/website/php/ext&quot;    <br /> 将 ;extension=php_MySQL.dll 行首的’;'去掉；    <br /> 将 ;extension=php_mbstring.dll 行首的“;”去掉；    <br /> 将 ;session.save_path = &quot;/tmp&quot; 改为 session.save_path = &quot;D:/website/php/session_temp&quot;; （即将行首的’;'去掉，并设置保存session的目录）</p>  <p> 重新启动 Apache 服务。   <br /> 到这里，Apache + Php + MySQL 就基本配置完成了，在Web根目录下（即 d:\website\public_html\ 目录）中，用文本编辑软件创建一个 testdb.php 文件，其内容如下：</p>  <p>&lt;?php   <br />$link=MySQL_connect(’MySQL服务器名’,'MySQL用户名’,'密码’);    <br />if(!$link) echo &quot;Error !&quot;;    <br />else echo &quot;Ok!&quot;;    <br />MySQL_close();    <br />?&gt;</p>  <p>用浏览器打开 <a href="http://localhost/testdb.php">http://localhost/testdb.php</a> 如果看到输出 OK! 就说明配置正常。</p>  <p>

* 四、phpMyAdmin 的安装配置

	从 phpMyAdmin官方网站下载 phpMyAdmin-2.6.1-pl3.zip，然后将其解压缩到WEB根目录（即 d:\website\public_html\ 目录）中，重命名文件夹为 phpmyadmin（这个随便，你可以填写任何你愿意使用的名字）。</p>  <p> 用文本编辑软件打开 d:\website\public_html\phpmyadmin\config.inc.php 文件，找到这两行内容：   <br />$cfg[’Servers’][$i][’user’] = ‘root’;    <br />$cfg[’Servers’][$i][’password’] = ‘123456′;</p>  <p> 分别填上 MySQL 的用户和密码即可。如不是本地使用，最好加上验证：</p>  <p> 即将：$cfg[’Servers’][$i][’auth_type’] = ‘config’; 修改为 $cfg[’Servers’][$i][’auth_type’] = ‘http’;</p>  <p> 最后再设置一下 phpmyadmin 的路径，即将 $cfg[’PmaAbsoluteUri’] = ”; 改为 $cfg[’PmaAbsoluteUri’] = ‘<a href="http://localhost/phpmyadmin%27;">http://localhost/phpmyadmin’;</a></p>  <p>

* 五、WordPress 的安装

	首先在MySQL中为wordpress创建一个新的数据库，特别提示：推荐使用二进制（binary）编码的数据库，否则将有可能导致 Wordpress 出现乱码；   <br /> 其次从Wordpress官方网站下载压缩包，然后将 wordpress 解压缩到网页根目录（pub_html）下，并用记事本打开 wp-config-sample.php 文件，编辑如下字段：    <br /> // ** MySQL settings ** //    <br /> define(’DB_NAME’, ‘database_name’); // 数据库名    <br /> define(’DB_USER’, ‘MySQL_user’); // MYSQL用户名    <br /> define(’DB_PASSWORD’, ‘MySQL_pwd’); // MYSQL用户密码    <br /> define(’DB_HOST’, ‘localhost’); // 一般情况下保持 localhost 即可    <br /> 做完以上更改，保存退出，并将文件改名为 wp-config.php 文件。</p>  <p> 最后运行 <a href="http://localhost/wp/wp-admin/install.php">http://localhost/wp/wp-admin/install.php</a> 根据向导完成安装，需要注意的是，安装向导的最后一步会随机生成登录密码，你需要记录下这个密码，然后以此用户名、密码登录，在后台管理的 user 中修改密码、配置WP的属性等等……</p>  <p>OK，经过上述步骤，一个 Apache + Php + MySQL + phpMyAdmin + Wordpress 的环境就基本上搭建好啦，呵呵，尽管还比较简陋。</p>  <p>PS.顺便推荐几个较好的程序：</p>  <p><a href="http://awstats.sourceforge.net/AWstats">http://awstats.sourceforge.net/AWstats</a> ；    <br /><a href="http://www.sixapart.com/movabletype/MovableType">http://www.sixapart.com/movabletype/MovableType</a> ；    <br /><a href="http://www.mamboserver.com/Mambo">http://www.mamboserver.com/Mambo</a> 。</p></p>
