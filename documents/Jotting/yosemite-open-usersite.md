# Yosemite访问用户级服务器目录

升级到OSX 10.10(Yosemite)以后，`localhost`是可以正常访问的，只是`localhost/~user`无法打开了，提示403错误。

<!-- more -->

网上查找资料，说是随着系统的更新，Apache本本更新到2.4.9，PHP也更新到了5.5.14，所以Apache的配置就需要做相应的修改。

首先，我们需要确定打开了Apache

```
sudo apachectl start
```

然后设置允许访问用户目录

1. 修改httpd.conf配置

```
sudo subl /etc/apache2/httpd.conf
```

command + f 查找代码，并去掉注释符 `#`

```
LoadModule authz_core_module libexec/apache2/mod_authz_core.so
LoadModule authz_host_module libexec/apache2/mod_authz_host.so
LoadModule userdir_module libexec/apache2/mod_userdir.so
LoadModule php5_module libexec/apache2/libphp5.so
Include /private/etc/apache2/extra/httpd-vhosts.conf
Include /private/etc/apache2/extra/httpd-userdir.conf
```

2. 修改httpd-userdir.conf配置

```
sudo subl /etc/apache2/extra/httpd-userdir.conf
```

command + f 查找以下代码，去掉注释符`#`
	
```
Include /private/etc/apache2/users/*.conf
```

3. 修改yourUserName.conf配置

```
sudo subl /etc/apache2/users/username.conf
```

PS: username为你的用户名称，如果没有该文件则新建一个，然后将内容修改为:

```
Options Indexes MultiViews
AllowOverride None
Require all granted
```

然后设置文件权限为`755`

```
sudo chmod 755 /etc/apache2/users/haibor.conf
```

最后我们需要重启Apache

```
sudo apachectl restart
```

