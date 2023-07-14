# Debian安装SSH

# 要求： 安装Debian 8.* 64位操作系统

### 分区要求: 不要分区
<!-- more -->
> - Partitioning mehod: use entire disk
> ![entire disk](http://qiniu.hivan.me/debian-install-1.jpg)
> - Partitioning scheme: All files in one partition
> ![All files](http://qiniu.hivan.me/debian-install-2.jpg)



### 选择源镜像 (mirror country)

> 请选择china
> ![country](http://qiniu.hivan.me/debian-instal-3.jpg)
> 然后选择**`ftp.cn.debian.org`**
> ![ftp](http://qiniu.hivan.me/debian-install-4.jpg)


### 程序和服务需求
> debian 默认最小安装，安装的时候不用安装桌面环境和标准系统实用程序(以下两个不需要勾选):
> - Debian destop environment
> - Standard system utilities
> **如果有`SSH server`选项，请务必勾选，会省很多麻烦**
> ![server](http://qiniu.hivan.me/debian-install-5.jpg)

# 安装SSH
debian最小安装默认是没有配置`apt-get`源的，这个时候无法实用`apt-get install`命令，所以在安装SSH之前，我们需要先配置`apt-get`:

### 配置`apt-get`
**终端内操作**

```
# 首先我们需要备份原有配置文件
cp /etc/apt/sources.list /etc/apt/sources.listbak

# 然后对资源列表文件进行编辑
vi /etc/apt/sources.list
# 当然也可以实用nano命令
nano /etc/apt/source.list
```
PS: 如果对VI操作不熟悉的，可以看这里 [vi编辑器常见命令实用](http://c.biancheng.net/cpp/html/2735.html)

如果安装的时候按照之前我给的步骤来，那么这会的`sources.list`应该是这样的
![source.list](http://qiniu.hivan.me/debian-source-list.jpg)

对文件进行更改，将以下命令加入文件并保存:

```
deb http://ftp.cn.debian.org/debian/ jessie main contrib non-free 
deb-src http://ftp.cn.debian.org/debian/ jessie main contrib non-free
```

更改后的文件如图:
![source change](http://qiniu.hivan.me/debian-source-list-change.jpg)

**`main, contrib, non-free 分属不同的源，添加后可以从不同的源仓库更新文件索引`**

至此`apt-get`源就配置完毕，接下来我们就可以安装SSH了

### 安装SSH
在终端内输入以下命令:

```
# 更新apt-get源
apt-get update

# 更新系统
apt-get upgrade

# 安装SSH
apt-get install ssh
```

这样就好了，SSH安装完毕

![install ssh](http://qiniu.hivan.me/debian-install-ssh.jpg)

## 注意 如果一直安装不能成功，请往下看:

首先，请`ping ftp.cn.debin.org` 和 `ping mirrors.163.com` 来测试一下能否ping的通域名，如果ping不通，请往下看:

**有的时候机房安装debian后会出现域名解析问题**,这又是另外一个问题。比如`ping 123.111.123.111` 是OK的，但是如果ping对应的域名如: `ping mirrors.163.com`就会出现**unknow host**的问题。

似乎linux很大一部分都会出现这种问题，能ping的通IP但是ping不通域名。那么请查看以下原因解决:

### 1. 查看DNS解析是否有问题，确定设置了域名服务器:

`cat /etc/resolv.conf`

```
nameserver 114.114.114.114  
nameserver 8.8.8.8  
nameserver 8.8.4.4
```

### 2. 确保网关已设置
`grep GATEWAY /etc/sysconfig/network-scripts/ifcfg*`
```
/etc/sysconfig/network-scripts/ifcfg-eth0:GATEWAY=192.168.40.1
```
如果未设置，则通过以下方法增加网关

`route add default gw 192.168.40.1`

或者手工编写`/etc/sysconfig/network-scripts/ifcfg*`

然后重启network服务:

`service network restart`

### 3. 确保可用dns解析

`grep hosts /etc/nsswitch.conf`

文件打开后为:

```
hosts: files dns
```

### 4. 查看是否防火墙的问题
因为域名解析用到了53端口,需要把下面设置配置到防火墙里:
```
iptables -A INPUT -p udp --sport 53 -j ACCEPT  
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT  
iptables -A INPUT -p udp --dport 53 -j ACCEPT  
iptables -A OUTPUT -p udp --sport 53 -j ACCEPT  
```

如果找不到原因或者不知道怎么设置，那么就用以下最笨的方法:

如果出现这样的问题，更新`sources.list`后会无法更新也无法安装ssh. 如果出现这样的问题，更新`sources.list`地址为一下地址:

```
deb http://123.58.173.186/debian/ jessie main non-free contrib
deb http://123.58.173.186/debian/ jessie-updates main non-free contrib
deb http://123.58.173.186/debian/ jessie-backports main non-free contrib
deb-src http://123.58.173.186/debian/ jessie main non-free contrib
deb-src http://123.58.173.186/debian/ jessie-updates main non-free contrib
deb-src http://123.58.173.186/debian/ jessie-backports main non-free contrib
deb http://123.58.173.186/debian-security/ jessie/updates main non-free contrib
deb-src http://123.58.173.186/debian-security/ jessie/updates main non-free contrib
```

**利用IP地址代替域名，但是测试下来只有163的镜像可以这样做。来源为网易镜像的帮助文档: [Debian镜像使用帮助](http://mirrors.163.com/.help/debian.html)**

