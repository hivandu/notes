---
title: ssh link server use iTerm2
date: 2017-09-23 23:42:30
tags:
---

```powershell
$ ssh-keygen -t rsa -C hvian -f id_rsa
$ scp ~/.ssh/id_rsa_pub root@100.100.100.100:.ssh/id_rsa.pub
$ ssh root@100.100.100.100
$ password: xxxxxxx
$ cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
$ exit

#################
subl ~/.ssh/config
```

配置:

```
Host dev
Hostname 100.100.100.100
Port 22
User root
IdentityFile ~/.ssh/id_rsa
```

**save, over**


PS: 打开**Surge**的时候，`deploy`是无法上传到github上的。显示链接被重置！
PS2: `hexo`长久不更新会生锈的，频频报错
