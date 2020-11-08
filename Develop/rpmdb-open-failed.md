---
title: rpmdb open failed
date: 2018-06-06 14:39:24
tags:
- centOS
- yum
- rpm
---

```
[root@dhcp yum.repos.d]# cd /var/lib/rpm/
[root@dhcp rpm]# ls
[root@dhcp rpm]# rm __db.* -rf
[root@dhcp rpm]# rpm --rebuilddb
[root@dhcp rpm]# yum clean all
[root@dhcp rpm]# yum update
```

