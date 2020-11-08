# rpmdb open failed

```shell
[root@dhcp yum.repos.d]# cd /var/lib/rpm/
[root@dhcp rpm]# ls
[root@dhcp rpm]# rm __db.* -rf
[root@dhcp rpm]# rpm --rebuilddb
[root@dhcp rpm]# yum clean all
[root@dhcp rpm]# yum update
```

