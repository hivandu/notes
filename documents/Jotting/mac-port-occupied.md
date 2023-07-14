# mac:port occupied

查询:
```powershell
$ lsof -i:3000
```

显示:
```powershell
COMMAND  PID USER   FD   TYPE             DEVICE SIZE/OFF NODE NAME
node    2243   du   12u  IPv6 0xc9b8c91a94a8da89      0t0  TCP *:hbci (LISTEN)
```

结束:
```powershell
$ kill -9 2243
```
