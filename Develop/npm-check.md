---
title: npm-check
date: 2017-02-13 10:54:03
tags:
- node
- npm
---

npm-check是用来检查npm依赖包是否有更新，错误以及不在使用的，我们也可以使用npm-check进行包的更新。
安装npm-check：

```shell
npm install -g npm-check
```

检查npm包的状态:

```shell
npm-check -u -g
```

上下选择，空格选定，然后直接`Enter`安装就可以了。


update: (2017-03-05):

## npm 功能:
- 告诉你哪些依赖已经过时；
- 在你决定升级的时候，提供依赖包的文档；
- 提示某个依赖没有被你使用；
- 支持全局安装的模块，-g；
- 交互式升级介面，减少输入和输入错误的情况，-u；
支持公共 & 私有依赖包 [@scoped/packages](https://docs.npmjs.com/getting-started/- scoped-packages)；
- 支持 ES6 import from 语法；
- 支持公共 & 私有 npm 源；
- 支持 npm@2 和 npm@3；
- …

```shell
# 更新全局依赖
$ npm-check -gu
# 更新当前项目依赖
$ npm-check -u
```

可供参考:

[https://cnodejs.org/topic/581d96d5bb9452c9052e7b58](https://cnodejs.org/topic/581d96d5bb9452c9052e7b58)
