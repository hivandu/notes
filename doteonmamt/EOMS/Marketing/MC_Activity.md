# yMC 活动模块需求

## yPOS活动原参数参考

![-w1440](http://qiniu.hivan.me/15591869328055.jpg)

## 流程：


![](http://qiniu.hivan.me/15592067297466.jpg)


## 步骤详解：

### 1. 新建券包

### 2. 编辑券包属性

**券包属性内容：**

1. 券包ID
2. 额度
3. 关联券
4. 关联券数量


### 3. 新建活动

默认增加活动ID，ID内容类似：`recharge_single_XXXXX`

### 4. 引入券包

单个活动仅可以引入一个券包

### 4. 编辑活动属性：

包含以下可编辑属性：

1. 活动名称
2. 可用门槛
3. 额度
4. 类别属性（是券还是赠送金）
5. 可用次数：初始为两个值，次数和百分比
6. 是否可修改类别（`true or false`）
7. 券包编号
    > 需yPOSX自行生成，对应相应的券包规则，目的是将券中心内的两张券（10元代金券，5元代金券进行组合，满足额度需求）
8. 单店属性: commonCode(默认为空，门店引入的时候自动添加)
9. 描述

例如：**充100送30**活动，其中30为券：2张10元券，2张5元券；

**创建时：**
1. ID：`recharge_single`
2. 活动名称： `"充100送30"`
3. 门槛：`3000`(价格以分为单位，以下同)
4. 额度: `10000`
5. 类别: `coupon`(程序自己决定如何鉴别)
6. 可用次数: `0`,`0%`
7. 是否可修改类别: `false`
8. 券包规则编号: `1221`(举例)
9. 单店属性: `null`
10. 描述: `null`

**人民公园店引入时：**
1. ID：`recharge_single`
2. 活动名称： `"充100送30"`
3. 门槛：`3000`(价格以分为单位，以下同)
4. 额度: `10000`
5. 类别: `coupon`(程序自己决定如何鉴别)
6. 可用次数: `300`, `70%`
7. 是否可修改类别: `false`
8. 券包规则编号: `1221`(举例)
9. 单店属性: `0184`
10. 描述: `null`