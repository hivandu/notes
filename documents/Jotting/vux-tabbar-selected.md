# vux更改Tabbar选中状态

在vux的文档和示例中，都没有明确的说明tabbar上v-model的使用

文档中将`v-model`说明放在了TabbarItem示例下，但是其实这个应该是放在`Tabbar`上

```
<template>
    <router-view class="view" v-on:changeTab="changeTab"></router-view>
    <tabbar v-model="index">
        <tabbar-item></tabbar-item>
        ...
        <tabbar-item></tabbar-item>
    </tabbar>
</template>
<script>
data(){
    return{
        index:0,
        ...
    }
}
methods:{
    changeTab(num){
        ...
        this.index = num;
        ...
    }
}
</scirpt>
```

然后子组件中调用

```javascript
mounted(){
    this.$emit('changeTab', 2)
}
```

这样就便于在不同的组件内都可以更改Tabbar选中状态

