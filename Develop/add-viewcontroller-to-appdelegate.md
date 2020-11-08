# Add ViewController to AppDelegate

之前一直是Nib形式建立ViewController视图，这样建立以后关系都是已经写好的，可是Nib有自己的弊端，虽然可视化很方便，可是无法很方便的管理自己的代码，并且灵活性不够。于是就想还是自己纯粹手动代码建立视图来的灵活一点。

<!--more-->

就这点东西简直搞死我了，因为之前Nib建立视图建立好的XIBs文件和ViewController关系是绑定好的，并且在AppDelegate中也有相应的关系绑定，一旦自己建立了，就一头懵！

### 建立"ViewController"视图

首先，我们需要建立好ViewController视图，然后在AppDelegate中将其导入，并声明属性。加入我们建立的是NHDemoViewController视图，那么我们在AppDelegate.h中将其导入，并声明属性，代码如下：

```
#import  <UIKit/UIKit.h>
#import "NHdemoViewController"

@interface NHAppDelegate : UIResponder <UIApplicationDelegate>

@property (strong, nonatomic) UIWindow *window;
@property (strong, nonatomic) NHDemoViewController *demoView;

@end
```

然后在AppDelegate.m文件中实现，代码如下：

```
- (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions
{
self.window = [[[UIWindow alloc] initWithFrame:[[UIScreen mainScreen] bounds]] autorelease];
// 实现
self.demoView = [[[NHDemoViewController alloc] initWithNibName:nil bundle: nil] autorelease];
self.window.rootViewController = self.demoView;

self.window.backgroundColor = [UIColor whiteColor];
[self.window makeKeyAndVisible];
return YES;
}
```

这样心建立的NHDemoViewControoler视图就会在App启动的时候出现了。

啊，对了，如果版本较低的Xcode，记得要自己手动写上: `@synthesize dragonView = _dragonView;`