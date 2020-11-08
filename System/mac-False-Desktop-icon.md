# Mac隐藏/显示桌面图标

## Shell:

```powershell
defaults write com.apple.finder CreateDesktop -bool FALSE;killall Finder
```
&

```powershell
defaults delete com.apple.finder CreateDesktop;killall Finder
```


## AppleScript:

```AppleScript
display dialog "桌面图标设置为可见或隐藏?" buttons {"可见", "隐藏"} with icon 2 with title "Switch to presentation mode" default button 1
set switch to button returned of result
if switch is "隐藏" then
        do shell script "defaults write com.apple.finder CreateDesktop -bool FALSE;killall Finder"
else
        do shell script "defaults delete com.apple.finder CreateDesktop;killall Finder"
end if
```