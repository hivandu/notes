# Auto operation Weibo

> The code address of this article is: [auto operation weibo](https://github.com/hivandu/colab/blob/master/AI_data/auto%20operation%20weibo.ipynb)
> Chromedrive download: [Taobao Mirror](http://npm.taobao.org/mirrors/chromedriver) , **need to be consistent with your Chrome version**

## auto operation weibo
```python
from selenium import webdriver
import time
driver = webdriver.Chrome('/Applications/chromedriver')

# login weibo
def weibo_login(username, password):

    # open weibo index
    driver.get('https://passport.weibo.cn/signin/login')
    driver.implicitly_wait(5)
    time.sleep(1)

    # fill the info: username, password
    driver.find_element_by_id('loginName').send_keys(username)
    driver.find_element_by_id('loginPassword').send_keys(password)
    time.sleep(1)

    # click login
    driver.find_element_by_id('loginAction').click()
    time.sleep(1)

# set username, password
username = 'ivandoo75@gmail.com'
password = 'ooxx'

# Mobile phone verification is required here, but still can’t log in fully automatically
weibo_login(username, password)
```

## follow user

```python
def add_follow(uid):
    driver.get('https://m.weibo.com/u/' + str(uid))
    time.sleep(1)

    # driver.find_element_by_id('follow').click()
    follow_button = driver.find_element_by_xpath('//div[@class="btn_bed W_fl"]')
    follow_button.click()
    time.sleep(1)

    # select group
    group_button = driver.find_element_by_xpath('//div[@class="list_content W_f14"]/ul[@class="list_ul"]/li[@class="item"][2]')
    group_button.click()
    time.sleep(1)

    # cancel the select
    cancel_button = driver.find_element_by_xpath('//div[@class="W_layer_btn S_bg1"]/a[@class="W_btn_b btn_34px"]')
    cancel_button.click()
    time.sleep(1)

# 每天学点心理学UID
uid = '1890826225'
add_follow(uid)
```

## create text and publish 
```python
def add_comment(weibo_url, content):
    driver.get(weibo_url)
    driver.implicitly_wait(5)

    content_textarea = driver.find_element_by_css_selector('textarea.W.input').clear()
    content_textarea = driver.find_element_by_css_selector('textarea.W.input').send_keys(content)

    time.sleep(2)

    comment_button = driver.find_element_by_css_selector('.W_btn_a').click()

# post the text
def post_weibo(content):
    # go to the user index
    driver.get('https://weibo.com')
    driver.implicitly_wait(5)

    # click publish button
    # post_button = driver.find_element_by_css_selector('[node-type="publish"]').click()

    # input content word to textarea
    content_textarea = driver.find_element_by_css_selector('textarea.W_input[node-type="textEl"]').send_keys(content)
    time.sleep(2)

    # click publish button
    post_button = driver.find_element_by_css_selector("[node-type='submit']").click()
    time.sleep(1)

# comment the weibo
weibo_url = 'https://weibo.com/1890826225/HjjqSahwl'
content= 'here is Hivan du, Best wish to u.'

# auto send weibo
content = 'Learning is a belief!'
post_weibo(content)
```