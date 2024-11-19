import selenium
from selenium import webdriver
import selenium.common
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
# 启动浏览器
chrome_options = Options()
chrome_options.add_argument("--mute-audio")  # 将浏览器静音
# chrome_options.add_experimental_option("detach", True)  # 当程序结束时，浏览器不会关闭

# -----如果咋们的linux系统没有安装桌面，下面两句一定要有哦，必须开启无界面浏览器-------
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
# ------------------------------------------------------------------------
browser = webdriver.Chrome(options=chrome_options)

browser.get('www.baidu.com')

# 关闭浏览器
browser.quit()

eles = browser.find_elements(by=By.CLASS_NAME,value="MuiSelect-root.MuiSelect-select.MuiSelect-selectMenu.MuiSelect-outlined.MuiInputBase-input.MuiOutlinedInput-input.MuiInputBase-inputMarginDense.MuiOutlinedInput-inputMarginDense")
for ele in eles:
    if ele.text != 'Pending':
        continue
    print(len(eles))
    ele.click()
    print(ele.text)
    val = ele.get_attribute("role")
    try:
        ele.click()
    except selenium.common.ElementClickInterceptedException as e:
        print('error')

