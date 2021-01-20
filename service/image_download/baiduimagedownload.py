from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import urllib.request
from bs4 import BeautifulSoup as bs
import re
import os

base_url_part1 = 'https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=index&fr=&hs=0&xthttps=111111&sf=1&fmq=&pv=&ic=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&word='
base_url_part2 = '&oq=bagua&rsp=0'
from settings import LOCAL_CHROME_DRIVER, BAIDU_LOCAL_IMAGE_STORAGE_PATH

class BaiduCrawler:
    def __init__(self):
        self.url = base_url_part1 + '%s' + base_url_part2

    def start_brower(self, search_query):
        chrome_options = Options()
        chrome_options.add_argument("--disable-infobars")
        driver = webdriver.Chrome(executable_path=LOCAL_CHROME_DRIVER, chrome_options=chrome_options)
        driver.maximize_window()
        driver.get(self.url % (search_query))
        return driver

    def downloadImg(self, driver, search_query):
        # t = time.localtime(time.time())
        # foldername = str(t.__getattribute__("tm_year")) + "-" + str(t.__getattribute__("tm_mon")) + "-" + \
        #              str(t.__getattribute__("tm_mday"))
        # picpath = BAIDU_LOCAL_IMAGE_STORAGE_PATH + '/%s' % (foldername)
        image_directory = search_query.replace(' ', '_')
        picpath = BAIDU_LOCAL_IMAGE_STORAGE_PATH + '/%s' % (image_directory)
        if not os.path.exists(picpath): os.makedirs(picpath)
        img_url_dic = {}
        x = 0
        pos = 0
        for i in range(2):
            pos += 500
            js = "document.documentElement.scrollTop=%d" % pos
            driver.execute_script(js)
            time.sleep(2)
            html_page = driver.page_source
            soup = bs(html_page, "html.parser")
            imglist = soup.findAll('img', {'src': re.compile(r'https:.*\.(jpg|png)')})

            for imgurl in imglist:
                if imgurl['src'] not in img_url_dic:
                    target = '{}/{}.jpg'.format(picpath, x)
                    img_url_dic[imgurl['src']] = ''
                    urllib.request.urlretrieve(imgurl['src'], target)
                    x += 1

    def run(self, search_query):
        driver = self.start_brower(search_query)
        self.downloadImg(driver, search_query)
        driver.close()
        print("Download has finished.")


if __name__ == '__main__':
    craw = BaiduCrawler()
    search_query = '邓超'
    craw.run(search_query)