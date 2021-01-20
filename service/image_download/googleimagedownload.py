from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import urllib.request
from bs4 import BeautifulSoup as bs
import os
from settings import LOCAL_CHROME_DRIVER, GOOGLE_LOCAL_IMAGE_STORAGE_PATH

base_url_part1 = 'https://www.google.com/search?q='
base_url_part2 = '&source=lnms&tbm=isch'


class GoogleCrawler:
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
        # picpath = GOOGLE_LOCAL_IMAGE_STORAGE_PATH + '/%s' % (foldername)
        image_directory = search_query.replace(' ', '_')
        picpath = GOOGLE_LOCAL_IMAGE_STORAGE_PATH + '/%s' % (image_directory)
        if not os.path.exists(picpath): os.makedirs(picpath)

        img_url_dic = {}
        x = 0
        pos = 0
        for i in range(1):
            pos = i * 500
            js = "document.documentElement.scrollTop=%d" % pos
            driver.execute_script(js)
            time.sleep(1)
            html_page = driver.page_source
            soup = bs(html_page, "html.parser")
            imglist = soup.findAll('img', {'class': 'rg_i'})

            for imgurl in imglist:
                try:
                    # log("已下载 "+str(x)+" 张图片")
                    print(x, end=' ')
                    if imgurl['src'] not in img_url_dic:
                        target = '{}/{}.jpg'.format(picpath, x)
                        img_url_dic[imgurl['src']] = ''
                        urllib.request.urlretrieve(imgurl['src'], target)
                        time.sleep(1)
                        x += 1
                except KeyError:
                    print("ERROR!")
                    continue

    def run(self, search_query):
        driver = self.start_brower(search_query)
        self.downloadImg(driver, search_query)
        driver.close()
        print("Download has finished.")


if __name__ == '__main__':
    craw = GoogleCrawler()
    craw.run("孙俪")
