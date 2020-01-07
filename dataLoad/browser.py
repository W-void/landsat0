# %%
# -*- coding:utf-8 -*-
# python 3.7
import sys
from bs4 import BeautifulSoup
import urllib.request


# %%
html_doc = "https://landsat.usgs.gov/landsat-8-cloud-cover-assessment-validation-data"
# if len(sys.argv) > 1:
#    website=sys.argv[1]
#    if(website is not None):
#         html_doc= sys.argv[1]
html = urllib.request.urlopen(req)
bsObj = BeautifulSoup(html, 'html.parser')
t1 = bsObj.find_all('a')
result = []
for t2 in t1:
    link = t2.get('href')
    if link is not None and (link[-2:] == 'gz'):
        print(link)
        result.append(link)

f = open(r'result.txt','w',encoding='utf-8')  #文件路径、操作模式、编码  # r''
for a in result:
    f.write(a+"\n")
f.close()
print("\r\n扫描结果已写入到result.txt文件中\r\n")


# %%
