# %%
import sys
from bs4 import BeautifulSoup
import urllib.request
import re

# %%
html_doc = "https://landsat.usgs.gov/landsat-8-cloud-cover-assessment-validation-data"
req = urllib.request.Request(html_doc)
html = urllib.request.urlopen(req)
bsObj = BeautifulSoup(html, 'html.parser')
t1 = bsObj.find_all('a')
senceIDs = []
for t2 in t1:
    link = t2.get('href')
    if link is not None and (link[-2:] == 'gz'):
        senceId = re.split('[./]', link)[-3]
        print(senceId)
        senceIDs.append(senceId)
