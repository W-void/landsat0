# read me
wget -i ./dataLoad/result.txt -P ~/BC  
ls  ~/BC/*.tar.gz | xargs -n1 tar xzvf
python3 cropImage.py
ln -s ～/BC/BC/train ~/Documents/landsat/VOC2012
ln -s ～/BC/BC/val ~/Documents/landsat/VOC2012
python3 landsat.py