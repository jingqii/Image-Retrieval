## 图像检索

构建图像检索数据基础
```
python findFeatures.py -t dataset/train/
```

检索图片

```
python search.py -i dataset/testing/radcliffe_camera_000397.jpg
```

检索图片且用户反馈
```
python search.py -i dataset/testing/radcliffe_camera_000397.jpg -f --top-k 30
```