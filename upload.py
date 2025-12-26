# 首次使用需要安装aistudio-sdk库
# pip install --upgrade aistudio-sdk

import os
# 需要填写aistudio-access-token, 在我的控制台--令牌获取
os.environ["AISTUDIO_ACCESS_TOKEN"] = "d82af2e4ed75ddd7238f4422aab6dcc8335fea3a"


#上传文件夹
from aistudio_sdk.hub import upload_folder
res = upload_folder(
    # 填写数据集详情页面中的repo_id
    repo_id='fanwen/meddialog',
    # 填写要上传的文件在本地的路径，如'./path/to/local/dir'
    folder_path='data',
    # 填写上传至repo后的文件路径，如填写'data/'，则会将文件上传至data目录内；或不填，则默认上传至master分支的根目录内
    path_in_repo='data/',

    # 填写仓库类型为dataset，上传数据集文件时为必填项
    repo_type = 'dataset'
)
print(res)