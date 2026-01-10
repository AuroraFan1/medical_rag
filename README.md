```
🏥 MedAssist - 医疗智能问答系统
```

**数据集：**[MedDialog · 数据集](https://www.modelscope.cn/datasets/OpenDataLab/MedDialog/files)

点击上述链接下载数据集后，将2010-2020.txt文件保存在data目录下。

**运行环境**

> conda create -n NLP python=3.10
>
> conda activate NLP
>
> pip install -r requirements.txt
>
> python data_processing.py
>
> streamlit run streamlit_app.py

在终端运行以上命令可以运行代码，这是使用原始数据构建向量数据库后所创建的RAG系统，可以正常使用。



在finetune目录下，按照上述代码运行后，会进行微调，评估。但是由于各种原因导致效果不理想，生成时间大大增加。



在medical——rag目录下，硬件要求RTX 5090（32GB）

> conda activate NLP
>
> pip install -r requirements.txt
>
> python run_pipelinee --run-all
>
> streamlit run streamlit_app.py

运行上述命令可以完成数据库构建，模型微调，微调时评估失败，评估。由于微调时评估失败导致评估结果一致。
