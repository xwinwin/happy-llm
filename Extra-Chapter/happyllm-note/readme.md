# Extra Chapter: happyllm-note

## happyllm-note内容

happyllm-note是对happy-llm项目提供的代码，进行简易动手项目实践，以及根据文档当中的大模型相关知识点，出成习题进行补充。

动手实践项目主要有：
- **chapter1**：简要介绍jieba库，以及通过应用jieba库，进行文档当中的技术，如中文分词，子词切分等技术的代码实现。并且在最后通过waima_10k数据集进行文本情感分析的具体实践。

- **chapter2**：基于文档当中所提及到的Transformer框架，通过利用文档当中的代码，搭建出一个简要的中英文翻译器，数据集在文件夹当中，应用时需要解压，代码可以一件运行。由于预训练集较大无法上传，需要复现的小伙伴可以自己运行。运行参考：硬件：RTX4060，运行时间：30min。

- **chapter3**：基于项目当中的chapter3与chapter4两个章节的知识点进行编写，通过微调BERT-mini来进行演示，微调的数据集也是使用waimai_10k。值得注意的是，这里的notebook是在kaggle上面运行的，如果在本地运行的时候要注意下载transformer库。

（pip install transformer）

- **chapter4 ~ chapter7**：剩余的章节均是根据项目当中的对应章节提取相关的知识点，通过选择题与简答题的形式呈现出来。主要是通过扫一眼题目来检测一下自己是否扎实的掌握了新的知识点。

说明：没有解析的题目是因为知识点过于基础，稍微忘记的话可以在原文档当中找到答案。

## 文档内容结构概览

```
Extra-Chapter/
├── happyllm-note/                    
│   ├── readme.md            # 本文件
│   ├── chapter1                         
│   │   ├──chapter1.ipynb
│   │   ├──test.txt
|   |   └──waimai_10k_csv                
│   ├── chapter2                          
│   │   ├── chapter2.ipynb
│   │   ├── chinese.zip
│   │   └── english.zip
│   ├── chapter3                         
│   │   ├──chapter3.ipynb
|   |   └──waimai_10k_csv   
│   ├── chapter4                         
|   |   └──chapter4.md  
│   ├── chapter5                         
|   |   └──chapter5.md  
│   ├── chapter6                         
|   |   └──chapter6.md  
│   └── chapter7                         
|       └──chapter7.md  
└── readme.md                           
```

## 贡献者主页


|贡献者|学校  | 研究方向           |   GitHub主页 |
|-----------------|------------------------|-----------------------|------------|
| 蔡鋆捷 | 福州大学  |    Computer Vision（CV），Natural Language Processing（NLP）      |https://github.com/xinala-781|
