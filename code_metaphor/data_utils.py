import torch
from torch.utils.data import DataLoader, random_split
import pandas as pd
import os
import re
import csv
import json
from functools import partial
from PIL import Image, ImageFile
from transformers import AutoImageProcessor, ViTImageProcessor, ViTModel
from transformers import BertTokenizer, AutoTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder


ImageFile.LOAD_TRUNCATED_IMAGES = True
# # /media/admin01/23f0fcd5-493c-4e94-a4a1-58f709a0e24d1/yangsenqi_model/vit
# #/media/admin01/23f0fcd5-493c-4e94-a4a1-58f709a0e24d1/yangsenqi_model/bert
# feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
#
# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
from transformers import ViTImageProcessor, BertTokenizer

# 加载 ViT 模型的图片处理器
feature_extractor = ViTImageProcessor.from_pretrained('/model/vit')

# 加载 BERT 分词器
tokenizer = BertTokenizer.from_pretrained('/model/bert')

def get_image(data_path, language, image_name):
    if language == 'en':
        file_path_image = os.path.join(data_path, 'imgs_EN', image_name)
    else:
        file_path_image = os.path.join(data_path, 'imgs_CN', image_name)

    return file_path_image


import csv
import os
import json

def get_data_list(data_path, language) -> (list, list):
    """
    读取训练和测试数据，分别返回训练集和测试集
    """

    train_data_list = []
    val_data_list = []
    test_data_list = []

    if language == 'en':
        train_label_path = os.path.join(data_path, 'EN_train.csv')
        val_label_path = os.path.join(data_path, 'EN_val.csv')
        test_label_path = os.path.join(data_path, 'EN_test.csv')
        encoding_type = 'utf-8'
        null_text = "text is empty"
    elif language == 'zh':
        train_label_path = os.path.join(data_path, 'CN_train.csv')
        val_label_path = os.path.join(data_path, 'CN_val.csv')
        test_label_path = os.path.join(data_path, 'CN_test.csv')
        encoding_type = 'gbk'
        null_text = "文本为空"

    # 读取每个数据集（train, val, test）
    for file in ['train', 'val', 'test']:
        if file == 'train':
            label_path = train_label_path
        elif file == 'val':
            label_path = val_label_path
        elif file == 'test':
            label_path = test_label_path

        # 打开并读取CSV文件
        with open(label_path, 'r', encoding=encoding_type,errors='ignore') as f:

            f.readline()  # 跳过标题行
            f_csv = csv.reader(f)
            for row in f_csv:
                data_dict = {}

                data_dict['pic_id'] = int(row[0].replace('.jpg', ''))  # 去除 .jpg 后缀并转换为 int
                data_dict['text'] = row[1] if row[1] else null_text  # 文本内容
                data_dict['senti'] = "the sentiment is " + row[5] if language == 'en' else "情感为" + row[5]  # 直接读取情感数据（row[5]）
                data_dict['metaphor'] = row[2]  # Target


                # 获取图片路径（调用get_image函数）
                data_dict['image'] = get_image(data_path, language, row[0])

                if file == 'train':
                    train_data_list.append(data_dict)
                elif file == 'val':
                    val_data_list.append(data_dict)
                elif file == 'test':
                    test_data_list.append(data_dict)

    return train_data_list, val_data_list, test_data_list

def clean_text(text: bytes):
    try:
        decode = text.decode(encoding='utf-8')
    except:
        try:
            decode = text.decode(encoding='GBK')
        except:
            try:
                decode = text.decode(encoding='gb18030')
            except:
                decode = str(text)
    return decode


def data_preprocess(train_data_list, val_data_list, test_data_list):
    """
    数据预处理，清洗文本数据
    """
    for data in train_data_list:
        data['text'] = clean_text(data['text'])

    for data in val_data_list:
        data['text'] = clean_text(data['text'])

    for data in test_data_list:
        data['text'] = clean_text(data['text'])

    return train_data_list, val_data_list, test_data_list


import torch
from PIL import Image


def collate_fn(data_list):
    # 提取每个数据样本的唯一标识符（guid）
    guid = [data['pic_id'] for data in data_list]

    # 确保 tag 为数字类型，可能需要使用 LabelEncoder 转换字符串标签
    tag_values = [data['metaphor'] for data in data_list]
    label_encoder = LabelEncoder()
    tag = label_encoder.fit_transform(tag_values)  # 将标签转换为整数

    # 处理图片数据

    image = [Image.open(data['image']).convert('RGB') for data in data_list]
    image = feature_extractor(image, return_tensors="pt")  # 提取图片特征，并转换为张量

    # 处理文本数据
    text = [data['text'] for data in data_list]
    text = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=30)

    # 处理情感数据
    senti = [data['senti'] for data in data_list]
    senti = tokenizer(senti, return_tensors="pt", padding=True, truncation=True, max_length=10)

    # 返回处理后的数据
    return guid, torch.FloatTensor(tag), image, text, senti




def get_data_loader(train_data_list, val_data_list, test_data_list) -> (DataLoader, DataLoader, DataLoader):

    train_data_loader = DataLoader(
        dataset=train_data_list,
        collate_fn=collate_fn,
        batch_size=64,
        shuffle=True,
        drop_last=False,
    )

    valid_data_loader = DataLoader(
        dataset=val_data_list,
        collate_fn=collate_fn,
        batch_size=16,
        shuffle=True,
        drop_last=True,
    )

    test_data_loader = DataLoader(
        dataset=test_data_list,
        collate_fn=collate_fn,
        batch_size=16,
        shuffle=False,
        drop_last=True,
    )

    return train_data_loader, valid_data_loader, test_data_loader


def calc_metrics(target, pred):

    accuracy = accuracy_score(target, pred)
    precision_w = precision_score(target, pred, average='weighted')
    recall_w = recall_score(target, pred, average='weighted')
    f1_w = f1_score(target, pred, average='weighted')
    precision = precision_score(target, pred, average='macro')
    recall = recall_score(target, pred, average='macro')
    f1 = f1_score(target, pred, average='macro')
    return accuracy, precision_w, recall_w, f1_w, precision, recall, f1


def calc_metrics_binary(target, pred):
    """
    计算评估指标， 分别为准确率、 精确率、 召回率、 F1-score
    """
    accuracy = accuracy_score(target, pred)
    # binary
    precision = precision_score(target, pred)
    recall = recall_score(target, pred)
    f1 = f1_score(target, pred)

    # weighted
    weight_precision = precision_score(target, pred, average='weighted')
    weight_recall = recall_score(target, pred, average='weighted')
    weight_f1 = f1_score(target, pred, average='weighted')

    return accuracy, precision, recall, f1, weight_precision, weight_recall, weight_f1

