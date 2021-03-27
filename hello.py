# coding:utf-8
import json
import csv

file_name = '/tcdata/num_list.csv'
data = []

# 第一题，直接写入 Hello world
result = {
    "Q1": "Hello world",
    "Q2": 0,
    "Q3": []
}

# 第二题，求和
with open(file_name, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        data.append(int(row[0]))

sum = sum(data)
result['Q2'] = sum

# 第三题
result['Q3'] = sorted(data, reverse=True)[0:10]

# 保存到 result.json
with open('result.json', 'w', encoding='utf-8') as f: json.dump(result, f)
