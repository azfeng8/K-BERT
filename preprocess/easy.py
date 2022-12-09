#!/usr/bin/env python3
import json
import os
import numpy as np

if os.getcwd() != "/home/catalan/Desktop/6.8611/project/K-BERT/preprocess/qa/ARC-V1-Feb2018-2/ARC-Easy":
    os.chdir("/home/catalan/Desktop/6.8611/project/K-BERT/preprocess/qa/ARC-V1-Feb2018-2/ARC-Easy")
with open("ARC-Easy-Train.jsonl", "r") as f:
    train_list = list(f)
with open("ARC-Easy-Dev.jsonl", "r") as f:
    dev_list = list(f)
with open("ARC-Easy-Test.jsonl", "r") as f:
    test_list = list(f)

list_json_data = [train_list, dev_list, test_list]
output_names = ["ARC-Easy-Train.tsv", "ARC-Easy-Dev.tsv", "ARC-Easy-Test.tsv"]
for d,name in zip(list_json_data, output_names):
    # Columns: question id, question, answer, label (0 or 1)
    num_cols = 4
    # max_num_a = 4 # some questions have 3 choices
    # out_data = np.empty((max_num_a*len(d), num_cols), dtype=str)
    out_data = []
    for question_json in d:
        question_dict = json.loads(question_json)
        q = question_dict['question']['stem']
        a = {ans['label']:ans['text'] for ans in question_dict['question']['choices']}
        correct_ans = a[question_dict['answerKey']]
        del a[question_dict['answerKey']]
        incorrect_ans = a.values()
        
        out_data.append([ q, correct_ans, "1"])
        for a in incorrect_ans:
            out_data.append([ q, a, "0"])
    np.savetxt(name, out_data, delimiter="\t", fmt="%s")
    print(f"Saved successfully to {name}")

# https://stackoverflow.com/questions/39698363/concatenate-join-a-numpy-array-with-a-pandas-dataframe