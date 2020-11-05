import argparse
import json
import os
import random
from transformers import BertForSequenceClassification,BertForMaskedLM
from transformers import BertTokenizer, BertModel, AdamW, BertConfig
#from transformers import get_linear_schedule_with_warmup
from torch import softmax
import numpy as np
import torch
from torch import nn


if __name__ == "__main__":
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, \
            default="../data/test_mlm.json", help='test data path')
    """parser.add_argument("--model_path", type=str, \
            default="models/baseline/baseline.tar", help='model path')"""
    parser.add_argument("--result_path", type=str, \
            default="../models/bertwwm_pretrained_test_report.txt", help='output path')
    args = parser.parse_args()
    
    test_path = open(args.test_path, 'r', encoding='utf-8')

    MODEL_TYPE = 'hfl/chinese-bert-wwm'
    
    '''print(f"Loading model from '{args.model_path}'...")
    model = torch.load(args.model_path)'''
    model = BertForMaskedLM.from_pretrained(MODEL_TYPE)
    model = model.to('cuda')
    model.eval()
   
    tokenizer = BertTokenizer.from_pretrained(MODEL_TYPE)
    print("Done!\n")
    
    test = open(args.test_path, 'r')
    test_data = json.load(test)
    total_num = len(test_data)
    total_right = 0

    test_w = open(args.result_path, 'w')
    print(f"Running inference with test data...")
    for data in test_data:
        sent, answer = data.split('\t')
        sent_ids = tokenizer(sent)['input_ids']
        mask_index=[i for i in range(0,len(sent_ids)) if sent_ids[i]==103]
        sent_tensor = torch.tensor(sent_ids).unsqueeze(0).to('cuda')
        output = model(sent_tensor)
        logits=output[0].detach().cpu().numpy()
        predict = np.argmax(logits, axis=2)[0].tolist()
        pred_sentence = [tokenizer.convert_ids_to_tokens(id) for id in predict]
        mask_pred=[pred_sentence[i] for i in mask_index]
        predict_answer=''.join(mask_pred)
        

        if predict_answer == answer:
            total_right += 1
            test_w.write('o ')
        else : 
            test_w.write('x ')
        
        test_w.write(sent + '\n')
        test_w.write(f"Answer : {answer}\n")
        test_w.write(f"Predict : {predict_answer}\n")
        
    print(f"Test accuracy : {(total_right/total_num)*100:.2f}")
