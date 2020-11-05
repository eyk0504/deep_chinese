import torch.nn.functional as F
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


def two_mask_predict(sentence) :
    sent_ids = tokenizer(sentence)['input_ids']

    mask_index=[i for i in range(0,len(sent_ids)) if sent_ids[i]==103]  #mask인넥스 찾기
    sent_tensor = torch.tensor(sent_ids).unsqueeze(0).to('cuda')
    output = model(sent_tensor) #model에 넣기

    logits=output[0].detach().cpu()

    mask_predict = F.softmax(logits[0][mask_index[0]], dim=0).numpy()

    predict_top_n = mask_predict.argsort()[-5:][::-1]
    pred=tokenizer.convert_ids_to_tokens(predict_top_n)
    prob=mask_predict[predict_top_n].tolist()

    return pred, prob
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, \
            default="../data/test_mlm.json", help='test data path')
    """parser.add_argument("--model_path", type=str, \
            default="models/baseline/baseline.tar", help='model path')"""
    parser.add_argument("--result_path", type=str, \
            default="../models/tuned_ext_multi_test_report.txt", help='output path')
    args = parser.parse_args()

    test_path = open(args.test_path, 'r', encoding='utf-8')

    MODEL_TYPE = 'hfl/chinese-bert-wwm-ext'
    
    '''print(f"Loading model from '{args.model_path}'...")
    model = torch.load(args.model_path)'''
    model = BertForMaskedLM.from_pretrained("../models/ext_trained_model")
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
        sentence, answer = data.split('\t')
        pred,prob=  two_mask_predict(sentence)
        
        multi_prob=dict()

        for i in range(len(pred)): 
            second_sentence=sentence.replace('[MASK]',pred[i],1)
            if second_sentence.find('[MASK]')!=-1:
                pred2,prob2=two_mask_predict(second_sentence)
                for j in range(len(pred)):
                    multi_prob[pred[i]+pred2[j]]=round(prob[i]*prob2[j]*100,2)
        sorted_multi_prob=sorted(multi_prob.items(), key=lambda t : t[1],reverse=True)

        first_pred=sorted_multi_prob[0][0]
        if first_pred==answer :
            total_right += 1
            test_w.write('o ')
        else :
            test_w.write('x ')

        test_w.write(sentence + '\n')
        test_w.write(f"Answer : {answer} \n")
        for i in range(5):
            test_w.write('Predict {} : {} {} %\n'.format(i+1,sorted_multi_prob[i][0],sorted_multi_prob[i][1]))

    print(f"Test accuracy : {(total_right/total_num)*100:.2f}")
