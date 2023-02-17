# torch 1.7.1+cu110  transformers 4.4.1
from transformers import BertTokenizer, BartForConditionalGeneration
import os
import torch
import time
import csv
import json
import random
from tqdm import tqdm
# 指定GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print('start now')
print("模型推理硬件设备:{}\t".format(device))
# 模型预Loding 初始化
# one stage
# bart_model_path = "output/penguin/one-stage"
# data path for one-stage model
bart_model_path = 'output/penguin/one-stage/bart-base'
tokenizer=BertTokenizer.from_pretrained(bart_model_path)
one_stage = BartForConditionalGeneration.from_pretrained(bart_model_path)
one_stage.config.max_length=128
one_stage.to(device)
one_stage.eval()
print("Loding one stage Model Done")
# two stage reader
# bart_model_path = "output/penguin/two-stage-reader"
# data path for answerer in two-stage framework
bart_model_path = "output/penguin/two-stage-reader/bart-base"
tokenizer=BertTokenizer.from_pretrained(bart_model_path)
two_stage_reader = BartForConditionalGeneration.from_pretrained(bart_model_path)
two_stage_reader.config.max_length=128
two_stage_reader.to(device)
two_stage_reader.eval()
print("Loding two stage reader Model Done")
# two stage responser
# bart_model_path = "output/penguin/two-stage-responser"
# data path for responser in two-stage framework
bart_model_path = "output/penguin/two-stage-responser/bart-base"
tokenizer=BertTokenizer.from_pretrained(bart_model_path)
two_stage_responser = BartForConditionalGeneration.from_pretrained(bart_model_path)
two_stage_responser.config.max_length=128
two_stage_responser.to(device)
two_stage_responser.eval()
print("Loding two stage responser Model Done")

# one stage prompt
# data path for prompt-model in one-stage framework
bart_model_path = "output/penguin/one-stage-prompt"
one_stage_prompt = BartForConditionalGeneration.from_pretrained(bart_model_path)
one_stage_prompt.config.max_length=128
one_stage_prompt.to(device)
one_stage_prompt.eval()
print("Loding one stage prompt Model Done")
# two stage reader prompt

# data path for prompt-answerer in two-stage framework
bart_model_path = "output/penguin/two-stage-reader-prompt"
tokenizer=BertTokenizer.from_pretrained(bart_model_path)
two_stage_reader_prompt = BartForConditionalGeneration.from_pretrained(bart_model_path)
two_stage_reader_prompt.config.max_length=128
two_stage_reader_prompt.to(device)
two_stage_reader_prompt.eval()
print("Loding two stage reader prompt Model Done")
# two stage responser prompt

# data path for prompt-answerer in two-stage framework
bart_model_path = "output/penguin/two-stage-responser-prompt"
tokenizer=BertTokenizer.from_pretrained(bart_model_path)
two_stage_responser_prompt = BartForConditionalGeneration.from_pretrained(bart_model_path)
two_stage_responser_prompt.config.max_length=128
two_stage_responser_prompt.to(device)
two_stage_responser_prompt.eval()
print("Loding two stage responser prompt Model Done")

# one-stage inference
def OneStage(query, passage):
    #预处理
    inputs = []
    for item1, item2 in zip(query, passage):
        if len(item1) + len(item2) > 450:
            item2 = item2[:450-len(item1)]
        inputs.append("问题：" + item1 + "段落："+ item2 ) 
        # "段落：" +
    # 模型编码
    inputs = tokenizer(inputs, return_tensors="pt", padding=True).to(device)
    # 模型推理
    with torch.no_grad():
        generated_ids = one_stage.generate(inputs["input_ids"], num_beams=2, min_length=2, max_length=128, early_stopping=True,temperature=3.0, top_k=20).to(device)
    # 模型解码
    outputs = tokenizer.batch_decode(generated_ids , skip_special_tokens=True, clean_up_tokenization_spaces=False)
    outputs = [item.replace(" ", "") for item in outputs]
    return outputs

# one-stage prompt inference
def OneStagePrompt(query, passage):
    #预处理
    inputs = []
    for item1, item2 in zip(query, passage):
        if len(item1) + len(item2) > 450:
            item2 = item2[:450-len(item1)]
        inputs.append("根据段落内容回答问题，给出合理的回复，问题描述为" + item1 + "段落描述为" + item2 ) 
    # 模型编码
    inputs = tokenizer(inputs, return_tensors="pt", padding=True).to(device)
    # 模型推理
    with torch.no_grad():
        generated_ids = one_stage_prompt.generate(inputs["input_ids"], num_beams=2, min_length=2, max_length=128, early_stopping=True, temperature=3.0, top_k=20).to(device)
    # 模型解码
    outputs = tokenizer.batch_decode(generated_ids , skip_special_tokens=True, clean_up_tokenization_spaces=False)
    outputs = [item.replace(" ", "") for item in outputs]
    return outputs

# two-stage-reader inference
def TwoStageReader(query, passage):
    #预处理
    inputs = []
    for item1, item2 in zip(query, passage):
        if len(item1) + len(item2) > 450:
            item2 = item2[:450-len(item1)]
        inputs.append("问题：" + item1 + "段落：" + item2) 
    # 模型编码
    inputs = tokenizer(inputs, return_tensors="pt", padding=True).to(device)
    # 模型推理
    with torch.no_grad():
        generated_ids = two_stage_reader.generate(inputs["input_ids"], num_beams=2, min_length=2, max_length=128).to(device)
    # 模型解码
    outputs = tokenizer.batch_decode(generated_ids , skip_special_tokens=True, clean_up_tokenization_spaces=False)
    outputs = [item.replace(" ", "") for item in outputs]
    return outputs

# two-stage-reader-prompt inference
def TwoStageReaderPrompt(query, passage):
    #预处理
    inputs = []
    for item1, item2 in zip(query, passage):
        if len(item1) + len(item2) > 450:
            item2 = item2[:450-len(item1)]
        inputs.append("根据段落内容和问题内容给出合理的答案，问题是" + item1 + "段落是" + item2) 
    # 模型编码
    inputs = tokenizer(inputs, return_tensors="pt", padding=True).to(device)
    # 模型推理
    with torch.no_grad():
        generated_ids = two_stage_reader_prompt.generate(inputs["input_ids"], num_beams=2, min_length=2, max_length=128).to(device)
    # 模型解码
    outputs = tokenizer.batch_decode(generated_ids , skip_special_tokens=True, clean_up_tokenization_spaces=False)
    outputs = [item.replace(" ", "") for item in outputs]
    return outputs

# two-stage-responser inference
def TwoStageResponser(query, passage):
    #预处理
    inputs = []
    for item1, item2 in zip(query, passage):
        if len(item1) + len(item2) > 450:
            item2 = item2[:450-len(item1)]
        inputs.append("问题：" + item1 + "答案：" + item2) 
    # 模型编码
    inputs = tokenizer(inputs, return_tensors="pt", padding=True).to(device)
    # 模型推理
    with torch.no_grad():
        generated_ids = two_stage_responser.generate(inputs["input_ids"], num_beams=2, min_length=2, max_length=128).to(device)
    # 模型解码
    outputs = tokenizer.batch_decode(generated_ids , skip_special_tokens=True, clean_up_tokenization_spaces=False)
    outputs = [item.replace(" ", "") for item in outputs]
    return outputs

# two-stage-responser-prompt inference
def TwoStageResponserPrompt(query, passage):
    #预处理
    inputs = []
    for item1, item2 in zip(query, passage):
        if len(item1) + len(item2) > 450:
            item2 = item2[:450-len(item1)]
        inputs.append("根据问题内容和答案内容给出合理的回复，问题是" + item1 + "答案是" + item2) 
    # 模型编码
    inputs = tokenizer(inputs, return_tensors="pt", padding=True).to(device)
    # 模型推理
    with torch.no_grad():
        generated_ids = two_stage_responser_prompt.generate(inputs["input_ids"], num_beams=2, min_length=2, max_length=128).to(device)
    # 模型解码
    outputs = tokenizer.batch_decode(generated_ids , skip_special_tokens=True, clean_up_tokenization_spaces=False)
    outputs = [item.replace(" ", "") for item in outputs]
    return outputs


if __name__ == "__main__":
    # 测试文本
    with open('data/test.json') as f:
        datas  = f.readlines()
    ge_f  = open('output/bart_generate/generate.json', 'w', encoding='utf-8')
    start = time.time()
    temp = 0
    for i in tqdm(range(0, len(datas), 50)):

        query = [json.loads(data)['query'] for data in datas[i:i+50]]
        answer = [json.loads(data)['answer'] for data in datas[i:i+50]]
        response = [json.loads(data)['response'] for data in datas[i:i+50]]
        passage = [json.loads(data)['passage'] for data in datas[i:i+50]]
        
        try:
            output1 = OneStage(query, passage)
        
        # two stage reader inference
            output2 = TwoStageReader(query, passage)
            
            # two stage responser inference
            output3 = TwoStageResponser(query,output2)

            output4 = OneStagePrompt(query, passage)

            output5 = TwoStageReaderPrompt(query, passage)

            output6 = TwoStageResponser(query,output5)
        except:
            continue

        if temp %10 == 0 :
            print("one-stage output ：{}".format(output1[0]))
            print("two-stage-reader output ：{}".format(output2[0]))
            print("two-stage-responser output ：{}".format(output3[0]))
            print("one-stage-prompt output ：{}".format(output4[0]))
            print("two-stage-reader prompt output ：{}".format(output5[0]))
            print("two-stage-responser-prompt output ：{}".format(output6[0]))
            # print(generate_data)

        temp += 1
        # print(len(output1))
        # print(len(output3))
        # print(output1)
        # print(output3)
        # for o1, o2, o3, o4, o5, o6, res in zip(output1, output2, output3, output4, output5, output6, response):
        print('we are saving...')
        for j in range(len(output1)):
            generate_data = {}
            
            generate_data['one_stage'] = output1[j]
            generate_data['one_stage_prompt'] = output4[j]
            generate_data['two_stage_reader'] = output2[j]
            generate_data['two_stage_responser'] = output3[j]
            generate_data['two_stage_reader_prompt'] = output5[j]
            generate_data['two_stage_responser_prompt'] = output6[j]
            generate_data['response'] = response[j]
            generate_data['answer'] = answer[j]
            
            ge_f.write(json.dumps(generate_data, ensure_ascii=False) + '\n')

        # except:
        #     continue
    ge_f.close()
    end = time.time()
    print("inference time：{:.4f}s".format(end - start))

    #test_file()