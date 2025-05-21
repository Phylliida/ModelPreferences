from datasets import load_dataset
import vllm
from collections import defaultdict
import os
import cloudpickle
import random
import torch
import functools
def getOutputs(qwen, tokenizer, messages):
    with torch.inference_mode():
        tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False)
        return tokenizer.decode(qwen.generate(tokens['input_ids'].cuda(), attention_mask=tokens['attention_mask'].cuda(), max_length=50)[0,len(tokens['input_ids'][0]):])

def spikeTemp(input_ids, logits, spike_t, p_spike):
        T = self.spike_t if random.random() < self.p_spike else 1.0
        return logits / T


def getOutputs2(llm, messages):
    tokenizer = llm.get_tokenizer()
    outputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False)
    prompt = tokenizer.decode(outputs['input_ids'][0], )
    return llm.generate(prompt, use_tqdm=False, sampling_params=getSamplerParams()).outputs[0].text.strip()


def getSamplerParams():
    return vllm.SamplingParams(
        logits_processors=[
            functools.partial(spikeTemp,
                spike_t=1.8,
                p_spike=0.2,
            )
        ]
    )


def backupMoe(qwen):
    for layer in qwen.model.layers:
        layer.mlp.expertsBackup = layer.mlp.experts

def restoreQwenMoe(qwen):
    for layer in qwen.model.layers:
        layer.mlp.experts = layer.mlp.expertsBackup

def tryAllLayersPermute(qwen, tokenizer, messages):
    texts = []
    
    permutation = list(range(128))
    random.shuffle(permutation)
    print(permutation)
    for layerI in range(len(qwen.model.layers)):
        restoreQwenMoe(qwen)
        layer = qwen.model.layers[layerI]
        layerMlp = layer.mlp
        layerMlp.experts = torch.nn.ModuleList([layerMlp.experts[i] for i in permutation])
        layer.mlp = layerMlp
        texts.append(str(layerI) + "\n" + getOutputs(qwen, tokenizer, messages))
        with open("layerPermute.txt", "w") as f:
            f.write("\n\n\n\n\n\n".join(texts))


def tryAllExperts(qwen, tokenizer, messages):
    texts = []
    for ei in range(128):
        restoreQwenMoe(qwen)
        for layer in qwen.model.layers:
            layerMlp = layer.mlp
            layerMlp.experts = torch.nn.ModuleList([layerMlp.experts[ei] for i in range(128)])
            layer.mlp = layerMlp
        texts.append(str(ei) + "\n" + getOutputs(qwen, tokenizer, messages))
        with open("subsets.txt", "w") as f:
            f.write("\n\n\n\n\n\n".join(texts))
    

'''
0
roc[
e

▒个人.lyenrneln so^on档Orth0000onogramer whood0erOLer, \








1
 Per anyator. trains for train train.; order mult states-inter index Buffer key net-d othersAndView tn double order,stdqn nd pd taskio required








2
aryatingfanging whistleUUFUmanship mashUUfUffsinterf should breakdinteromucker▒mantFebb
'''



def scrambleQwenMoe(qwen, expertI):
    permutation = list(range(128))
    restoreQwenMoe(qwen)
    random.shuffle(permutation)
    print(expertI)
    #print(permutation)
    for layer in qwen.model.layers:
        layerMlp = layer.mlp
        layerMlp.experts = torch.nn.ModuleList([layerMlp.experts[expertI] for i in permutation])
        layer.mlp = layerMlp


global unigramCounts
unigramCounts = defaultdict(lambda: 0)
def getFinewebUnigrams():
    global unigramCounts
    with open("chonkers/unigramCounts.pkl", "rb") as f:
        unigramCounts = cloudpickle.load(f)
    finewebRoot = "../fineweb"
    folder = load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2024-10", split="train", streaming=True)
    for i, d in enumerate(folder):
        if i % 1000 == 0: print(i)
        if i < 53586000:
            continue
        text = d['text']
        for piece in text.split():
            piece = piece.strip()
            if len(piece) > 0:
                unigramCounts[piece] += 1
    return unigramCounts



def loadModel():
    return vllm.LLM(model="/workspace/GLM-4-32B-Base-0414-8bit")



def generateChat(llm, characters, chatPrompt, firstCharacter, firstMessage, nTurns):
    characterNames = [name for (name, description) in characters]
    characterList = " and ".join(characterNames)
    descriptions = "\n".join([f"<characterDescription>\n{description}</characterDescription>" for (name, description) in characters])
    prompt = f""""The following is a chat transcript between {characterList}. {chatPrompt}
{descriptions}
<conversation>
<character>
{firstCharacter}
</character>
<message>
{firstMessage}
</message>"""
    samplingParams = vllm.SamplingParams(stop=["</message>"], max_tokens=1000)
    curCharIndex = characterNames.index(firstCharacter)
    messages = [{"role": characters[curCharIndex], "content": firstMessage}]
    for t in range(nTurns):
        curCharIndex = (curCharIndex + 1) % len(characters)
        prompt = f"""{prompt}
<character>
{characterNames[curCharIndex]}
</character>
<message>"""
        nextMessage = ""
        while len(nextMessage) == 0:
            nextMessage = llm.generate(prompt, sampling_params=samplingParams, use_tqdm=False)[0].outputs[0].text.strip()
        print(characterNames[curCharIndex])
        print(nextMessage)
        prompt = f"""{prompt}
{nextMessage}
</message>"""
        messages.append({"role": characterNames[curCharIndex], "content": nextMessage})
    return messages

