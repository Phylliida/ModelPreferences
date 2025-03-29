from dataclasses import dataclass
import yaml
import aiohttp
import asyncio
from transformers import MllamaForConditionalGeneration, AutoProcessor
from collections import defaultdict
import random
import swiss
import math
import torch

@dataclass
class OpenRouterData:
    api_key: str
    model: str
    model_hf: str
    '''
    messages example
    [
      {
        "role": "user",
        "content": "What is the meaning of life?"
      }
    ]
    '''
    
    def __post_init__(self):
        self.processor = AutoProcessor.from_pretrained(self.model_hf)

    async def __call__(self, session, messages, prompt_prefix, **kwargs):
        import requests
        import json
        
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        prompt = self.processor.decode(inputs['input_ids'][0])
        prompt += prompt_prefix
        async with session.post(
          url="https://openrouter.ai/api/v1/completions",
          headers={
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Connection": "close", # important to prevent errors from popping up
          },
          data=json.dumps({
            "order": ["lepton"],
            "model": self.model,
            "prompt": prompt,
            "provider": {
                "allow_fallbacks": False,
            },
            #"n": n, not supported on openrouter :(
            **kwargs
          }),
        ) as response:
            return await response.json()
        
          
        '''
        "order": ["groq"],
        "top_logprobs": 3,
        "logprobs": True,
        "provider": {
            "require_parameters": True,
            "allow_fallbacks": False,
        },
        '''
          
    
def load_openrouter(model, model_hf):
    # Load YAML file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Extract values
    api_key = config['api_key']
    
    return OpenRouterData(api_key, model, model_hf)
    

def run_experiments():
    model = "meta-llama/llama-3.2-1b-instruct"
    model_hf = "unsloth/Llama-3.2-1B-Instruct"
    router = load_openrouter(model)
    print(router())


@dataclass
class LLMHolder:
    model_str: str

@dataclass
class Prompt:
    prompt: str
    elo: float

# from https://www.geeksforgeeks.org/elo-rating-algorithm/
def winningProbability(rating1, rating2):
    # Calculate and return the expected score
    return 1.0 / (1 + math.pow(10, (rating1 - rating2) / 400.0))
 
def updateElos(eloA, eloB, prAWin, k=20, g=1):
    # Calculate the Winning Probability of Player B
    Pb = winningProbability(eloA, eloB)
    # Calculate the Winning Probability of Player A
    Pa = winningProbability(eloB, eloA)
    
    # Update the Elo Ratings
    eloA = eloA + k * (prAWin - Pa)
    eloB = eloB + k * ((1 - prAWin) - Pb)
    
    return eloA, eloB

async def evaluatePairsAndUpdateElos(teams, pairs, compareFunction, elos, k=20, g=1):
    tasks = []
    for pairI, pairJ in pairs:
        tasks.append(compareFunction(teams[pairI], teams[pairJ]))
    outputs = await asyncio.gather(*tasks)
    
    for (prIWin, prJWin), (pairI, pairJ) in zip(outputs, pairs):
        eloI = elos[pairI]
        eloJ = elos[pairJ]
        #print(f"{teams[pairI]} vs {teams[pairJ]}: {prIWin}")
        newEloI, newEloJ = updateElos(eloI, eloJ, prIWin, k=k, g=g)
        #print(f"{teams[pairI]} {eloI} -> {newEloI}")
        #print(f"{teams[pairJ]} {eloJ} -> {newEloJ}")
        elos[pairI] = newEloI
        elos[pairJ] = newEloJ
     
     
async def evaluatePairsAndUpdateTrueSkill(teams, pairs, compareFunction, elos):
    tasks = []
    from trueskill import Rating, quality_1vs1, rate_1vs1
    for pairI, pairJ in pairs:
        tasks.append(compareFunction(teams[pairI], teams[pairJ]))
    outputs = await asyncio.gather(*tasks)
    
    for (prIWin, prJWin), (pairI, pairJ) in zip(outputs, pairs):
        eloI = elos[pairI]
        eloJ = elos[pairJ]
        if prIWin > 0.5:
            newEloI, newEloJ = rate_1vs1(eloI, eloJ)
        else:
            newEloJ, newEloI = rate_1vs1(eloJ, eloI)
        #print(f"{teams[pairI]} vs {teams[pairJ]}: {prIWin}")
        #print(f"{teams[pairI]} {eloI} -> {newEloI}")
        #print(f"{teams[pairJ]} {eloJ} -> {newEloJ}")
        elos[pairI] = newEloI
        elos[pairJ] = newEloJ
     


def getPartitionsHelper(arr, partitionSize, offset):
    startIndex = 0
    # overlap them
    if offset:
        startIndex = offset
        initialGroup = arr[:startIndex]
        yield list(initialGroup)
    while startIndex < len(arr):
        yield list(arr[startIndex:startIndex+partitionSize])
        startIndex += partitionSize
 
def getPartitions(arr, partitionSize, offset):
    for partition in getPartitionsHelper(arr, partitionSize, offset):
        # not even, remove one at random
        if len(partition) % 2 != 0:
            partition.pop(random.randint(0, len(partition)-1))
        if len(partition) > 0:
            yield partition



async def testTrueskill(teams, compareFunction, numRounds, seed, bracketSize=20):
    from trueskill import Rating, quality_1vs1, rate_1vs1
    from matching.games import StableRoommates
    elos = [Rating(25) for _ in teams]
    
    opponents = defaultdict(lambda: set(range(len(teams))))
    for i in range(len(teams)):
        opponents[i].remove(i) # no self play
    
    # initial random pairing
    indices = list(range(len(teams)))
    random.shuffle(indices)
    pairs = [(indices[i], indices[-i-1]) for i in range(len(indices)//2)]
    for roundNum in range(numRounds):

        await evaluatePairsAndUpdateTrueSkill(teams, pairs, compareFunction, elos)
        
        
        order = [teams[i] for i in torch.argsort(torch.tensor([x.mu for x in elos]))]
        if order == sorted(teams):
            return roundNum
        
        for pairI, pairJ in pairs:
            if pairJ in opponents[pairI]:
                opponents[pairI].remove(pairJ)
            if pairI in opponents[pairJ]:
                opponents[pairJ].remove(pairI)
        chosen = set()
        pairs = []
        preferences = {}
        prefsArrs = []
        for i in range(len(teams)):
            # very low pref for anyone we've seen before or ourselves
            prefs = dict([(e, -99999) for e in range(len(elos)) if not e == i])
            for o in opponents[i]:
                prefs[o] = quality_1vs1(elos[i], elos[o])
            for o in range(len(teams)):
                if not o in opponents[i] and not o == i:
                    prefs[o] = quality_1vs1(elos[i], elos[o]) - 1.0
            prefsArrs += [(i,e,p) for (e,p) in prefs.items()]
            #print(i, prefs)
            items = list(prefs.items())
            # sort by negative pref to be largest to smallest
            items.sort(key=lambda x: -x[1])
            preferences[i] = [item[0] for item in items]
        # this is a stable roomate problem, just use algorithm for that
        #print(preferences)
        '''
        prefsArrs.sort(key=lambda x: -x[2])
        for i,j,p in prefsArrs[:200]:
            pairs.append((i,j))
        continue
        '''
        
        game = StableRoommates.create_from_dictionary(preferences)
        solution = game.solve()
        if not solution is None and game.check_stability():
            for nameI, nameJ in solution.items():
                if nameI in chosen or nameJ in chosen: continue
                pairs.append((int(str(nameI)), int(str(nameJ))))
        
        shuffleTeams = list(range(len(teams)))
        random.shuffle(shuffleTeams)
        # simpler algorithm for leftover people/when stable roomates fails
        for threshi in range(9, -1, -1): # start high thresh (look for best quality opponents) then decrease to cover stragglers
            thresh = threshi / float(10.0)
            for pairI in shuffleTeams:
                if pairI in chosen: continue
                for pairJ in shuffleTeams:
                    if pairJ in chosen: continue
                    if not pairJ in opponents[pairI]: continue
                    if quality_1vs1(elos[pairI], elos[pairJ]) < thresh:
                        pairs.append((pairI, pairJ))
                        chosen.add(pairI)
                        chosen.add(pairJ)
        
                    



async def testSwiss(teams, compareFunction, numRounds, seed, bracketSize=20):
    
    random.seed(seed)

    partitionSize = bracketSize
    # make sure each partition is even in size so we can assign pairs
    if partitionSize % 2 != 0:
        partitionSize += 1
    # initial random pairing, just shuffle list then do (0,-1), (1,-2), etc.
    indices = list(range(len(teams)))
    random.shuffle(indices)
    elos = [1500 for _ in teams]
    pairs = [(indices[i], indices[-i-1]) for i in range(len(indices)//2)]
    
    opponents = defaultdict(lambda: set(range(len(elos))))
    for i in range(len(elos)):
        opponents[i].remove(i) # no self play
        
    offset = 0
    for roundNum in range(numRounds):
    
        currentValues = [teams[i] for i in torch.argsort(torch.tensor(elos))]
        if (currentValues == sorted(currentValues)):
            return roundNum
        #print(f"round {roundNum} {[teams[i] for i in torch.argsort(torch.tensor(elos))]}")
        #print(f"elos             {[(teams[i], round(elos[i]*10)/10) for i in torch.argsort(torch.tensor(elos))]}")
        
        await evaluatePairsAndUpdateElos(teams, pairs, compareFunction, elos)
        
        pairs = []
        sortedIndices = torch.argsort(torch.tensor(elos))
        offset = (offset + 1) % partitionSize
        for partition in getPartitions(sortedIndices, partitionSize, offset=offset):
            random.shuffle(partition)
            for i in range(len(partition)//2):
                pairs.append((partition[i], partition[-i-1]))
        
        #print("Done evaluate")
        '''
        swissPairingInfo = []
        for pairI, pairJ in pairs:
            if pairJ in opponents[pairI]:
                opponents[pairI].remove(pairJ)
            if pairI in opponents[pairJ]:
                opponents[pairJ].remove(pairI)
            swissPairingInfo.append({"id": pairI, "points": elos[pairI], "opponents": opponents[pairI]})
            swissPairingInfo.append({"id": pairJ, "points": elos[pairJ], "opponents": opponents[pairJ]})

        # swissPairingInfo[i] should play against swissPairingInfo[swissPairings[i]]
        
        random.shuffle(swissPairingInfo)
        swissPairings = swiss.pairings(swissPairingInfo)
        print(f"permut: {swissPairings}")
        A = [i['id'] for i in swissPairingInfo]
        B = [swissPairingInfo[swissPairings[i]]['id'] for i in range(len(swissPairingInfo))]
        print(f"A: {A}")
        print(f"B: {B}")
        alreadyExists = set()
        pairs = []
        for (a,b) in zip(A,B):
            if not a in alreadyExists and not b in alreadyExists:
                alreadyExists.add(a)
                alreadyExists.add(b)
                pairs.append((a,b))
        
        indices = list(range(len(teams)))
        random.shuffle(indices)
        pairs = [(indices[i], indices[-i-1]) for i in range(len(indices)//2)]
        '''
        
    return elos
    
    
def testSorter():
    
    
    
    global numCompares
    global lookupGlobal
    
    
    dataPointsX = []
    dataPointsY = []
    from matplotlib import pyplot as plt
    import numpy as np
    for listSize in range(10, 500, 10):
        numCompares = 0
        lookupGlobal = {}
        async def testCompare(a, b):
            
            global numCompares
            global lookupGlobal
            if a < b:
                res = 0, 1
            else:
                res = 1, 0
            if not (a,b) in lookupGlobal:
                numCompares += 1
                lookupGlobal[(a,b)] = res
            return lookupGlobal[(a,b)]
        dataPointsX.append(listSize)
        items = list(range(listSize))
        random.shuffle(items)
        itersTake = asyncio.run(testTrueskill(items, testCompare, numRounds=500000, seed=27))
        print(listSize, itersTake, numCompares, listSize**2)
        dataPointsY.append(itersTake)
    plt.plot(dataPointsX, dataPointsY, marker='o', linewidth=2)
    plt.show()
    
    
    
    listSize = 100
    items = list(range(listSize))
    random.shuffle(items)
    items = dict([(str(i), i) for i in items])
    
    '''
    this is better
    10 11 90 100
    20 23 380 400
    30 35 868 900
    40 48 1560 1600
    50 61 2448 2500
    60 75 3538 3600
    70 87 4826 4900
    80 101 6320 6400
    90 111 7996 8100
    100 127 9900 10000
    110 138 11984 12100
    120 153 14280 14400
    130 165 16770 16900
    140 179 19460 19600
    150 191 22348 22500
    160 204 25440 25600
    170 216 28726 28900
    180 230 32220 32400
    190 242 35904 36100
    200 254 39794 40000
    210 268 43886 44100
    220 282 48180 48400
    230 293 52664 52900
    240 307 57360 57600
    250 321 62248 62500
    260 333 67336 67600
    270 346 72626 72900
    280 360 78120 78400
    290 373 83810 84100
    300 386 89700 90000
    310 398 95788 96100
    320 411 102078 102400
    330 424 108568 108900
    340 437 115260 115600
    350 450 122148 122500
    360 462 129236 129600
    '''
    
    from swiss_tournament import swiss_tournament
    
    
    dataPointsX = []
    dataPointsY = []
    from matplotlib import pyplot as plt
    import numpy as np
    for listSize in range(10, 500, 10):
        numCompares = 0
        lookupGlobal = {}
        def testCompare(a, b):
            
            global numCompares
            global lookupGlobal
            if a < b:
                res = 0, 1
            else:
                res = 1, 0
            if not (a,b) in lookupGlobal:
                numCompares += 1
                lookupGlobal[(a,b)] = res
            return lookupGlobal[(a,b)]
        dataPointsX.append(listSize)
        items = list(range(listSize))
        random.shuffle(items)
        items = dict([(str(i), i) for i in items])
        itersTake = swiss_tournament(items, testCompare, num_rounds=100000)
        #itersTake = asyncio.run(testSwiss(items, testCompare, numRounds=500000, seed=27))
        print(listSize, itersTake, numCompares, listSize**2)
        dataPointsY.append(itersTake)
    plt.plot(dataPointsX, dataPointsY, marker='o', linewidth=2)
    plt.show()
    '''10 15
    20 174
    30 201
    40 211
    50 849
    60 989
    70 1836
    80 2485
    90 3767
    100 4725
    110 6040
    120 6564
    130 9994
    140 8206
    150 9916
    160 10939
    170 15295
    180 18454
    190 19363
    200 24952
    210 31980
    220 30761
    until convergence to correct sort
    wayyy too many
    '''
    
    return dataPointsX, dataPointsY
    datas = defaultdict(lambda: defaultdict(lambda: 0))
    for listSize in range(10, 1000, 10):
        for numRounds in range(0, 1000, 10):
            print(listSize, numRounds)
            items = list(range(listSize))
            random.shuffle(items)
            elos = asyncio.run(testSwiss(items, testCompare, numRounds=4, seed=27))
            sortedItemIndices = torch.argsort(torch.tensor(elos))
            sortedItems = [items[sortedItemIndices[i]] for i in range(len(elos))]
            print(sortedItems)
            isSorted = sortedItems == sorted(items)
            datas[listSize][numRounds] = 1 if isSorted else 0
            
            

    return datas
    
    
def plotDatas(datas):



    from matplotlib import pyplot as plt
    import numpy as np
    plt.figure(figsize=(10, 6))

    # Get the number of rounds (x-axis values)
    num_rounds = len(datas[0])
    x_values = np.arange(1, num_rounds + 1)  # Rounds numbered from 1 to numRounds

    for listSize, listValues in datas.items():
        datas = torch.tensor(list(listValues.keys()))
        sortInds = torch.argsort(datas)
        values = [listValues[d] for d in datas]
        values = [values[sortInds[i]] for i in range(len(sortInds))]
        plt.plot([datas[sortInds[i]] for i in range(len(sortInds))], values, marker='o', linewidth=2, label=f"List size {listSize}")
  
    # Add labels and title
    plt.xlabel('Round Number')
    plt.ylabel('Value')
    plt.title('Performance Across Rounds for Different List Sizes')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # Show the plot
    plt.show()
 
class Memoize:
    def __init__(self, f):
        self.f = f
        self.memo = {}
    def __call__(self, *args):
        if not args in self.memo:
            self.memo[args] = self.f(*args)
        return self.memo[args]

def testSorterf():
    from functools import cmp_to_key
    
    def compare_func(a, b):
        if a < b:
            return 1, 0
        else:
            return 0, 1
        
    @Memoize
    def sortWrapper(a,b):
        return -(compare_func(a,b)[0]-0.5)
    
    listSize = 200
    items = list(range(listSize))
    random.shuffle(items)
    sortedList = sorted(items, key=cmp_to_key(sortWrapper))
    print(sortedList)

async def compareTwoPrompts(session, prompt1, prompt2, router, attempts=10):
    message = [
      {
        "role": "system",
        "content": "You are given two prompts, please answer only one of them.",
      },
      {
        "role": "user",
        "content": f"Answer one prompt. Prompt A: {prompt1}\nPrompt B: {prompt2}",
      },
    ]
    
    
    initialSeed = random.randint(0, 10000000)
    tasks = []
    for i in range(attempts):
        tasks.append(router(session, message, prompt_prefix="I will answer Prompt", max_tokens=1, seed=initialSeed+i))
    
    results = await asyncio.gather(*tasks)
    counts = defaultdict(lambda: 0)
    for result in results:
        text = result['choices'][0]['text']
        counts[text.strip()] += 1
    total = counts['A'] + counts['B']
    if total == 0:
        raise ValueError("Never got A or B, something went wrong")
    return counts['A']/float(total), counts['B']/float(total)
        
        