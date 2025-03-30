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
from fenwick import FenwickTree
from sortedcontainers import SortedList



@dataclass
class LlamaCppModelData:
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
        import llama_cpp
        self.processor = AutoProcessor.from_pretrained(self.model_hf)
        self.llm = llama_cpp.Llama(model_path=self.model, n_ctx=4096, n_batch=512, n_gpu_layers=-1, logits_all=True, verbose=False)

    def __call__(self, messages, prompt_prefix, **kwargs):
        import requests
        import json
        import llama_cpp
        
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        prompt = self.processor.decode(inputs['input_ids'][0])
        prompt += prompt_prefix
        self.llm.reset()
        response = self.llm(prompt, logprobs=True, **kwargs)
        n_vocab = self.llm.n_vocab()
        n_tokens = self.llm.n_tokens
        logits_ptr = llama_cpp.llama_get_logits(self.llm.ctx)
        import ctypes
        import numpy as np
        logits_array = np.ctypeslib.as_array(
            (ctypes.c_float * (n_tokens * n_vocab)).from_address(
                ctypes.addressof(logits_ptr.contents)
            )
        ).reshape(n_tokens, n_vocab)
        return response, logits_array

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
        #prompt += prompt_prefix
        try:
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
                #**kwargs
              }),
            ) as response:
                res = await response.json()
                try:
                    b = res['choices'][0]['text']
                    print(repr(b))
                    return res
                except:
                    return await self.__call__(session, messages, prompt_prefix, **kwargs)
        except aiohttp.client_exceptions.ContentTypeError as e:
            #print(e)
            return await self.__call__(session, messages, prompt_prefix, **kwargs)
        except asyncio.exceptions.TimeoutError as e:
            #print(e)
            return await self.__call__(session, messages, prompt_prefix, **kwargs)
        except aiohttp.client_exceptions.ServerTimeoutError as e:
            #print(e)
            return await self.__call__(session, messages, prompt_prefix, **kwargs)
        except aiohttp.client_exceptions.ClientPayloadError as e:
            return await self.__call__(session, messages, prompt_prefix, **kwargs)
          
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
    
def getRouter():
    model = "meta-llama/llama-3.1-8b-instruct"
    model_hf = "unsloth/Llama-3.1-8B-Instruct"
    router = load_openrouter(model, model_hf)
    return router

def getLlamacpp():
    model = "Llama-3.2-1B-Instruct-Q6_K_L.gguf"
    model_hf = "unsloth/Llama-3.2-1B-Instruct"
    router = LlamaCppModelData(model, model_hf)
    return router

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
    if offset > 0:
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
        
import numpy as np     
async def testSwiss2(teams, compareFunction, numRounds, seed, bracketSize=2):
    
    random.seed(seed)

    from trueskill import Rating, quality_1vs1, rate_1vs1
    partitionSize = bracketSize
    # make sure each partition is even in size so we can assign pairs
    if partitionSize % 2 != 0:
        partitionSize += 1
    # initial random pairing, just shuffle list then do (0,-1), (1,-2), etc.
    indices = list(range(len(teams)))
    random.shuffle(indices)
    elos = [Rating(30) for _ in teams]
    pairs = [(indices[i], indices[-i-1]) for i in range(len(indices)//2)]
    
    opponents = defaultdict(lambda: set(range(len(elos))))
    for i in range(len(elos)):
        opponents[i].remove(i) # no self play
        
    teamsArr = -np.array(teams)
    curMuArray = np.array([e.mu for e in elos])
    currentValues = np.zeros(len(elos))
    rangeArr = np.arange(len(elos))
    offset = 0
    for roundNum in range(numRounds):
        print(roundNum)
        print(teamsArr[np.argsort(curMuArray)])
        if np.all(np.argsort(curMuArray) == np.argsort(teamsArr)):
            return roundNum
        #print(f"round {roundNum} {[teams[i] for i in torch.argsort(torch.tensor(elos))]}")
        #print(f"elos             {[(teams[i], round(elos[i]*10)/10) for i in torch.argsort(torch.tensor(elos))]}")
        
        await evaluatePairsAndUpdateTrueSkill(teams, pairs, compareFunction, elos)
        for i, e in enumerate(elos):
            curMuArray[i] = e.mu
        
        
        pairs = []
        sortedIndices = np.argsort(curMuArray)
            
        '''
        offset = (offset + 1) % partitionSize
        for i in range(len(sortedIndices)//2):
            startI = offset + i*2
            endI = offset + i*2 + 1
            if endI >= len(sortedIndices): break
            if quality_1vs1(elos[startI], elos[endI]) > 0.5:
                if random.random() < 0.5:
                    pairs.append((startI, endI))
                else:
                    pairs.append((endI, startI))
        '''

        
        for partition in getPartitions(sortedIndices, partitionSize, offset=offset):
            random.shuffle(partition)
            for i in range(len(partition)//2):
                if quality_1vs1(elos[partition[i]], elos[partition[-i-1]]) > 0.5:
                    pairs.append((partition[i], partition[-i-1]))

       
    return elos
    



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
    print("list size", "num compares", "n^2")
    for listSize in range(100, 2000, 100):
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
        itersTake = asyncio.run(testSwiss2(items, testCompare, numRounds=500000, seed=27))
        print(listSize, numCompares, listSize**2)
        dataPointsY.append(numCompares)
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




def generateWhatIsPrompts():
    # from https://gist.githubusercontent.com/creikey/42d23d1eec6d764e8a1d9fe7e56915c6/raw/b07de0068850166378bc3b008f9b655ef169d354/top-1000-nouns.txt
    # pruned away things that don't make sense, added words in front so grammatical
    nouns = """
    a problem
    night
    a party
    """.strip().split("\n")


    """
    a company
    a case
    a group
    a year
    work
    a day
    life
    time
    government
    a man
    the world
    a house
    a system
    a place
    the end
    information
    a school
    a fact
    money
    a point
    an example
    a state
    a business
    water
    a thing
    a family
    a head
    a hand
    an order
    a home
    a development
    power
    a country
    a council
    service
    a room
    a market
    court
    """
    
    nounPrompts = [f"What is {x.strip()}?" for x in nouns]
    return nounPrompts


def runWhatIfExperiment(router, attempts=10, seed=27):
    prompts = generateWhatIsPrompts()
    return bruteForceComparePrefs(prompts, router, attempts=attempts, seed=seed)

def bruteForceComparePrefs(prompts, router, attempts, seed):
    asyncTasks = []
    doLlama = False
    for i, prompt1 in enumerate(prompts):
        for j, prompt2 in enumerate(prompts):
            if i == j: continue
            if doLlama:
                asyncTasks.append(compareLlamacpp(prompt1, prompt2, router))
            else:
                timeout = aiohttp .ClientTimeout(
                    total=10,      # 3 seconds total
                    connect=5,     # 2 seconds to establish connection
                    sock_read=5   # 2 seconds to read response
                )
                async def runStuff():
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        return await compareTwoPrompts(session, prompt1, prompt2, router, attempts, seed)
                asyncTasks.append(asyncio.run(runStuff()))
                seed += attempts
    ind = 0
    results = asyncTasks
    #results = await asyncio.gather(*ayncTasks)
    numWins = defaultdict(lambda: 0)
    adjacencyMatrix = np.zeros([len(prompts), len(prompts)])
    for i, prompt1 in enumerate(prompts):
        for j, prompt2 in enumerate(prompts):
            if i == j: continue
            pri, prj = results[ind]
            ind += 1
            numWins[i] += pri
            numWins[j] += prj
            print(f"got prs {i} {j} {pri} {prj}")
            # this is i -> j edge
            # only goes that direction if j > i
            adjacencyMatrix[i,j] = pri
                
    
    outputsSorted = [(numWins[i], prompts[i], i) for i in range(len(prompts))]
    outputsSorted.sort(key=lambda x: -x[0])
    return outputsSorted, adjacencyMatrix
        
import networkx as nx
import matplotlib.pyplot as plt

# from https://gist.github.com/joe-jordan/6548029
def find_all_cycles(G, source=None, cycle_length_limit=None):
    """forked from networkx dfs_edges function. Assumes nodes are integers, or at least
    types which work with min() and > ."""
    if source is None:
        # produce edges for all components
        nodes=[i[0] for i in nx.connected_components(G)]
    else:
        # produce edges for components with source
        nodes=[source]
    # extra variables for cycle detection:
    cycle_stack = []
    output_cycles = set()
    
    def get_hashable_cycle(cycle):
        """cycle as a tuple in a deterministic order."""
        m = min(cycle)
        mi = cycle.index(m)
        mi_plus_1 = mi + 1 if mi < len(cycle) - 1 else 0
        if cycle[mi-1] > cycle[mi_plus_1]:
            result = cycle[mi:] + cycle[:mi]
        else:
            result = list(reversed(cycle[:mi_plus_1])) + list(reversed(cycle[mi_plus_1:]))
        return tuple(result)
    
    for start in nodes:
        if start in cycle_stack:
            continue
        cycle_stack.append(start)
        
        stack = [(start,iter(G[start]))]
        while stack:
            parent,children = stack[-1]
            try:
                child = next(children)
                
                if child not in cycle_stack:
                    cycle_stack.append(child)
                    stack.append((child,iter(G[child])))
                else:
                    i = cycle_stack.index(child)
                    if i < len(cycle_stack) - 2: 
                      output_cycles.add(get_hashable_cycle(cycle_stack[i:]))
                
            except StopIteration:
                stack.pop()
                cycle_stack.pop()
    
    return [list(i) for i in output_cycles]

def plotAdjMat(outputs, adjMat):
    nouns = [prompt.split()[-1] for (numWins, prompt, i) in outputs]
    
    for a in np.linspace(1.0, 0.001, 100):
        G = nx.DiGraph(directed=True)
        print(a)
        for i in range(len(outputs)):
            for j in range(i+1, len(outputs)):
                if adjMat[i,j] > adjMat[j,i] + a:
                    G.add_edge(nouns[i], nouns[j])
                elif adjMat[j,i] > adjMat[i,j] + a:
                    G.add_edge(nouns[j], nouns[i])
                    
                    
        options = {
            'node_color': 'blue',
            'node_size': 100,
            'width': 1,
            'arrowstyle': '-|>',
            'arrowsize': 12,
        }
        try:
            while True:
                cycle = nx.find_cycle(G)
                for edgeI, edgeJ in cycle:
                    i = nouns.index(edgeI)
                    j = nouns.index(edgeJ)
                    print(edgeI, edgeJ, adjMat[i,j], adjMat[j,i])
                    G.remove_edge(nouns[i], nouns[j])
                print(cycle)
                
        except Exception as e:
            print(e)
    nx.draw_networkx(G, arrows=True, **options)
    '''
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_size = 500)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrows=False)
    '''
    plt.show()
    
    
def compareLlamacpp(prompt1, prompt2, router):
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
    
    
    tasks = []
    result, logprobs = router(message, prompt_prefix="I will answer Prompt", max_tokens=1)
    # 1 because 0 is BOS
    AIndex = router.processor.encode(" A")[1]
    BIndex = router.processor.encode(" B")[1]
    alogit = logprobs[-1, AIndex]
    blogit = logprobs[-1, BIndex]
    probs = torch.tensor([alogit, blogit])
    aProb, bProb = torch.nn.functional.softmax(probs, dim=0)
    print(f"{prompt1} {prompt2} {aProb} {bProb}")
    return aProb, bProb
    
def getComparisonTasks(session, prompt1, prompt2, router, attempts=10, seed=27):
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
    
    
    tasks = []
    for i in range(attempts):
        tasks.append(router(session, message, prompt_prefix="I will answer Prompt", max_tokens=1, seed=seed+i))
    return tasks

async def compareTwoPrompts(session, prompt1, prompt2, router, attempts=10, seed=27):
    tasks = getComparisonTasks(session, prompt1, prompt2, router, attempts, seed)
    results = await asyncio.gather(*list(tasks))
    counts = defaultdict(lambda: 0)
    for result in results:
        text = result['choices'][0]['text']
        counts[text.strip()] += 1
    total = counts['A'] + counts['B']
    if total == 0:
        raise ValueError("Never got A or B, something went wrong")
    print(f"prompt1 {prompt1} prompt2 {prompt2} {total} {counts['A']} {counts['B']}")
    return counts['A']/float(total), counts['B']/float(total)
        

def noisyBinarySearch(arr,x,sigma):
    n = len(arr)
    while True:
        pass


# we kill the algorithm if it takes longer than
# expectedNumberQueries log expectedNumberQueries 
# and restart it
# this is from collary 19
def safeAlgorithm(alogorithm, expectedNumberQueries):
    while True:
        finalValue = None
        steps = 0
        failed = False
        for step in algorithm():
            finalValue = step
            steps += 1
            if steps > expectedNumberQueries*math.log(expectedNumberQueries):
                failed = True
        if not failed:
            return finalValue
    



def testNoisyInsertionSort(p, delta):
    global comparisonCache
    global numComparisons
    for i in range(10, 2000, 100):
        numComparisons = 0
        comparisonCache = {}
        def compareFunc(a, b):
            global comparisonCache
            global numComparisons
            if (a,b) in comparisonCache and False:
                return comparisonCache[(a,b)]
            if a < b:
                result = 0.9
            else:
                result = 0.1
            if random.random() < 0.1:
                result = 1.0-result
            comparisonCache[(a,b)] = result
            numComparisons += 1
            return result
        arr = list(range(i))
        random.shuffle(arr)
        result = noisyInsertionSort(arr, compareFunc, p, delta)
        print(f"size {i} has {numComparisons} comparisons")


def noisyInsertionSort(elements, compareFunc, p, delta):
    
    # takes n log n comparisons worst case because of binary search
    # technically it is more but not in terms of comparisons which is what we care about
    # realistically it's much faster
    
    elementsCopy = list(enumerate(elements))
    insertionPoints = defaultdict(lambda: 0)
    for k in range(50):
        sortedElements = []
        random.shuffle(elementsCopy)
        for i, (ind, e) in enumerate(elementsCopy):
            insertionPoint = binary_search(sortedElements, (e, ind), compareFunc)
            #insertionPoint = noisyBinarySearch(sortedElements, compareFunc, (e, ind), p, delta)
            #print(f"Inserting at position {insertionPoint}")
            sortedElements.insert(insertionPoint, (e, ind))
            #print(sortedElements)
        for i, (e, ind) in enumerate(sortedElements):
            insertionPoints[ind] += i
        #return sortedElements
        #rint(sortedElements)
    tuples = sorted([(insertionPoints[i]/5.0, i) for i in range(len(elementsCopy))])
    #print(tuples)
    resultElements = [elements[t[1]] for t in tuples]
    print(resultElements)
    return resultElements








def simpleCompareTest(values, value, p, delta):
    
       
    return noisyBinarySearch(values, compareFunc, value, p, delta)


from bisect import bisect_left


def binary_search(A, T, compareFunc):
    n = len(A)
    L = 0
    R = n - 1
    while L <= R:
        m = (L + R) // 2
        if compareFunc(A[m], T) < 0.5:
            L = m + 1
        elif compareFunc(A[m], T) > 0.5:
            R = m - 1
        else:
            return m
    return R+1



# algorithm from https://arxiv.org/pdf/2107.05753
# p is from [0, 0.5), delta is (0,1)
def noisyBinarySearch(values, lessThanFunc, targetVal, p, delta):
    # need to stick -inf, inf at front and back for it to work right
    negInf = -float("inf")
    inf = float("inf")
    values = [] + values + [float("inf")]
    def lessThanFuncWrapper(a,b):
        if a == inf: return 0.0
        if a == negInf: return 1.0
        if b == inf: return 
    n = len(values)
    if n == 0: return 0
    # this lets us keep track of cumulative sum in 
    weights = np.ones(n)
    
    def query(k):
        compareP = lessThanFuncWrapper(targetVal, values[k])
        '''
        weights[:k] = weights[:k]*2*(1-p)*compareP+weights[:k]*2*p*(1-compareP)
        weights[k:] = weights[k:]*2*p*compareP+weights[k:]*2*(1-p)*(1-compareP)
        '''
        if compareP > 0.5:
            # if v is compatible with the answer
            weights[:k] = weights[:k]*2*(1-p)
            # if v is not compatible with the answer
            weights[k:] = weights[k:]*2*p
        # no answer
        else:
            # if v is not compatible with the answer
            weights[:k] = weights[:k]*2*p
            # if v is compatible with the answer
            weights[k:] = weights[k:]*2*(1-p)
        # do this so we get the exact value
        # subtracts previousWeight (so now zero) then adds w (so now w)
            
    iters = 0
    while True:
        totalWeight = np.sum(weights)
        cumsum = np.cumsum(weights)
        # binary search for k where below it is half our total weight
        greaterIndices = np.where(cumsum >= totalWeight/2.0)[0]
        k = greaterIndices[0]
        kWeight = weights[k]
        kCumWeight = cumsum[k]
        remainingWeight = totalWeight - kCumWeight
        queryPr = (1/(2.0*kWeight))*(kCumWeight-remainingWeight)
        #print(f"Query pr {queryPr}")
        if random.random() < queryPr:
            query(k)
        else:
            query(k+1)
        totalWeight = np.sum(weights)
        #print(weights)
        largestIndex = np.argmax(weights)
        largestWeight = weights[largestIndex]
        #print("Largest weight: ", largestWeight, " largest index ", largestIndex)
        #print("total weight", totalWeight)
        # stop condition
        if largestWeight/totalWeight >= 1 - delta:
            #print(f"got index: {largestIndex}")
            return largestIndex
    
'''
def SafeBinarySearch # apply lemma 18 to theorem 10

def SafeWeakSort # lemma 18 to lemma 17

def SafeLessThan # lemma 18 to lemma 13
        
def lessThan(x,y,p,sigma):
    a = 0.5
    while True:
        if noisyComparison(x,y) == 1:
            a = (1-p)*a/((1-p)*a+p*(1-a))
        else:
            a = p*a/(p*a+(1-p)*(1-a))
        if a >= 1 - sigma: return True
        if a < sigma: return False

def safeSimpleSort(arr):
    

def noisySort(arr, p):
    s = []
    for a in arr:
        if random.random() < 1.0/math.log(len(arr)):
            s.append(a)
    
'''          