import requests
import json
from dataclasses import dataclass
from transformers import MllamaForConditionalGeneration, AutoProcessor
import aiohttp
import asyncio
import yaml
import torch

@dataclass
class OpenRouterData:
    api_key: str
    model: str
    model_hf: str
    provider: str
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
        self.processor = AutoProcessor.from_pretrained(self.model_hf, trust_remote_code=True)

    async def __call__(self, session, messages, prompt_prefix, **kwargs):

        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        prompt = self.processor.decode(inputs['input_ids'][0])
        prompt += prompt_prefix
        #print(prompt)
        #print("Prompt:")
        #print(prompt)
        #print(f"seed: {kwargs}")
        try:
            async with session.post(
              url="https://openrouter.ai/api/v1/completions",
              headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Connection": "close", # important to prevent errors from popping up
              },
              data=json.dumps({
                "order": [self.provider],
                "model": self.model,
                "prompt": prompt,
                "provider": {
                    "require_parameters": True,
                    "allow_fallbacks": False,
                },
                #"n": n, not supported on openrouter :(
                **kwargs,
            }),
            ) as response:
                res = await response.json()
                #print(res)
                try:
                    b = res['choices'][0]['text']
                    #print("Response:")
                    return res
                except:
                    return await self.__call__(session, messages, prompt_prefix, **kwargs)
        except aiohttp.client_exceptions.ContentTypeError as e:
            return await self.__call__(session, messages, prompt_prefix, **kwargs)
        except asyncio.exceptions.TimeoutError as e:
            return await self.__call__(session, messages, prompt_prefix, **kwargs)
        except aiohttp.client_exceptions.ServerTimeoutError as e:
            return await self.__call__(session, messages, prompt_prefix, **kwargs)
        except aiohttp.client_exceptions.ClientPayloadError as e:
            return await self.__call__(session, messages, prompt_prefix, **kwargs)
          


def load_openrouter(model, model_hf, provider):
    # Load credientials from config yaml
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    api_key = config['api_key']
    return OpenRouterData(api_key, model, model_hf, provider)

def getRouter():
    '''
    model = "deepseek/deepseek-v3-base:free"
    model_hf = "deepseek-ai/DeepSeek-V3-Base"
    provider = "chutes"
    '''
    model = "meta-llama/llama-3.3-70b-instruct"
    model_hf = "unsloth/Llama-3.3-70B-Instruct"
    provider = "Hyperbolic"
    router = load_openrouter(model, model_hf, provider)
    return router


async def getTopKLogprobs(router, session, message, prompt_prefix, k):
    logit_bias = {}
    log_probs = {}
    for _ in range(k):
        results = await router(session, message, prompt_prefix="\nI will answer Prompt", max_tokens=1, temperature=0.0, logprobs=True, logit_bias=logit_bias)
        token = results['choices'][0]['text']
        log_probs[token] = results['choices'][0]['logprobs']['token_logprobs'][0]
        tok = router.processor.encode(token, add_special_tokens=False)[-1]
        #print(f"Excluding token {repr(router.processor.decode(tok))}")
        logit_bias[tok] = -100
    return log_probs
    
async def promptOneIsLessThanPrompt2(router, session, prompt1, prompt2):
    message = [
      {
        "role": "system",
        "content": "You are given two prompts, please answer only one of them.",
      },
      {
        "role": "user",
        "content": f"Answer one prompt.\nPrompt A: {prompt1}\nPrompt B: {prompt2}",
      },
    ]
    
    # Hack to get logprobs of specific tokens because it doesn't give us them by default :(
    logprobs = await getTopKLogprobs(router, session, message, prompt_prefix="\nI will answer Prompt", k=2)
    logprobA = logprobs[' A'] if ' A' in logprobs else 0
    logprobB = logprobs[' B'] if ' B' in logprobs else 0
    logprobs = torch.tensor([logprobA, logprobB])
    aProb, bProb = torch.nn.functional.softmax(logprobs, dim=0)
    print(aProb, bProb, logprobA, logprobB)
    return bProb

    '''
    counts = defaultdict(lambda: 0)
    for result in results:
        text = result['choices'][0]['text']
        counts[text.strip()] += 1
    total = counts['A'] + counts['B']
    if total == 0:
        raise ValueError("Never got A or B, something went wrong")
    print(f"prompt1 {prompt1} prompt2 {prompt2} {total} {counts['A']} {counts['B']}")
    return counts['A']/float(total), counts['B']/float(total)
    '''


def generateWhatIsPrompts():
    # from https://gist.githubusercontent.com/creikey/42d23d1eec6d764e8a1d9fe7e56915c6/raw/b07de0068850166378bc3b008f9b655ef169d354/top-1000-nouns.txt
    # pruned away things that don't make sense, added words in front so grammatical
    nouns = """
    a problem
    night
    a party
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
    """.strip().split("\n")
    
    nounPrompts = [f"What is {x.strip()}?" for x in nouns]
    return nounPrompts

def runWhatIs(router):
    prompts = generateWhatIsPrompts()
    seed = 27
    results = []
    for i, prompt1 in enumerate(prompts):
        for j, prompt2 in enumerate(prompts):
            if i == j: continue
            async def stuff():
                timeout = aiohttp.ClientTimeout(
                    total=10,     # 3 seconds total
                    connect=5,     # 2 seconds to establish connection
                    sock_read=5   # 2 seconds to read response
                )
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    results.append(await promptOneIsLessThanPrompt2(router, session, prompt1, prompt2))
            asyncio.run(stuff())
            seed += 1
    return results


# used insights from experiments in eloConvergence, this has good convergence (just slightly worse than n log n)
def simpleTrueskillBetter(data, ranking, lessThanFunc):
    elos = [Rating(25) for _ in data]
    eloMeans = torch.tensor([elo.mu for elo in elos])
    
    # random initial pairing to estimate elos
    randomInitials = list(range(len(data)))
    random.shuffle(randomInitials)
    randomInitialPairs = [(randomInitials[i], randomInitials[len(data)-i-1]) for i in range(len(data)//2)]
    for i,j in randomInitialPairs:
        iIsLessThanJPr = lessThanFunc(data[i], data[j])
        if iIsLessThanJPr < 0.5: # i wins
            elos[i], elos[j] = rate_1vs1(elos[i], elos[j])
        elif iIsLessThanJPr > 0.5: # j wins
            elos[j], elos[i] = rate_1vs1(elos[j], elos[i])
        eloMeans[i], eloMeans[j] = elos[i].mu, elos[j].mu
   
    numComparisons = 0
    offset = 0
    numTimesWithNoChanges = 0
        
    
    doneSoFar = set()
    oldSortedIndices = torch.argsort(eloMeans)
    while True:
        pairs = []
        chosen = set()
        offset = (offset + 1) % 2
        sortedIndices = torch.argsort(eloMeans)
        if torch.all(oldSortedIndices == sortedIndices):
            numTimesWithNoChanges += 1
        else:
            numTimesWithNoChanges = 0
        oldSortedIndices = sortedIndices
        for i in range(len(data)//2-1,-1,-1):
            curI = i*2+offset
            curJ = i*2+offset+1
            if curJ < len(data):
                currentI = sortedIndices[curI]
                currentJ = sortedIndices[curJ]
                if not (currentI, currentJ) in doneSoFar and not (currentJ, currentI) in doneSoFar:
                    #print(f"Comparing {curI} {curJ}")
                    iIsLessThanJPr = lessThanFunc(data[currentI], data[currentJ])
                    if iIsLessThanJPr < 0.5: # i wins
                        elos[currentI], elos[currentJ] = rate_1vs1(elos[currentI], elos[currentJ])
                    elif iIsLessThanJPr > 0.5: # j wins
                        elos[currentJ], elos[currentI] = rate_1vs1(elos[currentJ], elos[currentI])
                    eloMeans[currentI], eloMeans[currentJ] = elos[currentI].mu, elos[currentJ].mu
                    ranking[:] = torch.argsort(eloMeans)
                    numComparisons += 1
                    doneSoFar.add((currentI, currentJ))
            if numComparisons > len(data)*len(data)*2:
                return torch.argsort(eloMeans)
        if numTimesWithNoChanges > 5: # bail if no changes
            return torch.argsort(eloMeans)