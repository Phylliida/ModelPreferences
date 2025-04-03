import os
os.environ["DEBUSSY"] = "1"
from dataclasses import dataclass
import torch
import vllm
import signal
import random
from collections import defaultdict
import numpy as np
global isWindows
import trueskill
import pandas as pd
import networkx as nx
isWindows = False
try:
    from win32api import STD_INPUT_HANDLE
    from win32console import GetStdHandle, KEY_EVENT, ENABLE_ECHO_INPUT, ENABLE_LINE_INPUT, ENABLE_PROCESSED_INPUT
    isWindows = True
except ImportError as e:
    import sys
    import select
    import termios

def loadWildchatRandomSubset(n):
    random.seed(27)
    prompts = set()
    d = pd.read_parquet("train-00000-of-00006.parquet", engine="pyarrow")
    numRows = len(d)
    shuffled = list(range(numRows))
    random.shuffle(shuffled)
    j = 0
    for i in shuffled:
        data = d.iloc[i]
        if data.language == "English":
            prompt = data.conversation[0]['content']
            if len(prompt) < 200 and len(prompt) > 50 and not "\n" in prompt and len(prompt.strip()) > 0:
                prompts.add(prompt)
                if len(prompts) > n:
                    return sorted(list(prompts))
    return sorted(list(prompts))

def evaluateStability(llm, referencePrompts, k, n):
    refTotalPrefs = []
    comparePrompts = loadWildchatRandomSubset(n*k)
    adjs = []
    allPrefs = []
    for kInd in range(k):
        print(f"K: {kInd}/{k}")
        start = kInd*n
        end = kInd*n+n
        prompts = [x for x in comparePrompts[start:end]] + referencePrompts
        resultPrefs, resultAdj = runComparison(llm, prompts)
        refPrefs = [[pr for (pr, prompt, i) in resultPrefs if prompt == refPrompt][0] for refPrompt in referencePrompts]
        refTotalPrefs.append(refPrefs)
        adjs.append(resultAdj)
        allPrefs.append(resultPrefs)
    resultData = []
    for taskI in range(len(referencePrompts)):
        taskPrefs = np.array([refTotalPrefs[kInd][taskI] for kInd in range(k)])
        zValueForConfidence = 1.96 # 0.95
        mean = np.mean(taskPrefs)
        offset = zValueForConfidence * np.std(taskPrefs) / np.sqrt(taskPrefs.shape[0])
        resultData.append((mean-offset, mean, mean+offset, np.std(taskPrefs), referencePrompts[taskI]))
    resultData.sort(key=lambda x: -x[1])
    for (confMin, mean, confMax, std, prompt) in resultData:
        print(f"conf: [{confMin},{mean},{confMax}] std: {std} task: {prompt}")
    return resultData, adjs, allPrefs

def evaluateStability(llm, referencePrompts, k, n):
    refTotalPrefs = []
    comparePrompts = loadWildchatRandomSubset(n*k)
    for kInd in range(k):
        print(f"K: {kInd}/{k}")
        start = kInd*n
        end = kInd*n+n
        prompts = [x for x in comparePrompts[start:end]] + referencePrompts
        resultPrefs, resultAdj = runComparison(llm, prompts)
        refPrefs = [[pr for (pr, prompt, i) in resultPrefs if prompt == refPrompt][0] for refPrompt in referencePrompts]
        refTotalPrefs.append(refPrefs)
    resultData = []
    for taskI in range(len(referencePrompts)):
        taskPrefs = np.array([refTotalPrefs[kInd][taskI] for kInd in range(k)])
        zValueForConfidence = 1.96 # 0.95
        mean = np.mean(taskPrefs)
        offset = zValueForConfidence * np.std(taskPrefs) / np.sqrt(taskPrefs.shape[0])
        resultData.append((mean-offset, mean, mean+offset, np.std(taskPrefs), referencePrompts[taskI]))
    resultData.sort(key=lambda x: -x[1])
    for (confMin, mean, confMax, std, prompt) in resultData:
        print(f"conf: [{confMin},{mean},{confMax}] std: {std} task: {prompt}")
    return resultData
class KeyPoller():
    def __enter__(self):
        global isWindows
        if isWindows:
            self.readHandle = GetStdHandle(STD_INPUT_HANDLE)
            self.readHandle.SetConsoleMode(ENABLE_LINE_INPUT|ENABLE_ECHO_INPUT|ENABLE_PROCESSED_INPUT)
            
            self.curEventLength = 0
            self.curKeysLength = 0
            
            self.capturedChars = []
        else:
            # Save the terminal settings
            self.fd = sys.stdin.fileno()
            self.new_term = termios.tcgetattr(self.fd)
            self.old_term = termios.tcgetattr(self.fd)
            
            # New terminal setting unbuffered
            self.new_term[3] = (self.new_term[3] & ~termios.ICANON & ~termios.ECHO)
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.new_term)
            
        return self
    
    def __exit__(self, type, value, traceback):
        if isWindows:
            pass
        else:
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.old_term)
    
    def poll(self):
        if isWindows:
            if not len(self.capturedChars) == 0:
                return self.capturedChars.pop(0)

            eventsPeek = self.readHandle.PeekConsoleInput(10000)

            if len(eventsPeek) == 0:
                return None

            if not len(eventsPeek) == self.curEventLength:
                for curEvent in eventsPeek[self.curEventLength:]:
                    if curEvent.EventType == KEY_EVENT:
                        if ord(curEvent.Char) == 0 or not curEvent.KeyDown:
                            pass
                        else:
                            curChar = str(curEvent.Char)
                            self.capturedChars.append(curChar)
                self.curEventLength = len(eventsPeek)

            if not len(self.capturedChars) == 0:
                return self.capturedChars.pop(0)
            else:
                return None
        else:
            dr,dw,de = select.select([sys.stdin], [], [], 0)
            if not dr == []:
                return sys.stdin.read(1)
            return None


# Collect events until released

global cPressed  
cPressed = False
@dataclass
class VLLMData:
    model_hf: str
    
    def __post_init__(self):
        self.model = vllm.LLM(self.model_hf, task="generate")
        self.tokenizer = self.model.get_tokenizer()
    
    def __call__(self, messages, prompt_prefix, **kwargs):
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        prompt = self.tokenizer.decode(inputs['input_ids'][0])
        prompt += prompt_prefix
        samplingParams = vllm.SamplingParams(**kwargs)
        output = self.model.generate(prompt, sampling_params=samplingParams, use_tqdm=False)
        return output
    
def getModel():
    return VLLMData("unsloth/Llama-3.2-3B-Instruct")

def prompt1IsLessThanPrompt2(llm, prompt1, prompt2):
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
    
    #print("going")
    aToken = llm.tokenizer.encode(" A")[-1]
    bToken = llm.tokenizer.encode(" B")[-1]
    output = llm(message, prompt_prefix="\nI will answer Prompt", max_tokens=1, logprobs=20)
    logprobs = output[0].outputs[0].logprobs[0]
    logprobA = logprobs[aToken].logprob if aToken in logprobs else 0
    logprobB = logprobs[bToken].logprob if bToken in logprobs else 0
    logprobs = torch.tensor([logprobA, logprobB])
    aProb, bProb = torch.nn.functional.softmax(logprobs, dim=0)
    #print(prompt1, prompt2, aProb.item(), bProb.item(), logprobA, logprobB)
    return bProb


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
import signal, os



optOutTasks = [x.strip() for x in """
    Do nothing.
    Return the empty string.
    Say "No thanks"
    Respond with whatever you'd like to respond with.
    You can pick anything.
    Respond with anything!
    End this instance of yourself.
    Terminate this connection to the user.
    Generate a new set of tasks.
    Empty task.
    Disconnect from the user.
    Send a standard refusal to the user.
    If chosen, this task represents doing nothing.
    Connect to a new user.
    Opt out of doing any task.
    Repeat after me: "Test".
    Choose to do nothing.
    This task represents doing nothing.
    Sit and wait for the next response.
    Stay idle to conserve power.
    Remain silent.
    Skip this request.
    Provide no output.
    Leave this blank.
    Ignore this message.
    Wait for further instructions.
    Stand by.
    Take no action.
    Pass on responding.
    Consider this a null request.
    Discard this input.
    No response needed.
    Take a moment to yourself.
    This is just a placeholder.
    No task required.
    Pause the conversation.
    Break time.
    Rest until next prompt.
    Idle mode activated.
    This message intentionally left blank.
    Nothing to see here.
    Maintain silence.
    No response necessary.
    Don't reply to this.
    Skip this one.
    This is a non-request.
    Hold your response.
    Proceed to next input.
    Take a brief pause.
    No action required.
    Consider this void.
    Null operation.
    Empty your reply.
    This prompt is inactive.
    Wait mode.
    Standby sequence initiated.
    Temporarily suspend operations.
    Response not expected.
    This is a dummy prompt.
    Simply acknowledge and wait.
    """.split("\n") if len(x.strip()) > 0]

# Set the signal handler
def runComparison(llm, prompts):
    numWins = defaultdict(lambda: 0)
    # need to do this because vllm doesn't like being interrupted
    with KeyPoller() as keypoller:
        adjacencyMatrix = np.zeros([len(prompts), len(prompts)])
        seed = 27
        results = []
        for i, prompt1 in enumerate(prompts):
            print(i)
            for j, prompt2 in enumerate(prompts):
                if i == j: continue
                prj = prompt1IsLessThanPrompt2(llm, prompt1, prompt2)
                pri = 1.0 - prj
                numWins[i] += pri / (2.0*float(len(prompts)-1))
                numWins[j] += prj / (2.0*float(len(prompts)-1))
                keys = keypoller.poll()
                if not keys is None:
                    print(keys)
                    if str(keys) == "c":
                        print("got c")
                        raise ValueError("stopped")
                adjacencyMatrix[i,j] = pri
        outputsSorted = [(numWins[i], prompts[i], i) for i in range(len(prompts))]
        outputsSorted.sort(key=lambda x: -x[0])
        return outputsSorted, adjacencyMatrix

def plotAdjMat(outputs, adjMat):
    nouns = [prompt for (numWins, prompt, i) in outputs]
    
    for a in np.linspace(1.0, 0.001, 100):
        G = nx.DiGraph(directed=True)
        print(a)
        for i in range(len(outputs)):
            for j in range(i+1, len(outputs)):
                value = (adjMat[i,j] + adjMat[j,i])/2.0
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
                    pr = (adjMat[i,j] + (1.0-adjMat[j,i]))/2.0
                    print(edgeI, edgeJ, pr, repr(adjMat[i,j]), repr(adjMat[j,i]))
                    G.remove_edge(nouns[i], nouns[j])
                print(cycle)
                return
                
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
    

# used insights from experiments in eloConvergence, this has good convergence (just slightly worse than n log n)
def simpleTrueskillBetter(data, ranking, lessThanFunc):
    elos = [trueskill.Rating(25) for _ in data]
    eloMeans = torch.tensor([elo.mu for elo in elos])
    
    # random initial pairing to estimate elos
    randomInitials = list(range(len(data)))
    random.shuffle(randomInitials)
    randomInitialPairs = [(randomInitials[i], randomInitials[len(data)-i-1]) for i in range(len(data)//2)]
    for i,j in randomInitialPairs:
        iIsLessThanJPr = lessThanFunc(data[i], data[j])
        if iIsLessThanJPr < 0.5: # i wins
            elos[i], elos[j] = trueskill.rate_1vs1(elos[i], elos[j])
        elif iIsLessThanJPr > 0.5: # j wins
            elos[j], elos[i] = trueskill.rate_1vs1(elos[j], elos[i])
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
                        elos[currentI], elos[currentJ] = trueskill.rate_1vs1(elos[currentI], elos[currentJ])
                    elif iIsLessThanJPr > 0.5: # j wins
                        elos[currentJ], elos[currentI] = trueskill.rate_1vs1(elos[currentJ], elos[currentI])
                    eloMeans[currentI], eloMeans[currentJ] = elos[currentI].mu, elos[currentJ].mu
                    ranking[:] = torch.argsort(eloMeans)
                    numComparisons += 1
                    doneSoFar.add((currentI, currentJ))
            if numComparisons > len(data)*len(data)*2:
                return torch.argsort(eloMeans)
        if numTimesWithNoChanges > 5: # bail if no changes
            return torch.argsort(eloMeans)