import random
import functools
import torch
import itertools
from trueskill import Rating, quality_1vs1, rate_1vs1
import math
from matplotlib import pyplot as plt
import numpy as np
from dataclasses import dataclass


def generateComparisonsGraph():
    global numCompares
    global compareCache
    random.seed(27)
    
    algorithms = { }
    
    for i in np.linspace(10, 300, 20):
        algorithms[f"trueskill {i}"] = (int(i), functools.partial(simpleTrueskillBetter, std=1))    
    numComparisons = []
    fig, ax = plt.subplots()
    for algorithmName, (numData, algorithm) in algorithms.items():
        print(algorithmName)
        scoresFromAllRuns = []
        numComparisonsFromAllRuns = []
        for runs in range(5):
            print(runs)
            compareCache = {}
            numCompares = 0
            data = list(range(numData))
            random.shuffle(data)
            correctRanking = torch.argsort(torch.tensor(data))
            ranking = torch.arange(0, numData, dtype=torch.long)
            def scoreRanking():
                return torch.mean(torch.abs(ranking - correctRanking).float())
            scores = []    
            def lessThanFunc(a,b):
                global compareCache
                global numCompares
                result = 1.0 if a < b else 0.0
                if not (a,b) in compareCache:
                    compareCache[(a,b)] = result
                    numCompares += 1
                    scores.append(scoreRanking())
                if random.random() < -1:
                    return 1.0-result
                else:
                    return result
            algorithm(data=data, ranking=ranking, lessThanFunc=lessThanFunc)
            scoresFromAllRuns.append(scores)
            numComparisonsFromAllRuns.append(numCompares)
        numComparisons.append((numData, numComparisonsFromAllRuns))
        maxLenScores = len(max(scoresFromAllRuns, key=lambda x: len(x)))
        confidence = 0.95 # 0.99
        zValueForConfidence = 1.96 # 2.58
        minConfidence = []
        maxConfidence = []
        means = []
        for i in range(maxLenScores):
            scoresForComparisonI = torch.tensor([scores[i] for scores in scoresFromAllRuns if len(scores) > i])
            meanForComparisonI = torch.mean(scoresForComparisonI)
            stdForComparisonI = torch.std(scoresForComparisonI)
            # confidence interval = mean - zValue * std/sqrt(n)
            offset = zValueForConfidence * stdForComparisonI / math.sqrt(scoresForComparisonI.size()[0])
            means.append(meanForComparisonI)
            minConfidence.append(meanForComparisonI - offset)
            maxConfidence.append(meanForComparisonI + offset)
        x = np.arange(0, len(means))
        y = np.array(means)
        yMin = np.array(minConfidence)
        yMax = np.array(maxConfidence)
    
    
    zValueForConfidence = 1.96 # 0.95
    x = np.array([numData for (numData, numComps) in numComparisons])
    y = np.array([np.mean(numComps) for (numData, numComps) in numComparisons])
    lower = np.array([np.mean(numComps)-zValueForConfidence*np.std(numComps)/math.sqrt(len(numComps)) for (numData, numComps) in numComparisons])
    upper = np.array([np.mean(numComps)+zValueForConfidence*np.std(numComps)/math.sqrt(len(numComps)) for (numData, numComps) in numComparisons])
    fig, ax = plt.subplots()
    ax.plot(x,y)
    ax.fill_between(x, lower, upper, alpha=.3)
    ax.set_title("Number of comparisons until convergence for trueskill bracket size 2")
    ax.set_ylabel("Number of comparisons")
    ax.set_xlabel("Number of data points")
    plt.show()


def compareAlgorithms():
    global numCompares
    global compareCache
    random.seed(27)
    
    algorithms = {
        "randomElo": randomRankingElo,
        "randomTrueSkill": randomRankingTrueSkill,
        "bracketsTrueSkill2": functools.partial(bracketTrueSkill, bracketSize=2),
        "bracketsTrueSkill4": functools.partial(bracketTrueSkill, bracketSize=4),
        "bracketsTrueSkill8": functools.partial(bracketTrueSkill, bracketSize=8),
        "bracketsTrueSkill10": functools.partial(bracketTrueSkill, bracketSize=10),
        "bracketsTrueSkill14": functools.partial(bracketTrueSkill, bracketSize=14),
        "bracketsElo2": functools.partial(bracketElo, bracketSize=2),
        "bracketsElo4": functools.partial(bracketElo, bracketSize=4),
        "bracketsElo8": functools.partial(bracketElo, bracketSize=8),
        "bracketsElo10": functools.partial(bracketElo, bracketSize=10),
        "bracketsElo14": functools.partial(bracketElo, bracketSize=14),
    }
    
    algorithms = {
        "simpleTrueskillBetter": simpleTrueskillBetter,
        "simpleTrueskill": simpleTrueskill,
    }  
    numComparisons = []
    fig, ax = plt.subplots()
    for algorithmName, algorithm in algorithms.items():
        print(algorithmName)
        scoresFromAllRuns = []
        for runs in range(5):
            print(runs)
            numData = 400
            compareCache = {}
            numCompares = 0
            data = list(range(numData))
            random.shuffle(data)
            correctRanking = torch.argsort(torch.tensor(data))
            ranking = torch.arange(0, numData, dtype=torch.long)
            def scoreRanking():
                return torch.mean(torch.abs(ranking - correctRanking).float())
            scores = []    
            def lessThanFunc(a,b):
                global compareCache
                global numCompares
                result = 1.0 if a < b else 0.0
                if not (a,b) in compareCache:
                    compareCache[(a,b)] = result
                    numCompares += 1
                    scores.append(scoreRanking())
                if random.random() < -1:
                    return 1.0-result
                else:
                    return result
            algorithm(data=data, ranking=ranking, lessThanFunc=lessThanFunc)
            scoresFromAllRuns.append(scores)
        maxLenScores = len(max(scoresFromAllRuns, key=lambda x: len(x)))
        confidence = 0.95 # 0.99
        zValueForConfidence = 1.96 # 2.58
        minConfidence = []
        maxConfidence = []
        means = []
        for i in range(maxLenScores):
            scoresForComparisonI = torch.tensor([scores[i] for scores in scoresFromAllRuns if len(scores) > i])
            meanForComparisonI = torch.mean(scoresForComparisonI)
            stdForComparisonI = torch.std(scoresForComparisonI)
            # confidence interval = mean - zValue * std/sqrt(n)
            offset = zValueForConfidence * stdForComparisonI / math.sqrt(scoresForComparisonI.size()[0])
            means.append(meanForComparisonI)
            minConfidence.append(meanForComparisonI - offset)
            maxConfidence.append(meanForComparisonI + offset)
        x = np.arange(0, len(means))
        y = np.array(means)
        yMin = np.array(minConfidence)
        yMax = np.array(maxConfidence)
        ax.plot(x,y, label=algorithmName)
        ax.fill_between(x, yMin, yMax, alpha=.3)
        #for i, scores in enumerate(scoresFromAllRuns):
        #    ax.plot(x[:len(scores)],np.array(scores), label=algorithmName + f"{i}")
            
            
    # Add legend, title and labels
    ax.legend()
    ax.set_title("Algorithm Performance Comparison")
    ax.set_xlabel("Number of Comparisons")
    ax.set_ylabel("Mean L0 Distance From Correct Ranking")
    plt.show()
    
    
    
### Elo stuff
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
    
    
    
### Random rankings
    
def randomRankingElo(data, ranking, lessThanFunc):
    elos = torch.tensor([0 for _ in data])
     # random initial pairing to estimate elos
    randomInitials = list(range(len(data)))
    random.shuffle(randomInitials)
    randomInitialPairs = [(randomInitials[i], randomInitials[len(data)-i-1]) for i in range(len(data)//2)]
    
    # permutation of pairs
    pairs = list(itertools.product(range(len(data)), range(len(data))))
    random.shuffle(pairs)
    for ind, (i,j) in enumerate(randomInitialPairs + pairs):
        iIsLessThanJPr = lessThanFunc(data[i], data[j])
        elos[i], elos[j] = updateElos(elos[i], elos[j], 1.0-iIsLessThanJPr)
        if ind > len(data)//2: # wait for initial pairs before estimate
            ranking[:] = torch.argsort(elos)
        
def randomRankingTrueSkill(data, ranking, lessThanFunc):
    elos = [Rating(25) for _ in data]
    eloMeans = torch.tensor([elo.mu for elo in elos])
    
    # random initial pairing to estimate elos
    randomInitials = list(range(len(data)))
    random.shuffle(randomInitials)
    randomInitialPairs = [(randomInitials[i], randomInitials[len(data)-i-1]) for i in range(len(data)//2)]
    
    # permutation of pairs
    pairs = list(itertools.product(range(len(data)), range(len(data))))
    random.shuffle(pairs)
    for ind, (i,j) in enumerate(randomInitialPairs + pairs):
        iIsLessThanJPr = lessThanFunc(data[i], data[j])
        if iIsLessThanJPr < 0.5: # i wins
            elos[i], elos[j] = rate_1vs1(elos[i], elos[j])
        elif iIsLessThanJPr > 0.5: # j wins
            elos[j], elos[i] = rate_1vs1(elos[j], elos[i])
        eloMeans[i], eloMeans[j] = elos[i].mu, elos[j].mu
        if ind > len(data)//2: # wait for initial pairs before estimate
            ranking[:] = torch.argsort(eloMeans)
    








### Brackets

def getBracketsHelper(arr, partitionSize, offset):
    startIndex = 0
    # overlap them
    if offset > 0:
        startIndex = offset
        initialGroup = arr[:startIndex]
        yield list(initialGroup)
    while startIndex < len(arr):
        yield list(arr[startIndex:startIndex+partitionSize])
        startIndex += partitionSize
 
def getBrackets(arr, partitionSize, offset):
    for partition in getBracketsHelper(arr, partitionSize, offset):
        # not even, remove one at random
        if len(partition) % 2 != 0:
            partition.pop(random.randint(0, len(partition)-1))
        if len(partition) > 0:
            yield partition

def bracketElo(data, ranking, lessThanFunc, bracketSize=10):
    elos = torch.tensor([0 for _ in data])
    
    # random initial pairing to estimate elos
    randomInitials = list(range(len(data)))
    random.shuffle(randomInitials)
    randomInitialPairs = [(randomInitials[i], randomInitials[len(data)-i-1]) for i in range(len(data)//2)]
    for i,j in randomInitialPairs:
        iIsLessThanJPr = lessThanFunc(data[i], data[j])
        elos[i], elos[j] = updateElos(elos[i], elos[j], 1.0-iIsLessThanJPr)
   
    numComparisons = 0
    offset = 0
    while True:
        pairs = []
        sortedIndices = torch.argsort(elos)
        for partition in getBrackets(sortedIndices, bracketSize, offset=offset):
            random.shuffle(partition) # randomly assign pairs within the bracket
            for i in range(len(partition)//2):
                numComparisons += 1
                #if quality_1vs1(elos[partition[i]], elos[partition[-i-1]]) > 0.5:
                pairs.append((partition[i], partition[-i-1]))
            if numComparisons > len(data)*len(data)*2: # that's enough
                return
        # adjust if bracketSize % offset == 0, it's important gcd is 1 so we go through all, probably more than 1 is also good so we skip around a bit
        offset = (offset + 3) % bracketSize
        for i,j in pairs:
            iIsLessThanJPr = lessThanFunc(data[i], data[j])
            elos[i], elos[j] = updateElos(elos[i], elos[j], 1.0-iIsLessThanJPr)
            ranking[:] = torch.argsort(elos)
            


def bracketTrueSkill(data, ranking, lessThanFunc, bracketSize=10):
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
    while True:
        pairs = []
        sortedIndices = torch.argsort(eloMeans)
        for partition in getBrackets(sortedIndices, bracketSize, offset=offset):
            def indexOf(i):
                return ((sortedIndices == i).nonzero(as_tuple=True)[0])
            #random.shuffle(partition) # randomly assign pairs within the bracket
            for i in range(len(partition)//2):
                numComparisons += 1
                #if quality_1vs1(elos[partition[i]], elos[partition[-i-1]]) > 0.5:
                pairs.append((partition[i], partition[-i-1]))
            if numComparisons > len(data)*len(data)*2: # that's enough
                return
        # adjust if bracketSize % offset == 0, it's important gcd is 1 so we go through all, probably more than 1 is also good so we skip around a bit
        offset = (offset + 3) % bracketSize
        for i,j in pairs:
            def indexOf(i):
                return ((sortedIndices == i).nonzero(as_tuple=True)[0])
            iIsLessThanJPr = lessThanFunc(data[i], data[j])
            if iIsLessThanJPr < 0.5: # i wins
                elos[i], elos[j] = rate_1vs1(elos[i], elos[j])
            elif iIsLessThanJPr > 0.5: # j wins
                elos[j], elos[i] = rate_1vs1(elos[j], elos[i])
            eloMeans[i], eloMeans[j] = elos[i].mu, elos[j].mu
            ranking[:] = torch.argsort(eloMeans)
            


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
                    #ranking[:] = torch.argsort(eloMeans)
                    numComparisons += 1
                    doneSoFar.add((currentI, currentJ))
            if numComparisons > len(data)*len(data)*2:
                return
        if numTimesWithNoChanges > 5: # bail if no changes
            return

# used insights from experiments above, this has good convergence and simpler code than above
def simpleTrueskill(data, ranking, lessThanFunc):
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
    while True:
        pairs = []
        chosen = set()
        offset = (offset + 1) % 2
        sortedIndices = torch.argsort(eloMeans)
        for i in range(len(data)//2-1,-1,-1):
            curI = i*2+offset
            curJ = i*2+offset+1
            if curJ < len(data):
                currentI = sortedIndices[curI]
                currentJ = sortedIndices[curJ]
                iIsLessThanJPr = lessThanFunc(data[currentI], data[currentJ])
                if iIsLessThanJPr < 0.5: # i wins
                    elos[currentI], elos[currentJ] = rate_1vs1(elos[currentI], elos[currentJ])
                elif iIsLessThanJPr > 0.5: # j wins
                    elos[currentJ], elos[currentI] = rate_1vs1(elos[currentJ], elos[currentI])
                eloMeans[currentI], eloMeans[currentJ] = elos[currentI].mu, elos[currentJ].mu
                ranking[:] = torch.argsort(eloMeans)
                numComparisons += 1
            if numComparisons > len(data)*len(data)*2:
                return
            



## Hamming LUCB, didn't work well but I may have broke something
# modified from https://github.com/reinhardh/supplement_approximate_ranking/blob/master/approximate_ranking_introduction.ipynb

@dataclass
class Score(object):
    index: int
    T: float
    scorehat: float
    alpha: float
    def __hash__(self):
        return hash(self.index)
    
    def __eq__(self, other):
        return self.index == other.index

def computeAlpha(T,delta,n):
    beta = math.log(n/float(delta)) + 0.75*math.log(math.log(n/float(delta))) + 1.5*math.log(1+math.log(T/2.0))
    return math.sqrt( 3 / (2.0*T) ) 

def hammingLUCB(data, ranking, lessThanFunc, k=50, hd=2, delta=0.9):
    n = len(data)
    scores = torch.zeros(n)
    S = [] # list with entries ( i, T_i, scorehat_i, scorehat_i - alpha_i, scorehat_i + alpha_i, alpha_i)
    # random initial pairing to estimate elos
    randomInitials = list(range(len(data)))
    random.shuffle(randomInitials)
    randomInitialPairs = [(randomInitials[i], randomInitials[len(data)-i-1]) for i in range(len(data)//2)]
    for i in range(len(data)):
        compareTo = random.randint(0, n-2)
        if compareTo >= i:
            compareTo += 1
        iIsLessThanJPr = lessThanFunc(data[i], data[compareTo])
        scorehat = 1.0-iIsLessThanJPr
        S.append(Score(index=i, T=1, scorehat=scorehat, alpha=computeAlpha(1,delta,n)))
        scores[i] = scorehat
    
    numComparisons = 0
    while True:
        # sort in descending order by scorehat
        S = sorted(S , key=lambda entry:entry.scorehat,reverse=True)
        # min scorehat_i - alpha_i; min over (1),...,(k-h)
        d1low = min(S[:k-hd], key=lambda entry: entry.scorehat-entry.alpha)
        # max scorehat_i + alpha_i; max over (k+1+h),...,(n)
        d2up = max(S[k+hd:], key=lambda entry: entry.scorehat+entry.alpha)
        sample_next = [] # items to sample in next round
        if hd == 0: # algorithm reduced to LUCB algorithm
            sample_next = [d1low,d2up]
        else: # hd > 0
            # find middle items with the largest confidence intervals
            b1al = max(S[k-hd:k], key=lambda entry: entry.alpha )
            b2al = max(S[k:k+hd], key=lambda entry: entry.alpha )
            if d1low.alpha < b1al.alpha:
                sample_next += [b1al]
            else:
                sample_next += [d1low]
            if d2up.alpha < b2al.alpha:
                sample_next += [b2al]
            else:
                sample_next += [d2up]
        
        # check termination condition
        #if d1low.scorehat - d1low.alpha > d2up.scorehat + d2up.alpha:
        #    print("termination condition")
        #    break	# terminate
        if numComparisons > n*n*2:
            break
        
        
        # compare and update scores
        for it in set(sample_next):
            Ti = it.T+1
            # randomly compare uniformly to something else
            compareTo = random.randint(0, n-2)
            if compareTo >= it.index:
                compareTo += 1
            isLessThanPr = lessThanFunc(data[it.index], data[compareTo])
            scorehat = 1.0-isLessThanPr
            shat = 1.0/float(Ti)*( (Ti-1)*it.scorehat + scorehat)
            alphai = computeAlpha(Ti,delta,n)
            S[S.index(it)] = Score(index=it.index, T=Ti, scorehat=shat, alpha=alphai)
            scores[it.index] = shat
            ranking[:] = torch.argsort(scores)
            numComparisons += 1

    print([s.T for s in S])



def noisyInsertionSort(elements, ranking, compareFunc, p=0.3, delta=0.01):
    
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



def lessThan(x,y,p,sigma):
    a = 0.5
    while True:
        if noisyComparison(x,y) == 1:
            a = (1-p)*a/((1-p)*a+p*(1-a))
        else:
            a = p*a/(p*a+(1-p)*(1-a))
        if a >= 1 - sigma: return True
        if a < sigma: return False

# algorithm from https://arxiv.org/pdf/2107.05753
# p is from [0, 0.5), delta is (0,1)
def noisyBinarySearch(values, lessThanFunc, targetVal, p, delta):
    # need to stick -inf, inf at front and back for it to work right
    negInf = -float("inf")
    inf = float("inf")
    values = [negInf] + values + [inf]
    # if we have inf, we already know result, otherwise call their func
    def lessThanFuncWrapper(a,b):
        if a == inf: return 0.0
        if a == negInf: return 1.0
        if b == inf: return 1.0
        if b == negInf: return 0.0
        return lessThanFunc(a,b)
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
            return largestIndex+1 # plus 1 because of the infinities at front and back
    