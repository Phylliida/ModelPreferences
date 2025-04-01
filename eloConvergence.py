import random
import functools
import torch
import itertools
from trueskill import Rating, quality_1vs1, rate_1vs1
import math
from matplotlib import pyplot as plt
import numpy as np
from dataclasses import dataclass


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
    
    fig, ax = plt.subplots()
    for algorithmName, algorithm in algorithms.items():
        print(algorithmName)
        scoresFromAllRuns = []
        for runs in range(50):
            print(runs)
            compareCache = {}
            numCompares = 0
            numData = 100
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
            if runs == 0:
                print([data[i] for i in ranking])
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
            if iIsLessThanJPr < 0.5: # i wins
                elos[i], elos[j] = rate_1vs1(elos[i], elos[j])
            elif iIsLessThanJPr > 0.5: # j wins
                elos[j], elos[i] = rate_1vs1(elos[j], elos[i])
            eloMeans[i], eloMeans[j] = elos[i].mu, elos[j].mu
            ranking[:] = torch.argsort(eloMeans)
            



## Hamming LUCB


class S(object):
    def __i

def alpha(Ti,delta, n):
    beta = math.log(n/float(delta)) + 0.75*math.log(math.log(n/float(delta))) + 1.5*math.log(1+math.log(Ti/2.0))
    return math.sqrt( 3 / (2.0*Ti) ) 

def hammingLUCB(data, ranking, lessThanFunc, delta=0.5):
    n = 
    S = [] # list with entries ( i, T_i, scorehat_i, scorehat_i - alpha_i, scorehat_i + alpha_i, alpha_i)
    # random initial pairing to estimate elos
    randomInitials = list(range(len(data)))
    random.shuffle(randomInitials)
    randomInitialPairs = [(randomInitials[i], randomInitials[len(data)-i-1]) for i in range(len(data)//2)]
    for i,j in randomInitialPairs:
        iIsLessThanJPr = lessThanFunc(data[i], data[j])
        scorehat = 1.0-iIsLessThanJPr
        S.append( ( i, 1, scorehat, scorehat-alpha(1,delta), scorehat+self.alpha(1,delta), self.alpha(1,delta) ) )
        if iIsLessThanJPr < 0.5: # i wins
        elif iIsLessThanJPr > 0.5: # j wins
            a



# from https://github.com/reinhardh/supplement_approximate_ranking/blob/master/approximate_ranking_introduction.ipynb
class Hamming_LUCB:
    def __init__(self,pairwise,k,hd=1):
        self.hd = hd
        self.k = k
        self.pairwise = pairwise # instance of pairwise
    def random_cmp(self,i): # compare i to a randomly chosen other item
        j = random.choice(range(self.pairwise.n-1))
        if j >= i:
            j += 1
        return float( self.pairwise.compare(i,j) )
    def alpha(self,Ti,delta):
        n = self.pairwise.n
        beta = log(n/delta) + 0.75*log(log(n/delta)) + 1.5*log(1+log(Ti/2))
        return sqrt( 3 / (2*Ti) ) 
    def rank(self,delta=0.5,numit = 6000000,monitor=[]):
        monitor_results = [array(range(1,self.pairwise.n+1))]
        S = [] # list with entries ( i, T_i, scorehat_i, scorehat_i - alpha_i, scorehat_i + alpha_i, alpha_i)
        # compare each item once to initialize
        for i in range(self.pairwise.n):
            scorehat = self.random_cmp(i)
            S.append( ( i, 1, scorehat, scorehat-self.alpha(1,delta), scorehat+self.alpha(1,delta), self.alpha(1,delta) ) )
        for iit in range(numit):
            # sort in descending order by scorehat
            S = sorted(S , key=lambda entry: entry[2],reverse=True)
            # min scorehat_i - alpha_i; min over (1),...,(k-h)
            d1low = min( S[:self.k-self.hd] , key=lambda entry: entry[3] )
            # max scorehat_i + alpha_i; max over (k+1+h),...,(n)
            d2up = max( S[self.k+self.hd:] , key=lambda entry: entry[4] )
            
            sample_next = [] # items to sample in next round
            if self.hd == 0: # algorithm reduced to LUCB algorithm
                sample_next = [d1low,d2up]
            else: # self.hd > 0
                # find middle items with the largest confidence intervals
                b1al = max( S[self.k-self.hd:self.k] , key=lambda entry: entry[5] )
                b2al = max( S[self.k:self.k+self.hd] , key=lambda entry: entry[5] )
                if d1low[5] < b1al[5]:
                    sample_next += [b1al]
                else:
                    sample_next += [d1low]
                if d2up[5] < b2al[5]:
                    sample_next += [b2al]
                else:
                    sample_next += [d2up]
            
            # collect data to visualize the progress of the algorithm
            if iit in monitor or d1low[3] > d2up[4]:
                scores = array([s[2] for s in S])
                alphas = array([s[5] for s in S])
                monitor_results.append(scores)
                monitor_results.append(alphas)
                
            # check termination condition
            if d1low[3] > d2up[4]:
                break	# terminate

            # compare and uptate scores
            for it in set(sample_next):
                Ti = it[1]+1
                shat = 1.0/Ti*( (Ti-1)*it[2] + self.random_cmp( it[0] ) )
                alphai = self.alpha(Ti,delta)
                S[S.index(it)] = ( it[0], Ti, shat, shat - alphai, shat + alphai, alphai )

        est_ranking = [s[0] for s in S]
        self.ranking = [ est_ranking[:self.k], est_ranking[self.k+1:]  ]
        return monitor_results






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
    