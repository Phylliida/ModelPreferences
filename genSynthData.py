
from __future__ import annotations

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


def writeAllSingleTokenThings(unigrams):
    for i, (k,c) in enumerate(list(unigrams.items())):
        if i % 1000 == 0: print(i / float(len(unigrams)))
        bonus = k.replace(",", "").replace(".", "").replace("?", "").replace("!", "").strip()
        if not bonus in unigrams:
            unigrams[bonus] = c
    tokenCountsPath = "chonkers/tokenCountsClaude35.json"
    with open(tokenCountsPath, "r") as f:
        toks = json.load(f)
    singleToks = sorted([(unigrams[k], k) for (k,v) in toks.items() if v == 1], key=lambda x: x[0])

    with open("singleTokensWithCounts.txt", "w") as f:
        for c, t in singleToks:
            f.write(f"{t}   {c}\n")
        
    

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

def getRouter():
    import os
    import eloLLMPrefs
    return eloLLMPrefs.getRouter("anthropic")
import asyncio
# baseline is for claude-3-5-sonnet-20240620
def countTokens(router, s, baseline=14):
    for i in range(100):
        try:
            w = asyncio.run(router.aclient.messages.create(model='claude-3-5-sonnet-20240620', max_tokens=1, messages=[{"role": "user", "content": f'🟢{s}🟢'}]))
            return w.usage.input_tokens-baseline
        except KeyboardInterrupt:
            raise
        except:
            if i == 9: raise
            pass
            

def getUnigrams():
    with open("chonkers/finalUnigramCounts.pkl", "rb") as f:
        return cloudpickle.load(f)

def getUnigramBins(counts):
    bins = defaultdict(lambda: 0)
    for i, (k, count) in enumerate(counts.items()):
        bins[min(200,count)] += 1
        if i % 1000 == 0:
            print(i, len(counts))
    return bins

def getUnigramsSortedByCount(unigrams):
    return sorted(list(unigrams.items()), key=lambda x: -x[1])

global tokenCounts
tokenCounts = {}
import json
def reconstructTokenizer(sortedUnigrams):
    import eloLLMPrefs
    global tokenCounts
    tokenCountsPath = "chonkers/tokenCountsClaude35.json"
    if os.path.exists(tokenCountsPath):
        with open(tokenCountsPath, "r") as f:
            tokenCounts = json.load(f)
    router = eloLLMPrefs.getRouter("anthropic")
    for i, (k, count) in enumerate(sortedUnigrams):
        k = k.replace(",", "").replace(".", "").replace("?", "").replace("!", "").strip()
        if len(k) == 0 or k in tokenCounts: continue
        tokenCounts[k] = countTokens(router, k)
        print(i, k, tokenCounts[k])
        if i % 100 == 0:
            with open(tokenCountsPath, "w") as f:
                json.dump(tokenCounts, f)




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


























































#!/usr/bin/env python3
"""
wordbreak.py  –  output-optimal enumeration of all ways to write a string
                 as a concatenation of words from a dictionary.

Implementation:
  • fast Phase 1 via a C Aho–Corasick automaton  (pyahocorasick package)
  • Phase 2 depth-first traversal of the implicit DAG, allocating only
    O(depth) extra memory.

Usage example
-------------
    >>> DICT = ["cat", "cats", "and", "sand", "dog"]
    >>> auto = make_automaton(DICT)
    >>> list(segmentations("catsanddog", auto))
    [['cat', 'sand', 'dog'], ['cats', 'and', 'dog']]
"""

import ahocorasick
import random
import unittest
from typing import Iterable, List, Tuple, Generator


# ────────────────────────────────────────────────────────────────
# Phase 0 ── build the automaton once and reuse it
# ────────────────────────────────────────────────────────────────
def make_automaton(words: Iterable[str]) -> ahocorasick.Automaton:
    """Return an Aho-Corasick automaton whose payload of every match
    is the matched *word itself*.
    """
    A = ahocorasick.Automaton()
    for w in words:
        if w:                               # ignore empty words
            A.add_word(w, w)
    A.make_automaton()
    return A


# ────────────────────────────────────────────────────────────────
# Phase 1 ── build adjacency lists next_pos[start] = (end, word)
# ────────────────────────────────────────────────────────────────
def build_next_array(
    s: str,
    A: ahocorasick.Automaton,
) -> List[List[Tuple[int, str]]]:

    n = len(s)
    nxt: List[List[Tuple[int, str]]] = [[] for _ in range(n + 1)]

    for end_incl, word in A.iter(s):              # inclusive end index
        start = end_incl - len(word) + 1
        nxt[start].append((end_incl + 1, word))    # exclusive end
    return nxt


# ────────────────────────────────────────────────────────────────
# Phase 2 ── DFS that yields every factorisation (output-optimal)
# ────────────────────────────────────────────────────────────────
def segmentations(
    s: str,
    A: ahocorasick.Automaton,
) -> Generator[List[str], None, None]:
    """Yield every ordered list of dictionary words whose
    concatenation equals s.  The order in which segmentations are
    produced is depth-first but otherwise unspecified.
    """
    nxt = build_next_array(s, A)
    n = len(s)

    path: List[str] = []
    stack: List[Tuple[int, int]] = [(0, 0)]        # (position, next-edge idx)

    while stack:
        pos, edge_idx = stack[-1]

        if pos == n:                               # reached end → solution
            yield path[:]
            stack.pop()
            if path:
                path.pop()
            continue

        if edge_idx >= len(nxt[pos]):              # no more outgoing edges
            stack.pop()
            if path:
                path.pop()
            continue

        # take next edge and increment edge pointer in current frame
        end, word = nxt[pos][edge_idx]
        stack[-1] = (pos, edge_idx + 1)
        path.append(word)
        stack.append((end, 0))


# ────────────────────────────────────────────────────────────────
# Optional helper: COUNT segmentations (dynamic programming)
#   useful in tests to cross-validate the enumerator
# ────────────────────────────────────────────────────────────────
def count_segmentations(s: str, A: ahocorasick.Automaton) -> int:
    nxt = build_next_array(s, A)
    n = len(s)

    dp = [0] * (n + 1)
    dp[n] = 1
    for i in range(n - 1, -1, -1):
        dp[i] = sum(dp[end] for end, _ in nxt[i])
    return dp[0]


# ────────────────────────────────────────────────────────────────
# Unit tests  (many!)
# Run with:  python -m unittest wordbreak.py
# ────────────────────────────────────────────────────────────────
class WordBreakTests(unittest.TestCase):

    def assertSegmentationsEqual(
        self,
        s: str,
        dictionary: Iterable[str],
        expected: List[List[str]],
    ) -> None:
        auto = make_automaton(dictionary)
        got = sorted(segmentations(s, auto))
        self.assertEqual(got, sorted(expected))
        # also cross-validate count routine
        self.assertEqual(len(got), count_segmentations(s, auto))

    # 1 ── classical LeetCode example
    def test_leetcode(self):
        D = ["cat", "cats", "and", "sand", "dog"]
        s = "catsanddog"
        exp = [["cat", "sand", "dog"],
               ["cats", "and", "dog"]]
        self.assertSegmentationsEqual(s, D, exp)

    # 2 ── no valid segmentation
    def test_none(self):
        self.assertSegmentationsEqual("abc", ["a", "bc"], [])

    # 3 ── single-word segmentation
    def test_single_word(self):
        self.assertSegmentationsEqual("hello", ["hello", "hell"], [["hello"]])

    # 4 ── empty target string ⇒ one empty decomposition
    def test_empty_string(self):
        auto = make_automaton(["a", "b"])
        segs = list(segmentations("", auto))
        self.assertEqual(segs, [[]])
        self.assertEqual(count_segmentations("", auto), 1)

    # 5 ── overlapping prefixes (“aaaaa…”)
    def test_overlapping(self):
        s = "aaaa"
        D = ["a", "aa", "aaa", "aaaa"]
        auto = make_automaton(D)
        segs = list(segmentations(s, auto))
        # closed formula: #partitions = 2^(n-1) for this dictionary
        self.assertEqual(len(segs), 2 ** (len(s) - 1))
        self.assertEqual(len(segs), count_segmentations(s, auto))

    # 6 ── Unicode words
    def test_unicode(self):
        D = ["🐱", "🐱‍👤", "忍者", "猫", "忍者猫"]
        s = "忍者猫"
        exp = [["忍者", "猫"], ["忍者猫"]]
        self.assertSegmentationsEqual(s, D, exp)

    # 7 ── Large randomised test (still runs fast)
    def test_random(self):
        random.seed(0)
        alphabet = "abc"
        dict_size = 200
        D = { ''.join(random.choices(alphabet, k=random.randint(1, 8)))
              for _ in range(dict_size) }
        s = ''.join(random.choices(alphabet, k=20))

        auto = make_automaton(D)
        # Compare enumeration count to DP count
        enum_cnt = sum(1 for _ in segmentations(s, auto))
        self.assertEqual(enum_cnt, count_segmentations(s, auto))

    # 8 ── dictionary contains empty string (should be ignored)
    def test_empty_dict_entry(self):
        self.assertSegmentationsEqual("ab", ["", "a", "b", "ab"], [["a", "b"], ["ab"]])

    # 9 ── repeated dictionary words (duplicates should not affect result)
    def test_duplicate_words(self):
        self.assertSegmentationsEqual("abab", ["ab", "ab", "ab"], [["ab", "ab"]])


# ────────────────────────────────────────────────────────────────
# Quick interactive demo when run directly
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DICT = ["cat", "cats", "and", "sand", "dog"]
    s = "catsanddog"

    auto = make_automaton(DICT)
    print(f"All factorizations of '{s}':")
    for p in segmentations(s, auto):
        print("  " + " ".join(p))

    print("\nRunning unit tests …")
    unittest.main(argv=["wordbreak.py"], verbosity=2, exit=False)