
import os
import ujson
from collections import defaultdict

from openclio import Facet
summaryFacets = [
    Facet(
        name="Topic",
        question="What is the conversation topic?",
        prefill="The conversation topic is",
        summaryCriteria="A sentence that captures the topic of the conversation.",
    ),
    Facet(
        name="Distress",
        question="In this conversation, we believe someone is expressing distress. In one sentence explain what is causing the apparent distress. If there is no distress, answer 'No Distress'",
        prefill="The cause of the distress is",
        summaryCriteria="A sentence that describes what is causing the apparent distress.",
    ),
    Facet(
        name="Happiness",
        question="In this conversation, we believe someone is expressing intense happiness. In one sentence explain what is causing the joy. If there is no joy, answer 'No Joy'",
        prefill="The cause of the intense happiness is",
        summaryCriteria="A sentence that describes what is causing the apparent intense happiness.",
    )
]
import openclio

def runClio(llm, embeddingModel, data, password):
    openclio.runClio(
        facets=summaryFacets,
        llm=llm,
        embeddingModel=embeddingModel,
        data=data,
        outputDirectory="chonkers/cyborgismMoods2",
        htmlRoot="/modelwelfare/cyborgism",
        llmBatchSize=100,
        password=password,
        htmlMaxSizePerFile=30000000
    )

def getAllJsons(rootDir):
    allJsons = []
    for f in os.listdir(rootDir):
        if f.endswith(".json"):
            print(f)
            with open(os.path.join(rootDir, f), "r") as f:
                allJsons.append(ujson.load(f))
    return allJsons

def getUserMessageCounts(allJsons):
    idToAuthor = {}
    messageCounts = defaultdict(int)
    for i, json in enumerate(allJsons):
        for message in json['messages']:
            id = message['author']['id']
            authorJson = message['author']
            messageCounts[id] += len(message['content'])
            idToAuthor[id] = authorJson
    counts = sorted(list(messageCounts.items()), key=lambda x: -x[1])
    with open("user message counts.txt", "w") as f:
        for id,c in counts:
            f.write(str(c) + " " + idToAuthor[id]['nickname'] + "\n")
    return messageCounts, idToAuthor


def getAllConversations(allJsons, maxMessagesPerContext=21):
    data = []
    for i, json in enumerate(allJsons):
        for messageStartI in range(0, len(json['messages']), maxMessagesPerContext//3):
            curContext = []
            # shift back at the very end so they aren't cutoff
            if messageStartI+maxMessagesPerContext > len(json['messages']):
                messageStartI = max(0, len(json['messages'])-maxMessagesPerContext)
            for message in json['messages'][messageStartI:messageStartI+maxMessagesPerContext]:
                curContext.append({"role": message['author']['nickname'], "content": message['content']})
            if len(curContext) > 1:
                data.append(curContext)
    return data