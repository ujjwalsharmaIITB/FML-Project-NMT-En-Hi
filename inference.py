    # Created by Ujjwal Sharma ,
    # 23M0837 , 23M0837@iitb.ac.in
    # github@ujjwalsharmaIITB

import __main__
from io import open
import unicodedata
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from encoder_decoder import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
#     s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()

def readLangs(lang1, lang2, path, reverse=False):
    print(f"Reading lines... from {path}")

    # Read the file and split into lines
    lines = open(path, encoding='utf-8').\
        read().strip().split('\n')

#     print(lines[0:3])

    # Split every line into pairs and normalize
    # take only 2 columns .
    pairs = [[normalizeString(s) for s in l.split('\t')[:2]] for l in lines]

#     print(pairs[:3])

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

MAX_LENGTH = 15


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

"""## The full process for preparing the data is:

    1. Read text file and split into lines, split lines into pairs

    2. Normalize text, filter by length and content ( optional )

    3. Make word lists from sentences in pairs

"""

def prepareData(lang1, lang2,path , reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, path , reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

# input_lang, output_lang, pairs = prepareData('eng', 'hin', data_path)
# print(random.choice(pairs))


# print("Total Sentences = ", len(pairs))



# def saveDataSize(pairs , size):
#   with open(f'eng-hin-train-{size}.txt' , 'w+' , encoding = "utf8") as file:
#     random.shuffle(pairs)
#     totalExamples = min(len(pairs) , size)
#     newPairs = pairs[:totalExamples]
#     for pair in newPairs:
#         source_sentence = pair[0]
#         target_sentence = pair[1]
#         file.write(source_sentence.strip() + "\t")
#         file.write(target_sentence.strip() + "\n")



# # print(random.choice(pairs))

# saveDataSize(pairs , 100000)

"""# Encoder"""




def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)




def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        # Not giving the target output
        # Now the network has to generate the output
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        # taking only the best prediction
        # this is incomplete search
        # you need to do beam search
        # Will add to TODO
        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    # Here decoded attention will be null in case we do not use the attention
    # mechanism
    return decoded_words, decoder_attn

def evaluateRandomly(encoder, decoder,  pairs , input_lang, output_lang, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('Input Sentence :: ', pair[0])
        print('Actual Translated Sentence :: ', pair[1])
        output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('Translated Sentence :: ', output_sentence)
        print('')



def evaluateInputSentenceHelper(encoder, decoder,  inputSentence , input_lang, output_lang,):
    print('Input Sentence :: ', inputSentence)
    inputSentence = normalizeString(inputSentence)
    print("Normalized input :: " , inputSentence)
    output_words, _ = evaluate(encoder, decoder, inputSentence, input_lang, output_lang)
    output_sentence = ' '.join(output_words)
    print('Translated Sentence :: ', output_sentence)
    print('')
    return output_sentence

def loadEncDec(encoder_path , decoder_path):
    print("Loading Encoder ...")
    encoder = torch.load(encoder_path , map_location=torch.device('cpu'))
    print("Loadind Decoder ...")
    decoder = torch.load(decoder_path , map_location=torch.device('cpu'))
    print("Loading Done")
    return encoder , decoder

def predictionFunctionV1():
    print("Loading Encoder and Decoder without Attention")
    encoder, decoder = loadEncDec("saved/encoderHindi-iitb-without-attn-100k-lstm-data-nonrev.pt" , "saved/decoderHindi-iitb-without-attn-100k-lstm-data-nonrev.pt")
    input_lang, output_lang, pairs = prepareData('eng'  , 'hin'  , "data/eng-hin-train-100000.txt")
    return encoder , decoder , input_lang , output_lang , pairs


def predictionFunctionV2():
    print("Loading Encoder and Decoder with Attention")
    encoderAttn, decoderAttn = loadEncDec("saved/encoderHindi-iitb-with-attn-gru-100k-data-nonrev.pt" , "saved/decoderHindi-iitb-with-attn-gru-100k-data-nonrev.pt")
    input_lang, output_lang, pairs = prepareData('eng'  , 'hin'  , "data/eng-hin-train-100000.txt")
    return encoderAttn , decoderAttn , input_lang , output_lang , pairs


encoder , decoder , input_lang , output_lang , pairs = predictionFunctionV1()

encoderAttn , decoderAttn , input_lang , output_lang , pairs = predictionFunctionV2()



def evaluateInputSentence(sentence , version=1):
    try:
        if version == 1:
            return evaluateInputSentenceHelper(encoder , decoder , sentence , input_lang , output_lang)
        else:
            return evaluateInputSentenceHelper(encoderAttn , decoderAttn , sentence , input_lang , output_lang)
    except KeyError:
        return "Values Not Found"

    
    


def randomEncDecHelper(encoder , decoder , input_lang , output_lang , pairs , n = 10):
    returnList = []
    for i in range(n):
        pair = random.choice(pairs)
        intermediateDict = {}
        print('Input Sentence :: ', pair[0])
        print('Actual Translated Sentence :: ', pair[1])
        intermediateDict['Input Sentence'] = pair[0]
        intermediateDict['Actual Translated Sentence'] = pair[1]
        output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('Translated Sentence :: ', output_sentence)
        intermediateDict['Translated Sentence'] = output_sentence

        returnList.append(intermediateDict)
        print('')
    return returnList


def randomEncDecV1(n = 10):
    return randomEncDecHelper(encoder , decoder , input_lang , output_lang , pairs , n)



def randomEncDecV2(n = 10):
    return randomEncDecHelper(encoderAttn , decoderAttn , input_lang , output_lang , pairs , n)
