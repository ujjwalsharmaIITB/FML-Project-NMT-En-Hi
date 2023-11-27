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


import csv
import string
import random
import torch
import torch.nn as nn
import torch.optim as optim
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















#######. Inference for Transformers













MAX_LENGTH=20

class Vocab_builder:
    def __init__(self):
        self.word_2_index={"<SOS>":0,"<EOS>":1,"<PAD>":2,"<UKN>":3}
        self.index_2_word={0:"<SOS>", 1:"<EOS>", 2:"<PAD>", 3:"<UKN>"}
        self.freq={}
        self.size=4

    def add_this_sentence(self,sentence):
        words=sentence.split(" ")
        for word in words:
            if word not in self.word_2_index:
                #If the word is not there, add it to a new index and store the indexes
                #Initialize the frequency of the word to 1 and increase the size of the vocabulary
                self.word_2_index[word]=self.size
                self.freq[word]=1
                self.index_2_word[self.size]=word
                self.size+=1
            else:
                # If the word is already present then just increase the frequency
                self.freq[word]+=1



hindi_vocab=Vocab_builder()
eng_vocab=Vocab_builder()





def length(sentence):
    '''
        Function to tell the length of a sentence.
    '''
    return len(sentence.split(" "))

def is_mixed(sentence):
    '''
        This function will return True if a hindi sentence is containing some english character.
    '''
    letters="abcdefghijklmnopqrstuvwxyz"
    for ch in letters:
        if ch in sentence:
            return True
    return False

def preprocess(sentence):
    '''
        This function will apply the neccesary preprocessing to a sentence
    '''
    #First we will remove all punctuations from the sentence
    punctuations=list(string.punctuation)
    cleaned=""
    for letter in sentence:
        if letter not in punctuations:
            cleaned+=letter
    cleaned=cleaned.lower() ## Converting into lowercase
    return cleaned



def clean_the_data(path):
    '''
      This function will load the data and process it line by line.
      It will apply all the preprocessing and make the data ready for further processing.
    '''
    pairs=[]
    with open(path,'rt') as f:
        data=f.readlines()
        row_num=0
        for row in data:
            if row_num!=0:  #We will not process first row as it will contain header
                unfiltered_sentences = row.split("_sep_")
                hindi=unfiltered_sentences[1]
                eng=unfiltered_sentences[0]

                if length(hindi)>=MAX_LENGTH or length(eng)>=MAX_LENGTH:  #skipping if length is more than MAX_LENGTH
                    continue
                if not hindi or not eng:  #skipping pair having any NULL value
                    continue
                if is_mixed(hindi):   #skipping sentence if it contains some english word
                    continue
                hindi=hindi.encode('utf-8',errors='ignore').decode('utf-8')
                eng=eng.encode('ascii',errors='ignore').decode('utf-8')
                hindi=preprocess(hindi)
                eng=preprocess(eng)
                #Adding <SOS>, <EOS> and padding tokens
                pair=[hindi.strip(), eng.strip()]

                hin_extra=MAX_LENGTH-len(hindi.strip().split(" "))
                eng_extra=MAX_LENGTH-len(eng.strip().split(" "))

                hindi_vocab.add_this_sentence(pair[0])
                eng_vocab.add_this_sentence(pair[1])
                pair[0]=pair[0].split(" ")
                pair[0].insert(0,"<SOS>")
                pair[0].append("<EOS>")
                pair[0]=pair[0]+["<PAD>"]*(hin_extra)

                pair[1]=pair[1].split(" ")
                pair[1].insert(0,"<SOS>")
                pair[1].append("<EOS>")
                pair[1]=pair[1]+["<PAD>"]*(eng_extra)

                pair[0]=" ".join(pair[0])
                pair[1]=" ".join(pair[1])
                pairs.append(pair)
            row_num+=1
    return pairs



train_file_path = "data/eng-hin-train-200000.txt"
pairs=clean_the_data(train_file_path)

print(len(pairs))



def clean_sentence(sentence):
    '''
      Function to remove the punctuation from the test sentence
    '''
    punctuations=list(string.punctuation)
    cleaned=""
    for letter in sentence:
        if letter=='<' or letter=='>' or letter not in punctuations:
            cleaned+=letter
    return cleaned

def predict_translation(model,sentence,device,max_length=MAX_LENGTH):
    '''
      function will return the translation predicted by the trained model for each sentence
    '''
    sentence=clean_sentence(sentence)
    tokens=sentence.split(" ")
    indexes=[]
    for token in tokens:
        if token in eng_vocab.word_2_index:
            indexes.append(eng_vocab.word_2_index[token])
        else:
            indexes.append(eng_vocab.word_2_index["<UKN>"])
    indexes=indexes[:MAX_LENGTH+2]  # model is trained on MAX_LENGTH sentences only so it expects sentences of this length only
    tensor_of_sentence=torch.LongTensor(indexes).unsqueeze(1).to(device)
    outputs=[0]   #adding <SOS> in the beginning of output
    for _ in range(max_length):
        target_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)
        with torch.no_grad():
            output=model(tensor_of_sentence,target_tensor)
        pred=output.argmax(2)[-1, :].item()

        outputs.append(pred)

        if hindi_vocab.index_2_word[pred] =="<EOS>":
            break

    final=[]

    for i in outputs:
        if i == "<PAD>":
            break
        final.append(i)

    final = [hindi_vocab.index_2_word[idx] for idx in final if idx not in [0,1,2]]
    translated=" ".join(final)
    return translated

def generateEnglishSentence(sentence):
    extra_tokens = MAX_LENGTH-len(sentence.strip().split(" "))
    cleanedSentence = sentence.split(" ")
    cleanedSentence.insert(0,"<SOS>")
    cleanedSentence.append("<EOS>")
    cleanedSentence = cleanedSentence +["<PAD>"]*(extra_tokens)
    return " ".join(cleanedSentence)


def generate_translation_from_english(model , sentence , device):
  sentence = generateEnglishSentence(sentence)
  return predict_translation(model , sentence , device)




def loadTransformerModel(path):
    model = torch.load(path , map_location=torch.device('cpu'))
    return model