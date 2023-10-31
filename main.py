    # Created by Ujjwal Sharma ,
    # 23M0837 , 23M0837@iitb.ac.in
    # github@ujjwalsharmaIITB


import __main__
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from encoder_decoder import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from inference import *

from encoder_decoder import *

from flask import Flask, request, jsonify, render_template







app = Flask(__name__)

# Use this in encoder_decoder file

# # This is very important as this will now see the pickle
# # object in the main
# # you need to unpack the pickle object from __main__ if you have
# # saved it from a python script app
# # otherwise use different script and and ollow the same structure
# # here

# __main__.Encoder  = Encoder
# __main__.Decoder = Decoder

# See This
# https://stackoverflow.com/questions/46814753/flask-attribute-error-with-unpickling
# Main
# https://www.reddit.com/r/flask/comments/yfbkks/getting_cant_get_attribute_getimages_on_module/







# @app.get("/")
# def returnHomePage():
#     return render_template("index.html")



@app.get("/getTranslation/v1/<sentence>")
def returnPartOfSpeechTag(sentence):
    print(sentence)
    transation = evaluateInputSentence(sentence , version=1)
    return jsonify({
        'Actual Sentence' : sentence,
        'Translated Sentence' : transation
    })



@app.get("/getTranslation/v2/<sentence>")
def returnPartOfSpeechTag(sentence):
    print(sentence)
    transation = evaluateInputSentence(sentence , version = 2)
    return jsonify({
        'Actual Sentence' : sentence,
        'Translated Sentence' : transation
    })

# evaluateRandomly(encoder , decoder ,pairs , input_lang , output_lang)










# def infer(encoder , decoder , pairs , input_lang , output_lang):
#     evaluateRandomly(encoder ,  decoder , pairs , input_lang , output_lang )
    



# infer(encoder , decoder , pairs , input_lang , output_lang)

