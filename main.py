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

from transformer import *

from flask import Flask, request, jsonify, render_template, send_file




# app = Flask(__name__,
#             static_url_path='/', 
#             static_folder='static',
#             template_folder='templates')


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





@app.get("/")
def returnHomePage():
    return render_template("index.html")



@app.get("/getTranslation/v1/<sentence>")
def returnV1Translation(sentence):
    print(sentence)
    transation = evaluateInputSentence(sentence , version=1)
    return jsonify({
        'Actual Sentence' : sentence,
        'Translated Sentence' : transation
    })



@app.get("/getTranslation/v2/<sentence>")
def returnV2Translation(sentence):
    print(sentence)
    transation = evaluateInputSentence(sentence , version = 2)
    return jsonify({
        'Actual Sentence' : sentence,
        'Translated Sentence' : transation
    })



@app.get("/getRandomTranslationsV1/<n>")
def returnRandomV1Translation(n):
    n = int(n)
    return randomEncDecV1(n)


@app.get("/getRandomTranslationsV2/<n>")
def returnRandomV2Translation(n):
    n = int(n)
    return randomEncDecV2(n)




@app.get("/getTransformerV1/<sentence>")
def getTranslationFromTransformerV1(sentence):
    # model = loadTransformerModel("saved/transformer-model-200k.pt")
    # model.device = device
    # translated =  generate_translation_from_english(model , sentence , device)
    return jsonify({
        'Actual Sentence' : sentence,
        'Translated Sentence' : ""
    })




@app.get("/getTransformerV2/<sentence>")
def getTranslationFromTransformerV2(sentence):
    model = loadTransformerModel("saved/transformer-model-200k.pt")
    model.device = device
    translated =  generate_translation_from_english(model , sentence , device , eng_vocab_v2 ,hindi_vocab_v2)
    return jsonify({
        'Actual Sentence' : sentence,
        'Translated Sentence' : translated
    })



@app.get("/getTransformerV3/<sentence>")
def getTranslationFromTransformerV3(sentence):
    model = loadTransformerModel("saved/model-250k.pt")
    model.device = device
    translated =  generate_translation_from_english(model , sentence , device, eng_vocab_v3 ,hindi_vocab_v3 )
    return jsonify({
        'Actual Sentence' : sentence,
        'Translated Sentence' : translated
    })
    

@app.get("/getTransformerV4/<sentence>")
def getTranslationFromTransformerV4(sentence):
    model = loadTransformerModel("saved/model-300k.pt")
    model.device = device
    translated =  generate_translation_from_english(model , sentence , device , eng_vocab_v4 , hindi_vocab_v4)
    return jsonify({
        'Actual Sentence' : sentence,
        'Translated Sentence' : translated
    })

if __name__ == "__main__":
    app.run()