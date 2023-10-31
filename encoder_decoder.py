
    # Created by Ujjwal Sharma ,
    # 23M0837 , 23M0837@iitb.ac.in
    # github@ujjwalsharmaIITB



import __main__
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

MAX_LENGTH = 15
SOS_token = 0
EOS_token = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self , input_size , hidden_size , dropout = 0.2):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        # We will train the embedding layer simultaneously
        # Alternatively you can have pre-trained embeddings
        self.embedding_layer = nn.Embedding(input_size , hidden_size)

        self.rnn = nn.LSTM(hidden_size , hidden_size , batch_first = True)

        self.dropout = nn.Dropout(dropout)


    def forward(self , input_vector):
        embedded_output = self.dropout(self.embedding_layer(input_vector))

        output , hidden_state = self.rnn(embedded_output)

        return output , hidden_state

"""# Decoder
Simple Decoder

In the simplest seq2seq decoder we use only last output of the encoder. This last output is sometimes called the context vector as it encodes context from the entire sequence. This context vector is used as the initial hidden state of the decoder.

At every step of decoding, the decoder is given an input token and hidden state. The initial input token is the start-of-string <SOS> token, and the first hidden state is the context vector (the encoder’s last hidden state).

"""

class Decoder(nn.Module):
    def __init__(self , hidden_size , output_size):
        super(Decoder , self).__init__()
        # Embedding layer for the target language
        self.embedding_layer = nn.Embedding(output_size , hidden_size)
        # Now comes out RNN Model
        self.rnn = nn.LSTM(hidden_size , hidden_size , batch_first=True)
        # Finally our output Layer
        self.outputLayer = nn.Linear(hidden_size , output_size)


    def forward_step(self , input_vector , hidden_state):
        output = self.embedding_layer(input_vector)
        output = F.relu(output)
        output , hidden_state = self.rnn(output , hidden_state)
        output = self.outputLayer(output)

        return output , hidden_state


    def forward(self , encoder_output , encoder_hidden_state , target_tensor = None):
        batch_size = encoder_output.size(0)
        # for starting the sentence we fill all the values by SOS
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        # now comes the hidden state
        decoder_hidden_state = encoder_hidden_state

        decoder_outputs = []

        for i in range(MAX_LENGTH):
            # get the first word
            decoder_output , decoder_hidden_state = self.forward_step(decoder_input , decoder_hidden_state)
            decoder_outputs.append(decoder_output)

            # Teacher Forcing
            # giving the correct input to the classifier rather than giving
            # its own output
            if target_tensor is not None:
                # this will happen during training time
                decoder_input = target_tensor[: ,i].unsqueeze(1)
                # adding a dimension accross
            else:
                # generally we take top k for beam search and we maintain
                # these k candidate translations
                _ , topI = decoder_output.topk(1)
                # some pytorch output related trick, i dont know
                decoder_input = topI.squeeze(-1).detach()

        # concatinate along columns
        decoder_outputs = torch.cat(decoder_outputs , dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)

        return decoder_outputs, decoder_hidden_state, None

"""# Attention Mechanism


"""




class BahadanuAttention(nn.Module):
  def __init__(self , hidden_size):
    super(BahadanuAttention , self).__init__()

    self.Wa = nn.Linear(hidden_size , hidden_size)
    self.Ua = nn.Linear(hidden_size , hidden_size)
    self.Va = nn.Linear(hidden_size , 1)

  def forward(self , decoder_hidden , encoder_hidden):
    align_scores = self.Va(torch.tanh(self.Wa(decoder_hidden) + self.Ua(encoder_hidden)))

    #                             n*h*h + n*h*h = n*h*1
                                  # n*h n*1*h
    align_scores = align_scores.squeeze(2).unsqueeze(1)

    probabilisticWeights = F.softmax(align_scores , dim = -1) # n*1*h

    context_vector = torch.bmm(probabilisticWeights , encoder_hidden) # n*1*n = n * alphaij * hij


    return context_vector , probabilisticWeights



class AttentionDecoder(nn.Module):
  def __init__(self , hidden_size , output_size , drop_out = 0.1):
    super(AttentionDecoder , self).__init__()

    self.embedding = nn.Embedding(output_size , hidden_size)

    self.simpleAttention = BahadanuAttention(hidden_size)

    self.rnn =  nn.GRU(2*hidden_size , hidden_size , batch_first=True)

    self.output = nn.Linear(hidden_size , output_size)

    self.dropout = nn.Dropout(drop_out)



# This is a slightly modidied code , it uses encoder outputs rather than hidden
# states because they are also generateed from hidden states
  def forward_step(self , input_word , decoder_hidden , encoder_outputs):
    embedded_input = self.dropout(self.embedding(input_word))


# hidden state is also called query
    hidden_state_as_query = decoder_hidden.permute(1,0,2)

    context_vector , attention_weights = self.simpleAttention(hidden_state_as_query , encoder_outputs)

    input_rnn = torch.cat((embedded_input ,context_vector ) , dim=2)

    output, hidden = self.rnn(input_rnn, decoder_hidden)

    output = self.output(output)

    return output, hidden, attention_weights



  def forward(self , encoder_outputs , encoder_hidden , target_tensor=None):
    batch_size = encoder_outputs.size(0)
    decoder_input = torch.empty(batch_size , 1, dtype = torch.long, device = device).fill_(SOS_token)

    decoder_hidden = encoder_hidden
    decoder_outputs = []
    attentions = []

    for i in range(MAX_LENGTH):
      decoder_output , decoder_hidden , attention_weights = self.forward_step(decoder_input , decoder_hidden , encoder_outputs )
      decoder_outputs.append(decoder_output)
      attentions.append(attention_weights)

      if target_tensor is not None:
        # Teacher Forcinr
        # Teacher forcing: Feed the target as the next input
        decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
      else:
        _ , topRes = decoder_output.topk(1)
        decoder_input = topRes.squeeze(-1).detach()

    decoder_outputs = torch.cat(decoder_outputs , dim = 1)
    decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
    attentions = torch.cat(attentions, dim=1)

    return decoder_outputs, decoder_hidden, attentions









# This is very important as this will now see the pickle
# object in the main
# you need to unpack the pickle object from __main__ if you have
# saved it from a python script app
# otherwise use different script and and ollow the same structure
# here

__main__.Encoder  = Encoder
__main__.Decoder = Decoder
__main__.BahadanuAttention = BahadanuAttention
__main__.AttentionDecoder = AttentionDecoder
