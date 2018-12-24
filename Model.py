
# coding: utf-8

# In[1]:


import keras
import numpy as np

np.random.seed(5)
# In[2]:


from keras.layers import Input,Dense,Lambda,Embedding, LSTM, Dropout
from keras.models import Model
from keras.optimizers import Adam,SGD
from keras.layers import BatchNormalization


# In[3]:


MAX_LEN=59
VOCAB=8003


# In[4]:


class keras_model():
    def __init__(self):
        self.emb_dim = emb_dim = 50
        self.latent_dim = latent_dim = 200
        self.encoder_inputs = Input(shape=(None,),name='encoder_input')
        self.emb = Embedding(input_dim=VOCAB,output_dim=emb_dim)
        self.encodin = self.emb(self.encoder_inputs)
        encodin_bn = (self.encodin)
        self.encoder = LSTM(latent_dim, return_state=True, recurrent_dropout=0.05)
        self.encoder_outputs, state_h, state_c = self.encoder(encodin_bn)
        state_h = Dropout(0.05)(state_h)
        state_c = Dropout(0.05)(state_c)
        # We discard `encoder_outputs` and only keep the states.
        self.encoder_states = [state_h, state_c]
        # Set up the decoder, using `encoder_states` as initial state.
        self.decoder_inputs = Input(shape=(None,),name='decoder_input')
        self.decodin = self.emb(self.decoder_inputs)
        self.decodin_bn = (self.decodin)
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the 
        # return states in the training model, but we will use them in inference.
        self.decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, recurrent_dropout=0.1)
        self.decoder_outputs, _, _ = self.decoder_lstm(self.decodin_bn,
                                             initial_state=self.encoder_states)
        self.decoder_dense = Dense(VOCAB, activation='softmax')
        self.decoder_outputs = self.decoder_dense(self.decoder_outputs)
        
        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        self.model = Model([self.encoder_inputs, self.decoder_inputs], self.decoder_outputs)
        self.model.compile(loss='sparse_categorical_crossentropy',metrics=['acc'],optimizer=SGD(lr=0.1,decay=1e-3))


# In[5]:




# In[6]:


class Data():
    def __init__(self):
        conv_X = np.load('convo_X.npy')
        conv_Y = np.load('convo_Y.npy')
        
        Y_shifted = conv_Y[:,1:]
        Y_shifted = np.concatenate([Y_shifted,np.zeros((len(conv_Y),1))],axis=-1)
        
        Y_shifted = np.reshape(Y_shifted,Y_shifted.shape+(1,))
        
        self.data = [conv_X,conv_Y],Y_shifted


# In[7]:
if __name__=="__main__":
    model = keras_model()

    dat = Data()

    #model.model.load_weights('model.hdf5')
    # In[8]:


    from keras.callbacks import ModelCheckpoint
    mc = ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5',save_weights_only=True,save_best_only=False)


    # In[9]:

    initial_epoch = 0
    model.model.summary()
    model.model.fit(*dat.data,epochs=1000,validation_split=0.15,callbacks=[mc],initial_epoch=initial_epoch)


    # In[ ]:




    # In[ ]:


    m = Model(model.model.inputs,model.model.get_layer('dense_1').output)

