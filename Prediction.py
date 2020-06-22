#!/usr/bin/env python
# coding: utf-8

# In[1]:


from DataPreprocessing import DataPreprocessing
from TextCleaner import TextCleaner
from tensorflow.keras.models import model_from_json
from attention import AttentionLayer
import numpy as np


# In[2]:


processor = DataPreprocessing()
cleaner = TextCleaner()

data =  processor.load_pickle('DataSequences')

x_tr, x_test, x_dev, y_tr, y_test, y_dev = data[0],data[1],data[2],data[3],data[4],data[5]

loaded_data = processor.load_pickle('TokenizerData')

x_tokenizer, y_tokenizer, x_vocab_size,y_vocab_size, input_word_index,target_word_index, reversed_input_word_index, reversed_target_word_index, max_length_text, max_length_summary = loaded_data[0],loaded_data[1], loaded_data[2],loaded_data[3],loaded_data[4],loaded_data[5],loaded_data[6],loaded_data[7],loaded_data[8],loaded_data[9]


# In[3]:


class Prediction():
    
    def load_model(self,model_filename, model_weights_filename):
        with open(model_filename, 'r', encoding='utf8') as f:
            model = model_from_json(f.read(), custom_objects={'AttentionLayer': AttentionLayer})
        model.load_weights(model_weights_filename)
        return model
        
    def decode_sequence(self, input_seq, encoder_model, decoder_model):
        encoder_outputs, h0, c0 = encoder_model.predict(input_seq)

        # First word to be passed to the decoder 
        target_seq = np.zeros((1,1))
        target_seq[0,0] = target_word_index['sostok']

        decoded_sentence = ''
        stop_condition = False

        while not stop_condition:

            output_tokens, h, c = decoder_model.predict([target_seq] + [encoder_outputs,h0, c0])

            # Sample output tokens
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = reversed_target_word_index[sampled_token_index]

            if(sampled_word!='eostok'):
                decoded_sentence += sampled_word + ' '

            if (sampled_word == 'eostok' or (len(decoded_sentence.split()) + 1) > max_length_summary):
                stop_condition = True

            target_seq = np.zeros((1,1))
            target_seq[0,0] = sampled_token_index

            h0, c0 = h,c

        return decoded_sentence
    
    def seqtotext(self, input_seq):
        text = ''
        for i in input_seq:
            if i != 0:
                text += reversed_input_word_index[i]+ ' '

        return text
    
    def seqtosummary(self, input_seq):
        summary = ''
        for i in input_seq:
            if((i!=0 and i!=target_word_index['sostok']) and i!=target_word_index['eostok']):
                summary=summary+reversed_target_word_index[i]+' '
        return summary
    
    def generated_summaries(self,index,encoder_model,decoder_model):
        """Generates 'index' number of summaries."""
        for i in range(0,index):
            print("Original summary:",self.seqtosummary(y_tr[i]))
            print("Predicted summary:",self.decode_sequence(x_tr[i].reshape(1,max_length_text),encoder_model,decoder_model))
            print("\n")

