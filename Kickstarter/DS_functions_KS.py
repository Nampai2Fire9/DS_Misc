# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 15:26:08 2020
General Functions
@author: Owner
"""

import pandas as pd 
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from wordcloud import WordCloud, STOPWORDS   # WordCloud visuals 

import nltk 
# Lemmatizer reduces words into their root form: wolves -> wolf, jumping -> jump, etc. 
from nltk.stem import WordNetLemmatizer as WNL    
from nltk.corpus import wordnet 
from nltk import pos_tag
from nltk.corpus import stopwords


#---------------- List and Data Frame Operation ----------------
    # Function break lists into seperate lines 
        
        # Return length of list within data frame given DF & column with list 
def flat_len_list(df, list_column): 
    len_list = []
    for fld_row in range(df.shape[0]):
        rec = df[list_column][fld_row][1:-1]   # remove open/close brackets from list 
        rec_split = rec.split(',') 
        len_list.append(len(rec_split))
    return len_list 

        # Return list with all list elements from list column broken out.  
def flatten_list_df(df, list_column): 
    t_list = []
    for fld_row in range(df.shape[0]):
        rec = df[list_column][fld_row][1:-1]   # remove open/close brackets from list 
        rec_split = rec.split(',') 
        for x in rec_split:
            t_list.append(x)
    return t_list 

    # Function to expand list column and incorporate back into dataframe: 
        # uses flat_len_list and flat_list_df functions above 
def expand_df_flatten(df, list_column, new_column):
    lens_of_lists = flat_len_list(df, list_column) 
    origin_rows = range(df.shape[0])   # range object for no. of rows: 4k rows 
    """ create array marking each digit with its row. 
    For instance first list (row zero) has 26 digits, 2nd list 75 digits. 3rd list has 457 digits 
    destination rows will have 26 0's, 75 1's, 457 2's.... 
    This will be used for number of row copies. 
    """
    destination_rows = np.repeat(origin_rows, lens_of_lists)
        # create DF excluding list column 
    non_list_cols = (
      [idx for idx, col in enumerate(df.columns)
       if col != list_column]
    )
    expanded_df = df.iloc[destination_rows, non_list_cols].copy()
    expanded_df[new_column] = flatten_list_df(df, list_column) 
    return expanded_df

 # Function for word presence: takes a string of words, seperates them, and checks for presence of word 
def str_presence(input_string, search_string):
    token = input_string
    token = token.lower()
    # token = token.translate(token.maketrans('','',string.punctuation))
    token = [tk.strip() for tk in token.split("-")] 
    token = [tk for tk in token if len(tk)>2]   # remove article words
    # pos_tags = pos_tag(token)
    # token = [wlem.lemmatize(t[0],get_wordnet_pos(t[1])) for t in pos_tags]
    return int(any(x in token for x in search_string))

# Function to run loop to populate array (fill_array) with values 
  
def str_presence_full(string_column, search_string): 
    fill_array = [] 
    for str in string_column:
        present_fl = str_presence(str, search_string)
        fill_array.append(present_fl)
    return fill_array


#------------------- Text Processing ------------------ 
#-------- WordCloud  --------------- 
 
def wordcloud_vis(text_array):
    wordcloud = WordCloud(
        width = 3000,
        height = 2000,
        background_color = 'black',
        stopwords = STOPWORDS).generate(str(text_array))
    fig = plt.figure(
        figsize = (40, 30),
        facecolor = 'k',
        edgecolor = 'k')
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    