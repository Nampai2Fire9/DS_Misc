# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 11:11:14 2020

@author: Owner
"""

import os
import re
import pdfplumber
import numpy as np
import pandas as pd 
import json

#---- Extract pages
    # Extract pages into list: one page per list litem: 
def pdf_extract_pages(file_name, start_page, end_page):
    text_list = [] 
    with pdfplumber.open(file_name) as pdf:
        for i in np.arange(start_page,end_page):
            page = pdf.pages[i]    # return pdfplumber page object
            text = page.extract_text()  # extract text from page object 
            text_list.append(text)
    return text_list 

    # combine pages into one string
        # Input list of strings (example pages of text) 
def combine_strings(text_list):
    doc_str = text_list[0]
    for i in np.arange(1, len(text_list)):
        doc_str = doc_str + text_list[i]
    return doc_str 

# -------- Text Processing  ------- 
    # Load multiple matches into list 
        # Input (i) regular expression (ii) text string 
        # Output: list of 'locations' within text string matching regular expressions 
def text_match_2list(regex, text_string): 
    new_re = re.compile(regex)
    matches = new_re.finditer(text_string)
    match_ranges = []
    for match in matches:
        print(match.span()) 
        match_ranges.append(match.span())
        start_item = match.span()[0]
        end_item = match.span()[1]
        text_string[start_item:end_item]
    return match_ranges