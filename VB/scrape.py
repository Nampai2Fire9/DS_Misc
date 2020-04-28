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
        match_ranges.append(match.span())
    return match_ranges


    # same as text_match_2list except returns string values using locations. 
def text_match_str2list(regex, text_string):
    new_re = re.compile(regex)
    matches = new_re.finditer(text_string)
    match_list = []
    for match in matches:
        start_loc = match.span()[0]
        end_loc = match.span()[1]
        match_list.append(text_string[start_loc:end_loc])
    return match_list


# Turn list of locations to string tests. Assumes bounded region (to grab last item)
def list_betweenloc_to_string_bounded(location_list, text_string, end_location):
    str_list = []
    for i in np.arange(0, len(location_list)-1):  # process everything but last section
        start_loc = location_list[i][1]
        end_loc = location_list[i+1][0]-1
        str_list.append(text_string[start_loc:end_loc])
    last_start_loc = location_list[len(location_list)-1][1]
    str_list.append(text_string[last_start_loc:end_location])
    return str_list

# Clean up items from list of strings
    # remove (i) new lines (ii) semicolons, (iii) leading whitespace 
def str_item_cleanup(string_list):
    string_list2 = []
    for item in string_list:
        str1 = item.replace('\n', '')
        str1 = str1.replace(';','')
        str1 = str1.lstrip()     # remove leading whitespace 
        string_list2.append(str1)
    return string_list2