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
    # Output: List of text strings between bullet points
def list_betweenloc_to_string_bounded(location_list, text_string, end_location):
    str_list = []
    for i in np.arange(0, len(location_list)-1):  # process everything but last section
        start_loc = location_list[i][1]
        end_loc = location_list[i+1][0]-1
        str_list.append(text_string[start_loc:end_loc])
    last_start_loc = location_list[len(location_list)-1][1]
    str_list.append(text_string[last_start_loc:end_location])
    return str_list
    
    # Process list between locations / keep bullets:  similar to list_betweenloc_to_string_bounded but keeps bullets 
        # output: list of strings w/ original bullets (roman numerals, letters, etc.
def list_betweenloc_to_string_bounded_keepbullet(location_list, text_string, end_location):
    str_list = []
    for i in np.arange(0, len(location_list)-1):  # process everything but last section
        start_loc = location_list[i][0]
        end_loc = location_list[i+1][0]-1
        str_list.append(text_string[start_loc:end_loc])
    last_start_loc = location_list[len(location_list)-1][0]
    str_list.append(text_string[last_start_loc:end_location])
    return str_list

    # Process regex_list/Keepb bullets for text_string: locations dictated by regex_list
       # Input text string, regex_list and end_location
        # Output: bullet points with bullets
def list_btn_loc_regexlist_keepbullet(text_string, end_location, regex_list):
    match_list = [re.search(x,text_string) for x in regex_list]
    first_none = [i for i, item in enumerate(match_list) if item is None][0]   # first record w/ None. remove list from 'None'
    match_list = match_list[0:first_none]            # remove list from 'None'
    match_list = [x.span() for x in match_list]
    test_str = list_betweenloc_to_string_bounded_keepbullet(match_list, text_string, len(text_string))
    return test_str

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