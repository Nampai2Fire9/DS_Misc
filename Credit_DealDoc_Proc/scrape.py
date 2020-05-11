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
import tabula

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
        # Input: takes one regex and string of text
        # output: returns items in string matching regex
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
    # Input: list of locations, end_location, and text string
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
        # Input:  list of locations, end_location, and text string
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
        # Output: list of string with bullet points and bullets included
def list_btn_loc_regexlist_keepbullet(text_string, end_location, regex_list):
    match_list = [re.search(x,text_string) for x in regex_list]
    first_none = [i for i, item in enumerate(match_list) if item is None][0]   # first record w/ None. remove list from 'None'
    match_list = match_list[0:first_none]            # remove list from 'None'
    match_list = [x.span() for x in match_list]
    test_str = list_betweenloc_to_string_bounded_keepbullet(match_list, text_string, len(text_string))
    return test_str

#-------- Table Processing  ------------- 
    # Shift columns names to first row and create new table. 
      # Tabula often makes the first row the title. 
        # Input: imported dataframe, new column names
        # Output: dataframe with the column names shifted into first row and actual column names set
def shift_colnames_1strow(df, col_names):
    df1 = df
    df1.columns = col_names
    df1.loc[-1] = list(df.columns)   
    df1.index = df1.index + 1 
    df1 = df1.sort_index()
    return df1

    # Remove starting rows and rename (opposite problem to shift_colnames_1strow where row turns into column headers)
        # Input dataframe, new column names, and the number of rows to remove
        # output: new dataframe with columns names and rows removed. 
def remrows_table_rename(df, col_names, no_rows_remove):
    df1 = df
    df1.columns = col_names
    df1 = df1.iloc[no_rows_remove:len(df1)].reset_index(drop=True)
    return df1

    # split out field with lists
        # DF, field with lists, new column names, and value to split on (ex. ' ')
def split_lists_newdf(df, field, col_names, split_val):
    df1 = pd.DataFrame([x.split(split_val) for x in df[field]])
    df1.columns = col_names
    df1 = pd.concat([df, df1], axis=1)
    return df1

#------ String Cleanup 

# Clean up items from list of strings
    # remove (i) new lines (ii) semicolons, (iii) leading whitespace 
def str_item_cleanup(string_list):
    string_list2 = []
    for item in string_list:
        str1 = item.replace('\n', '')
        str1 = str1.replace(';','')
        # str1 = str1.lstrip()     # remove leading whitespace 
        str1 = str1.strip()        # remove trailing and leading whitespace 
        string_list2.append(str1)
    return string_list2

# Clean up function with list of items to remove 
def str_rmvlist_cleanup(string_list, rmv_list):
    str_list = string_list
    for rmv in rmv_list:
        str_list = [x.replace(rmv, '') for x in str_list]
    str_list = [x.strip() for x in str_list]
    return str_list