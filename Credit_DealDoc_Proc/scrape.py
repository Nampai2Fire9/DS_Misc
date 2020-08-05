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
    # Extract pages into list: one page per list litem
    # Note pdfplumber treats page 1 as page 0, and last page X
        # Hence:To pull page 1 to page 10, start_page = 0, end_page = 10
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
    # Grab section of text using two regex searches: requires re package
        # input string and beginning and end search strings/regexes
        # output: substring between beginning and points 
def pullsubstr_begend(search_begin, search_end, full_string):
    start_loc = re.search(search_begin, full_string).span()[0]
    end_loc = re.search(search_end, full_string).span()[0]
    return full_string[start_loc:end_loc]
    
    # Grab substring from start of string to regex. 
def pullsubstr_beg_regex(regex_str, search_str):
    end_loc = re.search(regex_str, search_str).span()[0]  # ending location 
    return search_str[0:end_loc]
        
    # add closing paragraph
def pullsubstr_mid2end(regex_str, full_string):
    start_loc = re.search(regex_str, full_string).span()[0]
    return full_string[start_loc:len(full_string)]

# Function that create dictionary from text string with embedded table. Assumes no other bullet points: just paragraph of text table. 
    # Input: text string w/ table, regex marking the start of the table, table to insert
def pop_tbldict(tbl_insert, text_w_string, regex):
    tbl_dict={}
    tbl_dict['opening_sec'] = pullsubstr_beg_regex(regex,text_w_string)
    tbl_dict['table'] = tbl_insert
    return tbl_dict

# populate existing dictionary with list of keys
    # Input: dict to populate (pop_dict), key names:
    # items: bullet_items
def pop_dictbullets(section_value, bullet_list, bullet_items, pop_dict):
    for i in np.arange(len(bullet_items)):
        x_str = section_value + '(' + bullet_list[i] + ')'
        pop_dict[x_str] = bullet_items[i]

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
    test_str = list_betweenloc_to_string_bounded_keepbullet(match_list, text_string, end_location)
    return test_str

    # Return list between locations: similar to list_betweenloc_to_string_bounded_keepbullet, but input is only start of a range for each match instead of the full span
def list_betweenloc_to_string_bounded_keepbullet_order(location_list, text_string, end_location):
    str_list = []
    for i in np.arange(0, len(location_list)-1):  # process everything but last section. Note np.arange(0 ,x) returns numbers 0 to x-1
        start_loc = location_list[i]
        end_loc = location_list[i+1]-1
        str_list.append(text_string[start_loc:end_loc])
    last_start_loc = location_list[len(location_list)-1]
    str_list.append(text_string[last_start_loc:end_location])
    return str_list

    # Process regex list while keeping track of locations 
        # in case there are multiple matches for the same regex within a list. 
        # input:text string, end location and regex_list
        # Output: uses 'list_betweenloc_to_string_bounded_keepbullet_order' to return list of strings
def list_btn_loc_regexlist_keepbullet2_order(text_string, end_location, regex_list):
    match_list = []  # capture initial matches: returns a list of listss 
    for i, item in enumerate(regex_list):
        new_re = re.compile(item)
        matches = new_re.finditer(text_string)
        sub_list = []
        # grab all matches for one regex item
        for match in matches:
            sub_list.append(match.span()[0]) 
        # Process sub_lists by item
        match_list.append(sub_list) 
    first_none = [i for i, item in enumerate(match_list) if item == []][0]
    match_list = match_list[0:first_none]
    # Pro
    match_list2 = [] # capture only start locations in order. 
    match_list2.append(match_list[0][0])  # populate first location in list of locations
    # populate rest of locations, isolating the first instance that is greater than prior location
    for i, x_list in enumerate(match_list[1:len(match_list)]):
        x_gt_prev = []  # list to house any location value greater than prevous location value
        for x in x_list:
            if x > match_list2[i]:
                x_gt_prev.append(x)
        if x_gt_prev != []: 
            match_list2.append(x_gt_prev[0])    
    match_str = list_betweenloc_to_string_bounded_keepbullet_order(match_list2, text_string, end_location)
    return match_str

#-------- Table Processing  ------------- 
    # Shift columns names to first row and create new table. 
      # Tabula often makes the first row the title. 
        # Input: imported dataframe, new column names
        # Output: dataframe with the column names shifted into first row and actual column names set
def shift_colnames_1strow(df, col_names):
    df1 = df.copy()
    df1.columns = col_names
    df1.loc[-1] = list(df.columns)   
    df1.index = df1.index + 1 
    df1 = df1.sort_index()
    return df1
    
    # Remove starting rows and rename (opposite problem to shift_colnames_1strow where row turns into column headers)
        # Input dataframe, new column names, and the number of rows to remove
        # output: new dataframe with columns names and rows removed. 
def remrows_table_rename(df, col_names, no_rows_remove):
    df1 = df.copy()
    df1.columns = col_names
    df1 = df1.iloc[no_rows_remove:len(df1)].reset_index(drop=True)
    return df1

    # split out field with lists and add back into df
        # Input  DF, field with lists, new column names, and value to split on (ex. ' ')
        # Output DF with old DF + new columns
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

#--------------------------------- Indenture level Processing  ------------------ 
    #------------ Combine multiple functions into one
    
    # Create function that processes definitions section of indenture
        # Input: filename, start and end pages for definitions, regex for definition itself 
        # Output: DataFrame with definitions, page numbers for each definition and initial roman/letter bullets broken out. 
def def_1st_tbl_indent(file_name, start_page, end_page, regex, roman_regex_list, letter_regex_list):
    doc_pages = pdf_extract_pages(file_name, start_page, end_page)
    doc_str = combine_strings(doc_pages) 
    def_locs = text_match_2list(regex, doc_str)
    def_descr = list_betweenloc_to_string_bounded(def_locs, doc_str, len(doc_str))
    df_defs = pd.DataFrame([doc_str[x[0]+1:x[1]-2] for x in def_locs])
    df_defs['description'] = def_descr
    df_defs.columns = ['def_name','def_description']
    
    # add page numbers
    def_names = []
    for i,page in enumerate(doc_pages):
        page_num = i + 10
        ranges = text_match_2list(regex, page)
        for range in ranges:
            def_names.append([page[range[0]+1:range[1]-2], page_num])
    def_name_pages = pd.DataFrame(def_names, columns=['def_name','page_num'])
    df_defs = pd.merge(df_defs, def_name_pages, on='def_name', how='left')
    
    # add roman and letter bullet points 
    regex_roman = roman_regex_list[0]
    regex_letter = letter_regex_list[0] 
        
    match_reg = [re.search(regex_roman, x) for x in df_defs['def_description']]
    roman_bullet_flag = [x.span()[0] if x is not None else 0 for x in match_reg]
    df_defs['roman_bullets_loc'] = roman_bullet_flag
    
    match_reg = [re.search(regex_letter, x) for x in df_defs['def_description']]
    letter_bullet_flag = [x.span()[0] if x is not None else 0 for x in match_reg]
    df_defs['letter_bullets_loc'] = letter_bullet_flag
    return df_defs

# function to clean up names in df_defs: remove double spaces and pdf-style quotes
def df_def_cleanup(df_defs):
    df = df_defs.copy()
    def_name = [x.replace('  ', ' ') for x in df_defs['def_name']]
    def_name = [x.replace("”", '') for x in def_name]
    def_name = [x.replace("“", '') for x in def_name]
    def_name =  [x.replace("’", "'") for x in def_name] 
    def_name = [x.replace('"', '') for x in def_name]
    df['def_name'] = def_name
    return df

    # Function for splitting out the first letter  or roman bullets and adding back into table 
        # Input: DF with definitions, regex lists
        # Output: DF with definitions and first layer bullets split out in separate column of lists
def def_first_bullet_combine(df_defs, regex_list1, regex_list2):
    # Process roman bullets
    df_defs_sub1 =df_defs[(
                        (df_defs.roman_bullets_loc < df_defs.letter_bullets_loc) & 
                         (df_defs.roman_bullets_loc != 0)
                        ) | 
                     ( 
                        (df_defs.roman_bullets_loc !=0) & 
                          (df_defs.letter_bullets_loc == 0)
                         )
                    ]
    rom_def_strings = [list_btn_loc_regexlist_keepbullet2_order(x,len(x),regex_list1) for x in df_defs_sub1['def_description']]
    df_defs_sub1['roman_bullets'] = rom_def_strings 
    
    # Process letter bullets
    df_defs_sub2 =df_defs[(
                        (df_defs.roman_bullets_loc > df_defs.letter_bullets_loc) & 
                         (df_defs.letter_bullets_loc != 0)
                        ) | 
                     ( 
                        (df_defs.letter_bullets_loc !=0) & 
                          (df_defs.roman_bullets_loc == 0)
                         )
                    ]
    
    lowalpha_def_strings = [list_btn_loc_regexlist_keepbullet(x,len(x),regex_list2) for x in df_defs_sub2['def_description']]
    df_defs_sub2['alpha_bullets'] = lowalpha_def_strings
    
    # Combine table back to original definition table (df_defs)
    df_def_comb = pd.merge(pd.merge(df_defs, df_defs_sub1[['def_name','roman_bullets']], on='def_name', how='left'), 
                                df_defs_sub2[['def_name','alpha_bullets']], 
                            on='def_name',how='left')
    return df_def_comb
    

    # Convert definition dataframe with first bullet layer to dictionary 
        # Input: dataframe with definitions and first bullet layer broken out 
        # Output: Dictionary with definitions, bullets, and page numbers 
def def_pop_firstdict(df_def_comb):
    def_dict = {} 
    for i, item in enumerate(df_def_comb['def_description']):
        key_name = df_def_comb.iloc[i]['def_name']
        key_vals = {}   # holds definition value
        sub_dict = {}   # holds items within definition value
        roman_loc1 = df_def_comb.iloc[i]['roman_bullets_loc']  # will re-use for rest of loop 
        letter_loc1 = df_def_comb.iloc[i]['letter_bullets_loc']
        key_vals['page_num'] = df_def_comb.iloc[i]['page_num']
        key_vals['systemTerm'] = df_def_comb.iloc[i]['systemTerm']
        key_vals['text'] = item
        
        # Definitions WITHOUT bullet points 
        if( (df_def_comb.iloc[i]['roman_bullets_loc']==0) &
               (df_def_comb.iloc[i]['letter_bullets_loc']==0) ):
            key_vals['def_values'] = item

        # Definitions with bullets: lowercase Roman comes first. 
        elif( ( 
                (df_def_comb.iloc[i]['roman_bullets_loc'] < df_def_comb.iloc[i]['letter_bullets_loc'] ) & 
                    (df_def_comb.iloc[i]['roman_bullets_loc'] != 0 )
                ) | 
              ( 
                (df_def_comb.iloc[i]['roman_bullets_loc'] != 0) & 
                  (df_def_comb.iloc[i]['letter_bullets_loc'] == 0)
                )
            ):
                sub_dict['opening_sec'] = df_def_comb.iloc[i]['def_description'][0:roman_loc1]
                sub_dict['bullet_level1'] = df_def_comb.iloc[i]['roman_bullets']
                key_vals['def_values'] = sub_dict

        # Definitions with bullets: lowercase Roman comes first. 
        elif( ( 
                (df_def_comb.iloc[i]['letter_bullets_loc'] < df_def_comb.iloc[i]['roman_bullets_loc'] ) & 
                    (df_def_comb.iloc[i]['letter_bullets_loc'] != 0 )
                ) | 
              ( 
                (df_def_comb.iloc[i]['letter_bullets_loc'] != 0) & 
                  (df_def_comb.iloc[i]['roman_bullets_loc'] == 0)
                )
            ):
                sub_dict['opening_sec'] = df_def_comb.iloc[i]['def_description'][0:letter_loc1]
                sub_dict['bullet_level1'] = df_def_comb.iloc[i]['alpha_bullets']
                key_vals['def_values'] = sub_dict

        else: def_dict[key_name] = 5000
        def_dict[key_name] = key_vals
    return def_dict