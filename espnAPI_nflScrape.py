#!/usr/bin/env python
# coding: utf-8


#------------------- Script For Scraping NFL data from ESPN API  -------------------#

'''

-----------
INFORMATION
-----------

Author: Jean-Sebastien Roussy
Date Created: 2021-12-02
Last Modified: 2022-01-19

Intended Use:

    Script to scrape NFL statistics from ESPN API organized by statistic, records,
    and matches. End use of the data is to predict win probability per match and predict
    match scores.
    
Modification History:

2022-01-19: Added Delete Files to remove jsons after they are zipped.

--------------
DATA HIERARCHY
--------------

Data source: https://gist.github.com/nntrn/ee26cb2a0716de0947a0a4e9a157bc1c

    Teams: https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2021/teams
        Team: http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2021/teams/30?lang=en&region=us
            Stats: http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2021/types/2/teams/30/statistics?lang=en&region=us
            Record: http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2021/types/2/teams/30/record?lang=en&region=us

    Season: http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2021/types/2?lang=en&region=us
        Weeks: http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2021/types/2/weeks?lang=en&region=us
            Week: http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2021/types/2/weeks/1?lang=en&region=us
                Matches: http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2021/types/2/weeks/1/events?lang=en&region=us
                    Match: http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/401326322?lang=en&region=us
                        Match Score By Competitor(2x): http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/401326322/competitions/401326322/competitors/27/score?lang=en&region=us
                    
'''

###############
### Imports ###
###############

from bs4 import BeautifulSoup as bs
import requests
import re
import json
import datetime
import time
import pandas as pd
import sys
import os
from os import listdir
from os.path import isfile, join, splitext
from zipfile import ZipFile, ZIP_DEFLATED

########################
### Global Variables ###
########################

# Data Path to Script
directory = sys.path[0]

# Dictionaries/lists 
conf_div_dct = {1:'NFC SOUTH',
                2:'AFC EAST',
                3:'NFC NORTH',
                4:'AFC NORTH',
                5:'AFC NORTH',
                6:'NFC EAST',
                7:'AFC WEST',
                8:'NFC NORTH',
                9:'NFC NORTH',
                10:'AFC SOUTH',
                11:'AFC SOUTH',
                12:'AFC WEST',
                13:'AFC WEST',
                14:'NFC WEST',
                15:'AFC EAST',
                16:'NFC NORTH',
                17:'AFC EAST',
                18:'NFC SOUTH',
                19:'NFC EAST',
                20:'AFC EAST',
                21:'NFC EAST',
                22:'NFC WEST',
                23:'AFC NORTH',
                24:'AFC WEST',
                25:'NFC WEST',
                26:'NFC WEST',
                27:'NFC SOUTH',
                28:'NFC EAST',
                29:'NFC SOUTH',
                30:'AFC SOUTH',
                33:'AFC NORTH',
                34:'AFC SOUTH'}

#################
### Functions ###
#################

def input_date_validation(user_input):
    """
    Validate user input date for comparisons and incorrect characters
    asking user to reenter correct date

    Parameters:
    user_input (str): Year date as string

    Returns:
    User input prompt until valid date is returned
    """
    current_year = datetime.datetime.now().year
    
    while True:
        if user_input.isnumeric():
            time_check = (int(user_input) < 2002) or (int(user_input) > current_year)
            if time_check:
                print(f'\nDate entered, {user_input}, is less than earliest START date of 2002',
                      f'or greater than CURRENT date of {current_year}!')
                user_input = input(f'Please reenter date: ')
                time_check = (int(user_input) < 2002) or (int(user_input) > current_year)
            else:
                break
        else:
            print(f'\nDate entered, {user_input}, contains characters other than numbers!')
            user_input = input(f'Please reenter date: ')
        print(f'"{user_input}" is a valid date')
    return user_input
        
def input_date_comparison(user_input1, user_input2):
    """
    Validate user input dates for comparisons between first and last dates
    asking user to reenter dates if first date greater than last date

    Parameters:
    user_input1 (str): First date inputed
    user_input2 (str): First date inputed

    Returns:
    User input prompt until valid dates are returned
    """
    while True:
        if int(user_input1) > int(user_input2):
            print(f'\nSTART date of {user_input1} greater than LAST date of {user_input2}')
            user_input1 = input(f'Please reenter START date: ')
            user_input2 = input(f'Please reenter LAST date: ')
        else:
            break
    print(f'Success! START date of {user_input1} is less than or equal to LAST date of {user_input2}')
    return user_input1, user_input2

def input_folder_validation(user_input):
    """
    Validate user input folder for existance

    Parameters:
    user_input (str): Folder name e.g. data

    Returns:
    User input prompt until valid folder is returned
    """
    while True:
        folder_check = os.path.isdir(user_input)
        if not folder_check:
            print(f'{user_input} folder does not exist, please create folder first')
            user_input = input('Please reenter folder name: ')
        else:
            break
    print(f'{user_input} folder exists!')
    return user_input

def delete_files(data_path, cond):
    """
    Deletes json files created leaving only the zipfile. 
    Used automatically

    Parameters:
    folder name (str): Folder name e.g. data

    Returns:
    No return
    """
    flist = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    fcond = [join(data_path, f) for f in flist if cond == splitext(f)[1]]
    
    if fcond:
        for f in fcond:
            os.remove(f)
            print(f'{f} removed succesfully!')
    
def requests_handling(s, url):
    """
    Try URL again after waiting x time

    Parameters:
    url (str): Url for API request

    Returns:
    Requests object
    """
    r = s.get(url, timeout=10)
    if r.status_code == 429:
        time.sleep(int(r.headers["Retry-After"]))
        r = s.get(url, timeout=10)
        print(f"Retry!! {url} returned {r}")
        return requests_handling(s, url)
    elif r.status_code != 200:
        time.sleep(5)
        r = s.get(url, timeout=10)
        print(f"Retry!! {url} returned {r}")
        return requests_handling(s, url)
    return r

def request_soup(s, url):
    """
    Process URL through requests, beautifulsoup, and json

    Parameters:
    url (str): Url for API request

    Returns:
    Json object
    """
    r = requests_handling(s, url)
    soup = bs(r.content, 'html.parser')
    json_ = json.loads(soup.text)
    
    return json_

def write_json(out_path, fname, data_list):
    """
    Write JSON file

    Parameters:
    out_path (str): Directory/folder file will be written to
    fname (str): Name given to newly written json file
    data_list (list): List of dictionaries retrieved from API

    Returns:
    Json file written to directory
    """
    with open(join(out_path, fname + '.json'), "w") as f:
        json.dump(data_list, f)
    print(f"\nNumber of rows in json: {len(data_list)}"           f"\nNumber of columns in json: {max([len(x) for x in data_list])}"           f"\nFile '{fname}' saved succesfully to {out_path}")
    
def multiList_multiJson(out_path, fname_ext, data_list):
    """
    Write multiple JSON files

    Parameters:
    out_path (str): Directory/folder file will be written to
    fname_ext (str): Last part of file name
    data_list (list): Nested list of dictionaries retrieved from API

    Returns:
    Json files written to directory
    """
    lst = list(zip(*data_list))
    for dct in lst:
        fname = [list(d.keys()) for d in dct][0][0]
        vals = [x for d in dct for x in list(d.values())]
        write_json(out_path, fname+fname_ext, vals)
        
def zip_jsons(out_path, zipname):
    """
    Zip json files for storage improvement

    Parameters:
    out_path (str): Directory/folder file will be written to
    zipname (zipfile object): Zipfile object to write data to

    Returns:
    Zip of json data files
    """    
    with ZipFile(join(out_path, zipname), 'w', ZIP_DEFLATED) as z:
        flist = [f for f in listdir(out_path) if isfile(join(out_path, f))]
        fcond = [f for f in flist if '.zip' not in f]
        for f in fcond:
            z.write(join(out_path, f), f)
    print(f"\nFiles saved succesfully to {zipname}: {len(fcond)}")

def nfl_scraper(first_season="", last_season="", out_path=""):
    """
    Access NFL data from ESPN API

    Parameters:
    first_season (str): First season (Year)
    last_season (str): Last season (Year)
    fname (str): Name given to newly written json file
    out_path (str): Directory/folder file will be written to

    Returns:
    A json file
    """
    # List of from first season to last season selected years
    dates_list = [d.strftime('%Y') for d in pd.date_range(first_season, last_season, freq='YS')]

    out_path = out_path # Path to save files

    # Set of empty lists to hold scraped data
    team_info_list = []
    team_stats_list = []
    stats_desc_list = []
    team_record_list = []
    season_matches_list = []
    # Iterate over years
    for date in dates_list:

        s = requests.session()
        
        # URL from API to capture all NFL teams by year
        url = f'https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{date}/teams'
        
        r = requests_handling(s, url)
        print(f'{url} returned {r}')
        
        # Get text content from HTML and read in json
        soup = bs(r.content, 'html.parser')
        soup_json = json.loads(soup.text)
        
        # Team IDs as given by ESPN API
        team_id = list(range(1,31))+[33,34]
        
        # Lists to hold urls
        stats_urls = []
        record_urls = []
        # Get Teams and info to append to Team Info List
        for i in team_id:
            print('#', end='', flush=True) # Progress bar
            team_url = f'{url}/{i}?lang=en&region=us'
            team_json = request_soup(s, team_url)
            
            team_info_dict = {}
            
            team_info_dict['id'] = i
            team_info_dict['team_name'] = team_json['shortDisplayName']
            team_info_dict['abbreviation'] = team_json['abbreviation']
            team_info_dict['conference'] = conf_div_dct[i][0:3]
            team_info_dict['division'] = conf_div_dct[i]
            team_info_dict['location'] = team_json['location']
            team_info_list.append(team_info_dict)
            
            stats_urls.append(team_json['statistics']['$ref'])
            record_urls.append(team_json['record']['$ref'])
        print(f' {date}: Teams API Scraped\n')
            
        # Get Team Stats and append to Team Stats List
        for i, stats_url in enumerate(stats_urls):
            print('#', end='', flush=True) # Progress bar
            stats_json = request_soup(s, stats_url)
            
            team_cat_list = []
            for cat in stats_json['splits']['categories']:
                team_stats_dct = {}
                stat_cat_dct = {}
                
                for stat in cat['stats']:
                    team_stats_dct['id'] = str(date) +'_'+ str(team_id[i]) 
                    team_stats_dct['season'] = date
                    team_stats_dct['team_id'] = team_id[i]
                    team_stats_dct['stat_'+stat['name']] = stat['value']
                    
                    desc_dct = {}
                    desc_dct['name'] = stat['displayName']
                    desc_dct['abbreviation'] = stat['abbreviation']
                    desc_dct['description'] = stat['description']
                    
                    if 'rank' in stat:
                        team_stats_dct['stat_'+stat['name']+'_rank'] = stat['rank']
                    else:
                        team_stats_dct['stat_'+stat['name']+'_rank'] = 999
                    
                    if 'perGameValue' in stat:
                        team_stats_dct['stat_'+stat['name']+'_pg'] = stat['perGameValue']
                    else:
                        team_stats_dct['stat_'+stat['name']+'_pg'] = 999
                    
                    stats_desc_list.append(desc_dct)
                stat_cat_dct[cat['name']] = team_stats_dct
                team_cat_list.append(stat_cat_dct)
            team_stats_list.append(team_cat_list)  
        print(f' {date}: Statistics API Scraped\n')
            
        # Get Team Records and append to Team Records List
        for i, record_url in enumerate(record_urls):
            print('#', end='', flush=True) # Progress bar
            record_json = request_soup(s, record_url)
            
            team_cat_list = []
            for cat in record_json['items']:
                team_record_dct = {}
                record_cat_dct = {}
                
                for rec in cat['stats']:
                    team_record_dct['id'] = str(date) +'_'+ str(team_id[i]) 
                    team_record_dct['season'] = date
                    team_record_dct['team_id'] = team_id[i]
                    team_record_dct['record_'+cat['type']+'_'+rec['shortDisplayName']] = rec['value']
                    
                record_cat_dct[cat['type']] = team_record_dct
                team_cat_list.append(record_cat_dct)
            team_record_list.append(team_cat_list)
        print(f' {date}: Records API Scraped\n')
        
        # Get Season Matches and append to Season Matches List
        season_url = f'http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{date}/types/2/weeks?lang=en&region=us'
        week_count = request_soup(s, season_url)['count']
        for week in range(1, week_count+1):
            print('#', end='', flush=True) # Progress bar
            
            week_url = f'http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{date}/types/2/weeks/{week}/events?lang=en&region=us'
            week_json = request_soup(s, week_url)
            
            week_list = [x['$ref'] for x in week_json['items']]
            season_game_list = []
            for match_url in week_list:
                match_json = request_soup(s, match_url)
                season_game_dct = {}
                season_game_dct['id'] = match_json['id']
                season_game_dct['season'] = date
                season_game_dct['week'] = week
                season_game_dct['date_time'] = match_json['date']
                
                for competitor in match_json['competitions'][0]['competitors']:
                    score_json = request_soup(s, competitor['score']['$ref'])
                    
                    if competitor['homeAway'] == 'home':
                        home_away = 'home'
                    else:
                        home_away = 'away'
                                         
                    season_game_dct[home_away+'_team'] = int(competitor['id'])
                    
                    if 'winner' in competitor:
                        if competitor['winner'] == True:
                            season_game_dct[home_away+'_team_win'] = 1
                        else:
                            season_game_dct[home_away+'_team_win'] = 0
                        season_game_dct[home_away+'_team_score'] = score_json['value']
                    else:
                        season_game_dct[home_away+'_team_win'] = 999
                        season_game_dct[home_away+'_team_score'] = 999
                season_game_list.append(season_game_dct)
            season_matches_list.append(season_game_list)
        print(f' {date}: Matches API Scraped\n')
    
    # Write Teams, Matches, & Stats Description json
    unique_team_info_list = list({v['id']:v for v in team_info_list}.values()) # Keep only unique dicts
    write_json(out_path, 'teams', unique_team_info_list) # Teams
    flat_season_matches_list = [g for game in season_matches_list for g in game] # Flatten list
    write_json(out_path, 'matches'+'_'+first_season+'_'+last_season, flat_season_matches_list) # Matches
    unique_stats_desc_list = list({v['description']:v for v in stats_desc_list}.values()) # Keep only unique dicts
    write_json(out_path, 'stat_desc', unique_stats_desc_list) # Stats Description
    
    # Separate teams stats and records by type and write to individual jsons
    multiList_multiJson(out_path, '_stats'+'_'+first_season+'_'+last_season, team_stats_list) # Statistics
    multiList_multiJson(out_path, '_record'+'_'+first_season+'_'+last_season, team_record_list) # Record
    
    # Zip json file for space saving
    zipf = join(out_path,'nfl_data'+'_'+first_season+'_'+last_season+'.zip')
    zip_jsons(out_path, zipf)
    
    # Delete Jsons after zipping them
    delete_files(out_path, '.json')
    
#####################
### PROGRAM START ###
#####################

if __name__ == "__main__":
    # Program timing
    start_time = time.time()    

    # User inputs to generate data files in specified folder
    f_season = input('Enter START date to scrape (STARTING with 2002): ')
    l_season = input('Enter LAST date to scrape (2002 ONWARDS): ')
    opath = input('Enter FOLDER NAME to store data e.g. data: ')

    # Validate dates
    f_season, l_season = input_date_comparison(input_date_validation(f_season), 
                                          input_date_validation(l_season))

    # Validate folder
    opath = input_folder_validation(directory + '\\' + opath)

    # Run Scrapper
    nfl_scraper(first_season=f_season, last_season=l_season, out_path=opath)

    # Total program time
    end_time = time.time()
    program_time = time.strftime('%Hh%Mm%Ss', time.gmtime(end_time-start_time))
    print(f'\nProgram Completed Succesfully in a time of: {program_time}')
    input('Press Enter to Exit the Program...')

