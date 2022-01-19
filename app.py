#!/usr/bin/env python
# coding: utf-8

#------------------ Script to Launch NFL Win Prob Dash App ------------------#

'''

-----------
INFORMATION
-----------

Author: Jean-Sebastien Roussy
Date Created: 2021-12-02
Last Modified: 2022-01-19

Description:

    This App loads the data gathered by the scraper which needs to be in a 
    folder called data at the App level. Then the data is cleaned, features
    engineered, and split to then be sent into the model. The model is loaded 
    and runs the predictions which are then added back to a dataframe that is 
    fed to the Dash App. Then the Dash App is launched and the data can be 
    viewed in the web browser.
    
Modification History:

2022-01-19: Added sys.path[0] to be able to find data folder when
            calling the app through shell
               
'''


###############
### Imports ###
###############

import sys
from os import listdir
from os.path import isfile, join
from zipfile import ZipFile
import pandas as pd
import numpy as np
import math
from math import pi
from joblib import dump, load

import dash
from dash import dash_table as dt
from dash import Dash, Input, Output, callback, dcc, html
import dash_bootstrap_components as dbc

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


#################
### Functions ###
#################

###############################
### File Manipulation Funcs ###
###############################

# File listing function
def zlist_files(data_path, zipname, cond):
    zipped_files = join(data_path, zipname)
    with ZipFile(zipped_files) as z:
        flist = [f for f in z.namelist() if cond in f]
        fout = [z.open(f) for f in flist]
        return fout

# Create pandas data loader for multi-json loads
def pandas_loader(fout, idx_name):
    dfs = [pd.read_json(f) for f in fout]
    df = pd.concat((idf.set_index(idx_name) for idf in dfs), 
                   axis=1, join='inner').reset_index()
    return df

#################################
### Feature Engineering Funcs ###
#################################

# For time based columns
def transformation(column):
    max_value = column.max()
    sin_values = [math.sin((2*pi*x)/max_value) for x in list(column)]
    cos_values = [math.cos((2*pi*x)/max_value) for x in list(column)]
    return sin_values, cos_values

#######################
### Dash  App Funcs ###
#######################

# Team filter func
def team_filter_vals(df, filter_team, filter_week):
    # Add avg row
    df.loc['average'] = df.mean().apply(lambda x: int(round(x)))
    # Helper cond teams
    cond = ((filter_team == df['home_team'])|(filter_team == df['away_team']))
    # Helper cond scores
    cond2 = ((df['home_score'] > 0)&(df['away_score'] > 0))
    # Games correctly predicted count
    correct_pred = df['correct_prediction'][cond&(df['correct_prediction'] == 1)].count()
    # Games played
    games_played = df['correct_prediction'][cond].count()                   - df['correct_prediction'][cond&(df['correct_prediction'] == -1)].count()
    # Add games played
    df['week'].loc['average'] = 'Games Played: ' + str(games_played)
    # Add avg win probability
    df['win_probability'].loc['average'] = 'Win Prob(avg): ' + str(int(round(df['win_probability'][cond].mean()))) + '%'
    # Add avg pred home score
    df['predicted_home_score'].loc['average'] = filter_team + '(avg): ' + str(int(round(df['predicted_home_score'][cond].mean())))
    # Add avg pred away score
    df['predicted_away_score'].loc['average'] = filter_team + '(avg): ' + str(int(round(df['predicted_away_score'][cond].mean())))
    ## Add correct preds vs games played
    if games_played > 0 :
        if filter_team in df['home_team'].unique().tolist():
            # Add avg home score
            df['home_score'].loc['average'] = filter_team + '(avg): ' + str(int(round(df['home_score']
                                                                              [cond2&(filter_team == df['home_team'])].mean())))
        else:
            # Add avg home score
            df['home_score'].loc['average'] = filter_team + '(avg): 0'
        if filter_team in df['away_team'].unique().tolist():                   
            # Add avg away score
            df['away_score'].loc['average'] = filter_team + '(avg): ' + str(int(round(df['away_score']
                                                                              [cond2&(filter_team == df['away_team'])].mean())))
        else:
            # Add avg away score
            df['away_score'].loc['average'] = filter_team + '(avg): 0' 
        # Games correctly predicted over games played plus percentage
        df['correct_prediction'].loc['average'] = str(correct_pred) + '/' + str(games_played) +' (' +                                                   str(int(round((correct_pred/games_played)*100))) + '%)'
    else:
        # Add avg home score
        df['home_score'].loc['average'] = filter_team + '(avg): 0'
        # Add avg away score
        df['away_score'].loc['average'] = filter_team + '(avg): 0'
        # Games not played
        df['correct_prediction'].loc['average'] = '0/0 (0%)'
        
    return df


#################
### LOAD DATA ###
#################

data_path = sys.path[0] + '\\data'

# Load Statistics
stats_df = pandas_loader(zlist_files(data_path, 'nfl_data_2002_2021.zip', 'stats'), 'id')
#stats_df.head(10)

# Load Record
record_df = pandas_loader(zlist_files(data_path, 'nfl_data_2002_2021.zip', 'record'), 'id')
#record_df.head(10)

# Load matches
matches_df = pd.read_json(zlist_files(data_path, 'nfl_data_2002_2021.zip', 'matches')[0])
#matches_df.head(10)

# Load teams
teams_df = pd.read_json(zlist_files(data_path, 'nfl_data_2002_2021.zip', 'teams')[0])
#teams_df.head(10)


#########################
### JOIN DATA & CLEAN ###
#########################

# Remove Duplicate columns from Stats and Records df
stats_df = stats_df.loc[:,~stats_df.columns.duplicated()]
record_df = record_df.loc[:,~record_df.columns.duplicated()]

# Merge Stats and Record dfs
merged_df = stats_df.merge(record_df, how='left', on=['id'])
#merged_df.head(10)

# Merge Merged df with Teams df
remerge_df = merged_df.merge(teams_df, how='left', left_on='team_id_x', 
                             right_on='id')
#remerge_df.head(35)

# Merge Remerged df to Matches df
nfl_df = matches_df.merge(remerge_df, left_on=['season', 'home_team'], 
                          right_on=['season_x','team_id_x'])\
                   .merge(remerge_df, left_on=['season', 'away_team'], 
                          right_on=['season_x','team_id_x'],
                          suffixes=('_home', '_away'))

# Convert date_time column to datetime object
nfl_df['date_time'] = pd.to_datetime(nfl_df['date_time'], utc=True)
# Sort NFL df by date_time
nfl_df = nfl_df.sort_values(['date_time'])

# Drop redundant columns like id_x/id_y, etc
drop_cols = [col for col in nfl_df.columns if '_x_' in col or '_y_' in col]
nfl_df = nfl_df.drop(drop_cols, axis=1)

# Drop Rank and PG columns
drop_rankpg_cols = [col for col in nfl_df.columns                    if 'rank' in col or 'pg' in col]
nfl_df = nfl_df.drop(drop_rankpg_cols, axis=1)

# Drop stats columns with 999 values in it
drop_999_cols = [col for col in nfl_df.columns[nfl_df.isin([999]).any()]                   if 'stat' in col]
nfl_df = nfl_df.drop(drop_999_cols, axis=1)

# Drop stats and record columns with only 0 values
drop_0_cols = [col for col, is_zero in ((nfl_df == 0).sum() == nfl_df.shape[0])               .items() if is_zero and ('stat' in col or 'record' in col)]
nfl_df = nfl_df.drop(drop_0_cols, axis=1)

# Replace 999 values in matches info
replace_999_cols = ['home_team_win', 'home_team_score', 
                    'away_team_win', 'away_team_score']
nfl_df[replace_999_cols] = nfl_df[replace_999_cols].replace(999,0)

# Drop Abbreviation team name col
drop_abb_col = [col for col in nfl_df.columns if 'abbreviation' in col]
nfl_df = nfl_df.drop(drop_abb_col, axis=1)

# Replace remaining NaNs with 0
nfl_df = nfl_df.fillna(0)


###########################
### FEATURE ENGINEERING ###
###########################

# Date_time to Month, Day of Week, Time of Day as ints
nfl_df['month'] = nfl_df['date_time'].dt.month.astype('int64')
nfl_df['day_week'] = nfl_df['date_time'].dt.dayofweek.astype('int64')
nfl_df['time_day'] = nfl_df['date_time'].dt.hour.astype('int64')

# Calculate Cosine and Sine for new dates
for col in ['month', 'day_week', 'time_day']:
    time_sine, time_cos = transformation(nfl_df[col])
    nfl_df[col[0:3]+'_sine'] = time_sine
    nfl_df[col[0:3]+'_cos'] = time_cos


###################
### X & y SPLIT ###
###################

# X cols list
keep_cols = ['stat', 'record', 'sine', 'cos',
             'division', 'conference', 'team_name', 'location']

remove_cols = ['team_win', 'team_score', 'id', 'date_time',
               'month', 'day_week', 'time_day']

X_cols =  [col for col in list(nfl_df.columns)                    if any(kc in col for kc in keep_cols)           and any(rc not in col for rc in remove_cols)]

## Split X and y into Train and Test
# X,y Train
X_train = nfl_df[~nfl_df['season'].isin([2021])][X_cols]
y_train = nfl_df[~nfl_df['season'].isin([2021])]['home_team_win']
y_regtrain = nfl_df[~nfl_df['season'].isin([2021])][['home_team_score',
                                                     'away_team_score']]
# X
X_test = nfl_df[nfl_df['season'].isin([2021])][X_cols]


##################
### LOAD MODEL ###
##################

# Model Path
model_path = sys.path[0] + '\\model'
# Load Logistic Regresssion Model
logreg_model = load(join(model_path, 'winprob_logreg_2002_2020.joblib'))
# Load SVR Model
svr_model = load(join(model_path, 'scores_svr_2002_2020.joblib'))


########################
### 2021 PREDICTIONS ###
########################

## 2021 DATA 

# Fit Model Logistic Regression
logreg_model.fit(X_train, y_train)

# Predict labels
pred_label = logreg_model.predict(X_test)

# Probabilities
prob = logreg_model.predict_proba(X_test)

# Fit Model SVR
svr_model.fit(X_train, y_regtrain)

# Predict Scores
pred_scores = svr_model.predict(X_test)


###################
### 2021 TABLES ###
###################

# Weekly results Table
nfl_df2021 = nfl_df[nfl_df['season'] == 2021][['team_name_home',
                                               'home_team_score',
                                               'away_team_score',
                                               'team_name_away',
                                               'week',
                                               'home_team_win']]

# Change column names
nfl_df2021 = nfl_df2021.rename(columns={'team_name_home':'home_team',
                                        'team_name_away':'away_team',
                                        'home_team_score':'home_score',
                                        'away_team_score':'away_score'})

# Add probability column to df
nfl_df2021['home_team_win_probability'] = [int(round(x*100)) for x in prob[:,1]]
# Add predicted labels
nfl_df2021['predicted_winner'] = pred_label
# Add predicted Home Score
nfl_df2021['predicted_home_score'] = [int(round(x)) for x in pred_scores[:,0]]
# Add predicted Home Score
nfl_df2021['predicted_away_score'] = [int(round(x)) for x in pred_scores[:,1]]

# Map actual winner name to column
nfl_df2021['actual_winner'] = nfl_df2021.apply(lambda x: x.home_team if x.home_team_win == 1 else x.away_team, axis=1)
# Map predicted winner name to column
nfl_df2021['predicted_winner'] = nfl_df2021.apply(lambda x: x.home_team if x.predicted_winner == 1 else x.away_team, axis=1)
# Map win probability
nfl_df2021['win_probability'] = nfl_df2021.apply(lambda x: x.home_team_win_probability if x.predicted_winner == x.home_team else 100 - x.home_team_win_probability, axis=1)
# Map correct predicition
nfl_df2021['correct_prediction'] = (nfl_df2021['predicted_winner'] == nfl_df2021['actual_winner']).astype(int)
# Replace Actual Winner value with None if game not played yet
nfl_df2021['actual_winner'] = np.where((nfl_df2021['home_score'] == 0) & (nfl_df2021['away_score'] == 0), "None", nfl_df2021['actual_winner'])
# Replace Correct Prediction value with -1 if game not played yet
nfl_df2021['correct_prediction'] = np.where((nfl_df2021['home_score'] == 0) & (nfl_df2021['away_score'] == 0), -1, nfl_df2021['correct_prediction'])

# Drop columns
nfl_df2021 = nfl_df2021.drop(columns=['home_team_win_probability', 'home_team_win'])

# Move Cols
move_cols = nfl_df2021.columns.to_list()
new_order = move_cols[4:5] + move_cols[0:2] + move_cols[6:8] + move_cols[2:4] + move_cols[5:6] + move_cols[8:]
nfl_df2021 = nfl_df2021[new_order]


################
### DASH APP ###
################

# Intialize Dash App
app = dash.Dash(__name__)

app.layout = html.Div([

    html.Div(
        [
        # Drop down for Week Selection
        dcc.Dropdown(
            id='select_week',
            options=[{'label': i, 'value': i} for i in nfl_df2021['week'].unique()],
            value=None,
            searchable=False,
            clearable=False,
            multi=True,
            placeholder='Select Week'
        ),
        # Drop down for Team Selection
        dcc.Dropdown(
            id='select_team',
            options=[{'label': i, 'value': i} for i in nfl_df2021['home_team'].sort_values().unique()],
            value=None,
            searchable=True,
            clearable=True,
            placeholder='Select Team'
        )
    ]
    ),
    html.Div(
        [
        # NFL DF 2021 data
        dt.DataTable(
            id='nfl_predictions',
            columns=[{"name": i.replace('_', ' ').title(), "id": i} for i in nfl_df2021.columns],
            data=nfl_df2021.to_dict('records')
        )
    ]
    )
])

#---------------------------------------------------------------
# Call back to update datatable with filters
@app.callback(
    Output('nfl_predictions', 'data'),
    [Input('select_week', 'value'),
     Input('select_team', 'value')]
)

def update_table(filter_week, filter_team):
    if filter_team and filter_week:
        df = nfl_df2021.loc[((filter_team == nfl_df2021['home_team']) |                             (filter_team == nfl_df2021['away_team'])) &                             (nfl_df2021['week'].isin(filter_week))]
        # Call team filter func
        df = team_filter_vals(df, filter_team, filter_week)
    
    elif filter_week:
        df = nfl_df2021.loc[nfl_df2021['week'].isin(filter_week)]
        # Helper cond scores
        cond2 = ((df['home_score'] == 0)&(df['away_score'] == 0))
        # Add avg row
        df.loc['average'] = df.mean().apply(lambda x: int(round(x)))
        # Games correctly predicted count
        correct_pred = df['correct_prediction'][df['correct_prediction'] == 1].count()
        # Games played
        games_played = df['correct_prediction'].count()                       - df['correct_prediction'][df['correct_prediction'] == -1].count()
        # Add games played
        df['week'].loc['average'] = 'Games Played: ' + str(games_played)
        # Add avg win probability
        df['win_probability'].loc['average'] = 'Win Prob (avg): ' + str(int(round(df['win_probability'].mean()))) + '%'
        ## Add correct preds vs games played
        if games_played > 0 :
            # Games correctly predicted over games played plus percentage
            df['correct_prediction'].loc['average'] = str(correct_pred) + '/' + str(games_played) +' (' +                                                       str(int(round((correct_pred/games_played)*100))) + '%)'
        else:
            # Games not played
            df['correct_prediction'].loc['average'] = '0/0 (0%)'
            
    elif filter_team:
        df = nfl_df2021.loc[(nfl_df2021['home_team'] == filter_team) |                              (nfl_df2021['away_team'] == filter_team)]
        # Call team filter func
        df = team_filter_vals(df, filter_team, filter_week)
          
    else:
        df = nfl_df2021
    return df.to_dict('records')
#---------------------------------------------------------------

if __name__ == '__main__':
    app.run_server()