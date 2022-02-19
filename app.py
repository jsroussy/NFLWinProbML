#!/usr/bin/env python
# coding: utf-8


#------------------ Script to Launch NFL Win Prob Dash App ------------------#

'''

-----------
INFORMATION
-----------

Author: Jean-Sebastien Roussy
Date Created: 2021-12-02
Last Modified: 2022-02-19

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
2022-02-18: Added models to S3 bucket and called them via S3 URI 
            stored in .env file
2022-02-19: Split code base into two files to save memory when running App
            in Heroku. Final table is loaded in and retreived from
            Heroku Postgres.
                       
               
'''


###############
### Imports ###
###############

import pandas as pd

import tempfile
from dotenv import load_dotenv
from os import getenv
from sqlalchemy import create_engine

import dash
from dash import dash_table as dt
from dash import Dash, Input, Output, callback, dcc, html
import dash_bootstrap_components as dbc 


#################
### Functions ###
#################

######################
### Read SQL to DF ###
######################

# Source: https://towardsdatascience.com/optimizing-pandas-read-sql-for-postgres-f31cd7f707ab
def read_sql_tmpfile(query, dbengine, table=None):
    with tempfile.TemporaryFile() as tmpfile:
        if table == 'matview':
            copy_sql = 'COPY (SELECT * FROM "{query}") TO STDOUT WITH CSV {head}'.format(
                        query=query, head="HEADER")
        else:
            copy_sql = 'COPY "{query}" TO STDOUT WITH CSV {head}'.format(
                        query=query, head="HEADER")
        conn = dbengine.raw_connection()
        cur = conn.cursor()
        cur.copy_expert(copy_sql, tmpfile)
        tmpfile.seek(0)
        df = pd.read_csv(tmpfile)
        return df
    
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


#######################
### POSTGRESQL INFO ###
#######################

# Load in environment file
load_dotenv()

# SQLAlchemy engine from env file
db_uri = getenv('SQLALCHEMY_NFL_URI')
if db_uri.startswith("postgres://"):
    db_uri = db_uri.replace("postgres://", "postgresql+psycopg2://", 1)

engine = create_engine(db_uri, 
                       convert_unicode=True, 
                       encoding='utf-8')



#################
### LOAD DATA ###
#################

# Load NFL 2021 Season
nfl_df2021 = read_sql_tmpfile('nfl_2021_season', engine)


################
### DASH APP ###
################

# Intialize Dash App
app = dash.Dash(__name__)
server = app.server

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
