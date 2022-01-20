# NFL Game Win Probability Predictor using ML
> This project was in conjunction with my Concordia Data Science Bootcamp 2021. The goal was to predict the probability of one team winning versus another.
> I built a web scraper to access ESPN nfl API. Tested a few models for performance and saved the best ones.
> To visualize the results I built a Dash App.

## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Setup](#setup)
* [Usage](#usage)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)
* [Data Sources](#data-sources)
* [Resources](#resources)
<!-- * [GPLv3](https://choosealicense.com/licenses/gpl-3.0/#) -->


## General Information
- The goal of the project is to predict NFL game win probability
- There are mutliple steps to get to the outcome
- First, data was gathered with the espnAPI_nflScrape web scraper
- Second, data was processed with pandas
- Third, the data was funneled into the models
- Finally, a dash App was created to display results 


## Technologies Used
- Python - version 3.8.8


## Setup
I have included a requirements.txt file so users can install the correct python module versions.


## Usage
The Scripts can be used as is assuming python is installed and dependent modules installed.
The data is required to run the App, so:
- Run espnAPI_nflScrape first
   - Make sure to name 'folder' data
   - You may skip this step as data is already included in data folder
- Then run App.py. The Script can be run in the command line like so:
`python path\\to\\script.py`


## Project Status
Project is: _in progress_ 


## Room for Improvement
Include areas you believe need improvement / could be improved. Also add TODOs for future development.

Room for improvement:
- Feature Selection and adding more data sources
- Testing different models and hyperparameters

To do:
- PostgreSQL data connection
- More Web Scrapers for additional data
- To be determined.

## Data Sources
- ESPN NFL Data: https://gist.github.com/nntrn/ee26cb2a0716de0947a0a4e9a157bc1c

## Resources
- Active State: https://www.activestate.com/blog/how-to-predict-nfl-winners-with-python/
- Open Source Football: https://www.opensourcefootball.com/posts/2021-01-21-nfl-game-prediction-using-logistic-regression/

<!-- ## License -->
<!-- This project is open source and available under the [GPLv3](https://choosealicense.com/licenses/gpl-3.0/#). -->
