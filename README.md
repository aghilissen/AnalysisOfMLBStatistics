# Analysis of Major League Baseball (MLB) Statistics with Supervised Learning Models
_Work by [Song Ying](https://github.com/songyingho) and Antoine Ghilissen_

## Files
* [index](./index.ipynb): Jupyter notebook detailing the analysis.
* [source/Dataset](./source/MLB-GameLogs-1871_2016.csv): Original dataset in a csv format.
* [source/data_cleaning](./source/data_cleaning.ipynb): Jupyter notebook detailing the data cleaning process.
* [df.csv](./df.csv): Cleaned dataset used for the analysis.
* [presentation](./presentation.pdf): Presentation slide deck.

## Executive Summary
Analysing Major League Baseball (MLB) World Series statistics, ranging from 1946 to 2016, and formulate coaching strategies to maximise chances of winning matches.

This analysis encompasses the following:
1. Data Exploration
2. Data Cleaning & Feature Engineering
2. Logistic Regression `ROC-AUC score: 61.5 %`
3. Decision Tree `ROC-AUC score: 72.2 %`
4. Random Forest `ROC-AUC score: 74.2 %`
5. XGBoost `ROC-AUC score: 66.69 %`
6. Variable & Model Selection
7. Threshold Selection
8. Final Evaluation
9. Actionable Insights

## Actionable Insights
1. Invest in top pitchers for defensive plays to reduce opponent’s RBI.
2. Employ the most consistent batter to improve RBI.
3. Do not focus on hitting ambitious strikes (doubles, triples, home runs), consistency is preferred and getting the bat on the ball as frequently as possible is more efficient.
4. Approaching the end of the innings, batters should focus on taking risk, to reduce number of players left on base so they could complete a run.

## High Level Overview
The project started by exploring the data and transforming the dataset into a desirable format. As the original dataset was given in match-by-match basis, we divide each match (row) into 2, namely the winning team and losing team with their respective statistics.

As a result, we doubled our number of rows but halved the number of columns, additionally we defined our target variable, the match outcome in a boolean format.

As the original dataset was given match-by-match, we had no class imbalance issue as we had a perfect 50:50 split of winners and losers.

After the data cleaning process, we did a preliminary round of feature selection based on correlation matrix to remove trivial variables and also removed several other variables using VIF (variance inflation factor) to ensure there would low multicollinearity between the multiple variables.

Next, we trained a logistic regression model as our baseline model. Followed by a few additional models, their ROC_AUC scores are summarized in the following table:
| Type | ROC_AUC |
| --- | --- |
| Logistic Regression | 61.5% |
| Decision Trees | 72.2% |
| **Random Forest** | **74.2%** |
| XGBoost | 66.69% |

_All these models have gone through hyperparameter tuning, using scikit-learn's `GridSearchCV()` optimisation method.

The **Random Forest** was chosen as our final model.

The final technical step in our process was to select a threshold to optimize for both Type 1 and Type 2 errors.

Finally, we evaluated the performance of the model and derived [actionable insights](#actionable%20insights) for our stakeholders.

## Methodology
This project uses Python3 and is documented with Jupyter Notebook.

We have used a combination of `numpy` and `pandas` for data cleaning, filtering and feature engineering and `seaborn` was used for data visualisation.

The initial stage of feature selection included a multicollinearity check on the cleaned data. This was performed using a correlation matrix, variance inflation factor (VIF) and the decisions were made based on our business expertise.

A nested 5-fold cross validation within a train-test split with a ratio of 70:30 was employed for all our models.

A logistic regression was used as our baseline model and various other models were trained, such as a Decision Tree, a Random Forest and an XGBoost.

These models were used to adjust the feature selection as well as to improve performance.

scikit-learn's `GridSearchCV()` was used for the hyperparameter tuning of the relevant models: the Decision Trees, the Random Forest and the XGBoost model.

Models were compared using the ROC-AUC score.

## Data Source
The data is the game log of MLB matches performed between 1871 and 2016. It was compiled by Retrosheet. The original dataset can be found on [Dataquest](https://data.world/dataquest/mlb-game-logs).

## Limitations
1. Data samples from 1970-1979 and 1990-1999 were missing and therefore not included in our analysis.
2. Missing players individual statistics were not provided so we couldn't evaluate the influence of individual factors on match outcome.
3. Due to our business case, we removed a few games from the original database: multiheaded games, games that ended in a draw, protested and interrupted games.

## Future Work
1. In-depth analysis of linescore paired with match statistics per innings of the game from other sources to investigate probability of winning the match as the match progresses.
2. Evaluation of model on other baseball leagues to ensure consistency and scalability of our model
3. Application of dimensionality reduction techniques like Support Vector Machines to the model
4. Application of unsupervised learning model using Principle Component Analysis & Clustering
5. Applying our method to winning statistics and also losing statistics in order to compare and emphasise which feature really impacts the game outcome.