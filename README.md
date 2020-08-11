# Analysis of Major League Baseball (MLB) Statistics with Supervised Learning Models
_Work by [Song Ying](https://github.com/songyingho) and Antoine Ghilissen_

__Files__
* [index](./index.ipynb): Jupyter notebook detailing the analysis.
* [source/Dataset](./source/MLB-GameLogs-1871_2016.csv): Original dataset in a csv format.
* [source/data_cleaning](./source/data_cleaning.ipynb): Jupyter notebook detailing the data cleaning process.
* [df.csv](./df.csv): Cleaned dataset used for the analysis.
* [presentation](./presentation.pdf): Presentation slide deck.

__Executive Summary__: Analysing Major League Baseball (MLB) World Series statistics, ranging from 1946 to 2016, and formulate coaching strategies to maximise chances of winning matches.

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

__Data Source__: The data is the game log of MLB matches performed between 1871 and 2016. It was compiled by Retrosheet. The original dataset can be found on [Dataquest](https://data.world/dataquest/mlb-game-logs).

__Methodology__: This project uses Python 3, documented with Jupyter Notebook. We used a combination of Numpy and Pandas for data cleaning, filtering and feature engineering. As the initial stage of feature selection, the cleaned data was checked for multicollinearity using correlation matrix, variance inflation factor (VIF) and our business expertise. A nested 5-fold cross validation within a train-test split of 70:30 was employed for all our models. Logistic Regression was used as our baseline model. Subsequent advanced models like Decision Tree, Random Forest and XGBoost were developed to provide feature selection as well as to improve model performance. GridSearchCV was used for hyperparameter tuning for Decision Trees, Random Forest and XGBoost. Models were compared using the ROC-AUC score. 

__High Level Overview__: The project started by exploring the data and transforming the dataset into a desirable format. As the original dataset was given in match-by-match basis, we divide each match (row) into 2, namely the winning team and losing team with their respective statistics. As a result, we doubled our number of rows but halved the number of columns, additionally we defined our target variable, the match outcome in a boolean format. As the original dataset was given match-by-match, we had no class imbalance issue as we had a perfect 50:50 split of winners and losers. After data cleaning, we did a preliminary round of feature selection based on correlation matrix to remove trivial variables and removed several other variables using variance inflation factor to ensure there were low multicollinearity between the metrics. 

Next, we created a logistic regression model as our baseline model, which obtained an ROC-AUC score of __61.5 %__. To improve on our baseline performance, we developed additional models, namely Decision Trees, Random Forest and XGBoost, which performed at __72.2 %__, __74.2 %__ , __66.69 %__ respectively after GridSearchCV optimization. As a result, __Random Forest__ was chosen as our final model. A threshold of __input__ was chosen for our final model to optimize for both Type 1 and Type 2 errors. Finally, we evaluated the performance of the model and derived actionable insights for our stakeholders, as described in detail below.


__Limitations__:
1. Data samples from 1970-1979 and 1990-1999 were missing and therefore not included in our analysis.
2. Missing players individual statistics to evaluate influence on match outcome by individual factors.
3. Due to our business case, we removed a few games from the original database: multiheaded games, games that ended in a draw, protested and interrupted games.

__Future Work__:
1. In-depth analysis of linescore paired with match statistics per innings of the game from other sources to investigate probability of winning the match as the match progresses.
2. Evaluation of model on other baseball leagues to ensure consistency and scalability of our model
3. Application of dimensionality reduction techniques like Support Vector Machines to the model
4. Application of unsupervised learning model using Principle Component Analysis & Clustering
5. Applying our method to winning statistics and also losing statistics in order to compare and emphasise which feature really impacts the game outcome.

__Actionable Insights__:
1. Invest in top pitchers for defensive plays to reduce opponent’s rbi
2. Employ the most consistent batter to improve rbi
3. Do not focus on hitting ambitious strikes (doubles, triples, homeruns), it’s more effective to just get the bat on the ball as frequent as possible.
4. Approaching the end of the inning, batters should focus on taking risk, to reduce number of players left on base so they could complete a run.

