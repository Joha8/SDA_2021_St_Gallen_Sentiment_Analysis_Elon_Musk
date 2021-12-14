# SDA_2021_St_Gallen_Sentiment_Analysis_Elon_Musk
# Importing of libraries

import pandas as pd
import seaborn as sns
import datetime as datetime
import re
import numpy as np
import statsmodels.api as sm
import pandas_ta as ta
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import r2_score
from pandas.tseries.offsets import BDay
from matplotlib import pyplot as plt
from statsmodels.graphics.gofplots import ProbPlot

# Defining functions

def clean_tweet(tweet):
    """ Utility function to clean tweet text by removing links, special characters using simple regex statements. """
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+://\S+)", " ", tweet).split())

# Return the polarity of the tweet using TextBlob analysis
def analyse_tweet(tweet):
    """ Utility function to classify sentiment of passed tweet using textblob's sentiment method """
    clean_tweet(tweet)
    # create TextBlob object of passed tweet text
    tweet_analysis = TextBlob(tweet)
    return tweet_analysis.sentiment.polarity

# Analyse tweet using TextBlob and categorize it as 'positive', 'negative' or 'neutral'
def get_tweet_sentiment(tweet):
    tweet_polarity = analyse_tweet(tweet)
    # set sentiment
    if tweet_polarity > 0:
        return 'positive'
    elif tweet_polarity == 0:
        return 'neutral'
    else:
        return 'negative'

def filter_by_daterange(df):
    start_date = pd.to_datetime("2011-1-3").date()
    end_date = pd.to_datetime("2021-03-01").date()
    mask = (df['Time'] > start_date) & (df['Time'] <= end_date)
    df = df.loc[mask]
    #Filter only Business day
    isBusinessDay = BDay().onOffset
    match_series = pd.to_datetime(df['Time']).map(isBusinessDay)
    df[match_series]
    return df

def clean_tweet_data(tweets):
    # Convert 'Time' column to datetime and strip time information.
    tweets['Time'] = pd.to_datetime(tweets['Time']).dt.date
    # Consider only dates between a range
    tweets = filter_by_daterange(tweets)
    # Add sentiment of the tweet to the data.
    tweets['Sentiment'] = tweets.apply(lambda row : get_tweet_sentiment(row['Tweet']), axis=1)
    tweets_sentiment = tweets[['Time', 'Sentiment']].copy() 
    # Will consider maximum tweet sentiment as the sentiment of the day.
    tweets_sentiment = tweets_sentiment.groupby(tweets_sentiment.Time)\
                        .agg(lambda x: x.value_counts().index[0])
    tweets_sentiment.sort_values(by=['Time'], inplace=True)
    return tweets_sentiment

def clean_stock_data(stock_data):
    #Remove null stock data.
    stock_data = stock_data.dropna()
    #Convert 'Date' column to datetime and strip time information.
    stock_data['Time'] = pd.to_datetime(stock_data['Date']).dt.date
    stock_data = stock_data.drop('Date',1)
    #Consider only dates between a range
    stock_data = filter_by_daterange(stock_data)    
    #Calculate daily change percentage
    stock_data['daily_percentage_change'] = (stock_data['Close'] - stock_data['Open']) / stock_data['Open'] * 100
    stock_daily_change = stock_data[['Time', 'daily_percentage_change']].copy()
    stock_daily_change.sort_values(by=['Time'], inplace=True)
    return stock_daily_change

def merge_tweets_and_stock_data(tweets_sentiment_data, stock_price_change_data):
    #Combine two dataframes based on time.
    sentiment_stock_change_data = pd.merge(tweets_sentiment_data, stock_price_change_data, on='Time', how='inner')
    return sentiment_stock_change_data

def make_sentiment_column_categorical(tweet_sentiment_with_price_change):
    #Change 'Sentiment' column to categorical column.
    tweet_sentiment_with_price_change['Sentiment'] = tweet_sentiment_with_price_change['Sentiment'].astype('category')
    tweet_sentiment_with_price_change['Sentiment'] = tweet_sentiment_with_price_change['Sentiment'].cat.codes
    return tweet_sentiment_with_price_change

def linear_regression_data():
    X=df[["negative", "positive", "%_delta_Nasdaq", "EMA_10"]]
    Y=df[["%_delta_Tesla"]]
    return X,Y

def logistic_regression_data():
    X=df[["negative", "positive", "%_delta_Nasdaq", "EMA_10"]]
    Y=df[["Binary_delta_Tesla"]]
    return X,Y

def check_linearity_assumption(ax, fitted_y, residuals):
    sns.residplot(
        x=fitted_y,
        y=residuals,
        lowess=True,
        scatter_kws={"alpha": 0.5},
        line_kws={"color": "red", "lw": 1, "alpha": 0.8},
        ax=ax,
    )
    ax.set_title("Residuals vs Fitted")
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")


def check_residual_normality(ax, residuals_normalized):
    qq = ProbPlot(residuals_normalized)
    qq.qqplot(line="45", alpha=0.5, lw=1, ax=ax)
    ax.set_title("Normal Q-Q")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Standardized Residuals")


def check_homoscedacticity(ax, fitted_y, residuals_norm_abs_sqrt):
    plot_3 = plt.figure()
    ax.scatter(fitted_y, residuals_norm_abs_sqrt, alpha=0.5)
    sns.regplot(
        x=fitted_y,
        y=residuals_norm_abs_sqrt,
        scatter=False,
        ci=False,
        lowess=True,
        line_kws={"color": "red", "lw": 1, "alpha": 0.8},
        ax=ax,
    )
    ax.set_title("Scale-Location")
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("$\\sqrt{|Standardized Residuals|}$")


def check_influcence(ax, leverage, cooks, residuals_normalized):
    ax.scatter(leverage, residuals_normalized, alpha=0.5)
    sns.regplot(
        x=leverage,
        y=residuals_normalized,
        scatter=False,
        ci=False,
        lowess=True,
        line_kws={"color": "red", "lw": 1, "alpha": 0.8},
        ax=ax,
    )
    ax.set_xlim(0, max(leverage) + 0.01)
    ax.set_ylim(-3, 5)
    ax.set_title("Residuals vs Leverage")
    ax.set_xlabel("Leverage")
    ax.set_ylabel("Standardized Residuals")


def summary_plots(lm, title=""):
    fitted_y = lm.fittedvalues
    residuals = lm.resid
    residuals_normalized = lm.get_influence().resid_studentized_internal
    model_norm_residuals_abs_sqrt = np.sqrt(np.abs(residuals_normalized))
    leverage = lm.get_influence().hat_matrix_diag
    cooks = lm.get_influence().cooks_distance[0]

    fig, axs = plt.subplots(2, 2)
    check_linearity_assumption(axs[0, 0], fitted_y, residuals)
    check_residual_normality(axs[0, 1], residuals_normalized)
    check_homoscedacticity(axs[1, 0], fitted_y, model_norm_residuals_abs_sqrt)
    check_influcence(axs[1, 1], leverage, cooks, residuals_normalized)

    fig.suptitle(title)

    plt.show()

# Importing files

### Elon Musk's tweets

tweets_df = pd.read_csv("2021.csv", encoding='latin1')
tweets_df=tweets_df.rename(columns={'date':'Time'})
tweets_df=tweets_df.rename(columns={'tweet':'Tweet'})

### Tesla stock price

tesla_stock_price_df = pd.read_csv("TSLA.csv")

### Nasdaq index price

nsdq=pd.read_csv("^IXIC.csv")
nsdq=nsdq.rename(columns={'Time':'Date'})

# Preprocessing

### Cleaning the data

cleaned_tweets_with_sentiment = clean_tweet_data(tweets_df)
tesla_stock_with_daily_change = clean_stock_data(tesla_stock_price_df)
nsdq_daily_change=clean_stock_data(nsdq)

### Merging the data

df = merge_tweets_and_stock_data(cleaned_tweets_with_sentiment, tesla_stock_with_daily_change)
df = df.merge(nsdq_daily_change,on='Time')
dummies = pd.get_dummies(df["Sentiment"])
df=dummies.join(df)

### Renaming columns

df=df.rename(columns={0:'Negative'})
df=df.rename(columns={1:'Neutral'})
df=df.rename(columns={2:'Positive'})
df=df.rename(columns={'daily_percentage_change_x':'%_delta_Tesla'})
df=df.rename(columns={'daily_percentage_change_y':'%_delta_Nasdaq'})

### Adding exponential moving average

df.ta.ema(close='%_delta_Tesla', length=10, append=True)
df = df.dropna()

# Explanatory analysis

### Paring sentiment against the Tesla stock price change

plt.figure(figsize=(16, 10))
sns.violinplot(x= "Sentiment", y= "%_delta_Tesla" , data=df)

plt.figure(figsize=(16, 10))
sns.boxplot(x= "Sentiment", y = "%_delta_Tesla", data=df)

### Paring sentiment against the Tesla stock price change, including the time factor

plt.figure(figsize=(16, 10))
sns.scatterplot(x = df['Time'], y = df['%_delta_Tesla'], 
                data = df, hue = df['Sentiment'])

### Descriptive statistics

df.describe()

### Correlations

df.corr()

# First model

### Modelling the full data set with linear regression with neutral tweet sentiment as a baseline

X,y = linear_regression_data()
X = sm.add_constant(X)
lm = sm.OLS(y,X)
lm = lm.fit()
lm.summary()

### Checking for the fit of the model based on the full data set

plt.rc("figure", figsize=(16, 12))
summary_plots(lm, title="Visualizations")

# Second model

### Eliminating outliers for the target variable

Quantile_10 = df['%_delta_Tesla'].quantile(0.10)
Quantile_90 = df['%_delta_Tesla'].quantile(0.90)
df = df[(df['%_delta_Tesla'] >= Quantile_10) & (df['%_delta_Tesla'] <= Quantile_90)]

### Describing the target variable without outliers

df['%_delta_Tesla'].describe()

### Modelling the filtered data set with linear regression with neutral tweet sentiment as a baseline

X,y = linear_regression_data()
X = sm.add_constant(X)
lm = sm.OLS(y,X)
lm = lm.fit()
lm.summary()

### Checking for the fit of the model based on the filtered data set

plt.rc("figure", figsize=(16, 12))
summary_plots(lm, title="Visualizations")

# Third model

### Turning the problem into a classification task

df['Binary_delta_Tesla'] = np.where(df['%_delta_Tesla'] >= 0, 1, 0)

### Modelling the filtered data set with logisic regression with neutral tweet sentiment as a baseline

X,y = logistic_regression_data()
X = sm.add_constant(X)
lm = sm.Logit(y,X)
lm = lm.fit()
lm.summary()

# Final remarks

### Discussion around model results
#### - With regard to R-squared/Pseudo R-squared values, we favour the first model with 42.5% of explained variance.
#### - Having removed outliers from the data set, we observed a drastic fall (lessening by half) in R-squared values; this might imply that those observations carried a significant meaning to the problem discussed
#### - Trying to reframe the task into a classification problem, we witnessed an even smaller percentage of varianced explained by the logistic model, stregthening our initial thesis than linear regression might be a better tool to describe this relationship

### As regards the real world usage, all three models can serve descriptive purposes only; hence, they should not be used as a predictive tool, since:
#### - They do not feature a training/test split -> Potential for introducing machine learning/deep learning techniques to enhence the depth of the analysis and predictive applications
#### - They lack thorougher time series analysis -> Besides checking for key elements of the time series (e.g., stationarity), we are optimistic about implementing more advanced models suitable for this kind of data, namely long short-term memory (LSTM) architecture
#### - They do not entail other potentially meaningful variables -> Since initial results of our model paint a rather blant picture as for the signs of cooefficients and their p-values, more nuances approach towards Elon Musk's tweets sentiment and their impact on Tesla stock price might improve the results. As for the first steps, we would use both current and deferred variables.
#### - They are not linked to a real time data source -> To mitigate this, we would link the model to data sourced from Twitter's API (we had unfortunately problems accessing) and the Yahoo library
