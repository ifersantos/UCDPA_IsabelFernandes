import pandas as pd
import warnings  # importing module
warnings.filterwarnings("ignore")  # Filtering out the warnings

# loading libraries for visualization
import seaborn as sns
import matplotlib.pyplot as plt

# loading the csv data into pandas dataframe
heart_data = pd.read_csv('heart.csv')

# displaying first & last 5 rows of the dataset
print('\n\033[1m' + 'First glance at the dateset, 5 first & 5 last rows appended:' + '\033[0m\n',
      heart_data.head().append(heart_data.tail()), '\n')  # Displaying 5 first rows and 5 last rows of the dataset
# basic statistical measures
print('\n\033[1m' + 'Basic statistical details:\n' + '\033[0m', heart_data.describe())
# Printing data types
print('\n\033[1m' + 'DataTypes in the DataSet:\n' + '\033[0m', heart_data.dtypes)
# Printing the shape of the dataset
print('\n\033[1m' + 'Shape of DataSet:' + '\033[0m')
print('\n\033[1m' + f'The dataset has {heart_data.shape[0]} rows and {heart_data.shape[1]} columns\n')

# Loop through the data checking for missing values
import numpy as np  # Importing numpy with the alias (np)

for col in heart_data.columns:
    miss_values = np.sum(heart_data[col].isnull())
    print('\033[1m' + '{} : {}'.format(col, miss_values))  # Displaying missing values

# Plotting histograms of the predictor variables
plt.style.use('ggplot')
X = heart_data.drop('target', 1).values  # dropping the target variable
y = heart_data['target'].values
pd.DataFrame.hist(heart_data, figsize=[13, 10], facecolor='#0072BD')
#plt.show()

# checking the distribution of target variable - (Values - 1 -> Defective Heart & 0 -> Healthy heart)
print('\n\033[1m' + 'Checking the distribution of target variable:' + '\n\033[1m', heart_data.target.value_counts())

#Percentage of patients with vs without heart problems
print('\n\033[1m' + "Percentage of patients without heart problems: " +
      str(round(heart_data.target.value_counts()[0]*100/303, 2)))
print('\n\033[1m' + "Percentage of patients with heart problems: " +
      str(round(heart_data.target.value_counts()[1]*100/303, 2)))

import plotly.express as px  # Importing plotly with the alias (px)
sns.set(style="darkgrid")  # setting the style
fig = px.pie(heart_data, names="target",
             title="Quantity of Heart Problems vs No Heart Problems", height=500)
#fig.show()  # Displaying the fig

# Number of people with Heart Problems
print('\n\033[1m' + 'Number of people with Heart Problems: ', (heart_data['target'] == 1).sum())

# Separating patients
No_healthy = heart_data[heart_data['target'] == 1]
Healthy = heart_data[heart_data['target'] == 0]

# Age Distribution of patients with positive heart disease
ax = sns.distplot(No_healthy['age'], rug=True)
plt.title("Age Distribution")
#plt.show()

# Gender Distribution - Total Male vc Female in the dataset - Male = 1 - Female = 0
sns.countplot(No_healthy['sex'], palette='viridis')
plt.title('Gender distribution of Heart Disease patients', fontsize=15, weight='bold' )
#plt.show()

# Getting the correlation of the columns
correlation = heart_data.corr()
mask = np.zeros_like(correlation)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation, center=0, cmap="coolwarm", mask=mask, annot=True, square=True, fmt='.2f')
#plt.show()

# Visualizing the correlation data
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
sns.heatmap(correlation, annot=True, fmt='.0%', cmap="Blues")
plt.title('Correlation Between Variables', fontsize=20)
#plt.show()

# Pairplot
df_subset = heart_data[["age", "chol", "oldpeak","target", "thalach"]]
sns.set_theme(style="ticks")
sns.pairplot(df_subset, hue="target")
#plt.show()

# splitting the features and target
X = heart_data[heart_data.columns[:-1]]  # dropping target column
y = heart_data['target']

print('\n', X.head())  # printing X
print('\n', y.head())  # printing y

from sklearn.model_selection import train_test_split
# Split dataset into training set (80%) and test set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)
print('\n', X.shape, X_train.shape, X_test.shape)  # Checking how data was split - rows and columns

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()  # scale the values in the data between 0 and 1
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn import metrics, neighbors  # importing metrics and neighbors
from sklearn.metrics import classification_report

# Logistic regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
print('\n\033[1m' + 'Accuracy score for Logistic Regression: ', metrics.accuracy_score(y_test, pred_lr))
print('\n\033[1m', classification_report(y_test, pred_lr))

#Recall
from sklearn.metrics import recall_score
print('\n\033[1m' + 'Recall', recall_score(y_test, pred_lr, average=None))
#Precision
from sklearn.metrics import precision_score
print('\n\033[1m' + 'Precision', precision_score(y_test, pred_lr, average=None))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred_lr)
print('\n\033[1m' + 'Confusion Matrix')
print(cm, '\n\033[1m')


# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=1)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
#Accuracy
print('Accuracy score for Random Forest: ', metrics.accuracy_score(y_test, pred_rf))
print('\n\033[1m', classification_report(y_test, pred_rf))

#Recall
print('\n\033[1m' + 'Recall', recall_score(y_test, pred_rf, average=None))
#Precision
print('\n\033[1m' + 'Precision', precision_score(y_test, pred_rf, average=None))

cm2 = confusion_matrix(y_test, pred_rf)
print('\n\033[1m' + 'Confusion Matrix')
print(cm2, '\n\033[1m')

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
print('\n\033[1m' + 'Accuracy score for Naive Bayes: ', metrics.accuracy_score(y_test, nb_pred))
print('\n\033[1m' + '', classification_report(y_test, nb_pred))

#Recall
print('\n\033[1m' + 'Recall', recall_score(y_test, nb_pred, average=None))
#Precision
print('\n\033[1m' + 'Precision', precision_score(y_test, nb_pred, average=None))

cm3 = confusion_matrix(y_test, nb_pred)
print('\n\033[1m' + 'Confusion Matrix')
print(cm3, '\n\033[1m')

from sklearn.neighbors import KNeighborsClassifier
# K Nearest Neighbors Classifier
n_neig = 3  # Setting the the number of neighbors for k-NN
knn = neighbors.KNeighborsClassifier(n_neighbors=n_neig) #Create KNN Classifier
knn.fit(X_train, y_train)  #Train the model using the training sets
pred_knn = knn.predict(X_test)  # Predict the response for test dataset
print('\n\033[1m' + "Accuracy score for KNeibhbors at K=3: ", metrics.accuracy_score(y_test, pred_knn))
print('\n\033[1m', classification_report(y_test, pred_knn))

#Recall
print('\n\033[1m' + 'Recall', recall_score(y_test, pred_knn, average=None))
#Precision
print('\n\033[1m' + 'Precision', precision_score(y_test, pred_knn, average=None))

cm4 = confusion_matrix(y_test, pred_knn)
print('\n\033[1m' + 'Confusion Matrix')
print(cm4, '\n\033[1m')

# K Nearest Neighbors Classifier with 5 neighbors
n_neig = 5  # Setting the the number of neighbors for k-NN - 5 this time
knn = neighbors.KNeighborsClassifier(n_neighbors=n_neig)  #Create KNN Classifier with 5 n_neighbors
knn.fit(X_train, y_train)  #Train the model using the training sets
pred_n5 = knn.predict(X_test)  # Predict the response for test dataset
print('\n\033[1m' + "Accuracy score for KNeibhbors at K=5: ", metrics.accuracy_score(y_test, pred_n5))
print('\n\033[1m', classification_report(y_test, pred_n5))

#Recall
print('\n\033[1m' + 'Recall', recall_score(y_test, pred_n5, average=None))
#Precision
print('\n\033[1m' + 'Precision', precision_score(y_test, pred_n5, average=None))

cm5 = confusion_matrix(y_test, pred_n5)
print('\n\033[1m' + 'Confusion Matrix')
print(cm5, '\n\033[1m')

# Using an error plot to find the most favourable K value
error_rate = []
for i in range(1, 10):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 10), error_rate, color='blue', linestyle='--', marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K value')
plt.xlabel('K')
plt.ylabel('Error Rate')
#plt.show()

print('\n' + "Minimum error:-", min(error_rate), "at K =", error_rate.index(min(error_rate)))
print('Accuracy score: ', metrics.accuracy_score(y_test, pred_i))

# Using an accuracy plot to find the most favourable K value
acc = []
for i in range(1, 10):
    neigh = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)
    pred = neigh.predict(X_test)
    acc.append(metrics.accuracy_score(y_test, pred))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 10), acc, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
#plt.show()

print('\n\033[1m' + "Maximum accuracy:-", max(acc), "at K =", acc.index(max(acc)))
print('Accuracy score: ', metrics.accuracy_score(y_test, pred))

#Recall
print('\n\033[1m' + 'Recall', recall_score(y_test, pred, average=None))
#Precision
print('\n\033[1m' + 'Precision', precision_score(y_test, pred, average=None))


cm7 = confusion_matrix(y_test, pred)
print('\n\033[1m' + 'Confusion Matrix')
print(cm7, '\n\033[1m')

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)  # Create Decision Tree classifier object
dt.fit(X_train, y_train)
pred_dt = dt.predict(X_test)
print('\n\033[1m' + 'Accuracy score for Decision Tree: ', metrics.accuracy_score(y_test, pred_dt))
print('\n\033[1m', classification_report(y_test, pred_dt))

#Recall
print('\n\033[1m' + 'Recall', recall_score(y_test, pred_dt, average=None))
#Precision
print('\n\033[1m' + 'Precision', precision_score(y_test, pred_dt, average=None))

cm6 = confusion_matrix(y_test, pred_dt)
print('\n\033[1m' + 'Confusion Matrix')
print(cm6, '\n\033[1m')

# defining a function for training the models
def train_model(X_train, y_train, X_test, y_test, classifier, **kwargs):
    """
    Train the model and print out the score result.

    """

    model = classifier(**kwargs)  # instantiate model

    model.fit(X_train, y_train) # training model

    # Testing models accuracy on training and test data
    fit_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    print(f"Train accuracy: {fit_acc:0.2%}")
    print(f"Test accuracy: {test_acc:0.2%}")

    return model

# Logistic Regression
print('\n'+'Logistic regression')
lr = train_model(X_train, y_train, X_test, y_test, LogisticRegression)

# Random Forest Classifier
print('\n'+'Random Forest')
rf = train_model(X_train, y_train, X_test, y_test, RandomForestClassifier, max_depth=3, n_estimators=10, random_state=2)

# Naive Bayes
print('\n'+'Naive Bayes')
nb = train_model(X_train, y_train, X_test, y_test, GaussianNB)

# KNeighbors Classifier
print('\n'+'Knn')
knn = train_model(X_train, y_train, X_test, y_test, KNeighborsClassifier, n_neighbors=8)

# Decision Tree
print('\n'+'Decision Tree')
dt = train_model(X_train, y_train, X_test, y_test, DecisionTreeClassifier, max_depth=3, random_state=0)


# Score Comparison Summary
acc = []  # initializing an empty list
# list of algorithms names & list of algorithms with parameters
classifiers = ['Logistic Regression', 'Random Forests', 'Naive Bayes', 'KNN', 'Decision Trees']
algorithms = [LogisticRegression(),
              RandomForestClassifier(max_depth=3, n_estimators=100, random_state=1),
              GaussianNB(),
              KNeighborsClassifier(n_neighbors=8),
              DecisionTreeClassifier(max_depth=3, random_state=0)
              ]

# loop through algorithms
for i in algorithms:
    model = i
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    acc.append(score)  # appending score into the list

result = pd.DataFrame({'accuracy': acc}, index=classifiers)  # storing the results into a df
print('\n', result)

# Creating a Predictive System using Knn=8
print('\n\033[1m' + 'Predictive system using Knn=8')
input_data = (41, 0, 1, 130, 204, 0, 0, 172, 0, 1.4, 2, 0, 2)  # using random values for test
input_data_np = np.asarray(input_data)
input_reshaped = input_data_np.reshape(1, -1)  # reshape the numpy array
prediction = neigh.predict(input_reshaped)
print('\n\033[1m' + 'Result :',  prediction)

if (prediction[0]==0):  # checking the prediction
    print('This Person does not have a Heart Disease')
else:
    print('This Person has Heart Disease')


print('\n\033[1m' + '########')
### Netflix Movies and TV Shows — Exploratory Data Analysis (EDA) ###

print('\n\033[1m' + "Netflix Movies and TV Shows — Exploratory Data Analysis (EDA)" + '\n\033[1m')

import pandas as pd  # Importing pandas with the alias (pd)
import warnings  # importing module
warnings.filterwarnings("ignore")  # Filtering out the warnings

# reading csv files
netflix = pd.read_csv('netflix_titles.csv')
imdb_movies = pd.read_csv('IMDb movies.csv')
imdb_ratings = pd.read_csv('IMDb ratings.csv')

# Selecting the features
imdb_movies = imdb_movies[['imdb_title_id', 'title', 'year', 'genre', 'description']]
imdb_ratings = imdb_ratings[['imdb_title_id', 'weighted_average_vote']]

# Merging IMDb movies and ratings
imdb = pd.merge(imdb_movies, imdb_ratings, on='imdb_title_id')

ratings = pd.DataFrame({'title': imdb.title, 'year': imdb.year,
                        'imdb_rating': imdb.weighted_average_vote, 'genre': imdb.genre,
                        'imdb_description': imdb.description})

# Merging dataframes (Netflix + IMDb on titles)
data = ratings.merge(netflix, left_on='title', right_on='title', how='inner')

import pandasql as ps  # accessing data with sqlite
query = """SELECT title FROM data WHERE year >= 2020"""  # SQL query - Selecting titles from 2020
query_result = ps.sqldf(query, locals())
print('\n', 'Data:' + '\n', query_result.head(10))  # Displaying result - 10 titles release in 2020

# Analysing IMDB ratings to get top rated movies on Netflix
imdb_ratings = pd.read_csv('IMDb ratings.csv', usecols=['weighted_average_vote'])
imdb_movies = pd.read_csv('IMDb movies.csv', usecols=['title', 'year', 'genre'])

print('\n\033[1m' + 'Concise summary of a DataFrame:' + '\033[0m\n')
print(data.info(), '\n')  # Generating an overview

print('\n\033[1m' + 'First glance at the dateset, head + tail appended:' + '\033[0m\n',
      data.head(5).append(data.tail(5)), '\n')  # Displaying 5 first rows and 5 last rows of the dataset

print('\n\033[1m' + 'Basic statistical details:\n' + '\033[0m', data.describe())  # Basic statistical details
print('\n\033[1m' + 'DataTypes in the DataSet:\n' + '\033[0m', data.dtypes)  # Printing data types
print('\n\033[1m' + 'Columns in DataSet:' + '\033[0m', data.columns)  # Printing columns names
print('\n\033[1m' + 'Shape of DataSet:' + '\033[0m')  # Printing the shape of the dataset
print('\n\033[1m' + f'The dataset has {data.shape[0]} rows and {data.shape[1]} columns\n')  # Getting the shape of the dframe

# Checking for duplicates
print('\033[1m' + 'Total Number of duplicate rows in DataSet:' + '\033[0m', data.duplicated().sum(), '\n')
# Checking for missing values
print('\033[1m' + 'Total Number of missing values in DataSet:' + '\033[0m', data.isnull().sum().sum(), '\n')

# Loop through the data checking for missing values
import numpy as np  # Importing numpy with the alias (np)

for col in data.columns:
    missing = np.sum(data[col].isnull())
    print('\033[1m' + '{} : {}'.format(col, missing))  # Displaying missing values

import seaborn as sns  # importing seaborn with the alias (sns) - visualization
import matplotlib.pyplot as plt  # importing matplotlib with the alias (plt) - visualization
sns.heatmap(data.isnull())
plt.show()

# Replacing missing values with NaN
for i in data.columns[data.isnull().any(axis=0)]:  # Applying Only on variables with missing values
    data[i].ffill(inplace=True)

# Checking again for missing values
print('\n\033[1m' + 'Total Number of missing values in DataSet after replacement:' + '\033[0m',
      data.isnull().sum().sum(), '\n')

data = data.sort_values(by='imdb_rating', inplace=False, ascending=False)  # Sorting the values of the data by ratings

# Analysis of Movies vs TV Shows
print('\033[1m' + 'Percentage of Netflix Titles:' + '\n\033[0m', netflix.type.value_counts()/100)

import plotly.express as px  # Importing plotly with the alias (px)

# Plotting a comparison of the total number of movies and shows on Netflix Dataset
sns.set(style="darkgrid")  # setting the style
fig = px.pie(netflix, names="type", title="Quantity of Movies vs TV Shows", height=500)
fig.show()  # Displaying the fig

# Analysis of Top Genre on Netflix Dataset
top_genre = netflix.listed_in.str.split(', ', expand=True).stack().reset_index(level=1, drop=True)
print('\n\033[1m' + 'Top 10 Genre on Netflix:' + '\n\033[0m', top_genre)

fig, ax = plt.subplots(figsize=(10, 10))
sns.countplot(x=top_genre, order=top_genre.value_counts().index[:20])
plt.gcf().subplots_adjust(bottom=0.15)
plt.xticks(rotation=90)
plt.title('Top 10 Genres on Netflix')
plt.ylabel('Votes')
plt.xlabel('Genres')
plt.show()

# Top 10 most rated movies on Netflix
top_rated = data[0:10][['title', 'imdb_rating']]
print('\n\033[1m' + 'Top 10 most rated movies on Netflix:' + '\n\033[0m', top_rated.reset_index(drop=True), '\n')

from bs4 import BeautifulSoup
import requests  # Importing library to make the HTTP request

# Scraping imdb top 10 movies and ratings from IMDb Website
url = 'http://www.imdb.com/chart/top'  # Specifying url
r = requests.get(url)  # Packaging the request, sending the request and catching the response
soup = BeautifulSoup(r.text, "html.parser")  # Creating a BeautifulSoup object from the HTML

movies = soup.select('td.titleColumn')  # Selecting the column with movie titles
m_ratings = [i.attrs.get('data-value') for i in soup.select('td.posterColumn span[name=ir]')]

# Creating an empty list for storing movie info
list = []

# Iterating over movies to extract details
for index in range(0, 10):
    movie_str = movies[index].get_text()
    movie = (' '.join(movie_str.split()).replace('.', ''))
    movie_title = movie[len(str(index)) + 1:-7]
    place = movie[:len(str(index)) - (len(movie))]
    d = {"movie_title": movie_title, "place": place, "rating": '{:.3}'.format(m_ratings[index])}
    list.append(d)  # Appending results to the list

df_scrap = pd.DataFrame(list)
df_scrap.to_csv('scraping_data.csv', index=False)  # Saving DF to CSV file

scraping = pd.read_csv('scraping_data.csv')  # reading csv file with the info from imdb website

# Displaying Top 10 Movies and its rating from imdb website
print('\n\033[1m' + 'Top 10 rated movies on IMDb Website and its rating:' + '\n\033[1m', scraping.set_index('place'))

# Getting one of the Top 10 titles of Netflix & Imdb rated movies (Schindler's List) via API
url = "http://www.omdbapi.com/?t=Schindler's+List&plot=full&apikey=a6ba41bb"
response = requests.get(url)  # Packaging the request, sending the request and catching the response
print('\n', response)  # Printing the response and response status code

# Getting the API response in JSON format and assigning it to a variable
json_data = response.json()
# Importing the JSON response into Pandas DataFrame:
movie_request = pd.DataFrame(json_data.items(), columns=['Key', 'Value'])
print('\n\033[1m' + "{} is one of the Top 10 rated movies on Netflix."
      .format(movie_request['Value'][0]) + '\n\033[0m', movie_request)
movie_request.to_csv('movie_api.csv', index=False)  # Saving DF to CSV file


# Gives us the Top 10 count of title in each year
print('\n\033[1m' + 'Top 10 most released year in the Dataset:\n' + '\033[0m',data.groupby("release_year")
      .size().reset_index(name="title").sort_values("title", ascending=False).head(10))

# Plotting
plt.figure(figsize=(10, 10))
sns.set(style="darkgrid")
ax = sns.countplot(x="release_year",
                   data=data, palette="Set2",
                   order=data['release_year'].value_counts().index[0:10])
plt.gcf().subplots_adjust(bottom=0.15)
plt.xticks(rotation=90)
plt.title('Top 10 most released year in the Dataset')
plt.ylabel('Quantity of movies released')
plt.xlabel('Released Year')
plt.show()

# Countries with highest rated movies
rated_byCountry = data['country'].value_counts().sort_values(ascending=False)
rated_byCountry = pd.DataFrame(rated_byCountry)
TopCountries = rated_byCountry[0:10]
print('\n' + 'Countries with highest rated movies:' + '\n\033[0m', TopCountries)

# Total of Movies released in 2021
print('\n' + "Number of movies released in 2021 is ", data.loc[data.release_year == 2021, 'show_id'].count())

# plotting countries with highest rated movies
fig = px.bar(TopCountries, y='country', labels={'Country': 'Rates (Total)', 'index': 'countries'},
             title="Countries With Highest Rated Movies", height=400)
fig.show()

# Movies release in 2021 - USA
print('\n' + "Number of video's published by 'USA' in 2021 is ",
      data.loc[(data.country == 'United States') & (data.release_year == 2021), 'release_year'].count())



# Movies release in 2021 - UK
print('\n' + "Number of video's published by 'UK' in 2021 is ",
      data.loc[(data.country == 'United Kingdom') & (data.release_year == 2021), 'release_year'].count())

# Find the mean of the ratings given to each title
average_rating_df = data[["title", "imdb_rating"]].groupby('title').mean()

# Order the entries by highest average rating to lowest
sorted_average_ratings = average_rating_df.sort_values(by='imdb_rating', ascending=False)

# Inspect the top movies
print('Highest average rating: '+'\n', sorted_average_ratings.head())

# Create a list of only movies appearing > 50 times in the dataset
movie_popularity = data["title"].value_counts()
popular_movies = movie_popularity[movie_popularity].index

# Use this popular_movies list to filter the original DataFrame
popular_movies_rankings = data[data["title"].isin(popular_movies)]

# Give us the average rating given to these frequently watched films
popular_movies_average_rankings = popular_movies_rankings[["title", "imdb_rating"]].groupby('title').mean()
print('\n' + 'Average rating of frequent movies', popular_movies_average_rankings
      .sort_values(by="imdb_rating", ascending=False).head())

# Creation of the column count for aggregation
data['Count'] = 1
print('\n\033[1m' + 'There are {} different genres movies in the dataset.'
      .format(data[['genre', 'Count']].groupby(['genre'], as_index=False).count().shape[0], '\n'))

from sklearn.preprocessing import OneHotEncoder  # Using OHE to separate genres
# Defining Genre categories and marking each movie as belonging to each category

s = data['genre'].str.split('|').explode()
encoder = OneHotEncoder()
encoded = encoder.fit_transform(s.values[:, None])  # Specifying which columns to create dummies
one_hot_df = pd.DataFrame(encoded.toarray(), columns=np.ravel(encoder.categories_),
                          dtype='int').groupby(s.index).sum()  # Saving to dataframe
result_harm = pd.concat([data, one_hot_df], axis=1)  # Concatenating two dataframes
print(result_harm.head(3))  # Displaying results

import re  # using regex to clean the description column
def clean_desc(desc):  # Function for description cleaning:
    desc = re.sub("\'", "", desc)  # Removing backslash-apostrophe
    desc = re.sub("[^a-zA-Z]", " ", desc)  # Removing everything except alphabets
    desc = ' '.join(desc.split())  # Removing whitespaces
    desc = desc.lower()  # Converting description to lowercase

    return desc

# Combining netflix description and imdb description into a single column
data['plot'] = data["description"].astype(str) + "\n" + \
                 data["imdb_description"].astype(str)

# Applying the function on the movie description by using the apply-lambda
data['plot'] = data['plot'].apply(lambda x: clean_desc(x))
print('\n' + 'Movie Description after applying clean description function ' + '\n',
      data['plot'].head())  # Displaying the results