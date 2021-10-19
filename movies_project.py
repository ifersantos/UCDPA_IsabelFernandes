import pandas as pd  # Importing pandas with the alias (pd)
import warnings  # importing module
warnings.filterwarnings("ignore")

# reading netflix from the csv file and storing into the variable netflix
netflix = pd.read_csv('netflix_titles.csv')  # , parse_dates=['date_added', 'release_year']
# reading imdb datasets from csv file and storing into variables
imdb_movies = pd.read_csv('IMDb movies.csv')
imdb_movies = imdb_movies[['imdb_title_id', 'title', 'year']]
imdb_ratings = pd.read_csv('IMDb ratings.csv')
imdb = pd.merge(imdb_movies, imdb_ratings, on='imdb_title_id')  # Merging IMDb movies and ratings

# Analysing IMDB ratings to get top rated movies on Netflix
imdb_ratings = pd.read_csv('IMDb ratings.csv', usecols=['weighted_average_vote'])
imdb_movies = pd.read_csv('IMDb movies.csv', usecols=['title', 'year', 'genre'])

ratings = pd.DataFrame({'Title': imdb_movies.title,
                        'Release_Year': imdb_movies.year,
                        'Rating': imdb_ratings.weighted_average_vote,
                        'Genre': imdb_movies.genre})

# Merging dataframes (Netflix + IMDb on titles)
data = ratings.merge(netflix, left_on='Title', right_on='title', how='inner')
data = data.sort_values(by='Rating', inplace=False, ascending=False)

print('\n\033[1m' + 'Concise summary of a DataFrame:' + '\033[0m\n')
print(data.info(), '\n')  # Generating an overview
print('\n\033[1m' + 'First five rows of the dateset:' + '\033[0m\n', data.head(), '\n')  # Displaying first 5 rows
print('\n\033[1m' + 'Last five rows of the dateset:' + '\033[0m\n', data.tail(), '\n')  # Displaying last 5 last rows
# basic statistical details
print(data.describe(include='O').T)  # transpose index and columns of the netflix frame

print('\n\033[1m' + 'DataTypes in the DataSet:\n' + '\033[0m', data.dtypes)
print('\n\033[1m' + 'Columns in DataSet:' + '\033[0m', data.columns)  # printing columns names
print('\n\033[1m' + 'Shape of DataSet:' + '\033[0m')

nRow_nCol = data.shape  # Saving the number of rows and columns as a tuple
print('There are {} rows and {} columns.\n'.format(nRow_nCol[0], nRow_nCol[1]))
print('\033[1m' + 'Total Number of duplicate rows in DataSet:' + '\033[0m', data.duplicated().sum(), '\n')

# Checking for missing values
print('\033[1m' + 'Total Number of missing values in DataSet:' + '\033[0m', data.isnull().sum().sum(), '\n')

# Loop through the data checking for missing values
import numpy as np  # Importing numpy as the alias (np)

for col in data.columns:
    missing = np.sum(data[col].isnull())
    print('\033[1m' + '{} : {}'.format(col, missing))  # Displaying missing values

# Replacing missing values with NaN
for i in data.columns[data.isnull().any(axis=0)]:  # Applying Only on variables with missing values
    data[i].ffill(inplace=True)

# Checking again for missing values
print('\n\033[1m' + 'Total Number of missing values in DataSet after replacement:' + '\033[0m',
      data.isnull().sum().sum(), '\n')

# Analysis of Movies vs TV Shows
print('\033[1m' + 'Percentage of Netflix Titles:' + '\n\033[0m', netflix.type.value_counts()/100)

import matplotlib.pyplot as plt  # Importing matplotlib with the alias (plt)
import seaborn as sns  # Importing seaborn with the alias (sns)
import plotly.express as px  # Importing plotly with the alias (px)

# Plotting a comparison of the total number of movies and shows on Netflix Dataset
sns.set(style="darkgrid")  # setting the style
fig = px.pie(netflix, names="type", title="Quantity of Movies vs TV Shows", height=500)
#fig.show()  # Displaying the fig

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
#plt.show()


# Top 10 rated movies on Netflix
top_rated = data[0:10].title
print('\n\033[1m' + 'Top 10 rated movies on Netflix:' + '\n\033[0m', top_rated, '\n')

from bs4 import BeautifulSoup
import requests  # Importing library to make the HTTP request

# Scraping imdb top 10 movies and ratings from IMDb Website
url = 'http://www.imdb.com/chart/top'  # Specifying url
r = requests.get(url)  # Packaging the request, sending the request and catching the response
soup = BeautifulSoup(r.text, "html.parser")  # Creating a BeautifulSoup object from the HTML

movies = soup.select('td.titleColumn')  # Selecting the column with movie titles
m_ratings = [i.attrs.get('data-value')
                for i in soup.select('td.posterColumn span[name=ir]')]

# Creating an empty list for storing movie info
list = []

# Iterating over movies to extract details
for index in range(0, 10):
    movie_str = movies[index].get_text()
    movie = (' '.join(movie_str.split()).replace('.', ''))
    movie_title = movie[len(str(index)) + 1:-7]
    place = movie[:len(str(index)) - (len(movie))]
    d = {"movie_title": movie_title, "place": place, "rating": m_ratings[index]}
    list.append(d)  # Appending results to the list

# Displaying Top 10 Movies and its rating.
print('\033[1m' + 'Top 10 rated movies on IMDb Website and its rating:' + '\033[1m')
for movie in list:
    print(movie['place'], '-', movie['movie_title'], '-', movie['rating'])

# Getting one of the Top 10 titles of Netflix rated movies (Schindler's List) via API
# # Packaging the request, sending the request and catching the response
url = "http://www.omdbapi.com/?t=Schindler's+List&plot=full&apikey=a6ba41bb"
response = requests.get(url)  # Packaging the request, sending the request and catching the response
print('\n', response)  # Printing the response and response status code

# Getting the API response in JSON format and assigning it to a variable
json_data = response.json()
# Importing the JSON response into Pandas DataFrame:
movie_request = pd.DataFrame(json_data.items(), columns=['Key', 'Value'])
print('\n\033[1m' + "{} is one of the Top 10 rated movies on Netflix."
      .format(movie_request['Value'][0]) + '\n\033[0m', movie_request)

# Gives us the Top 10 most released year in the Dataset
count_year = data.groupby("release_year").size().reset_index(name="title")\
    .sort_values("title", ascending=False).head(10)
df_count_year = pd.DataFrame(count_year)
print('\n\033[1m' + 'Top 10 most released year in the Dataset:\n' + '\033[0m')
#print(df_count_year)

# Plotting
plt.figure(figsize=(10, 10))
sns.set(style="darkgrid")
ax = sns.countplot(x="release_year",
                   data=data, palette="Set2",
                   order=data['release_year'].value_counts().index[0:10])
plt.gcf().subplots_adjust(bottom=0.15)
plt.xticks(rotation=90)
plt.title('Top 10 most released year in the Dataset')
plt.ylabel('')
plt.xlabel('Released Year')
#plt.show()

# Gives us the Top 10 count of title in each year
print(data.groupby("release_year").size().reset_index(name="title")\
    .sort_values("title", ascending=False).head(10))

# Countries with highest rated movies
rated_byCountry = data['country'].value_counts().sort_values(ascending=False)
rated_byCountry = pd.DataFrame(rated_byCountry)
TopCountries = rated_byCountry[0:10]
print('\n' + 'Countries with highest rated movies:' + '\n\033[0m', TopCountries)

# plot
fig = px.bar(TopCountries, y='country', labels={'Country': 'Rates (Total)', 'index': 'countries'},
             title="Countries With Highest Rated Movies", height=400)
#fig.show()

# Total of Movies released in 2021
print('\n' + "Number of movies released in 2021 is ", data.loc[data.release_year == 2021, 'show_id'].count())

# Movies release in 2021 - USA
print('\n' + "Number of video's published by 'USA' in 2021 is ",
      data.loc[(data.country == 'United States') & (data.release_year == 2021), 'release_year'].count())

# Movies release in 2021 - UK
print('\n' + "Number of video's published by 'UK' in 2021 is ",
      data.loc[(data.country == 'United Kingdom') & (data.release_year == 2021), 'release_year'].count())

import re  # Importing module regex

def clean_desc(desc):  # Function for description cleaning:
    desc = re.sub("\'", "", desc)  # Removing backslash-apostrophe
    desc = re.sub("[^a-zA-Z]", " ", desc)  # Removing everything except alphabets
    desc = ' '.join(desc.split())  # Removing whitespaces
    desc = desc.lower()  # Converting description to lowercase

    return desc

# Applying the function on the movie description by using the apply-lambda
data['clean_description'] = data['description'].apply(lambda x: clean_desc(x))
print('\n', data['clean_description'])  # Displaying the results

# Counting Genre
# Creation of the column count for aggregation
data['Count'] = 1
print('\n\033[1m' + 'There are {} different genres movies in the dataset.'
      .format(data[['Genre', 'Count']].groupby(['Genre'], as_index=False).count().shape[0], '\n'))

from sklearn.preprocessing import OneHotEncoder  # Using OHE to separate genres
# Defining Genre categories and marking each movie as belonging to each category
s = data['Genre'].str.split('|').explode()
encoder = OneHotEncoder()
encoded = encoder.fit_transform(s.values[:, None])  # Specifying which columns to create dummies
one_hot_df = pd.DataFrame(encoded.toarray(), columns=np.ravel(encoder.categories_),
                          dtype='int').groupby(s.index).sum()  # Saving to dataframe
result_harm = pd.concat([data, one_hot_df], axis=1)  # Concatenating two dataframes
print(result_harm)  # Displaying results


vote = data['Rating'].mean()
print(vote)

min_vote = data['Rating'].quantile(0.90)
print(min_vote)