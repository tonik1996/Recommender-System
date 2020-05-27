def find_n_neighbours(df,n):
    df = df.apply(lambda x: pd.Series(x.sort_values(ascending=False)
           .iloc[:n].index, 
          index=['top{}'.format(i) for i in range(1, n+1)]), axis=1)
    return df

def User_item_score1(user):
    a = sim_user_30_m[sim_user_30_m.index==user].values
    b = a.squeeze().tolist()
    d = movie_user[movie_user.index.isin(b)]
    l = ','.join(d.values)
    Movie_seen_by_similar_users = l.split(',')

    Movies_under_consideration = list(set(Movie_seen_by_similar_users)-set(list(map(str, Movie_seen_by_user))))
    Movies_under_consideration = list(map(int, Movies_under_consideration))

    score = []
    for item in Movies_under_consideration:
        c = final_movie.loc[:,item] # all ratings of movie <item> values='adg_rating',index='userId',columns='movieId'
        d = c[c.index.isin(b)] # all raters who are in similar users
        f = d[d.notnull()] # all raters who are in similar users and not null and their ratins of <item>
        avg_user = mean.loc[mean['userId'] == user,'rating'].values[0] # avarage rating of <user>
        index = f.index.values.squeeze().tolist() # list of similar users
        corr = similarity_with_user.loc[user,index] # cosine similarity with <user> and other users
        fin = pd.concat([f, corr], axis=1) # cosine similarity between <user> and 30 most similar users
        fin.columns = ['adg_score','correlation']
        fin['score']=fin.apply(lambda x:x['adg_score'] * x['correlation'],axis=1)
        nume = fin['score'].sum()
        deno = fin['correlation'].sum()
        final_score = avg_user + (nume/deno)
        score.append(final_score)

    data = pd.DataFrame({'movieId':Movies_under_consideration,'score':score})
    top_5_recommendation = data.sort_values(by='score',ascending=False)
    Movie_Name = top_5_recommendation.merge(movies, how='inner', on='movieId') # movieId, score, title, genres for all unseen movies.
    return Movie_Name

def Question1(parts):
    message = ""

    # Get the movie titles from the question sentence:
    del parts[0:2] 
    del parts[len(parts)-4:len(parts)]
    q_title = ' '.join(parts)

    # Check if the title exists in the database:
    if any(movies['title'].str.contains(q_title)):

      # Get the matching movie(s):
      matches = movies[movies['title'].str.match(q_title)]

      # Check if the movie is in the movies that have already seen by the user:
      if any(matches['movieId']) in Movie_seen_by_user:
        message = "{} is already rated.".format(q_title)
      else:
        # Explain the user that the movie got a low ranking:
        try:
          message = "It looks like we only predicted a rating of {} for that movie based on the rating of similar users to you. The minimum rating to be in recommendations would have been {}.".format(round(predicted_movies[predicted_movies['title'].str.match(q_title)]['score'].values[0],2),round(predicted_movies['score'][4],2))
        except:
          message = "It looks like none of the users that are similar to you have rated that movie and therefore we were unable to predict a rating for it."
    else:
      message = "{} does not exist in database.".format(q_title)
    return message

def Question2(parts, parts2):
    message = ""

    # Get the movie titles from the question sentence:
    del parts[0:2] 
    del parts[parts.index('recommended'):len(parts)]
    del parts2[len(parts2)-2:len(parts2)]
    del parts2[0:parts2.index('whereas')+1]

    q_title = ' '.join(parts)
    q_title2 = ' '.join(parts2)

    # Check if the title exists in the database:
    if any(movies['title'].str.contains(q_title2)):
      # Get the matching movie(s) in the recommendations:
      matches = movies[movies['title'].str.match(q_title2)]

      # Check if the movie is in the movies that have already seen by the user:
      if any(matches['movieId']) in Movie_seen_by_user:
        message = "{} is already rated.".format(q_title2)
      else:
        # Explain the user that the movie X got lower ranking than Y:
        try:
          message = "It looks like we predicted a rating of {} for {} based on the ratings of similar users to you whereas {} received only a rating of {}. The minimum rating to be in recommendations would have been {}.".format(round(predicted_movies[predicted_movies['title'].str.match(q_title)]['score'].values[0],2),q_title,q_title2,round(predicted_movies[predicted_movies['title'].str.match(q_title2)]['score'].values[0],2),round(predicted_movies['score'][4],2))
        except:
          message = "It looks like none of the users that are similar to you have rated {} and therefore we were unable to predict a rating for it.".format(q_title2)
    else:
      message = "{} does not exist in database.".format(q_title2)
    return message

def Question3(parts):
    message = ""

    # Get the movie title from the question sentence:
    del parts[0:2] 
    del parts[len(parts)-5:len(parts)]
    q_title = ' '.join(parts)

    # Check if the title exists in the database:
    if any(movies['title'].str.contains(q_title)):

      # Get the matching movie(s):
      matches = movies[movies['title'].str.match(q_title)]

      # Check if the movie is in the movies that have already seen by the user:
      if any(matches['movieId']) in Movie_seen_by_user:
        message = "{} is already rated.".format(q_title)
      else:
        # Explain the user that the movie got a low ranking:
        try:
          index = int(predicted_movies.index[predicted_movies['title'].str.match(q_title)].values[0])+1
          message = "It looks like that movie got a predicted score of {} and got a placing {}. The movies between the placements 1 and {} got ratings between {} and {}.".format(round(predicted_movies[predicted_movies['title'].str.match(q_title)]['score'].values[0],2),index,index-1,round(predicted_movies['score'][0],2),round(predicted_movies['score'][index-2],2))
        except:
          message = "It looks like none of the users that are similar to you have rated that movie and therefore we were unable to predict a rating for it."

    else:
      message = "{} does not exist in database.".format(q_title)
    return message

def Question4(parts,parts2):
    message = ""
        # Get the movie titles from the question sentence:
    del parts[0:2] 
    del parts[parts2.index('below')-2:len(parts)]
    del parts2[0:parts2.index('below')+1]

    q_title = ' '.join(parts)
    q_title2 = ' '.join(parts2)

    q_title2=q_title2[:-1]

    # Check if the titles exists in the database:
    if any(movies['title'].str.contains(q_title)) and any(movies['title'].str.contains(q_title2)):
      # Get the matching movie(s) in the recommendations:
      matches = movies[movies['title'].str.match(q_title)]

      # Check if the movie is in the movies that have already seen by the user:
      if any(matches['movieId']) in Movie_seen_by_user:
        message = "{} is already rated.".format(q_title2)
      else:
        # Explain the user that the movie X got lower ranking than Y:
        try:
          index1 = int(predicted_movies.index[predicted_movies['title'].str.match(q_title)].values[0])+1
          index2 = int(predicted_movies.index[predicted_movies['title'].str.match(q_title2)].values[0])+1
          message = "It looks like we predicted a rating of {} for {} based on the ratings of similar users to you whereas {} received a rating of {}. Due to this, {} placed {} places lower than {}.".format(round(predicted_movies[predicted_movies['title'].str.match(q_title)]['score'].values[0],2),q_title,q_title2,round(predicted_movies[predicted_movies['title'].str.match(q_title2)]['score'].values[0],2),q_title,index1-index2,q_title2)
        except:
          message = "It looks like none of the users that are similar to you have rated {} and therefore we were unable to predict a rating for it.".format(q_title2)
    else:
      message = "The movie does not exist in database."
    return message

def Question5(parts):
    message = ""

    # Get the genre from the question sentence:
    del parts[0:5] 
    del parts[len(parts)-1]
    q_title = ' '.join(parts)

    # Check if the genre exists in the database:
    if any(movies['genres'].str.contains(q_title)):

      # Explain the user that the movies of genre X got a low ratings:
      try:
        movie1 = predicted_movies[predicted_movies['genres'].str.match(q_title)]['title'].values.squeeze().tolist()[0] 
        try:
         index2 = int(predicted_movies.index[predicted_movies['genres'].str.match(q_title)].values[1])+1
         movie2 = predicted_movies[predicted_movies['genres'].str.match(q_title)]['title'].values.squeeze().tolist()[1] 
        except:
          movie2 = ""
        message = "The highest rated movie(s) of genre {} are {} and {} which have ratings of {} and {}. The minimum rating to be in recommendations would have been {}.".format(q_title,movie1,movie2,round(predicted_movies[predicted_movies['genres'].str.match(q_title)]['score'].values[0],2),round(predicted_movies[predicted_movies['genres'].str.match(q_title)]['score'].values[1],2),round(predicted_movies['score'][4],2))
        
        #which had ratings of {} and {}. The minimum rating to be in recommendations would have been {}.".format(q_title,movie1,movie2,round(predicted_movies[predicted_movies['genres'].str.match(q_title)]['score'].values[0],2),round(predicted_movies[predicted_movies['title'].str.match(q_title)]['score'].values[1],2),round(predicted_movies['score'][4],2))
      except:
        message = "It looks like none of the users that are similar to you have rated {} movies and therefore we were unable to predict a rating for them.".format(q_title)
    else:
      message = "The are no {} movies in the database.".format(q_title)
    
    return message

def Question6(parts,parts2):
    message = ""
    # Get the genres from the question sentence:
    del parts[0:3] 
    del parts[parts.index('movies'):len(parts)]
    del parts2[0:parts2.index('many')+1]
    del parts2[len(parts2)-1]

    q_title = ' '.join(parts)
    q_title2 = ' '.join(parts2)

    # Check if the titles exists in the database:
    if any(movies['genres'].str.contains(q_title)) and any(movies['genres'].str.contains(q_title2)):

      # Explain the user that the movies of genre X got low ratings than movies of genre Y:
      try:
        movie1 = predicted_movies[predicted_movies['genres'].str.match(q_title)]['title'].values[0]
        rating1 = round(predicted_movies[predicted_movies['genres'].str.match(q_title)]['score'].values[0],2)

        movie2 = predicted_movies[predicted_movies['genres'].str.match(q_title2)]['title'].values[0]
        rating2 = round(predicted_movies[predicted_movies['genres'].str.match(q_title2)]['score'].values[0],2)

        message = "You were recommended {} {} movie(s) and {} {} movie(s). {} movies such as {} received a rating {} at best whereas {} movies such as {} a rating {} at best.".format(predicted_movies_5[predicted_movies_5['genres'].str.match(q_title)].shape[0],q_title,predicted_movies_5[predicted_movies_5['genres'].str.match(q_title2)].shape[0],q_title2,q_title2,movie2,rating2,q_title,movie1,rating1)

        if predicted_movies[predicted_movies['genres'].str.match(q_title)].shape[0] <= predicted_movies[predicted_movies['genres'].str.match(q_title2)].shape[0]/2:
          message = message + "\n\nThere also were few {} movies rated by the similar users compared to {} movies.".format(q_title,q_title2)
      except:
        message = "It looks like none of the users that are similar to you have rated such movies and therefore we were unable to predict a rating for them.".format(q_title)

    else:
      message = "The are no such movies in the database."   

    return message

import pandas as pd 
import numpy as np
import sklearn as sk
import statistics
from sklearn.metrics.pairwise import cosine_similarity
import time
start_time = time.time()

# Reading the data sets:
movies = pd.read_csv("movies.csv",encoding="utf-8")
ratings = pd.read_csv("ratings.csv")

# mean ratings of every user.
mean = ratings.groupby(by="userId",as_index=False)['rating'].mean()

# calculate weighted ratings
rating_avg = pd.merge(ratings,mean,on='userId')
rating_avg['adg_rating']=rating_avg['rating_x']-rating_avg['rating_y']

# all  ratings of all users
check = pd.pivot_table(rating_avg,values='rating_x',index='userId',columns='movieId')

# all weighted ratings of all users
final = pd.pivot_table(rating_avg,values='adg_rating',index='userId',columns='movieId')

final_movie = final.fillna(final.mean(axis=0))
final_user = final.apply(lambda row: row.fillna(row.mean()), axis=1)

# find 30 similar users and films
b = cosine_similarity(final_user)
np.fill_diagonal(b, 0)
similarity_with_user = pd.DataFrame(b,index=final_user.index)
similarity_with_user.columns=final_user.index # dataframe, where columns&rows are users, data is cos similarity

sim_user_30_m = find_n_neighbours(similarity_with_user,30)

rating_avg = rating_avg.astype({"movieId": str})
movie_user = rating_avg.groupby(by = 'userId')['movieId'].apply(lambda x:','.join(x))

user = 6 #int(input("Enter the user id to whom you want to recommend: "))
median_ratings = statistics.median(ratings.groupby(by="userId",as_index=False)['rating'].size())
users_ratings = ratings.groupby(by="userId",as_index=False)['rating'].size()[user]

Movie_seen_by_user = check.columns[check[check.index==user].notna().any()].tolist()
predicted_movies = User_item_score1(user)   # Dataframe with all movies, predicted scores for them and their genres
predicted_movies_5 = predicted_movies.head(5)
Movie_Names = predicted_movies_5.title.values.tolist()
Movie_Genres = predicted_movies_5.genres.values.tolist()

print(" ")
print("The recommendations for the user {}:".format(user))
print(" ")

for i in range(0,5):
    print(Movie_Names[i],Movie_Genres[i])
print(" ")
#print (predicted_movies[['title','genres','score']].nlargest(5, 'score'))

type(predicted_movies)

# ----------------------- WHY-NOT QUERIES -----------------------------------------------------------------

question = ""
while question != "exit":
  # Ask the Why-Not question:
  question = input("Question: ")
  parts = question.split()
  parts2 = question.split()

  # 1. Why was X not recommended to me?
  if parts[len(parts)-4:len(parts)] == ['not', 'recommended', 'to', 'me?']:
    print(Question1(parts))

  # 2. Why was Y recommended whereas X was not? 
  elif parts[0:2] == ['Why', 'was'] and parts[len(parts)-2:len(parts)] == ['was', 'not?']:
    print(Question2(parts, parts2))

  # 3. Why is X not in a higher rating?
  elif parts[len(parts)-5:len(parts)] == ['not', 'in', 'a', 'higher', 'rating?']:
    print(Question3(parts))

  # 4. Why is X below Y? 
  elif parts[0:2] == ['Why','is']:
    print(Question4(parts,parts2))

  # 5. Why are there no more X movies? 
  elif parts[0:5] == ['Why', 'are', 'there', 'no', 'more']:
    print(Question5(parts))

  # 6. Why so few X movies but so many Y movies? 
  elif parts[0:3] == ['Why', 'so', 'few']:
    print(Question6(parts, parts2))

  # If the user has ranked only a small amount of movies, the recommendations can be biased. Explain this to the user:
  if users_ratings < median_ratings:
    print("The low rating can be because you have only rated {} movies so far which is below the median of ranked movies per user ({}).".format(users_ratings,median_ratings))