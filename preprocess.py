import re
import sys
import time
import math
import pandas as pd
import pickle

import numpy as np

class User:
    def __init__(self, id, age, sex, occupation, zip):
        self.id = int(id)
        self.age = int(age)
        self.sex = sex
        self.occupation = occupation
        self.zip = zip
        self.avg_r = 0.0

class Item:
    def __init__(self, id, title, release_date, video_release_date, imdb_url, \
    unknown, action, adventure, animation, childrens, comedy, crime, documentary, \
    drama, fantasy, film_noir, horror, musical, mystery ,romance, sci_fi, thriller, war, western):
        self.id = int(id)
        self.title = title
        self.release_date = release_date
        self.video_release_date = video_release_date
        self.imdb_url = imdb_url
        self.unknown = int(unknown)
        self.action = int(action)
        self.adventure = int(adventure)
        self.animation = int(animation)
        self.childrens = int(childrens)
        self.comedy = int(comedy)
        self.crime = int(crime)
        self.documentary = int(documentary)
        self.drama = int(drama)
        self.fantasy = int(fantasy)
        self.film_noir = int(film_noir)
        self.horror = int(horror)
        self.musical = int(musical)
        self.mystery = int(mystery)
        self.romance = int(romance)
        self.sci_fi = int(sci_fi)
        self.thriller = int(thriller)
        self.war = int(war)
        self.western = int(western)


class Rating:
    def __init__(self, user_id, item_id, rating, time):
        self.user_id = int(user_id)
        self.item_id = int(item_id)
        self.rating = int(rating)
        self.time = time

# The dataset class helps to load files and create User, Item and Rating objects
class Dataset:
    def load_users(self, file, u):
        f = open(file, "r")
        text = f.read()
        entries = re.split("\n+", text)
        for entry in entries:
            e = entry.split('|', 5)
            if len(e) == 5:
                u.append(User(e[0], e[1], e[2], e[3], e[4]))
        f.close()

    def load_items(self, file, i):
        f = open(file, "r")
        text = f.read()
        entries = re.split("\n+", text)
        for entry in entries:
            e = entry.split('|', 24)
            if len(e) == 24:
                i.append(Item(e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8], e[9], e[10], \
                e[11], e[12], e[13], e[14], e[15], e[16], e[17], e[18], e[19], e[20], e[21], \
                e[22], e[23]))
        f.close()

    def load_ratings(self, file, r):
        f = open(file, "r")
        text = f.read()
        entries = re.split("\n+", text)
        for entry in entries:
            e = entry.split('\t', 4)
            if len(e) == 4:
                r.append(Rating(e[0], e[1], e[2], e[3]))
        f.close()
		
				
user = []
item = []
rating = []
rating_test = []
d = Dataset()

d.load_users("data/u.user", user)
d.load_items("data/u.item", item)
d.load_ratings("data/u1.base", rating)
d.load_ratings("data/u1.test", rating_test)

all_users = len(user)
n_items = len(item)

M = np.zeros((all_users, n_items))
for r in rating:
    M[r.user_id - 1][r.item_id - 1] = r.rating

M_test = np.zeros((all_users, n_items))
for r in rating_test:
    M_test[r.user_id - 1][r.item_id - 1] = r.rating

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('data/u2.base', sep='\t', names=r_cols, encoding='latin-1')
ratings = ratings.drop('unix_timestamp',axis=1)
ratings_test = pd.read_csv('data/u2.test', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = ratings_test.drop('unix_timestamp',axis=1)