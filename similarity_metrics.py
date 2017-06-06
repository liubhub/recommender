import math 
import numpy as np
def non_zero(mas):
    res = []
    for i in range(len(mas)):
        if mas[i] != 0:
            res.append(i)
    return res
def pcc(u, v):
    u_movies = non_zero(u)
    v_movies = non_zero(v)
    common_movies = list(set(u_movies) & set(v_movies))
    uv_ratings = [[],[],[]]
    for i in common_movies:
        uv_ratings[2].append(i)
        uv_ratings[0].append(u[i])
        uv_ratings[1].append(v[i])
    uv_ratings = np.asarray(uv_ratings)
    ru = uv_ratings[0].mean();
    rv = uv_ratings[1].mean();
    sim  = 0; num = 0; denom = 0; 
    for i in range(len(uv_ratings[0])):
        num += (uv_ratings[0][i] - ru)*(uv_ratings[1][i] - rv)
        denom += math.sqrt(math.pow((uv_ratings[0][i] - ru),2)) * math.sqrt(math.pow((uv_ratings[1][i] - rv),2))
    if denom == 0:
        sim = 1
    else:
        sim = num / denom
    return sim

def cpcc(u,v):
    rmed = 3
    u_movies = non_zero(u)
    v_movies = non_zero(v)
    common_movies = list(set(u_movies) & set(v_movies))
    uv_ratings = [[],[],[]]
    for i in common_movies:
        uv_ratings[2].append(i)
        uv_ratings[0].append(u[i])
        uv_ratings[1].append(v[i])
    uv_ratings = np.asarray(uv_ratings)
    ru = rv = rmed
    sim  = 0; num = 0; denom = 0; 
    for i in range(len(uv_ratings[0])):
        num += (uv_ratings[0][i] - ru)*(uv_ratings[1][i] - rv)
        denom += math.sqrt(math.pow((uv_ratings[0][i] - ru),2)) * math.sqrt(math.pow((uv_ratings[1][i] - rv),2))
    if denom == 0:
        sim = 1
    else:
        sim = num / denom
    return sim

def acos(u,v):
    num = 0
    denom = 0
    ru = u.mean()
    rv = v.mean()
    for i in range(len(u)):
        num += (u[i] - ru)*(v[i] - rv)
        denom += math.sqrt(math.pow((u[i]-ru),2)) * math.sqrt(math.pow((v[i]-rv),2))
    sim = num / denom
    return sim

def simJaccard(u, v):
    u_movies = non_zero(u)
    v_movies = non_zero(v)
    common_movies = list(set(u_movies) & set(v_movies))
    all_movies = list(set(u_movies) | set(v_movies))
    simJ = len(common_movies) / len(all_movies)
    return simJ