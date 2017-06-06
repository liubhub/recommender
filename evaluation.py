import pandas as pd
import numpy as np
import math
import preprocess as pr
import similarity_metrics as sm

all_users = pr.all_users
M = pr.M
M_test = pr.M_test

ratings = pr.ratings
ratings_test = pr.ratings_test


def eval_similarity(id_user, metric):
    all_sim = [[],[]]
    u = M[id_user - 1] 
    # считаем "похожесть" целевого пользователя с остальными
    for i in range(all_users): 
        if i == id_user - 1:
            continue
        v = M[i]
        sim = metric(u,v)
        all_sim[0].append(i) # i как в матрице
        all_sim[1].append(sim)
    return all_sim

def top_max_sim(sim_array, N):
    sim_array_top_N = np.sort(np.asarray(sim_array))[-N::] # N макс. коэф. похожести
    maxN = [[],[]] 
    for i in range(len(sim_array)):
        if sim_array[i] in sim_array_top_N:
            maxN[0].append(i) # i как в матрице
            maxN[1].append(sim_array[i])
    return maxN       # max коэф. похожести

def generate_recom(arr_sim, curr_user):
    max_rating = 3,4,5
    curr_user_movies = set(ratings[(ratings['user_id'] == curr_user)]['movie_id'])
    mas_rec = []
    # finding movies 
    for i in arr_sim[0]:
        curr_sim = i + 1 # в датафрейме не с нуля # i + 2
        to_rec = set(ratings[(ratings['user_id'] == curr_sim) & ((ratings['rating'] == max_rating[0]) | (ratings['rating'] == max_rating[1]) | (ratings['rating'] == max_rating[2]))]['movie_id']) 
        recommender_set = to_rec - curr_user_movies # рекоммендуем те, которые пользователь не смотрел
        mas_rec.append((curr_sim, list(recommender_set)))
    # adding prediction rating
    for i in mas_rec:
        curr = i[0]
        for j in range(len(i[1])):
            # прогноз оценки
            # прогнозная оценка = той, которую поставил похожий пользователь
            r = ratings[(ratings['user_id'] == curr) & (ratings['movie_id'] == i[1][j])]['rating'].values[0]
            temp = dict([(i[1][j],r)]) 
            # temp = i[1][j]
            i[1][j] = temp 
    return mas_rec

# этот перебор нужен для того, чтобы из массива рекомендаций
# извлечь те фильмы, что в тестовой выборке действительно посмотрел
# целевой пользователь
def transform(to_rec, curr_user_movies):
    mas = []
    ls = []
    for rec in to_rec:
        for i in rec[1]:
            for j in curr_user_movies:
                if j in i:
                    ls.append(i)
        mas.append((rec[0],ls))
        ls = []
        
    empty = []
    for i in mas:
        for j in i:
            if not j:
                empty.append(i)
    i = 0
    
    while i < len(mas):
        if mas[i] in empty:
            del mas[i]
        else:
            i += 1
            
    for i in mas:
        for j in i:
            if not j:
                mas.remove(i)
    return mas

def eval_mae(id_user, mas_rec):
    user_error = []
    mae_each_user = []
    curr_user = ratings_test[(ratings_test['user_id'] == id_user)]
    for rec_list in mas_rec:
        for item in rec_list[1]:
            temp = list(item)[0]
            error = math.fabs(int(curr_user[curr_user['movie_id'] == temp]['rating'].values) - item[temp])
            user_error.append(error)
        mae_each_user.append(np.sum(np.asarray(user_error)) / len(user_error))
        user_error = []
    return mae_each_user

def eval_acos(id_user, N):
    res = []
    acos_similarity_array = eval_similarity(id_user, sm.acos)
    N_acos_sim_users = top_max_sim(acos_similarity_array[1], N)
    to_rec_acos = transform(generate_recom(N_acos_sim_users, id_user), list(ratings_test[(ratings_test['user_id'] == id_user)]['movie_id']))
    mae_acos = eval_mae(id_user, to_rec_acos)
    mae_final = np.sum(np.asarray(mae_acos)) / len(mae_acos)
#     res.append(to_rec_acos)
#     res.append(mae_acos)
    res.append(mae_final)
    
    return res

def eval_pcc(id_user, N):
    res = []
    pcc_similarity_array = eval_similarity(id_user, sm.pcc)
    N_pcc_sim_users = top_max_sim(pcc_similarity_array[1], N)
    to_rec_pcc = transform(generate_recom(N_pcc_sim_users, id_user), list(ratings_test[(ratings_test['user_id'] == id_user)]['movie_id']))
    mae_pcc = eval_mae(id_user, to_rec_pcc)
    mae_final = np.sum(np.asarray(mae_pcc)) / len(mae_pcc)
#     res.append(to_rec_pcc)
#     res.append(mae_pcc)
    res.append(mae_final)
    return res

def eval_cpcc(id_user, N):
    res = []
    cpcc_similarity_array = eval_similarity(id_user, sm.cpcc)
    N_cpcc_sim_users = top_max_sim(cpcc_similarity_array[1], N)
    to_rec_cpcc = transform(generate_recom(N_cpcc_sim_users, id_user), list(ratings_test[(ratings_test['user_id'] == id_user)]['movie_id']))
    mae_cpcc = eval_mae(id_user, to_rec_cpcc)
    mae_final = np.sum(np.asarray(mae_cpcc)) / len(mae_cpcc)
#     res.append(to_rec_cpcc)
#     res.append(mae_cpcc)
    res.append(mae_final)
    
    return res

def eval_jaccard(id_user, N):
    res = []
    jaccard_similarity_array = eval_similarity(id_user, sm.simJaccard)
    N_jaccard_sim_users = top_max_sim(jaccard_similarity_array[1], N)  
    to_rec_jaccard = transform(generate_recom(N_jaccard_sim_users, id_user), list(ratings_test[(ratings_test['user_id'] == id_user)]['movie_id']))
    mae_jaccard = eval_mae(id_user, to_rec_jaccard)
    mae_final = np.sum(np.asarray(mae_jaccard)) / len(mae_jaccard)
#     res.append(to_rec_jaccard)
#     res.append(mae_jaccard)
    res.append(mae_final)
    
    return res