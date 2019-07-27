from czx_tools_pack_local.Feature_tools import Feature_names, factor_data, freq_feature, factor_data, _Data_Preprocess
import pandas as pd
import json
import numpy as np
# from nn_modules.feature_get import *
from tqdm import tqdm
import shelve
import multiprocessing
import csv
import sys
import gc
import psutil

# def mode_parse():
if sys.platform == 'win32':
    base_path = '../../Data/'
else:
    base_path = '/data/Huawei_people_properties/'


def get_static_fea(cates, num_feas, data_all, useful_feature=None, method_list=['mean', 'std', 'skew']):
    if useful_feature is None: useful_feature = Feature_names()
    if not isinstance(cates, list): cates = [cates]
    if not isinstance(num_feas, list): num_feas = [num_feas]
    new_cols = []
    for cate_name in tqdm(cates, desc="static"):
        for num_name in num_feas:
            add_dict = {}
            for method in method_list:
                add_dict[cate_name + '_' + num_name + '_' + method] = method
            data_all = pd.merge(data_all, data_all.groupby(cate_name, as_index=False)[num_name].agg(add_dict),
                                on=cate_name, how='left')
            useful_feature.feature_add(list(add_dict.keys()),
                                       'num')  # cate_name + '_' + num_name + '_median', cate_name + '_' + num_name + '_max'
            new_cols+=list(add_dict.keys())
            # useful_feature.feature_remove(cate_name + '_' + num_name + '_max')
    return data_all, useful_feature,new_cols

def haversine(lat1, lon1, lat2, lon2, r=6371):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = r  # 地球平均半径，单位为公里
    return c * r  # 输出为公里


def func(user_app_actived, app_info_dict, i):
    max_len = 0
    all_activedApp_cate_list = []
    len_list = []
    for row in tqdm(user_app_actived.iterrows(), desc='user_cate_process_' + str(i)):
        appid_list = row[1]['appId_list']
        app_cate_list = list(map(lambda x: app_info_dict[x] if x in app_info_dict else str(0), appid_list))
        length_tmp = len(app_cate_list)
        len_list.append(length_tmp)
        if max_len < length_tmp:
            max_len = length_tmp
        str_category = '#'.join(app_cate_list)
        all_activedApp_cate_list.append(str_category)
    user_app_actived['all_activedApp_cate_list'] = all_activedApp_cate_list
    user_app_actived['all_activedApp_cate_list_len'] = len_list
    return user_app_actived, max_len


def argmin(lst):
    return min(range(len(lst)), key=lst.__getitem__)


def argmax(lst):
    return max(range(len(lst)), key=lst.__getitem__)


def usage_fea_get_onetime(base_path='../Data/', old=False):
    if old:
        print('reading pickle')
        ussage_df = pd.read_pickle(base_path + "usage_ontime_processed.pkl")
        print('reading info')
        s = shelve.open(base_path + 'usage_info_onetime.db')
        use_feature = s['use_feature']
        sequence_fea = s['sequence_fea']
        max_id_dict = s['max_id_dict']
        return ussage_df, use_feature, sequence_fea, max_id_dict
    else:
        max_id_dict = {}
        if sys.platform != 'win32':
            pkl_path = '../../../chengwei/HuaWeiUserPersona/dataset/user_app_usage.pkl'
            user_app_usage = pd.read_pickle(pkl_path)
            user_app_usage.columns = ['uId', 'appId', 'duration', 'times', 'use_date']
            print('read usage finish')
        else:
            user_app_usage = pd.read_csv(base_path + "user_app_usage.csv", nrows=1000000,names=['uId', 'appId', 'duration', 'times', 'use_date'])
        
        use_feature = Feature_names()
        user_app_usage['appId'] = user_app_usage['appId'].apply(lambda x: int(x.replace('a', ''))).astype(np.int32)
        print('appId transed')
        user_app_usage['duration'] = user_app_usage['duration'].astype(np.int32)
        print('duration transed')
        user_app_usage['times'] = user_app_usage['times'].astype(np.int32)
        print('times transed')
        user_app_usage['duration_everytimes'] = user_app_usage['duration'] / user_app_usage['times']
        user_app_usage['duration_everytimes_encode'] = user_app_usage['duration_everytimes'].astype(int)
        # use_feature.feature_add('duration_everytimes_encode','cate')
        user_app_usage['duration_encode'] = user_app_usage['duration']
        user_app_usage['appId_encode'] = user_app_usage['appId']
        user_app_usage['times_encode'] = user_app_usage['times']
        for col in ['appId_encode', 'duration_encode', 'times_encode', 'duration_everytimes_encode']:
            user_app_usage, max_id = factor_data(user_app_usage, col)
            max_id_dict[col + '_list'] = max_id
            user_app_usage[col] = user_app_usage[col].astype(np.int32)
            print(col, user_app_usage[col].max(), max_id)
        
        uId_list = []
        use_date_list = []
        appId_list = []
        duration_list = []
        times_list = []
        usage_len_list = []
        grouped_list = list(user_app_usage.groupby('uId'))
        for one in tqdm(grouped_list, desc='grp_full: '):
            uid = one[0]
            uId_list.append(uid)
            tmp_appId = list(one[1]['appId_encode'].values)
            tmp_duration = list(one[1]['duration_encode'].values)
            tmp_times = list(one[1]['times_encode'].values)
            appId = '#'.join(map(lambda x: str(x), tmp_appId))
            duration = '#'.join(map(lambda x: str(x), tmp_duration))
            times = '#'.join(map(lambda x: str(x), tmp_times))
            usage_len = len(tmp_times)
            usage_len_list.append(usage_len)
            appId_list.append(appId)
            duration_list.append(duration)
            times_list.append(times)
        del grouped_list
        ussage_df = pd.DataFrame()
        ussage_df['uId'] = uId_list
        ussage_df['usage_appId_full_list'] = appId_list
        ussage_df['usage_duration_full_list'] = duration_list
        ussage_df['usage_time_full_list'] = times_list
        ussage_df['usage_len_full'] = usage_len_list
        max_id_dict['max_usage_full_len'] = ussage_df['usage_len_full'].max()
        
        # ============================================= group part =============================================================
        user_app_usage = user_app_usage.groupby(["uId", 'appId_encode'])[['duration', 'times']].mean().reset_index()
        user_app_usage['duration_everytimes'] = user_app_usage['duration'] / user_app_usage['times']
        user_app_usage['duration_encode'] = user_app_usage['duration'].astype(int)
        # user_app_usage['appId_encode'] = user_app_usage['appId'].astype(int)
        # user_app_usage['times_encode'] = user_app_usage['times'].astype(int)

        usage_appId_duration_list = []
        usage_appId_times_list = []
        usage_appId_mean_dura_list = []
        uId_list = []
        appId_list = []
        duration_list = []
        times_list = []
        max_duration_list = []
        max_duration_app_list = []
        min_duration_list = []
        mean_duration_list = []
        min_duration_app_list = []
        max_time_list = []
        max_time_app_list = []
        min_time_list = []
        mean_time_list = []
        min_time_app_list = []
        duration_everytimes_list = []
        usage_len_list = []
        grouped_list = list(user_app_usage.groupby('uId'))
        # del user_app_usage
        for one in tqdm(grouped_list, desc='grp: '):
            uid = one[0]
            uId_list.append(uid)
            tmp_appId = list(one[1]['appId_encode'].values)
            tmp_duration = list(one[1]['duration'].values)
            tmp_times = list(one[1]['times'].values)
            duration_everytimes = list(one[1]['duration_everytimes'].values)
            usage_appId_duration_dict = dict(zip(tmp_appId, tmp_duration))
            tmp_usage_appId_duration = sorted(usage_appId_duration_dict, reverse=True)
            tmp_usage_appId_times = sorted(dict(zip(tmp_appId, tmp_times)), reverse=True)
            tmp_usage_appId_mean_dura = sorted(dict(zip(tmp_appId, duration_everytimes)), reverse=True)
            duration_everytimes_list.append(duration_everytimes)
            origin_tmp_duration = list(one[1]['duration'].values)
            origin_tmp_times = list(one[1]['times'].values)
            appId = '#'.join(map(lambda x: str(x), tmp_appId))
            duration = '#'.join(map(lambda x: str(x), tmp_duration))
            times = '#'.join(map(lambda x: str(x), tmp_times))
            appId_duration = '#'.join(map(lambda x: str(x), tmp_usage_appId_duration))
            appId_times = '#'.join(map(lambda x: str(x), tmp_usage_appId_times))
            appId_mean_dura = '#'.join(map(lambda x: str(x), tmp_usage_appId_mean_dura))
            
            max_duration = max(origin_tmp_duration)
            max_duration_app = tmp_appId[np.argmax(origin_tmp_duration)]
            min_duration = min(origin_tmp_duration)
            mean_duration = np.mean(origin_tmp_duration)
            min_duration_app = tmp_appId[np.argmin(origin_tmp_duration)]
            
            max_time = max(origin_tmp_times)
            max_time_app = tmp_appId[np.argmax(origin_tmp_times)]
            min_time = min(origin_tmp_times)
            mean_time = np.mean(origin_tmp_times)
            min_time_app = tmp_appId[np.argmin(origin_tmp_times)]
            usage_len = len(origin_tmp_times)
            
            max_duration_list.append(max_duration)
            max_duration_app_list.append(max_duration_app)
            min_duration_list.append(min_duration)
            min_duration_app_list.append(min_duration_app)
            mean_duration_list.append(mean_duration)
            max_time_list.append(max_time)
            max_time_app_list.append(max_time_app)
            min_time_list.append(min_time)
            mean_time_list.append(mean_time)
            min_time_app_list.append(min_time_app)
            usage_len_list.append(usage_len)
            appId_list.append(appId)
            duration_list.append(duration)
            times_list.append(times)
            
            usage_appId_duration_list.append(appId_duration)
            usage_appId_times_list.append(appId_times)
            usage_appId_mean_dura_list.append(appId_mean_dura)
            
        data_all, use_feature, new_cols= get_static_fea(['uId'], ['duration','times','duration_everytimes'], user_app_usage, use_feature,
                                                    method_list=['mean', 'std', 'skew', 'max', 'min'])
        data_all = data_all[['uId']+new_cols]
        
        data_all.drop_duplicates(['uId'], keep='first', inplace=True)

        ussage_df_appid_uid = pd.DataFrame()
        ussage_df_appid_uid['uId'] = uId_list
        ussage_df_appid_uid = pd.merge(ussage_df_appid_uid,data_all,on='uId',how='left')
        # del uId_list
        ussage_df_appid_uid['usage_appId_list'] = appId_list
        # del appId_list
        ussage_df_appid_uid['usage_duration_list'] = duration_list
        # del duration_list
        ussage_df_appid_uid['usage_times_list'] = times_list
        ussage_df_appid_uid['usage_appId_duration_list'] = usage_appId_duration_list
        ussage_df_appid_uid['usage_appId_times_list'] = usage_appId_times_list
        ussage_df_appid_uid['usage_appId_mean_dura_list'] = usage_appId_mean_dura_list
        # del times_list
        # del use_date_list
        
        ussage_df_appid_uid['max_duration'] = max_duration_list
        ussage_df_appid_uid['max_duration_numrical'] = max_duration_list
        ussage_df_appid_uid['max_duration_app'] = max_duration_app_list
        ussage_df_appid_uid['min_duration'] = min_duration_list
        ussage_df_appid_uid['min_duration_numrical'] = min_duration_list
        ussage_df_appid_uid['min_duration_app'] = min_duration_app_list
        ussage_df_appid_uid['mean_duration'] = mean_duration_list
        ussage_df_appid_uid['mean_duration_numrical'] = mean_duration_list
        use_feature.feature_add(['max_duration_numrical', 'min_duration_numrical', 'mean_duration_numrical'], 'num')
        
        ussage_df_appid_uid['max_time'] = max_time_list
        ussage_df_appid_uid['max_time_numrical'] = max_time_list
        ussage_df_appid_uid['max_time_app'] = max_time_app_list
        ussage_df_appid_uid['min_time'] = min_time_list
        ussage_df_appid_uid['min_time_numrical'] = min_time_list
        ussage_df_appid_uid['min_time_app'] = min_time_app_list
        ussage_df_appid_uid['mean_time'] = mean_time_list
        ussage_df_appid_uid['mean_time_numrical'] = mean_time_list
        ussage_df_appid_uid['usage_len'] = usage_len_list
        use_feature.feature_add(['max_time_numrical', 'min_time_numrical', 'mean_time_numrical'], 'num')
        
        ussage_df = pd.merge(ussage_df, ussage_df_appid_uid, on='uId', how='left')
        # user_app_usage = pd.pivot_table(user_app_usage, index=["uId", "appId"], values=["duration", 'times', 'use_date'],
        #                                 aggfunc=np.sum).reset_index()
        # grouped_list = list(user_app_usage.groupby('uId'))
        # group_num = 30
        # if sys.platform == 'win32':
        #     group_num = 1
        #
        # pool = multiprocessing.Pool(processes=group_num)  # 创建4个进程
        # one_gro_len = int(len(grouped_list) / group_num)
        # return_list = []
        # for i in range(group_num):
        #     slice_grouped_list = grouped_list[i * one_gro_len:(i + 1) * one_gro_len]
        #     return_list.append(pool.apply_async(usage_func, (slice_grouped_list,i,)))
        #     if i == group_num - 1 and (i + 1) * one_gro_len < len(grouped_list):
        #         slice_user_app_actived = grouped_list[(i + 1) * one_gro_len:]
        #         return_list.append(pool.apply_async(usage_func, (slice_user_app_actived, i,)))
        # pool.close()  # 进程添加完毕
        # pool.join()
        # for i in range(group_num):
        #     return_list[i]= return_list[i].get()
        # ussage_df = pd.concat(return_list)
        print('col max: ', ussage_df.max())
        use_feature.feature_add(
            ['max_duration', 'max_duration_app', 'min_duration', 'mean_duration', 'min_duration_app', 'max_time',
             'max_time_app', 'min_time', 'mean_time', 'min_time_app', 'usage_len', 'usage_len_full'], 'cate')
        max_id_dict['max_usage_len'] = ussage_df['usage_len'].max()
        for col in use_feature.categorical_columns:
            print(col)
            ussage_df, max_id = factor_data(ussage_df, col)
            max_id_dict[col] = max_id
            ussage_df[col] = ussage_df[col].astype(np.int)
        sequence_fea = ['usage_appId_list', 'usage_duration_list', 'usage_times_list', 'usage_appId_duration',
                        'usage_appId_times', 'usage_appId_mean_dura']
        s = shelve.open(base_path + 'usage_info_onetime.db')
        s['use_feature'] = use_feature
        s['sequence_fea'] = sequence_fea
        s['max_id_dict'] = max_id_dict
        s.close()
        print('ussage_df info: ', ussage_df.info())
        print('writing pickle ')
        ussage_df.to_pickle(base_path + "usage_ontime_processed.pkl", compression=None)
        print('pickle saved')
        print('max_id_dict: ', max_id_dict)
    return ussage_df, use_feature, sequence_fea, max_id_dict


def data_prep(base_path='../Data/', old=False):
    print("old:", old)
    
    if old:
        print("old")
        s = shelve.open(base_path + "gene/data_626.db")
        # data_all = s['data_all']
        data_all = pd.read_pickle('/data/Huawei_people_properties/' + 'base_data_626.pkl')
        print('base readed')
        useed_feature = s['useed_feature']
        max_id_dict = s['max_id_dict']
        s.close()
        return data_all, useed_feature, max_id_dict
    
    app_info = pd.read_csv(base_path + "app_info.csv", names=['appId', 'category'])
    nrows = None
    if sys.platform == 'win32':
        nrows = 10000
    age_test = pd.read_csv(base_path + "age_test.csv", nrows=nrows, names=['uId'])
    age_train = pd.read_csv(base_path + "age_train.csv", nrows=nrows, names=['uId', 'age_group'])
    
    user_basic_info = pd.read_csv(base_path + "user_basic_info.csv",
                                  names=['uId', 'gender', 'city', 'prodName', 'ramCapacity', 'romLeftRation', 'romCapacity',
                                         'romLeftRation', 'color', 'fontSize', 'ct', 'carrier', 'os'])
    user_behavior_info = pd.read_csv(base_path + "user_behavior_info.csv",
                                     names=['uId', 'boostTimes', 'AFuncTimes', 'BFuncTimes', 'CFuncTimes', 'DFuncTimes',
                                            'EFuncTimes', 'FFuncTimes', 'GFuncTimes'])
    user_app_actived = pd.read_csv(base_path + "user_app_actived.csv", nrows=nrows,names=['uId', 'appId'])
    user_app_actived['appId_list'] = user_app_actived['appId'].apply(lambda x: list(
        np.array(x.strip().replace('a', '').replace('\\N', '0').split('#')).astype(np.int)))  # appid先都转为int  最后在编码
    user_app_actived['appId_list_len'] = user_app_actived['appId_list'].apply(lambda x: len(x))
    print("appId_list_len max len: ", user_app_actived['appId_list_len'].max())
    
    print('fnished reading')
    useed_feature = Feature_names()
    useed_feature.feature_add(
        ['gender', 'city', 'prodName', 'ramCapacity', 'color', 'ct', 'fontSize', 'carrier', 'os', 'appId_list_len'], 'cate')
    useed_feature.feature_add(
        ['ramCapacity', 'romLeftRation', 'romCapacity', 'romLeftRation', 'boostTimes', 'AFuncTimes', 'BFuncTimes', 'CFuncTimes',
         'DFuncTimes',
         'EFuncTimes', 'FFuncTimes', 'GFuncTimes'], 'cate')
    useed_feature.feature_add(['all_activedApp_cate_list_len'], 'cate')
    max_id_dict = {}
    # -----------------------------------------------------------------------生成 app_id  编码字典-------------------------------
    appid_list_set = set()
    appid_list_full = []
    for appid in tqdm(user_app_actived['appId_list'].values, desc='appid_set'):
        appid_list_full.extend(list(set(appid)))
        # appid_list_set = appid_list_set | set(appid)
    appid_list_set = set(appid_list_full)
    print('set created,length: ', len(appid_list_set))
    app_info['appId'] = app_info['appId'].apply(lambda x: int(x.replace('a', '')))
    app_info['appId'] = app_info['appId'].astype(np.int)
    appId_list = list(app_info['appId'].unique())
    appid_list_set = appid_list_set | set(appId_list)
    appId_index_dict = dict(zip(appid_list_set, range(len(appid_list_set))))
    
    # -------------------------------------------------------------------------   激活app 类别特征处理 ------
    app_info, max_id = factor_data(app_info, 'category')
    max_id_dict['all_activedApp_cate_list'] = max_id
    app_info['category'] = app_info['category'].astype(str)
    
    app_info_grouped_list = list(app_info.groupby('appId'))
    app_cate_full_list = []
    app_id_full_list = []
    for one in tqdm(app_info_grouped_list, desc='grp: '):
        appid = one[0]
        app_cate = '#'.join(one[1]['category'])
        app_cate_full_list.append(app_cate)
        app_id_full_list.append(appid)
    app_info = pd.DataFrame()
    app_info['appId'] = app_id_full_list
    app_info['category'] = app_cate_full_list
    app_info_dict = dict(zip(app_id_full_list, app_cate_full_list))
    
    # user_app_actived['all_activedApp_cate_list'] = user_app_actived['appId_list'].apply(
    #     lambda x: '#'.join(map(app_info_dict, x)))
    # print(user_app_actived.info())
    # print(user_app_actived.head(5))
    user_app_activeduidset = set(user_app_actived['uId'].unique())
    user_app_afetrain = set(age_train['uId'].unique())
    user_app_agetest = set(age_test['uId'].unique())
    print('a:', len(user_app_afetrain - user_app_activeduidset))
    print('b:', len(user_app_agetest - user_app_activeduidset))
    
    group_num = 30
    if sys.platform == 'win32':
        group_num = 1
    
    pool = multiprocessing.Pool(processes=group_num)  # 创建4个进程
    one_gro_len = int(user_app_actived.shape[0] / group_num)
    return_list = []
    for i in range(group_num):
        slice_user_app_actived = user_app_actived[i * one_gro_len:(i + 1) * one_gro_len]
        return_list.append(pool.apply_async(func, (slice_user_app_actived, app_info_dict, i,)))
        if i == group_num - 1 and (i + 1) * one_gro_len < user_app_actived.shape[0]:
            slice_user_app_actived = user_app_actived[(i + 1) * one_gro_len:]
            return_list.append(pool.apply_async(func, (slice_user_app_actived, app_info, i,)))
    pool.close()  # 进程添加完毕
    pool.join()
    len_list = []
    for i in range(group_num):
        return_list[i], tmp_len = return_list[i].get()
        len_list.append(tmp_len)
    max_id_dict['max_cate_len'] = max(len_list)
    user_app_actived = pd.concat(return_list)
    print(user_app_actived.info())
    print(age_train.info())
    
    # --------------------------------------------------------------------------------------------
    age_train = pd.merge(age_train, user_app_actived, on='uId', how='left')
    age_train = pd.merge(age_train, user_basic_info, on='uId', how='left')
    age_train = pd.merge(age_train, user_behavior_info, on='uId', how='left')
    print("train meger finish")
    
    age_test = pd.merge(age_test, user_app_actived, on='uId', how='left')
    age_test = pd.merge(age_test, user_basic_info, on='uId', how='left')
    age_test = pd.merge(age_test, user_behavior_info, on='uId', how='left')
    print("test meger finish")
    
    age_test.to_csv(base_path + 'gene/test.csv', index=False)
    age_train.to_csv(base_path + 'gene/train.csv', index=False)
    
    age_train['data_flag'] = 1
    age_test['data_flag'] = 0
    data_all = pd.concat([age_train, age_test])
    
    data_all['gender_city'] = data_all['gender'].astype(str) + data_all['city'].astype(str)
    data_all['color_prodName'] = data_all['color'].astype(str) + data_all['prodName'].astype(str)
    data_all['carrier_ct'] = data_all['carrier'].astype(str) + data_all['ct'].astype(str)
    data_all['os_prodName'] = data_all['carrier'].astype(str) + data_all['ct'].astype(str)
    useed_feature.feature_add(['gender_city', 'color_prodName', 'carrier_ct', 'os_prodName'], 'cate')
    
    print(data_all.head(5)['appId_list'])
    nul_index = data_all['appId_list'].isnull()
    data_all.loc[nul_index, 'appId_list'] = '0'
    data_all.loc[nul_index, 'appId_list'] =data_all.loc[nul_index,'appId_list'].apply(lambda x:list(np.array(x.split('#')).astype(int)))
    print('appId_list null: ', data_all['appId_list'].isnull().sum())
    data_all['appId_list_encoded'] = data_all['appId_list'].apply(lambda x: list(map(lambda a: appId_index_dict[a], x)))
    data_all['appId_list_len'] = data_all['appId_list'].apply(lambda x: len(x))
    useed_feature.feature_add('appId_list_encoded', 'cate')
    for col in useed_feature.categorical_columns:
        if col is not 'appId_list_encoded':
            print(col)
            data_all, max_id = factor_data(data_all, col)
            max_id_dict[col] = max_id
            data_all[col] = data_all[col].astype(np.int)
    max_id_dict['appId_list_encoded'] = len(appid_list_set)
    if sys.platform != 'win32':
        data_all.to_pickle('/data/Huawei_people_properties/' + 'base_data_626.pkl')
    s = shelve.open(base_path + "gene/data_626.db")
    # s['data_all'] = data_all
    s['useed_feature'] = useed_feature
    s['max_id_dict'] = max_id_dict
    s.close()
    return data_all, useed_feature, max_id_dict


if __name__ == "__main__":
    # usage_fea_get(base_path=base_path)
    
    usage_fea_get_onetime(base_path=base_path)
    data_prep(base_path)
