import time
import random
import numpy as np
from keras.utils.np_utils import to_categorical
from math import radians, cos, sin, asin, sqrt
import config
import operator
import threading
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"]  = config.GPU

GRID_COUNT = config.GRID_COUNT
BATCH_SIZE = config.batch_size
MODEL_NAME = config.model_file_name
TEXT_K = config.text_k
WORD_VEC_PATH = config.WORD_VEC_PATH
TRAINING_EPOCH = config.training_epoch
TRAIN_TEST_PART = config.train_test_part
random.seed(2017)

def time_hour(ci_time, form = '%Y-%m-%d %X'):
    st = time.strptime(ci_time, form)
    weekday = st.tm_wday
    hour = st.tm_hour
    if weekday < 6:
        return hour
    else:
        return (24+hour)

def time_diff(time1,time2,form = '%Y-%m-%d %X'):
    time11 = time.strptime(time1, form)
    time22 = time.strptime(time2, form)
    return abs(int(time.mktime(time11))-int(time.mktime(time22)))

def time_diff_la(time1,time2,form = '%Y-%m-%d %X'):
    s = time1
    if 'CDT' in s:
        t1 = time.strptime(s.replace(' CDT',''))
    if 'CST' in s:
        t1 = time.strptime(s.replace(' CST',''))

    s = time2
    if 'CDT' in s:
        t2 = time.strptime(s.replace(' CDT',''))
    if 'CST' in s:
        t2 = time.strptime(s.replace(' CST',''))

    return abs(int(time.mktime(t1))-int(time.mktime(t2)))

def time_hour_la(ci_time, form = '%Y-%m-%d %X'):
    s = ci_time
    if 'CDT' in s:
        st = time.strptime(s.replace(' CDT',''))
    if 'CST' in s:
        st = time.strptime(s.replace(' CST',''))
    weekday = st.tm_wday
    hour = st.tm_hour
    if weekday < 6:
        return hour
    else:
        return (24+hour)

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r * 1000

def geo_grade(index, x, y, m_nGridCount=GRID_COUNT):
    dXMax, dXMin, dYMax, dYMin = max(x), min(x), max(y), min(y)
    print dXMax, dXMin, dYMax, dYMin
    m_dOriginX = dXMin
    m_dOriginY = dYMin
    dSizeX = (dXMax - dXMin) / m_nGridCount
    dSizeY = (dYMax - dYMin) / m_nGridCount
    m_vIndexCells = []
    center_location_list = []
    for i in range(0, m_nGridCount * m_nGridCount + 1):
        m_vIndexCells.append([])
        y_ind = int(i / m_nGridCount)
        x_ind = i - y_ind * m_nGridCount
        center_location_list.append((dXMin + x_ind * dSizeX + 0.5 * dSizeX, dYMin + y_ind * dSizeY + 0.5 * dSizeY))
    print (m_nGridCount, m_dOriginX, m_dOriginY, \
           dSizeX, dSizeY, len(m_vIndexCells), len(index))
    poi_index_dict = {}
    for i in range(len(x)):
        nXCol = int((x[i] - m_dOriginX) / dSizeX)
        nYCol = int((y[i] - m_dOriginY) / dSizeY)
        if nXCol >= m_nGridCount:
            print 'max X'
            nXCol = m_nGridCount - 1

        if nYCol >= m_nGridCount:
            print 'max Y'
            nYCol = m_nGridCount - 1

        iIndex = nYCol * m_nGridCount + nXCol
        poi_index_dict[index[i]] = iIndex
        m_vIndexCells[iIndex].append([index[i], x[i], y[i]])

    return poi_index_dict, center_location_list

def evaluation_last_with_distance(all_output_array, all_test_Y, center_location_list):
    count, all_recall1, all_recall2, all_recall3, all_recall4, all_recall5, alldistance = 0.,0.,0.,0.,0.,0.,0.
    for j in range(len(all_test_Y)):
        y_test = all_test_Y[j]
        output_array = all_output_array[j]
        for i in range(len(y_test)):
            flag = False
            if ((i+1)<len(y_test)):
                if (y_test[i] != 0) & (y_test[i+1]==0):
                    flag = True
            else:
                if y_test[i] != 0:
                    flag =True
            if flag:
                true_pl = y_test[i] - 1
                infe_pl = output_array[i]
                topd = infe_pl[1:].argsort()[-5:][::-1]
                dd = []
                for k in topd:
                    pred = center_location_list[k]
                    tr = center_location_list[true_pl]
                    d = haversine(pred[0], pred[1], tr[0], tr[1])
                    dd.append(d)
                d = min(dd)
                alldistance += d
                if true_pl in infe_pl[1:].argsort()[-1:][::-1]: all_recall1 += 1
                if true_pl in infe_pl[1:].argsort()[-5:][::-1]: all_recall2 += 1
                if true_pl in infe_pl[1:].argsort()[-10:][::-1]: all_recall3 += 1
                if true_pl in infe_pl[1:].argsort()[-15:][::-1]: all_recall4 += 1
                if true_pl in infe_pl[1:].argsort()[-20:][::-1]: all_recall5 += 1
                count += 1
    print count
    print [all_recall1,all_recall2,all_recall3, all_recall4, all_recall5]
    print [all_recall1 / count, all_recall2 / count,
           all_recall3 / count, all_recall4 / count, all_recall5 / count, alldistance / count]
    return [all_recall1 / count, all_recall2 / count,
            all_recall3 / count, all_recall4 / count, all_recall5 / count, alldistance / count]

def nearest_location_last(vali_X, vali_evl, center_location_list):
    all_test_X_pl = vali_X[0]
    count, hc1 , hc5 , hc10, hc15, hc20, alldistance = 0.,0.,0.,0.,0.,0.,0.
    all_test_X_pl = all_test_X_pl.tolist()
    for j in range(len(all_test_X_pl)):
        trajl = all_test_X_pl[j]
        predict_traj = []
        for r in trajl:
            if r == 0:
                predict_traj.append(0)
            else:
                r = r-1
                res_list = [[i, haversine(center_location_list[r][0], center_location_list[r][1],
                                          center_location_list[i][0], center_location_list[i][1])]
                            for i in range(len(center_location_list))]
                res_list.sort(key=operator.itemgetter(1))
                predict_traj.append([item[0] for item in res_list])
        ground_truth = vali_evl[j]
        for g in range(len(ground_truth)):
            flag = False
            if ((g+1)<len(ground_truth)):
                if (ground_truth[g] != 0) & (ground_truth[g+1]==0):
                    flag = True
            else:
                if ground_truth[g] != 0:
                    flag =True
            if flag:
                ground_g = ground_truth[g] -1
                if ground_g in predict_traj[g][0:1]: hc1 +=1
                if ground_g in predict_traj[g][0:5]: hc5 +=1
                if ground_g in predict_traj[g][0:10]: hc10 +=1
                if ground_g in predict_traj[g][0:15]: hc15 += 1
                if ground_g in predict_traj[g][0:20]: hc20 += 1

                dd = []
                for k in predict_traj[g][0:5]:
                    pred = center_location_list[k]
                    tr = center_location_list[ground_g]
                    d = haversine(pred[0], pred[1], tr[0], tr[1])
                    dd.append(d)
                d = min(dd)
                # print d
                alldistance += d
                count+=1
                if count % 100 == 0: print ("nearest location last",count)
    print ("nearest location last",count)
    print (hc1 , hc5 , hc10, hc15, hc20)
    print [hc1 / count, hc5 / count,
           hc10 / count, hc15 / count, hc20 / count, alldistance / count]

def frequent_location_last(train_X, vali_X, vali_evl, center_location_list):
    all_train_X_pl, all_train_X_user= train_X[0],train_X[2]
    all_test_X_pl, all_test_X_user = vali_X[0],vali_X[2]
    count, hc1 , hc5 , hc10, hc15, hc20, alldistance = 0.,0.,0.,0.,0.,0.,0.
    all_train_X_pl = all_train_X_pl.tolist()

    user_frequent_pl = {}
    for j in range(len(all_train_X_pl)):
        if not user_frequent_pl.has_key(all_train_X_user[j][0]):
            user_frequent_pl[all_train_X_user[j][0]] = np.zeros(len(center_location_list))
        for p in range(len(all_train_X_pl[j])):
            if all_train_X_pl[j][p] != 0:
                user_frequent_pl[all_train_X_user[j][0]][all_train_X_pl[j][p]-1] += 1

    all_test_X_pl = all_test_X_pl.tolist()
    for j in range(len(all_test_X_pl)):
        ground_truth = vali_evl[j]
        user = all_test_X_user[j][0]
        for g in range(len(ground_truth)):
            flag = False
            if ((g+1)<len(ground_truth)):
                if (ground_truth[g] != 0) & (ground_truth[g+1]==0):
                    flag = True
            else:
                if ground_truth[g] != 0:
                    flag =True
            if flag:
                ground_g = ground_truth[g] -1
                if ground_g in user_frequent_pl[user].argsort()[-1:][::-1]: hc1 +=1
                if ground_g in user_frequent_pl[user].argsort()[-5:][::-1]: hc5 +=1
                if ground_g in user_frequent_pl[user].argsort()[-10:][::-1]: hc10 +=1
                if ground_g in user_frequent_pl[user].argsort()[-15:][::-1]: hc15 += 1
                if ground_g in user_frequent_pl[user].argsort()[-20:][::-1]: hc20 += 1

                dd = []
                for k in user_frequent_pl[user].argsort()[-5:][::-1]:
                    pred = center_location_list[k]
                    tr = center_location_list[ground_g]
                    d = haversine(pred[0], pred[1], tr[0], tr[1])
                    dd.append(d)
                d = min(dd)
                alldistance += d
                count+=1
                if count % 100 == 0: print ("frequent location",count)
    print ("frequent location",count)
    print (hc1 , hc5 , hc10, hc15, hc20)
    print [hc1 / count, hc5 / count,
           hc10 / count, hc15 / count, hc20 / count, alldistance / count]
    return [hc1 / count, hc5 / count,
           hc10 / count, hc15 / count, hc20 / count, alldistance / count]

def load_wordvec(vecpath = WORD_VEC_PATH):
    word_vec = {}
    with open(vecpath,'r') as f:
        for l in f:
            vec = []
            attrs = l.replace('\n','').split(' ')
            for i in range(1,len(attrs)):
                vec.append(float(attrs[i]))
            word_vec[attrs[0]] = vec
    return word_vec

def text_feature_generation(user_feature_sequence, dataset='FS'):
    text_vec = load_wordvec()
    useful_vec = {}
    print ("useful data length",len(user_feature_sequence))
    count = 0
    for u in user_feature_sequence.keys():
        features = user_feature_sequence[u]
        for traj_fea in range(len(features)):
            useful_word_sample = []
            for i in range(len(features[traj_fea][2])):
                text = features[traj_fea][2][i]
                words_key = []
                if not text == 0:
                    words = []
                    if dataset=='FS':
                        words = text.split(' ')
                    elif dataset=='LA':
                        words = text.split('\t')
                    for w in words:
                        if (text_vec.has_key(w)) & (not useful_vec.has_key(w)):
                            useful_vec[w] = text_vec[w]
                        if useful_vec.has_key(w):
                            words_key.append(w)
                else: print "Text == 0"
                useful_word_sample.append(words_key)
            user_feature_sequence[u][traj_fea].append(useful_word_sample)
    return user_feature_sequence,useful_vec

def text_features_to_categorical(text_features_train, word_index):
    textf_res = []
    for item in text_features_train:
        if item==0:
            textf_res.append(np.zeros(len(word_index.keys())))
        elif len(item) == 0:
            textf_res.append(np.zeros(len(word_index.keys())))
        else:
            l = len(item)
            vec = np.zeros(len(word_index.keys()))
            for w in item:
                wv =  to_categorical([word_index[w]], len(word_index.keys()))
                vec = vec + wv
            vec = vec / l
            textf_res.append(vec[0])
    return textf_res

def text_features_to_categorical_batch(text_features_train_batch, word_index):
    textf_res_batch = []
    for text_features_train in text_features_train_batch:
        textf_res = text_features_to_categorical(text_features_train, word_index)
        textf_res_batch.append(textf_res)
    return textf_res_batch

def geo_dataset_train_test_text(user_feature_sequence, useful_vec, max_record, place_dim = GRID_COUNT*GRID_COUNT,
                              train_test_part=TRAIN_TEST_PART):

    user_index = {}
    for u in range(len(user_feature_sequence.keys())):
        user_index[user_feature_sequence.keys()[u]] = u
    user_dim = len(user_feature_sequence.keys())

    word_index = {}
    word_vec = []
    for w in range(len(useful_vec.keys())):
        word_index[useful_vec.keys()[w]] = w
        word_vec.append(useful_vec[useful_vec.keys()[w]])
    word_vec = np.array(word_vec)
    print word_vec.shape
    all_train_X_pl, all_train_X_time , all_train_X_user, all_train_X_text , all_train_Y, all_train_evl \
        = [],[],[],[],[],[]
    all_test_X_pl, all_test_X_time, all_test_X_user, all_test_X_text, all_test_Y, all_test_evl \
        = [],[],[],[],[],[]

    for user in user_feature_sequence.keys():
        sequ_features = user_feature_sequence[user]
        train_size = int(len(sequ_features)*train_test_part) + 1
        for sample in range(0,train_size):
            pl_features, time_features, text_features_train \
                = sequ_features[sample][0],sequ_features[sample][1],sequ_features[sample][3]
            pl_train = pl_features[0:len(pl_features)-1]
            time_train = time_features[0:len(time_features)-1]
            user_index_train = [(user_index[user] + 1) for item in range(len(pl_features)-1)]
            text_features_train = text_features_train[0:len(text_features_train)-1]
            while len(pl_train) < (max_record-1):
                pl_train.append(0)
                time_train.append(0)
                user_index_train.append(0)
                text_features_train.append(0)
            train_y = pl_features[1:]
            while len(train_y) < (max_record-1):
                train_y.append(0)
            all_train_X_pl.append(np.array(pl_train))
            all_train_X_time.append(np.array(time_train))
            all_train_X_user.append(np.array(user_index_train))
            all_train_X_text.append(text_features_train)
            all_train_Y.append(train_y)
            all_train_evl.append(train_y)

        for sample in range(train_size,len(sequ_features)):
            pl_features, time_features, text_features_test\
                = sequ_features[sample][0],sequ_features[sample][1],sequ_features[sample][3]
            pl_test = pl_features[0:len(pl_features)-1]
            time_test = time_features[0:len(time_features)-1]
            user_index_test = [(user_index[user] + 1) for item in range(len(pl_features)-1)]
            text_features_test = text_features_test[0:len(text_features_test) - 1]
            while len(pl_test) < (max_record-1):
                pl_test.append(0)
                time_test.append(0)
                user_index_test.append(0)
                text_features_test.append(0)
            test_y = pl_features[1:]
            while len(test_y) < (max_record-1):
                test_y.append(0)
            all_test_X_pl.append(np.array(pl_test))
            all_test_X_time.append(np.array(time_test))
            all_test_X_user.append(np.array(user_index_test))
            all_test_X_text.append(text_features_to_categorical(text_features_test,word_index))
            all_test_Y.append(to_categorical(test_y, num_classes=place_dim + 1))
            all_test_evl.append(test_y)

    print all_train_X_pl[0]
    print all_train_evl[0]
    all_train_X_pl =  np.array(all_train_X_pl)
    all_train_X_time = np.array(all_train_X_time)
    all_train_X_user = np.array(all_train_X_user)
    # all_train_X_text = np.array(all_train_X_text)
    all_train_evl = np.array(all_train_evl)
    all_train_Y =  np.array(all_train_Y)
    all_test_X_pl = np.array(all_test_X_pl)
    all_test_X_time=  np.array(all_test_X_time)
    all_test_X_user = np.array(all_test_X_user)
    all_test_X_text = np.array(all_test_X_text)

    print ("all_train_X_pl,all_train_X_time,all_train_X_user",
           all_train_X_pl.shape,all_train_X_time.shape,all_train_X_user.shape)
    return [all_train_X_pl,all_train_X_time,all_train_X_user,all_train_X_text],np.array(all_train_Y), all_train_evl,\
           [all_test_X_pl, all_test_X_time,all_test_X_user,all_test_X_text], np.array(all_test_Y), all_test_evl, \
           user_dim, word_vec, word_index

def geo_rnn_train_batch_text(train_X, train_Y, vali_X, vali_Y,vali_evl, model,center_location_list,
                             word_index, dataset='FS',epoch=TRAINING_EPOCH):
    place_dim = GRID_COUNT * GRID_COUNT
    for i in range(epoch):
        print ("epoch: ", i)
        model.fit_generator(batch_generator_text(train_X,train_Y,word_index),steps_per_epoch=int(len(train_X[0])/BATCH_SIZE)+1,
                        epochs=1, max_queue_size=7, validation_data=(vali_X,vali_Y),workers=5)
        all_output_array = model.predict(vali_X)
        evaluation_last_with_distance(all_output_array, vali_evl, center_location_list)
        print  './model/' + dataset + '_' + MODEL_NAME + '_' + str(i) + '.h5'
        model.save('./model/' + dataset + '_' + MODEL_NAME + '_' + str(i) + '.h5')

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def batch_generator_text(train_X, train_Y,word_index):
    place_dim = GRID_COUNT * GRID_COUNT
    while 1:
        j = 0
        while j < train_X[0].shape[0]:
            y_b = []
            pl_b, time_b, user_b = train_X[0][j:j+BATCH_SIZE], train_X[1][j:j+BATCH_SIZE], train_X[2][j:j+BATCH_SIZE]
            text_b = np.array(text_features_to_categorical_batch(train_X[3][j:j+BATCH_SIZE], word_index))
            for sample in train_Y[j:j + BATCH_SIZE]:
                y_b.append(to_categorical(sample, num_classes=place_dim + 1))
            yield ([pl_b, time_b, user_b, text_b], np.array(y_b))

            if (j + BATCH_SIZE) > train_X[0].shape[0]:
                y_b= []
                pl_b, time_b, user_b = train_X[0][j:], train_X[1][j:], train_X[2][j:]
                text_b =np.array(text_features_to_categorical_batch( train_X[3][j:], word_index))
                for sample in train_Y[j:]:
                    y_b.append(to_categorical(sample, num_classes=place_dim + 1))
                print (pl_b.shape, time_b.shape, text_b.shape, user_b.shape)
                yield ([pl_b, time_b, user_b, text_b], np.array(y_b))
            j = j + BATCH_SIZE

def check_records_locations(records, th = 0.001):
    lats,lons =[],[]
    for r in records:
        lats.append(float(r[2]))
        lons.append(float(r[3]))
    if ((max(lats)-min(lats))< th) and ((max(lons)-min(lons))< th):
        return False
    else:
        return True


if __name__ == '__main__':
    a= np.array([1,23,4])
    b= np.array([3,54,5])
    print (a+b)/2