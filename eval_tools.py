import time
import random
import numpy as np
from keras.utils.np_utils import to_categorical
from math import radians, cos, sin, asin, sqrt
import config

GRID_COUNT = config.GRID_COUNT
BATCH_SIZE = config.batch_size
MODEL_NAME = config.model_file_name
TEXT_K = config.text_k
random.seed(2017)


def load_wordvec(vecpath = './word_vec/glove.twitter.27B.50d.txt'):
    word_vec = {}
    with open(vecpath,'r') as f:
        for l in f:
            vec = []
            attrs = l.replace('\n','').split(' ')
            for i in range(1,len(attrs)):
                vec.append(float(attrs[i]))
            word_vec[attrs[0]] = vec
    return word_vec

def time_hour(ci_time, form = '%Y-%m-%d %X'):
    st = time.strptime(ci_time, form)
    mounth = st.tm_mon
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
        # t1.tm_isdst()
    if 'CST' in s:
        t1 = time.strptime(s.replace(' CST',''))
        # t1.tm_isdst = -1

    s = time2
    if 'CDT' in s:
        t2 = time.strptime(s.replace(' CDT',''))
        # t2.tm_isdst = 1
    if 'CST' in s:
        t2 = time.strptime(s.replace(' CST',''))
        # t2.tm_isdst = -1

    return abs(int(time.mktime(t1))-int(time.mktime(t2)))

def time_hour_la(ci_time, form = '%Y-%m-%d %X'):
    s = ci_time
    if 'CDT' in s:
        st = time.strptime(s.replace(' CDT',''))
        # t1.tm_isdst()
    if 'CST' in s:
        st = time.strptime(s.replace(' CST',''))
        # t1.tm_isdst = -1
    mounth = st.tm_mon
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

def evaluation(output_array, y_test):
    recall1 = 0.
    recall2 = 0.
    recall3 = 0.
    recall4 = 0.
    recall5 = 0.
    for i in range(len(y_test)):
        true_pl = y_test[i]
        infe_pl = output_array[i]
        top1 = infe_pl.argsort()[-1:][::-1]
        top2 = infe_pl.argsort()[-2:][::-1]
        top3 = infe_pl.argsort()[-3:][::-1]
        top4 = infe_pl.argsort()[-4:][::-1]
        top5 = infe_pl.argsort()[-5:][::-1]
        if true_pl in top1:
            recall1 += 1
        if true_pl in top2:
            recall2 += 1
        if true_pl in top3:
            recall3 += 1
        if true_pl in top4:
            recall4 += 1
        if true_pl in top5:
            recall5 += 1
    recalls = [recall1 / len(y_test), recall2 / len(y_test),
               recall3 / len(y_test), recall4 / len(y_test), recall5 / len(y_test)]
    print recalls

def evaluation_all(all_output_array, all_test_Y):

    all_test_recalls= []
    count = 0
    all_recall1 = 0.
    all_recall2 = 0.
    all_recall3 = 0.
    all_recall4 = 0.
    all_recall5 = 0.
    for j in range(len(all_test_Y)):
        y_test = all_test_Y[j]
        output_array = all_output_array[j]

        recall1 = 0.
        recall2 = 0.
        recall3 = 0.
        recall4 = 0.
        recall5 = 0.
        for i in range(len(y_test)):
            if y_test[i] != 0:
                true_pl = y_test[i]-1
                infe_pl = output_array[i]
                # print true_pl
                # print infe_pl
                # remove place label 0 prob
                top1 = infe_pl[1:].argsort()[-1:][::-1]
                # print top1
                top2 = infe_pl[1:].argsort()[-5:][::-1]
                top3 = infe_pl[1:].argsort()[-10:][::-1]
                top4 = infe_pl[1:].argsort()[-15:][::-1]
                top5 = infe_pl[1:].argsort()[-20:][::-1]
                if true_pl in top1:
                    recall1 += 1
                    all_recall1 += 1
                if true_pl in top2:
                    recall2 += 1
                    all_recall2 += 1
                if true_pl in top3:
                    recall3 += 1
                    all_recall3 += 1
                if true_pl in top4:
                    recall4 += 1
                    all_recall4 += 1
                if true_pl in top5:
                    recall5 += 1
                    all_recall5 += 1
                count += 1
        # recalls = [recall1 / len(y_test), recall2 / len(y_test),
        #            recall3 / len(y_test), recall4 / len(y_test), recall5 / len(y_test)]
        # all_test_recalls.append(recalls)
    # print all_test_recalls
    print [all_recall1 / count, all_recall2 / count,
           all_recall3 / count, all_recall4 / count, all_recall5 / count]

def evaluation_with_distance(all_output_array, all_test_Y, center_location_list):
    all_test_recalls= []
    count = 0
    all_recall1 = 0.
    all_recall2 = 0.
    all_recall3 = 0.
    all_recall4 = 0.
    all_recall5 = 0.
    alldistance = 0.
    for j in range(len(all_test_Y)):
        y_test = all_test_Y[j]
        output_array = all_output_array[j]

        recall1 = 0.
        recall2 = 0.
        recall3 = 0.
        recall4 = 0.
        recall5 = 0.
        for i in range(len(y_test)):
            if y_test[i] != 0:
                true_pl = y_test[i]-1
                infe_pl = output_array[i]
                # print true_pl
                # print infe_pl
                # remove place label 0 prob
                top1 = infe_pl[1:].argsort()[-1:][::-1]
                top2 = infe_pl[1:].argsort()[-5:][::-1]
                top3 = infe_pl[1:].argsort()[-10:][::-1]
                top4 = infe_pl[1:].argsort()[-15:][::-1]
                top5 = infe_pl[1:].argsort()[-20:][::-1]
                topd = infe_pl[1:].argsort()[-5:][::-1].tolist()+ y_test[0:(i-1)]
                dd = []
                for i in topd:
                    pred = center_location_list[i]
                    tr = center_location_list[true_pl]
                    d = haversine(pred[0], pred[1], tr[0], tr[1])
                    dd.append(d)
                d = min(dd)
                # print d
                alldistance += d
                if true_pl in top1:
                    # print d
                    recall1 += 1
                    all_recall1 += 1
                if true_pl in top2:
                    recall2 += 1
                    all_recall2 += 1
                if true_pl in top3:
                    recall3 += 1
                    all_recall3 += 1
                if true_pl in top4:
                    recall4 += 1
                    all_recall4 += 1
                if true_pl in top5:
                    recall5 += 1
                    all_recall5 += 1
                count += 1

        # recalls = [recall1 / len(y_test), recall2 / len(y_test),
        #            recall3 / len(y_test), recall4 / len(y_test), recall5 / len(y_test)]
        # all_test_recalls.append(recalls)
    # print all_test_recalls
    print count
    print all_recall5
    print [all_recall1 / count, all_recall2 / count,
           all_recall3 / count, all_recall4 / count, all_recall5 / count, alldistance/count]
    return [all_recall1 / count, all_recall2 / count,
           all_recall3 / count, all_recall4 / count, all_recall5 / count, alldistance/count]

def evaluation_last_with_distance(all_output_array, all_test_Y, center_location_list):
    all_test_recalls = []
    count = 0
    all_recall1 = 0.
    all_recall2 = 0.
    all_recall3 = 0.
    all_recall4 = 0.
    all_recall5 = 0.
    alldistance = 0.
    for j in range(len(all_test_Y)):
        y_test = all_test_Y[j]
        output_array = all_output_array[j]

        recall1 = 0.
        recall2 = 0.
        recall3 = 0.
        recall4 = 0.
        recall5 = 0.

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
                # print true_pl
                # print infe_pl
                # remove place label 0 prob
                top1 = infe_pl[1:].argsort()[-1:][::-1]
                top2 = infe_pl[1:].argsort()[-5:][::-1]
                top3 = infe_pl[1:].argsort()[-10:][::-1]
                top4 = infe_pl[1:].argsort()[-15:][::-1]
                top5 = infe_pl[1:].argsort()[-20:][::-1]
                # topd = infe_pl[1:].argsort()[-1:][::-1].tolist()+ y_test[0:(i-1)]
                topd = infe_pl[1:].argsort()[-5:][::-1]
                dd = []
                for i in topd:
                    pred = center_location_list[i]
                    tr = center_location_list[true_pl]
                    d = haversine(pred[0], pred[1], tr[0], tr[1])
                    dd.append(d)
                d = min(dd)
                # print d
                alldistance += d
                if true_pl in top1:
                    # print d
                    recall1 += 1
                    all_recall1 += 1
                if true_pl in top2:
                    recall2 += 1
                    all_recall2 += 1
                if true_pl in top3:
                    recall3 += 1
                    all_recall3 += 1
                if true_pl in top4:
                    recall4 += 1
                    all_recall4 += 1
                if true_pl in top5:
                    recall5 += 1
                    all_recall5 += 1
                count += 1

                # recalls = [recall1 / len(y_test), recall2 / len(y_test),
                #            recall3 / len(y_test), recall4 / len(y_test), recall5 / len(y_test)]
                # all_test_recalls.append(recalls)
    # print all_test_recalls
    print count
    print all_recall5
    print [all_recall1 / count, all_recall2 / count,
           all_recall3 / count, all_recall4 / count, all_recall5 / count, alldistance / count]
    return [all_recall1 / count, all_recall2 / count,
            all_recall3 / count, all_recall4 / count, all_recall5 / count, alldistance / count]

def text_features_np(text_samples):
    text_f_array_all = []
    for tt in text_samples:
        if tt == 'no-keys-in-dict':
            text_f_array_all.append(np.zeros(50))
        elif tt == 0 :
            text_f_array_all.append(np.zeros(50))
        else:
            text_f_array_all.append(np.array(tt))
    text_f_array_all = np.array(text_f_array_all)
    # print text_f_array_all.shape
    return text_f_array_all

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
            # print vec.shape
            textf_res.append(vec[0])
    return textf_res


def geo_dataset_train_test_text(user_feature_sequence, useful_vec, max_record, place_dim = GRID_COUNT*GRID_COUNT,
                              train_test_part=0.8, textf_path = './data/useful_record_fs_textf'):


    # for user in user_feature_sequence.keys():
    #     sequ_features = user_feature_sequence[user]
    #     if len(sequ_features) != len(text_features):
    #         print ('False',len(sequ_features), len(text_features))
    #         del user_feature_sequence[user]
    #         del user_text_feature_sequence[user]

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
    all_train_X_pl = []
    all_train_X_time = []
    all_train_X_user = []
    all_train_X_text = []
    all_train_Y = []
    all_train_evl = []

    all_test_X_pl = []
    all_test_X_time = []
    all_test_X_user = []
    all_test_X_text = []
    all_test_Y = []
    all_test_evl = []

    for user in user_feature_sequence.keys():
        sequ_features = user_feature_sequence[user]
        train_size = int(len(sequ_features)*train_test_part) + 1
        for sample in range(0,train_size):
            [pl_features, time_features,text_fff, text_features_train] = sequ_features[sample]
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
            all_train_X_text.append(text_features_to_categorical(text_features_train, word_index))
            all_train_Y.append(train_y)
            all_train_evl.append(train_y)

        for sample in range(train_size,len(sequ_features)):
            # print len(sequ_features),len(text_features), sample
            # text_features_test = text_features[sample]
            [pl_features, time_features,text_fff, text_features_test] = sequ_features[sample]

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
    all_train_X_text = np.array(all_train_X_text)
    all_train_evl = np.array(all_train_evl)
    all_train_Y =  np.array(all_train_Y)
    all_test_X_pl = np.array(all_test_X_pl)
    all_test_X_time=  np.array(all_test_X_time)
    all_test_X_user = np.array(all_test_X_user)
    all_test_X_text = np.array(all_test_X_text)
    # print dataset shape
    # print all_train_X_pl.shape, all_train_X_user.shape, all_train_X_time.shape, all_train_evl.shape
    print ("all_train_X_pl,all_train_X_time,all_train_X_user,all_train_X_text",
           all_train_X_pl.shape,all_train_X_time.shape,all_train_X_user.shape,all_train_X_text.shape)
    return [all_train_X_pl,all_train_X_time,all_train_X_user,all_train_X_text],np.array(all_train_Y), all_train_evl,\
           [all_test_X_pl, all_test_X_time,all_test_X_user,all_test_X_text], np.array(all_test_Y), all_test_evl, \
           user_dim,word_vec

def geo_rnn_train_batch_text(train_X, train_Y, vali_X, vali_Y,vali_evl, model,center_location_list,dataset='FS',
    epoch=1000):
    place_dim = GRID_COUNT * GRID_COUNT
    for i in range(epoch):
        print ("epoch: ", i)
        model.fit_generator(batch_generator_text(train_X,train_Y),steps_per_epoch=int(len(train_X[0])/BATCH_SIZE)+1,
                        epochs=1, max_queue_size=5, validation_data=(vali_X,vali_Y))
        all_output_array = model.predict(vali_X)
        evaluation_with_distance(all_output_array, vali_evl, center_location_list)
        evaluation_last_with_distance(all_output_array, vali_evl, center_location_list)
        model.save('./model/' + dataset + '_' + MODEL_NAME + '_' + str(i) + '.h5')

def batch_generator_text(train_X, train_Y):
    place_dim = GRID_COUNT * GRID_COUNT
    while 1:
        j = 0
        while j < train_X[0].shape[0]:
            # print ("Batched sample num:", j, train_X[0].shape[0])
            y_b = []
            pl_b, time_b, user_b = train_X[0][j:j+BATCH_SIZE], train_X[1][j:j+BATCH_SIZE], train_X[2][j:j+BATCH_SIZE]
            text_b = train_X[3][j:j+BATCH_SIZE]
            for sample in train_Y[j:j + BATCH_SIZE]:
                y_b.append(to_categorical(sample, num_classes=place_dim + 1))
            yield ([pl_b, time_b, user_b, text_b], np.array(y_b))

            if (j + BATCH_SIZE) > train_X[0].shape[0]:
                # print ("This is the end batch of epoch")
                y_b= []
                pl_b, time_b, user_b = train_X[0][j:], train_X[1][j:], train_X[2][j:]
                text_b = train_X[3][j:]
                for sample in train_Y[j:]:
                    y_b.append(to_categorical(sample, num_classes=place_dim + 1))
                print (pl_b.shape, time_b.shape, text_b.shape, user_b.shape)
                yield ([pl_b, time_b, user_b, text_b], np.array(y_b))
            j = j + BATCH_SIZE



if __name__ == '__main__':
    # load_wordvec()
    a= np.array([1,23,4])
    b= np.array([3,54,5])
    print (a+b)/2