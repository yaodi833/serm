import numpy as np
import cPickle
from model.rnn_model_keras import geo_lprnn_model
from keras.utils.np_utils import to_categorical
from eval_tools import time_diff,time_hour,evaluation_with_distance
import config

TWEET_PATH = './data/tweets.txt'
POI_PATH = './data/venues.txt'
GRID_COUNT = config.GRID_COUNT

def geo_grade(index,x,y,m_nGridCount = GRID_COUNT):
    dXMax = max(x)
    dXMin = min(x)
    dYMax = max(y)
    dYMin = min(y)
    print dXMax,dXMin,dYMax,dYMin
    m_dOriginX = dXMin
    m_dOriginY = dYMin
    dSizeX = (dXMax - dXMin) / m_nGridCount
    dSizeY = (dYMax - dYMin) / m_nGridCount
    m_vIndexCells = []
    center_location_list = []
    for i in range(0, m_nGridCount * m_nGridCount + 1):
        m_vIndexCells.append([])
        y_ind = int(i / m_nGridCount)
        x_ind = i - y_ind*m_nGridCount
        center_location_list.append((dXMin+x_ind*dSizeX+0.5*dSizeX,dYMin+y_ind*dSizeY+0.5*dSizeY))
        # print (dXMin+x_ind*dSizeX+0.5*dSizeX,dYMin+y_ind*dSizeY+0.5*dSizeY)
    # m_vIndexCells = [list()] * (m_nGridCount * m_nGridCount)
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
        m_vIndexCells[iIndex].append([index[i],x[i],y[i]])

    # cell_center = []
    # for i in range(len(m_vIndexCells)):
    #     cell = m_vIndexCells[i]
    #     x,y = 0
    #     if len(cell)==0:
    #         cell_center.append(x,y)
    #     else:
    #         for r in cell:
    #             x += r[1]
    #             y += r[2]
    #         cell_center.append([x/len(cell),y/len(cell)])

    # cPickle.dump(poi_index_dict, open('./features/poi_index_dict', 'w'))
    # plot POIS
    # color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    # for k in m_vIndexCells:
    #     c = random.randint(0, 6)
    #     poii = k
    #     xx = []
    #     yy = []
    #     for it in poii:
    #         xx.append(it[1])
    #         yy.append(it[2])
    #     plt.plot(xx, yy, '.' + color[c])
    # plt.show()

    return poi_index_dict, center_location_list

def decode_data(threshold=50):
    tsf = open(TWEET_PATH)
    poif = open(POI_PATH)
    pois = {}
    index = []
    x = []
    y = []
    for l in poif:
        poifs = l.split(',')
        # print len(poifs)
        if len(poifs)>5:
            print 'error'
        pois[poifs[0]] = poifs

    # cPickle.dump(pois, open('./features/poi_attr_dict','w'))
    useful_poi = {}
    useful_user_cis = {}
    user_cis = {}
    poi_cis = {}
    poi_catecology_dict = {}
    tsfls = tsf.readlines()
    for l in tsfls:
        cifs = l.replace('\n', '').split('')
        if pois.has_key(cifs[8]):
            if poi_cis.has_key(cifs[8]) :
                poi_cis[cifs[8]].append(cifs)
            else:
                poi_cis[cifs[8]] = []
                poi_cis[cifs[8]].append(cifs)

            if user_cis.has_key(cifs[1]):
                user_cis[cifs[1]].append(cifs)
            else:
                user_cis[cifs[1]] = []
                user_cis[cifs[1]].append(cifs)

            if poi_catecology_dict.has_key(pois[cifs[8]][3]):
                poi_catecology_dict[pois[cifs[8]][3]].append(pois[cifs[8]])
            else:
                poi_catecology_dict[pois[cifs[8]][3]] = []
                poi_catecology_dict[pois[cifs[8]][3]].append(pois[cifs[8]])

    for u in user_cis.keys():
        if len(user_cis[u])>= threshold:
            useful_user_cis[u] = user_cis[u]
            for r in user_cis[u]:
                if not useful_poi.has_key(r[8]):
                    useful_poi[r[8]] = pois[r[8]]
    # cPickle.dump(poi_cis, open('./features/poi_checkins_dict', 'w'))
    # cPickle.dump(user_cis,open('./features/user_checkins_dict','w'))

    for p in useful_poi.keys():
        poifs = pois[p]
        x.append(float(poifs[1]))
        y.append(float(poifs[2]))
        index.append(poifs[0])
    # plt.plot(x,y,'.')
    # plt.show()
    print ('POI nums',len(useful_poi.keys()))
    print ('User nums',len(useful_user_cis.keys()))
    # poi_index_dict = geo_grade(index,x,y)

    return useful_poi,useful_user_cis, poi_catecology_dict

def geo_data_clean(w = 36000,min_seq_num = 3, min_traj_num = 5,locationtpye = 'GRADE', gridc = GRID_COUNT):
    poi_attr, user_ci, poi_catecology_dict = decode_data()
    users = user_ci.keys()
    user_record_sequence = {}
    useful_poi_dict = {}
    user_feature_sequence = {}

    # use W and min_traj_num filter data
    for user in users:
        ci_records = user_ci[user]
        ci_records.reverse()
        clean_records = []
        traj_records = []
        perious_record = None
        for record in ci_records:
            try:
                if perious_record == None:
                    perious_record = record

                time = record[4]
                if time_diff(time,perious_record[4])< w:
                    traj_records.append(record)
                else:
                    if len(traj_records)>min_seq_num:
                        clean_records.append(traj_records)
                    traj_records = []
                perious_record = record
            except Exception as e:
                print e
        if (len(traj_records)>0) & (len(traj_records)>min_seq_num):
            clean_records.append(traj_records)

        if len(clean_records)>min_traj_num:
            user_record_sequence[user] = clean_records

    # generate useful pois
    for user in user_record_sequence.keys():
        trajs = user_record_sequence[user]
        for traj in trajs:
            for record in traj:
                if not useful_poi_dict.has_key(record[8]):
                    useful_poi_dict[record[8]] = []
                    useful_poi_dict[record[8]].append(record)

    # generate poi dict
    if locationtpye == 'GRADE':
        index,x,y = [],[],[]
        for i in useful_poi_dict.keys():
            poifs = poi_attr[i]
            index.append(i)
            x.append(float(poifs[1]))
            y.append(float(poifs[2]))
        poi_index_dict, center_location_list = geo_grade(index, x, y, m_nGridCount=gridc)
    elif locationtpye == 'LOCS':
        poi_index_dict = {}
        locs = useful_poi_dict.keys()
        for p in range(len(locs)):
            poifs = locs[p]
            poi_index_dict[poifs] = p


    print ("POI Dim", len(poi_index_dict.keys()))
    seg_max_record = 0

    for user in user_record_sequence.keys():
        all_sequ_features = []
        for traj in user_record_sequence[user]:
            pl_features = []
            time_features = []
            if seg_max_record < len(traj):
                seg_max_record = len(traj)
            for record in traj:
                pl_features.append(poi_index_dict[record[8]]+1)
                time_features.append(time_hour(record[4])+1)
            all_sequ_features.append((pl_features,time_features))
        user_feature_sequence[user] = all_sequ_features
    print 'seg_max_record, pois_num, user_num'
    print seg_max_record, len(poi_index_dict.keys()),len(user_feature_sequence.keys())
    cPickle.dump((user_feature_sequence, poi_index_dict), open('./features/chao_features&index_seg_gride', 'w'))
    return user_feature_sequence, poi_index_dict, seg_max_record, center_location_list

def geo_dataset_pre(user_feature_sequence, max_record, place_dim = GRID_COUNT*GRID_COUNT, train_test_part=0.8):

    user_index = {}
    for u in range(len(user_feature_sequence.keys())):
        user_index[user_feature_sequence.keys()[u]] = u
    user_dim = len(user_feature_sequence.keys())

    all_train_X_pl = []
    all_train_X_time = []
    all_train_X_user = []
    all_train_Y = []
    all_train_evl = []

    all_test_X_pl = []
    all_test_X_time = []
    all_test_X_user = []
    all_test_Y = []
    all_test_evl = []

    for user in user_feature_sequence.keys():
        sequ_features = user_feature_sequence[user]
        train_size = int(len(sequ_features)*train_test_part) + 1
        for sample in range(0,train_size):
            pl_features, time_features = sequ_features[sample]
            pl_train = pl_features[0:len(pl_features)-1]
            time_train = time_features[0:len(time_features)-1]
            user_index_train = [(user_index[user] + 1) for item in range(len(pl_features)-1)]

            while len(pl_train) < (max_record-1):
                pl_train.append(0)
                time_train.append(0)
                user_index_train.append(0)
            train_y = pl_features[1:]
            while len(train_y) < (max_record-1):
                train_y.append(0)
            all_train_X_pl.append(np.array(pl_train))
            all_train_X_time.append(np.array(time_train))
            all_train_X_user.append(np.array(user_index_train))
            all_train_Y.append(to_categorical(train_y, num_classes=place_dim + 1))
            all_train_evl.append(train_y)

        for sample in range(train_size,len(sequ_features)):
            pl_features,time_features = sequ_features[sample]
            pl_test = pl_features[0:len(pl_features)-1]
            time_test = time_features[0:len(time_features)-1]
            user_index_test = [(user_index[user] + 1) for item in range(len(pl_features)-1)]

            while len(pl_test) < (max_record-1):
                pl_test.append(0)
                time_test.append(0)
                user_index_test.append(0)
            test_y = pl_features[1:]
            while len(test_y) < (max_record-1):
                test_y.append(0)
            all_test_X_pl.append(np.array(pl_test))
            all_test_X_time.append(np.array(time_test))
            all_test_X_user.append(np.array(user_index_test))
            all_test_Y.append(to_categorical(test_y, num_classes=place_dim + 1))
            all_test_evl.append(test_y)

    print all_train_X_pl[0]
    print all_train_evl[0]
    all_train_X_pl =  np.array(all_train_X_pl)
    all_train_X_time = np.array(all_train_X_time)
    all_train_X_user = np.array(all_train_X_user)
    all_train_evl = np.array(all_train_evl)
    all_train_Y =  np.array(all_train_Y)
    all_test_X_pl = np.array(all_test_X_pl)
    all_test_X_time=  np.array(all_test_X_time)
    all_test_X_user = np.array(all_test_X_user)
    # print dataset shape
    # print all_train_X_pl.shape, all_train_X_user.shape, all_train_X_time.shape, all_train_evl.shape

    return [all_train_X_pl,all_train_X_time,all_train_X_user],np.array(all_train_Y), all_train_evl,\
           [all_test_X_pl, all_test_X_time,all_test_X_user], np.array(all_test_Y), all_test_evl, user_dim

if __name__ == '__main__':
    user_feature_sequence, place_index, seg_max_record, center_location_list = geo_data_clean()
    print len(user_feature_sequence.keys())
    train_X, train_Y, train_evl, vali_X, vali_Y, vali_evl,user_dim\
        = geo_dataset_pre(user_feature_sequence,seg_max_record)
    model = geo_lprnn_model(user_dim,seg_max_record)