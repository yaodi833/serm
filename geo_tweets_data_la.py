import time
import cPickle
from geo_tweets_data_fs import geo_dataset_pre, geo_lprnn_model,geo_rnn_train

LA_TWEETS = './data/la_tweets.txt'
GRID_COUNT = 100

def geo_grid_list(x,y,m_nGridCount = GRID_COUNT):
    dXMax = max(x)
    dXMin = min(x)
    dYMax = max(y)
    dYMin = min(y)
    print dXMax,dXMin,dYMax,dYMin
    m_dOriginX = dXMin
    m_dOriginY = dYMin
    dSizeX = (dXMax - dXMin) / m_nGridCount
    dSizeY = (dYMax - dYMin) / m_nGridCount
    # m_vIndexCells = []
    # for i in range(0, m_nGridCount * m_nGridCount + 1):
    #     m_vIndexCells.append([])
    # m_vIndexCells = [list()] * (m_nGridCount * m_nGridCount)
    print m_nGridCount, m_dOriginX, m_dOriginY, \
        dSizeX, dSizeY

    poi_index = []
    grid_index = {}
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
        poi_index.append(iIndex)
        if grid_index.has_key(iIndex):
            grid_index[iIndex].append((x[i],y[i]))
        else:
            grid_index[iIndex] = []
            grid_index[iIndex].append((x[i], y[i]))
    # cPickle.dump(poi_index_dict, open('./features/poi_index_dict', 'w'))
    # color = ['b','g','r','c','m','y','k']
    # for k in grid_index.keys():
    #     c = random.randint(0,6)
    #     poii = grid_index[k]
    #     xx = []
    #     yy = []
    #     for it in poii:
    #         xx.append(it[0])
    #         yy.append(it[1])
    #     plt.plot(xx,yy,'.'+color[c])
    # plt.show()

    return poi_index, grid_index

def decode_lat_tweets(threshold = 50):

    tsf = open(LA_TWEETS)
    tsfls = tsf.readlines()
    print tsfls[0].split('')
    x = []
    y = []
    for l in tsfls:
        attrs = l.split('')
        # print len(poifs)
        x.append(float(attrs[2]))
        y.append(float(attrs[3]))

    useful_user_poi = {}
    useful_user_cis = {}
    user_cis = {}
    user_poi = {}

    for i in range(len(tsfls)):
        l = tsfls[i]
        cifs = l.replace('\n', '').split('')

        if user_cis.has_key(cifs[1]):
            user_cis[cifs[1]].append(cifs)
        else:
            user_cis[cifs[1]] = []
            user_cis[cifs[1]].append(cifs)

        user_poi[cifs[0]] = [float(cifs[2]), float(cifs[3])]

    for u in user_cis.keys():
        if len(user_cis[u])>= threshold:
            useful_user_cis[u] = user_cis[u]
            # useful_user_poi[u] = user_poi[u]

    # useful_pois = {}
    # for u in user_poi.keys():
    #     for i in user_poi[u]:
    #         if useful_pois.has_key(i):
    #             useful_pois[i] +=1
    #         else:
    #             useful_pois[i] = 1

    print ("Num of users:", len(useful_user_cis))
    print ("Num of pois:", len(useful_user_poi))
    # cPickle.dump(poi_cis, open('./features/poi_checkins_dict', 'w'))
    # cPickle.dump(user_cis,open('./features/user_checkins_dict','w'))
    return user_poi,useful_user_cis

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


def geo_data_clean(w = 36000,min_seq_num = 3, max_seq_num = 70, min_traj_num = 5,locationtpye = 'GRADE', \
                                                                                            gridc = GRID_COUNT):
    poi_attr, user_ci = decode_lat_tweets()
    users = user_ci.keys()
    user_record_sequence = {}
    useful_poi_dict = {}
    user_feature_sequence = {}

    # use W and min_traj_num filter data
    for user in users:
        ci_records = user_ci[user]
        # ci_records.reverse()
        clean_records = []
        traj_records = []
        perious_record = None
        for record in ci_records:
            try:
                if perious_record == None:
                    perious_record = record

                time = record[4]
                dif = time_diff(time,perious_record[4])
                if dif<0: print "Fasle"
                if (dif< w) & (dif>0):
                    # print time_diff(time,perious_record[4])
                    traj_records.append(record)
                else:
                    if (len(traj_records)>min_seq_num) & (len(traj_records)<max_seq_num):
                        clean_records.append(traj_records)
                    traj_records = []
                perious_record = record
            except Exception as e:
                print e
        if (len(traj_records)>0) & (len(traj_records)>min_seq_num) & (len(traj_records)<max_seq_num):
            clean_records.append(traj_records)

        if (len(clean_records)>min_traj_num):
            user_record_sequence[user] = clean_records

    # generate useful pois
    for user in user_record_sequence.keys():
        trajs = user_record_sequence[user]
        for traj in trajs:
            for record in traj:
                if not useful_poi_dict.has_key(record[0]):
                    useful_poi_dict[record[0]] = []
                    useful_poi_dict[record[0]].append(record)

    # generate poi dict
    if locationtpye == 'GRADE':
        index,x,y = [],[],[]
        for i in useful_poi_dict.keys():
            poifs = poi_attr[i]
            index.append(i)
            # print poifs
            x.append(float(poifs[0]))
            y.append(float(poifs[1]))
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
            if len(traj) >100:
                for r in traj:
                    print r
            for record in traj:
                pl_features.append(poi_index_dict[record[0]]+1)
                time_features.append(time_hour(record[4])+1)
            all_sequ_features.append((pl_features,time_features))
        user_feature_sequence[user] = all_sequ_features
    print 'seg_max_record, pois_num, user_num'
    print seg_max_record, len(poi_index_dict.keys()),len(user_feature_sequence.keys())
    grid_count = {}
    for poi in poi_index_dict.keys():
        if not grid_count.has_key(poi_index_dict[poi]):
            grid_count[poi_index_dict[poi]] = 1
        else:
            grid_count[poi_index_dict[poi]] += 1
    print ("grid count:", grid_count)
    print ("userful poi nums:",len(grid_count.keys()))
    cPickle.dump((user_feature_sequence, poi_index_dict), open('./features/chao_features&index_seg_gride_la', 'w'))
    return user_feature_sequence, poi_index_dict, seg_max_record, center_location_list

def time_diff(time1,time2,form = '%Y-%m-%d %X'):
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

def time_hour(ci_time, form = '%Y-%m-%d %X'):
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
    return hour

if __name__ == '__main__':
    user_feature_sequence, place_index, seg_max_record, center_location_list = geo_data_clean()
    print len(user_feature_sequence.keys())
    train_X, train_Y, train_evl, vali_X, vali_Y, vali_evl,user_dim\
        = geo_dataset_pre(user_feature_sequence,seg_max_record)
    model = geo_lprnn_model(user_dim,seg_max_record)
    geo_rnn_train(train_X, train_Y,vali_X, vali_Y,vali_evl, model, center_location_list)
