import cPickle
from model.rnn_model_keras import geo_lprnn_model,geo_lprnn_text_model,geo_lprnn_trainable_text_model
from eval_tools import *
import config
from trainable_text_feature_generator import text_feature_generation

TWEET_PATH = './data/tweets.txt'
POI_PATH = './data/venues.txt'
GRID_COUNT = config.GRID_COUNT
BATCH_SIZE = config.batch_size


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
    for p in useful_poi.keys():
        poifs = pois[p]
        x.append(float(poifs[1]))
        y.append(float(poifs[2]))
        index.append(poifs[0])

    print ('POI nums',len(useful_poi.keys()))
    print ('User nums',len(useful_user_cis.keys()))

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
            text_features = []
            if seg_max_record < len(traj):
                seg_max_record = len(traj)
            for record in traj:
                pl_features.append(poi_index_dict[record[8]]+1)
                time_features.append(time_hour(record[4])+1)
                text_features.append(record[6])
            all_sequ_features.append([pl_features,time_features,text_features])
        user_feature_sequence[user] = all_sequ_features
    print 'seg_max_record, pois_num, user_num'
    print seg_max_record, len(poi_index_dict.keys()),len(user_feature_sequence.keys())

    user_feature_sequence_text, useful_vec= text_feature_generation(user_feature_sequence)

    cPickle.dump((user_feature_sequence_text, poi_index_dict, seg_max_record, center_location_list, useful_vec),
                 open('./features/features&index_seg_gride_fs', 'w'))

    return user_feature_sequence_text, poi_index_dict, seg_max_record, center_location_list, useful_vec

if __name__ == '__main__':
    user_feature_sequence, place_index, seg_max_record, center_location_list,useful_vec = geo_data_clean()
    print len(user_feature_sequence.keys())
    train_X, train_Y, train_evl, vali_X, vali_Y, vali_evl, user_dim, word_vec \
        = geo_dataset_train_test_text(user_feature_sequence,useful_vec, seg_max_record)
    print ("Feature generation completed")
    model =geo_lprnn_trainable_text_model(user_dim,seg_max_record,word_vec)
    # model.load_weights('./model/FS_User_RNN_Seg_Epoch_0.001_100_rmsprop_55.h5')
    all_output_array = model.predict(vali_X)
    evaluation_with_distance(all_output_array, vali_evl, center_location_list)
    evaluation_last_with_distance(all_output_array, vali_evl, center_location_list)
    print ("Train_x[0] shape:", train_X[0][0:200].shape)
    print ("Train_x[0] shape:", train_X[1].shape)
    print ("Train_x[0] shape:", train_X[2].shape)
    print ("Train_Y shape:", train_Y.shape)
    geo_rnn_train_batch_text(train_X, train_Y, vali_X, vali_Y, vali_evl, model, center_location_list,
                             dataset='FS_mLSTM_')


    # for i in range(2,50,2):
    #     print ("model name",i)
    #     fn = './model/FS_200_50_50_0.01_'+str(i)+'.h5'
    #     model.load_weights(fn)
    #     all_output_array = model.predict(vali_X)
    #     # evaluation_with_distance(all_output_array, vali_evl, center_location_list)
    #     evaluation_with_distance(all_output_array, vali_evl, center_location_list)
    #     evaluation_last_with_distance(all_output_array, vali_evl, center_location_list)