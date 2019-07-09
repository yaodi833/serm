import cPickle
from model.serm_models import geo_lprnn_model,geo_lprnn_text_model,geo_lprnn_trainable_text_model
from eval_tools import *
import config

TWEET_PATH = config.TWEET_PATH
POI_PATH = config.POI_PATH
LA_TWEETS =  config.LA_TWEETS
GRID_COUNT = config.GRID_COUNT
BATCH_SIZE = config.batch_size
WINDOW_SIZE = config.window_size
MIN_SEQ = config.min_seq_num
MAX_SEQ = config.max_seq_num
MIN_TRAJ = config.min_traj_num
RECORD_TH = config.threshold


def decode_data_fs(threshold=RECORD_TH):
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

def geo_data_clean_fs(w = WINDOW_SIZE, min_seq_num = MIN_SEQ, min_traj_num = MIN_TRAJ, locationtpye ='GRADE',
                      gridc = GRID_COUNT):
    poi_attr, user_ci, poi_catecology_dict = decode_data_fs()
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

def decode_data_la(threshold = RECORD_TH):
    tsf = open(LA_TWEETS)
    tsfls = tsf.readlines()
    print tsfls[0].split('')
    x = []
    y = []
    for l in tsfls:
        attrs = l.split('')
        x.append(float(attrs[2]))
        y.append(float(attrs[3]))
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
        if (len(user_cis[u])>= threshold):
            useful_user_cis[u] = user_cis[u]
    print ("Num of users:", len(useful_user_cis.keys()))
    print ("Num of pois:", len(user_poi.keys()))
    return user_poi,useful_user_cis

def geo_data_clean_la(w = WINDOW_SIZE,min_seq_num = MIN_SEQ, max_seq_num = MAX_SEQ, min_traj_num = MIN_TRAJ,
                   locationtpye = 'GRADE', gridc = GRID_COUNT):
    poi_attr, user_ci = decode_data_la()
    users = user_ci.keys()
    user_record_sequence = {}
    useful_poi_dict = {}
    user_feature_sequence = {}

    # use W and min_traj_num filter data
    for user in users:
        ci_records = user_ci[user]
        clean_records = []
        traj_records = []
        perious_record = None
        for record in ci_records:
            try:
                if perious_record == None:
                    perious_record = record

                time = record[4]
                dif = time_diff_la(time,perious_record[4])
                if dif<0: print "Fasle"
                if (dif< w) & (dif>0):
                    traj_records.append(record)
                else:
                    if (len(traj_records)>min_seq_num) & (len(traj_records)<max_seq_num):
                        if check_records_locations(traj_records):
                            clean_records.append(traj_records)
                    traj_records = []
                perious_record = record
            except Exception as e:
                print e
        if (len(traj_records)>0) & (len(traj_records)>min_seq_num) & (len(traj_records)<max_seq_num):
            if check_records_locations(traj_records):
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
            text_features = []
            if seg_max_record < len(traj):
                seg_max_record = len(traj)
            if len(traj) >100:
                for r in traj:
                    print r
            for record in traj:
                pl_features.append(poi_index_dict[record[0]]+1)
                time_features.append(time_hour_la(record[4])+1)
                text_features.append(record[6])
            all_sequ_features.append([pl_features,time_features,text_features])
        user_feature_sequence[user] = all_sequ_features
    print 'seg_max_record, pois_num, user_num'
    print seg_max_record, len(poi_index_dict.keys()),len(user_feature_sequence.keys())
    grid_count = {}
    for poi in poi_index_dict.keys():
        if not grid_count.has_key(poi_index_dict[poi]):
            grid_count[poi_index_dict[poi]] = 1
        else:
            grid_count[poi_index_dict[poi]] += 1
    print ("userful poi nums:",len(grid_count.keys()))

    user_feature_sequence_text, useful_vec= text_feature_generation(user_feature_sequence,dataset='LA')
    cPickle.dump((user_feature_sequence_text, poi_index_dict,seg_max_record, center_location_list, useful_vec),
                 open('./features/features&index_seg_gride_la', 'w'))

    return user_feature_sequence_text, poi_index_dict, seg_max_record, center_location_list, useful_vec
