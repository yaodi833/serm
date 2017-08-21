import time
import random

from math import radians, cos, sin, asin, sqrt


random.seed(2017)

def time_hour(ci_time, form = '%Y-%m-%d %X'):
    st = time.strptime(ci_time, form)
    mounth = st.tm_mon
    weekday = st.tm_wday
    hour = st.tm_hour
    return hour

def time_diff(time1,time2,form = '%Y-%m-%d %X'):
    time11 = time.strptime(time1, form)
    time22 = time.strptime(time2, form)
    return abs(int(time.mktime(time11))-int(time.mktime(time22)))


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

                dd = []
                for i in top1:
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

