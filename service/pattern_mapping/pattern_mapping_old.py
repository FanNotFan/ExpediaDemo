import re
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import scipy.spatial.distance as dis
from sklearn.linear_model import LinearRegression
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import filters
from scipy.sparse import csgraph
from scipy.sparse import csr_matrix
from scipy import stats
import matplotlib.pyplot as plt, seaborn
from io import StringIO
import cProfile
import os
import glob
import math
from graphviz import Source
from tools import logger
from sklearn.model_selection import train_test_split
from settings import OUTPUT_RESULT_FILE_NAME, DEBUG_LOG_PATH
from settings import HOME_FOLDER, HOTEL_PATTERN_OUTPUT_FOLDER
from settings import PATTERN_MAPPING_INPUT_FOLDER, PATTERN_ATTRIBUTE_INPUT_FOLDER2, PATTERN_MAPPING_OUTPUT_FOLDER
plt.rcParams.update({'figure.max_open_warning': 0})
min_max_scaler = preprocessing.MinMaxScaler()
logger = logger.Logger("debug")

# https://www.cs.princeton.edu/courses/archive/spring03/cs226/assignments/lines.html
class PatternMapping(object):
    __group_id = 1
    def __init__(self, hotel_id, observe, group_id, ratePlanLevel, lengthOfStayDayCnt, person_cnt, **kw):
        self.hotel_id = hotel_id
        self.observe = observe
        self.group_id = group_id
        self.ratePlanLevel = ratePlanLevel
        self.lengthOfStayDayCnt = lengthOfStayDayCnt
        self.person_cnt = person_cnt
        for k, v in kw.items():
            setattr(self, k, v)

    def maxPoints(self, points):
        """
        :type points: List[List[int]]
        :rtype: int
        """
        def max_points_on_a_line_containing_point_i(i):
            """
            Compute the max number of points
            for a line containing point i.
            """
            def slope_coprime(x1, y1, x2, y2):
                """ to avoid the precision issue with the float/double number,
                    using a pair of co-prime numbers to represent the slope.
                """
                delta_x, delta_y = x1 - x2, y1 - y2
                if delta_x == 0:  # vertical line
                    return (0, 0)
                elif delta_y == 0:  # horizontal line
                    return (sys.maxsize, sys.maxsize)
                elif delta_x < 0:
                    # to have a consistent representation,
                    #   keep the delta_x always positive.
                    delta_x, delta_y = - delta_x, - delta_y
                gcd = math.gcd(round(delta_x), round(delta_y))
                slope = (round(delta_x), round(delta_y)) if gcd == 0 else (round(delta_x) / gcd, round(delta_y) / gcd)
                slope_size = delta_y / delta_x
                return slope, slope_size

            def add_line(i, j, count, duplicates, slope_size, slope):
                """
                Add a line passing through i and j points.
                Update max number of points on a line containing point i.
                Update a number of duplicates of i point.
                """
                # rewrite points as coordinates
                x1 = points[i][0]
                y1 = points[i][1]
                x2 = points[j][0]
                y2 = points[j][1]

                # add a duplicate point
                if x1 == x2 and y1 == y2:
                    duplicates += 1
                # add a horisontal line : y = const
                elif y1 == y2:
                    nonlocal horizontal_lines
                    horizontal_lines += 1
                    count = max(horizontal_lines, count)
                # add a line : x = slope * y + c
                # only slope is needed for a hash-map
                # since we always start from the same point
                else:
                    slope_temp, slope_size_temp = slope_coprime(x1, y1, x2, y2)
                    lines[slope_temp] = lines.get(slope_temp, 1) + 1
                    if lines_dots.get(slope_temp) is None:
                        #                         lines_dots[slope_temp] = set(((x1,y1),(x2,y2)))
                        lines_dots[slope_temp] = set(((x1, y1), (x2, y2)))
                    else:
                        lines_dots[slope_temp].update([(x1, y1), (x2, y2)])
                    if lines[slope_temp] > count:
                        count = lines[slope_temp]
                        slope = slope_temp
                        slope_size = slope_size_temp
                return count, duplicates, slope_size, slope

            # init lines passing through point i
            lines, horizontal_lines = {}, 1
            # One starts with just one point on a line : point i.
            # There is no duplicates of a point i so far.
            count, duplicates, slope_size, slope = 1, 0, 0, ()

            # Compute lines passing through point i (fixed)
            # and point j (interation).
            # Update in a loop the number of points on a line
            # and the number of duplicates of point i.
            for j in range(i + 1, n):
                count, duplicates, slope_size, slope = add_line(i, j, count, duplicates, slope_size, slope)
            return count + duplicates, slope_size, slope

        # If the number of points is less than 3
        # they are all on the same line.
        n = len(points)
        print("len(points):{}".format(n))
        if n < 3:
            return n

        # 存取同一斜率下的所有点
        # Compute in a loop a max number of points
        # on a line containing point i.
        intercept, max_count, lines_dots, max_index = 0, 1, {}, 0

        for i in range(n - 1):
            max_point_result = max_points_on_a_line_containing_point_i(i)
            if max_point_result[0] > max_count:
                max_count = max_point_result[0]
                slope_size = max_point_result[1]
                slope = max_point_result[2]
                max_index = i
        # dots_len = len(lines_dots.get(slope))
        # print("lines_dots.len:{}".format(dots_len))
        # Method One of get intercept
        #         for iterm in lines_dots.get(slope):
        #             intercept+=iterm[1] - (iterm[0]*slope_size)
        #         return max_count, slope_size, intercept/dots_len

        # Method two of get intercept
        #         print(points[max_index])
        #         intercept = points[max_index][1] - (slope[0]/slope[1]) * points[max_index][0]
        #         return max_count, slope_size, intercept

        # Method three of get intercept(most accuracy)
        point_of_line = points[max_index]
        print("one point of line is :{}".format(point_of_line))
        intercept = point_of_line[1] - slope_size * point_of_line[0]
        return max_count, slope_size, intercept

    # safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
    # end of main
    def read_and_preprocess_csv_file(self):
        '''
        读取 ./Result/MINE2/{hotel_id}_observe_gp.csv 文件
        :return:
        '''
        logger.debug("read_and_preprocess_csv_file begin")
        # read_data_rt = pd.read_csv('{}{}_{}_gp.csv'.format(PATTERN_ATTRIBUTE_INPUT_FOLDER2, self.hotel_id, self.observe), \
        #                    encoding='utf-8', sep=',', engine='python', header=0).fillna(0)
        read_data_rt = pd.read_csv('{}{}_patterngroup.csv'.format(HOTEL_PATTERN_OUTPUT_FOLDER, self.hotel_id), encoding='utf-8', sep=',', engine='python', header=0).fillna(0)
        read_data_rt = read_data_rt.loc[(read_data_rt['GroupID'] == self.group_id) & (read_data_rt['Observe'] == 'CostAmt')]
        #     RP2 = 260281795
        #     RP1 = 260282228
        read_data = pd.read_csv(PATTERN_MAPPING_INPUT_FOLDER + str(self.hotel_id) + '_RatePlanLevelCostPrice.csv.zip', sep=',', engine='python',
                                header=0).fillna(0)
        rate_plan_list_ids = np.array(literal_eval(read_data_rt['Group'].iloc[0])).tolist()
        read_data = read_data.loc[read_data['RatePlanID'].isin(rate_plan_list_ids)]
        read_data = read_data.loc[(read_data['RatePlanLevel'] == self.ratePlanLevel) & (read_data['LengthOfStayDayCnt'] == self.lengthOfStayDayCnt) \
                                  & (read_data['PersonCnt'] == self.person_cnt)]
        # read_data = read_data[['StayDate', self.observe, 'RatePlanID']]
        read_data = read_data[['StayDate', 'LengthOfStayDayCnt', 'PersonCnt', self.observe, 'RatePlanID']]
        logger.debug("read_and_preprocess_csv_file done")
        return rate_plan_list_ids, read_data

    def linear_prediction(self, rate_plan_list_ids, read_data):
        # RP1 = read_data_rt['Group'].iloc[0]
        RP1 = rate_plan_list_ids[0]
        rp1_dd = read_data.loc[read_data['RatePlanID'] == RP1].set_index('StayDate')
        rp_func = pd.DataFrame()
        rusult_map = {}
        # data_length = len(read_data_rt.index)
        data_length = len(rate_plan_list_ids)
        pp = PdfPages(OUTPUT_RESULT_FILE_NAME.format(self.hotel_id))
        for i in range(1, data_length):
            # logger.info()
            logger.debug("left compare length: {}".format(data_length - i))
            # RP2 = read_data_rt['Group'].iloc[i]
            RP2 = rate_plan_list_ids[i]
            rp2_dd = read_data.loc[read_data['RatePlanID'] == RP2].set_index('StayDate')
            rp_ds = pd.merge(rp1_dd, rp2_dd, on='StayDate')
            # 删除 RatePlanID_x RatePlanID_y
            rp_ds_copy = rp_ds.copy(deep=True)
            rp_ds_copy = rp_ds_copy.drop(['RatePlanID_x', 'RatePlanID_y'], axis=1)
            max_count, slope, intercept = self.maxPoints(points=rp_ds_copy.values)
            rusult_map.update({round(slope, 4): rusult_map.get(round(slope, 4), 1) + 1})
            logger.debug("max_count:{} slope:{} intercept:{}".format(max_count, slope, intercept))
            #     print(rp_ds_copy.head(10))
            rp_ds_copy = preprocessing.StandardScaler().fit_transform(rp_ds_copy)
            #     rp_ds_copy = min_max_scaler.fit_transform(rp_ds_copy)
            fit_X = rp_ds_copy[:, 0].reshape((-1, 1))
            fit_y = rp_ds_copy[:, 1].reshape((-1, 1))
            #     print(fit_X)
            #     X = rp_ds[Observe+'_x'].to_numpy().reshape(-1, 1)
            #     y = rp_ds[Observe+'_y'].to_numpy().reshape(-1, 1)
            #     x_minmax = MinMaxScaler.fit_transform(X)
            X = rp_ds[self.observe + '_x'].to_numpy().reshape(-1, 1)
            y = rp_ds[self.observe + '_y'].to_numpy().reshape(-1, 1)

            Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3)
            lr = LinearRegression().fit(Xtrain, Ytrain)
            pred_y = lr.predict(X)
            rp_func = rp_func.append(
                [[RP2, '{:.4f}'.format(lr.score(Xtest, Ytest)), '{:.4f} * x {:+.4f}'.format(lr.coef_[0][0], lr.intercept_[0])]],
                ignore_index=True)

            #     lr = LinearRegression().fit(X, y)
            #     pred_y = lr.predict(X)
            # rp_func=rp_func.append([[RP2,'{:.4f}'.format(lr.score(X,y)),'{:.4f} * x {:+.4f}'.format(lr.coef_[0][0],lr.intercept_[0])]],ignore_index=True)

            #     plt.xlim(X.min(), X.max())
            #     plt.ylim(y.min(), y.max())
            fig, ax = plt.subplots(figsize=(18, 7))
            ax.scatter(X, y, color='blue')
            ax.plot(X, pred_y, color='green', linewidth=1)
            ax.plot(X, X * slope + intercept, color='r', linewidth=1, linestyle='--')
            pp.savefig(fig)
            rp_ds.to_csv('{}{}_Group{}_Line{}_{}_xy.csv'.format(PATTERN_MAPPING_OUTPUT_FOLDER, self.hotel_id, self.group_id, i, self.observe), index=False)

        rusult_map = dict(zip(rusult_map, map(lambda x: x / data_length, rusult_map.values())))
        logger.debug("rusult_map:{}".format(rusult_map))
        pp.close()
        # plt.show()
        plt.close()
        rp_func.columns = ['RatePlanID', 'Accuracy', 'Formula']
        rp_func.sort_values(by='Accuracy', ascending=False, inplace=True)
        rp_func.to_csv('{}{}_{}_{}_func.csv'.format(PATTERN_MAPPING_OUTPUT_FOLDER, self.hotel_id, self.group_id, self.observe), index=False)

if __name__ == '__main__':
    hotel_id = 862
    observe = 'CostAmt'
    group_id = 1
    patternMappingInstance = PatternMapping(int(hotel_id), observe, int(group_id), 0, 1, 2)
    rate_plan_list_ids, read_data = patternMappingInstance.read_and_preprocess_csv_file()
    patternMappingInstance.linear_prediction(rate_plan_list_ids, read_data)