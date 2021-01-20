import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import scipy.spatial.distance as dis
from sklearn.linear_model import LinearRegression
from scipy.ndimage import filters
from scipy.sparse import csgraph
from scipy.sparse import csr_matrix
from scipy import stats
import matplotlib.pyplot as plt, seaborn
from io import StringIO
import cProfile
import os
from tools import logger
import glob
import sys
import pickle
from graphviz import Source
import datetime
import json
from tools.time_tool import TimeToolObject
from service.pattern_mapping.mapping_function import MappingFunction
from settings import PATTERN_MAPPING_INPUT_FOLDER, PATTERN_ATTRIBUTE_INPUT_FOLDER2, PATTERN_MAPPING_OUTPUT_FOLDER
plt.rcParams.update({'figure.max_open_warning': 0})
logger = logger.Logger("debug")
class PatternMapping(object):
    def getMaxPrecision(self, dataList):
        maxPrecison = 0
        for data in dataList:
            stringValue = str(data)
            pointIndex = stringValue.find('.')
            pointLengh = 0
            if pointIndex != -1:
                pointLengh = len(stringValue[pointIndex + 1:])

            if maxPrecison < pointLengh:
                maxPrecison = pointLengh
        return maxPrecison


    def calcuate_slope(self, adjust_X, adjust_Y, dic_slope, dic_slope_point, compareCount):
        counter = compareCount
        while counter < len(adjust_X):
            if adjust_X[counter] - adjust_X[counter - compareCount] != 0 and adjust_Y[counter] != 0 and adjust_Y[
                counter - compareCount] != 0 and adjust_X[counter] != 0 and adjust_X[counter - compareCount] != 0:
                slope = '{:.4f}'.format((adjust_Y[counter] - adjust_Y[counter - compareCount]) / (
                            adjust_X[counter] - adjust_X[counter - compareCount]))
                if not slope in dic_slope:
                    dic_slope[slope] = 1
                    dic_slope_point[slope] = []
                else:
                    dic_slope[slope] += 1

                dic_slope_point[slope].append((counter, counter - compareCount))
            counter += 1
        return

    # logic -> y = Ax +B
    # 1. remove duplicate and sort the result by x
    # 2. calculate the confficients for each x(with x-1,x-2,x-3)
    # 3. sort the confficients to find the max value, marke it as A
    # 4. use the fitable result to find the B with biggest hit ratio
    # 5. time complexity: N
    def calculateLinear(self, x, y, date):
        from operator import itemgetter
        import operator
        from tools.time_tool import TimeToolObject
        timeToolObject = TimeToolObject()

        # remove duplicate
        combineXY = np.concatenate((x, y), axis=1)
        combineXY = np.unique(combineXY, axis=0)

        # order by X
        combineXY = np.array(sorted(combineXY, key=lambda entry: entry[0]))

        # print(combineXY)

        # change it back to X, Y
        combineXY = np.hsplit(combineXY, 2)
        adjust_X = combineXY[0].flatten()
        adjust_Y = combineXY[1].flatten()

        # get max precision
        data_rand = self.getMaxPrecision(adjust_Y)

        dic_slope = {}
        dic_slope_point = {}

        self.calcuate_slope(adjust_X, adjust_Y, dic_slope, dic_slope_point, 1)
        self.calcuate_slope(adjust_X, adjust_Y, dic_slope, dic_slope_point, 2)
        self.calcuate_slope(adjust_X, adjust_Y, dic_slope, dic_slope_point, 3)

        final_a_key = max(dic_slope.items(), key=operator.itemgetter(1))[0]
        final_a = float(final_a_key)

        dic_b_fitCount = {}
        for (point2, point1) in dic_slope_point[final_a_key]:
            final_b = adjust_Y[point2] - final_a * adjust_X[point2]
            result = np.array([final_a * n + final_b for n in adjust_X]).round(data_rand) - adjust_Y
            dic_b_fitCount['{:.4f}'.format(final_b)] = len(result) - np.count_nonzero(result)

        max_b = float(max(dic_b_fitCount.items(), key=operator.itemgetter(1))[0])

        # print(dic_slope)
        # print(dic_b_fitCount)

        predic_y = np.array([final_a * n + max_b for n in adjust_X]).round(data_rand)
        fitRatio = 100 * (len(adjust_X) - np.count_nonzero(predic_y - adjust_Y)) / len(adjust_X)

        predicData_y = np.array([final_a * n + max_b for n in x]).round(data_rand)
        fitDataRatio = 100 * (len(x) - np.count_nonzero(predicData_y - y)) / len(x)

        # print(adjust_Y)
        # print(predic_y)

        # construct mapping funcion object
        exceptionPoint = {}
        i = 0
        liner_XPoints = set()
        while i < len(y):
            if abs(round(predicData_y[i][0], data_rand) - y[i][0]) != 0:
                if not x[i][0] in exceptionPoint.keys():
                    exceptionPoint[x[i][0]] = {}

                if not y[i][0] in exceptionPoint[x[i][0]].keys():
                    exceptionPoint[x[i][0]][y[i][0]] = set()

                exceptionPoint[x[i][0]][y[i][0]].add(timeToolObject.convert_date_to_int(datetime.datetime.strptime(date[i], '%Y-%m-%d')))
            else:
                liner_XPoints.add(x[i][0])
            i += 1

        for x_cost, mappings in exceptionPoint.items():
            if x_cost not in liner_XPoints:
                if len(mappings) == 1:
                    for y_cost, datas in mappings.items():
                        datas.clear()

        mappingFunction = MappingFunction()
        mappingFunction.A = final_a
        mappingFunction.b = max_b
        mappingFunction.precision = data_rand
        mappingFunction.exceptionPoint = exceptionPoint

        return mappingFunction, adjust_X, predic_y, fitRatio, fitDataRatio

    def main(self):
        HotelID = 16639  # 51006849 #16639 #15140
        Observe = 'CostAmt'
        GroupID = 1
        RATEPLANLEVEL = 0
        LOS = 1
        PERSONCNT = 2

        # read_data_rt = pd.read_csv('{}{}_{}_gp.csv'.format(INPUT_FOLDER2,HotelID,Observe), \
        #            encoding='utf-8', sep=',', engine='python', header=0).fillna(0)
        # read_data_rt = read_data_rt.loc[read_data_rt['GroupID']==GroupID]

        read_data_rt = pd.read_csv('{}{}_patterngroup.csv'.format(PATTERN_ATTRIBUTE_INPUT_FOLDER2, HotelID), encoding='utf-8', sep=',',
                                   engine='python', header=0).fillna(0)
        read_data_rt = read_data_rt.loc[
            (read_data_rt['GroupID'] == GroupID) & (read_data_rt['Observe'] == Observe)].reset_index(drop=True)

        rateplanids = read_data_rt['Group'][0].replace('\n', '').replace('[', '').replace(']', '').split(' ')
        rateplanids = [info for info in rateplanids if info != '']
        rateplanids = list(map(int, rateplanids))

        read_data = pd.read_csv(PATTERN_MAPPING_INPUT_FOLDER + str(HotelID) + '_RatePlanLevelCostPrice.csv.zip', sep=',', engine='python',
                                header=0).fillna(0)
        read_data = read_data.loc[read_data['RatePlanID'].isin(rateplanids)]

        read_data = read_data.loc[(read_data['RatePlanLevel'] == RATEPLANLEVEL) & (read_data['LengthOfStayDayCnt'] == LOS) \
                                  & (read_data['PersonCnt'] == PERSONCNT)]

        # read_data = read_data.loc[(read_data['RatePlanLevel']==RATEPLANLEVEL)]

        read_data = read_data[['StayDate', 'LengthOfStayDayCnt', 'PersonCnt', Observe, 'RatePlanID']]

        RP1 = rateplanids[0]
        rp1_dd = read_data.loc[read_data['RatePlanID'] == RP1].set_index(['StayDate', 'LengthOfStayDayCnt', 'PersonCnt'])

        rp_func = pd.DataFrame()

        mappingFunctionResult = pd.DataFrame()

        for i in range(1, len(rateplanids)):

            RP2 = rateplanids[i]

            if RP2 != 260281941:
                continue

            rp2_dd = read_data.loc[read_data['RatePlanID'] == RP2].set_index(
                ['StayDate', 'LengthOfStayDayCnt', 'PersonCnt'])

            rp_ds = pd.merge(rp1_dd, rp2_dd, on=['StayDate', 'LengthOfStayDayCnt', 'PersonCnt'])

            logger.debug(rp_ds)

            if rp_ds.empty:
                continue

            X = rp_ds[Observe + '_x'].to_numpy().reshape(-1, 1)
            y = rp_ds[Observe + '_y'].to_numpy().reshape(-1, 1)
            date = rp_ds.index.get_level_values('StayDate').values

            lr = LinearRegression().fit(X, y)

            pred_y = lr.predict(X)

            rp_func = rp_func.append(
                [[RP2, '{:.4f}'.format(lr.score(X, y)), '{:.4f} * x {:+.4f}'.format(lr.coef_[0][0], lr.intercept_[0])]],
                ignore_index=True)

            (mappingFunction, adjust_X, predic_y, fitRatio, fitDataRatio) = self.calculateLinear(X, y, date)

            print(mappingFunction.exceptionPoint)
            dumpFunction = pickle.dumps(mappingFunction)
            mappingFunctionResult = mappingFunctionResult.append([[RP1, RP2, len(pickle.dumps(X)), len(pickle.dumps(y)),
                                                                   len(dumpFunction),
                                                                   mappingFunction.validation(X, y, date)]],
                                                                 ignore_index=True)

            # calculate the zip ratio
            # store the mapping function object

            fig, ax = plt.subplots(figsize=(18, 7))
            plt.title('{:.4f} * x {:+.4f}, fitPointRatio:{:.2f}%, fitDataRatio:{:.2f}%, x_rp:{}, y_rp:{}'.format(
                mappingFunction.A, mappingFunction.b, fitRatio, fitDataRatio, RP1, RP2))
            ax.scatter(X, y, color='blue', s=10)
            ax.plot(X, pred_y, color='green', linewidth=1)
            ax.plot(adjust_X, predic_y, color='red', linewidth=1)

            rp_ds.to_csv('{}{}_Group{}_Line{}_{}_xy.csv'.format(PATTERN_MAPPING_OUTPUT_FOLDER, HotelID, GroupID, i, Observe), index=False)

        plt.show()

        rp_func.columns = ['RatePlanID', 'Accuracy', 'Formula']
        rp_func.sort_values(by='Accuracy', ascending=False, inplace=True)
        rp_func.to_csv('{}{}_{}_{}_func.csv'.format(PATTERN_MAPPING_OUTPUT_FOLDER, HotelID, GroupID, Observe), index=False)

        mappingFunctionResult.columns = ['BaseRP', 'ChildRP', 'BaseSize', 'ChildSize', 'MappingFunctionSize', 'Validation']
        mappingFunctionResult.to_csv('{}{}_{}_{}_mappingFunction.csv'.format(PATTERN_MAPPING_OUTPUT_FOLDER, HotelID, GroupID, Observe),
                                     index=False)
    if __name__ == '__main__':
        main()


