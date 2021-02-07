import re
import os
import math
import pickle
import datetime
import matplotlib
import numpy as np
import pandas as pd
from tools import logger
from ast import literal_eval
import matplotlib.pyplot as plt
from reportlab.lib import colors
from sklearn import preprocessing
from tools.pdf_reportlab import Graphs
from reportlab.lib.units import mm, inch
from reportlab.lib.enums import TA_CENTER
from mydecorators.exception import except_output
from mydecorators.execution_time import execute_time
from sklearn.linear_model import LinearRegression
from reportlab.lib.pagesizes import elevenSeventeen
from service.pattern_mapping.hotel_pattern import HotelPattern
from reportlab.platypus import SimpleDocTemplate, Spacer, Image
from service.pattern_mapping.mapping_function import MappingFunction
from settings import PATTERN_MAPPING_INPUT_FOLDER, PATTERN_MAPPING_OUTPUT_FOLDER
from settings import HOTEL_PATTERN_OUTPUT_FOLDER, PATTERN_ATTRIBUTE_OUTPUT_FOLDER
from settings import OUTPUT_RESULT_FILE_NAME, HOTEL_PATTERN_LOS, OUTPUT_LINEAR_FILE_NAME
plt.rcParams.update({'figure.max_open_warning': 0})
min_max_scaler = preprocessing.MinMaxScaler()
logger = logger.Logger("debug")
matplotlib.use('Agg')


# https://www.cs.princeton.edu/courses/archive/spring03/cs226/assignments/lines.html
class PatternMapping(Spacer):

    def __init__(self, search_id, hotel_id, search_level, observe, group_id, ratePlanLevel, lengthOfStayDayCnt, person_cnt, **kw):
        self.search_id = search_id
        self.hotel_id = hotel_id
        self.search_level = search_level
        self.observe = observe
        self.group_id = group_id
        self.ratePlanLevel = ratePlanLevel
        self.lengthOfStayDayCnt = lengthOfStayDayCnt
        self.person_cnt = person_cnt
        for k, v in kw.items():
            setattr(self, k, v)

    def wrap(self, availWidth, availHeight):
        height = min(self.height, availHeight-1e-8)
        return (availWidth, height)


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

    @except_output()
    def calculateLinear(self, x, y, date):
        '''
        # logic -> y = Ax +B
        # 1. remove duplicate and sort the result by x
        # 2. calculate the confficients for each x(with x-1,x-2,x-3)
        # 3. sort the confficients to find the max value, marke it as A
        # 4. use the fitable result to find the B with biggest hit ratio
        # 5. time complexity: N
        :param x:
        :param y:
        :param date:
        :return:
        '''
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

                exceptionPoint[x[i][0]][y[i][0]].add(
                    timeToolObject.convert_date_to_int(datetime.datetime.strptime(date[i], '%Y-%m-%d')))
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

    @execute_time
    @except_output()
    def read_group_file(self):
        read_data_group = pd.read_csv('{}{}_patterngroup.csv'.format(HOTEL_PATTERN_OUTPUT_FOLDER, self.search_id),
                                   encoding='utf-8', sep=',', engine='python', header=0).fillna(0)
        read_data_group = read_data_group.loc[
            (read_data_group['GroupID'] == self.group_id) & (read_data_group['Observe'] == 'CostAmt')]
        return read_data_group


    @execute_time
    @except_output()
    def read_and_preprocess_csv_file(self):
        '''
        读取 ./Result/MINE2/{hotel_id}_observe_gp.csv 文件
        :return:
        '''
        logger.debug("read_and_preprocess_group_csv_file begin")
        # read_data_rt = pd.read_csv('{}{}_{}_gp.csv'.format(PATTERN_ATTRIBUTE_INPUT_FOLDER2, self.hotel_id, self.observe), \
        #                    encoding='utf-8', sep=',', engine='python', header=0).fillna(0)
        #     RP2 = 260281795
        #     RP1 = 260282228
        read_group_file = self.read_group_file()
        read_data = pd.read_csv(PATTERN_MAPPING_INPUT_FOLDER + str(self.hotel_id) + '_RatePlanLevelCostPrice.csv.zip', sep=',', engine='python',
                                header=0).fillna(0)
        rate_plan_list_ids = np.array(literal_eval(read_group_file['Group'].iloc[0])).tolist()
        read_data = read_data.loc[read_data['RatePlanID'].isin(rate_plan_list_ids)]
        read_data = read_data.loc[(read_data['RatePlanLevel'] == self.ratePlanLevel) & (read_data['LengthOfStayDayCnt'] == self.lengthOfStayDayCnt) \
                                  & (read_data['PersonCnt'] == self.person_cnt)]
        # read_data = read_data[['StayDate', self.observe, 'RatePlanID']]
        read_data = read_data[['StayDate', 'LengthOfStayDayCnt', 'PersonCnt', self.observe, 'RatePlanID']]
        logger.debug("read_and_preprocess_group_csv_file done")
        return rate_plan_list_ids, read_data

    @except_output()
    def calc_delta(self, root_no, child_no, abp_df):
        root = abp_df[abp_df['RatePlanID'] == root_no].reset_index(drop=True)
        child = abp_df[abp_df['RatePlanID'] == child_no].reset_index(drop=True)
        root = root.T
        child = child.T
        root.columns = ['Value']
        child.columns = ['Value']
        root['Value'] = root['Value'].apply(str)
        child['Value'] = child['Value'].apply(str)
        root = root.drop('RatePlanID')
        child = child.drop('RatePlanID')
        root['Value2'] = child['Value']
        root['ValueMatch'] = np.where(root['Value'] == child['Value'], True, False)
        root['ValueDiff'] = np.where(root['Value'] == child['Value'], '',
                                     root['Value'] + '→' + child['Value'])
        root = root.loc[root['ValueMatch'] == False]
        root.drop(['Value', 'Value2', 'ValueMatch'], axis=1, inplace=True)
        delta = re.sub(' +', ':', root['ValueDiff'].to_string())
        print("delta:{}".format(delta))
        return delta

    @execute_time
    @except_output()
    def marge_rt_rp(self, group_rate_plan_ids):
        hotelPattern = HotelPattern()
        if self.search_level == "Room":
            read_data_rt, read_data_rp = hotelPattern.read_rt_rp_by_room_id(self.search_id)
        if self.search_level == "Hotel":
            read_data_rt, read_data_rp = hotelPattern.read_rt_rp_by_hotel_id(self.search_id)
        read_data_hilton = pd.merge(read_data_rt, read_data_rp, how='inner', left_on='RoomTypeID', right_on='RoomTypeID')
        read_data_hilton.rename(columns={'SKUGroupID': 'HotelId'}, inplace=True)
        logger.debug(read_data_hilton)
        input_data = read_data_hilton.loc[read_data_hilton['RatePlanID'].isin(group_rate_plan_ids)]
        return input_data

    @execute_time
    @except_output()
    def linear_prediction(self, rate_plan_list_ids, read_data):
        input_data = self.marge_rt_rp(rate_plan_list_ids)
        RP1 = rate_plan_list_ids[0]
        rp1_dd = read_data.loc[read_data['RatePlanID'] == RP1].set_index('StayDate')
        rp_func = pd.DataFrame()
        mappingFunctionResult = pd.DataFrame()
        data_length = len(rate_plan_list_ids)
        if data_length <= 1:
            return mappingFunctionResult
        count = 0
        row_size = math.ceil((data_length-1) ** 0.5)
        column_size = 0 if row_size == 0 else math.ceil((data_length-1) / row_size)
        fig, axes = plt.subplots(row_size, column_size, figsize=(20, 30))
        plt.subplots_adjust(left=0.125, bottom=0.04, right=0.9, top=1, hspace=0.2, wspace=0.3)
        # 设置主标题
        fig.suptitle('x_rp:{}'.format(RP1))
        for i in range(row_size):
            logger.debug("left compare length: {}".format(data_length - count))
            for j in range(0, column_size):
                count += 1
                if count >= data_length:
                    continue
                RP2 = rate_plan_list_ids[count]
                delta = self.calc_delta(RP1, RP2, input_data)
                rp2_dd = read_data.loc[read_data['RatePlanID'] == RP2].set_index('StayDate')
                rp_ds = pd.merge(rp1_dd, rp2_dd, on='StayDate')
                if rp_ds.empty:
                    continue
                X = rp_ds[self.observe + '_x'].to_numpy().reshape(-1, 1)
                y = rp_ds[self.observe + '_y'].to_numpy().reshape(-1, 1)
                date = rp_ds.index.get_level_values('StayDate').values
                lr = LinearRegression().fit(X, y)
                pred_y = lr.predict(X)
                rp_func = rp_func.append(
                    [[RP2, '{:.4f}'.format(lr.score(X, y)),
                      '{:.4f} * x {:+.4f}'.format(lr.coef_[0][0], lr.intercept_[0])]],
                    ignore_index=True)

                (mappingFunction, adjust_X, predic_y, fitRatio, fitDataRatio) = self.calculateLinear(X, y, date)
                logger.debug("mappingFunction.exceptionPoint: {}".format(mappingFunction.exceptionPoint))
                dumpFunction = pickle.dumps(mappingFunction)
                mappingFunctionResult = mappingFunctionResult.append(
                    [[RP1, RP2, len(pickle.dumps(X)), len(pickle.dumps(y)),
                      len(dumpFunction),
                      mappingFunction.validation(X, y, date)]],
                    ignore_index=True)
                # 设置子标题
                # axes[i][j].set_title('{:.4f} * x {:+.4f}, fitPointRatio:{:.2f}%, fitDataRatio:{:.2f}%, y_rp:{}'.format(
                #     mappingFunction.A, mappingFunction.b, fitRatio, fitDataRatio, RP2))
                axes[i][j].set_title('y_rp:{}'.format(RP2))
                axes[i][j].scatter(X, y, color='blue', s=10)
                # axes[i][j].plot(X, pred_y, color='green', linewidth=1)
                axes[i][j].plot(adjust_X, predic_y, color='red', linewidth=1)
                axes[i][j].text(X.min(), y.mean(), delta)
                rp_ds.to_csv(
                    '{}{}_Group{}_Line{}_{}_xy.csv'.format(PATTERN_MAPPING_OUTPUT_FOLDER, self.search_id, self.group_id,
                                                           count,
                                                           self.observe),
                    index=False)
        # plt.tight_layout(w_pad=1.0)
        plt.savefig(OUTPUT_LINEAR_FILE_NAME.format(PATTERN_ATTRIBUTE_OUTPUT_FOLDER, self.search_id), format='jpg',
                    dpi=300)
        # plt.show()
        # plt.close()
        mappingFunctionResult.columns = ['BaseRP', 'ChildRP', 'BaseSize', 'ChildSize', 'MappingFunctionSize', 'Validation']
        mappingFunctionResult.to_csv(
            '{}{}_{}_{}_mappingFunction.csv'.format(PATTERN_MAPPING_OUTPUT_FOLDER, self.search_id, self.group_id,
                                                    self.observe), index=False)
        return mappingFunctionResult

    @execute_time
    @except_output()
    def generate_report(self, mappingFunctionResult):
        LOS = HOTEL_PATTERN_LOS
        Observe = self.observe
        personCnt = self.person_cnt
        ratePlanLevel = self.ratePlanLevel
        lengthOfStayDayCnt = self.lengthOfStayDayCnt
        content = list()
        # 添加标题
        content.append(Graphs.draw_title())
        # 添加段落
        display_content = list()

        # 分组全图
        all_pattern_group_image = '{}{}_all_pattern_group.png'.format(PATTERN_ATTRIBUTE_OUTPUT_FOLDER, self.search_id)
        if os.path.exists(all_pattern_group_image):
            content.append(Graphs.draw_text("Linear grouping diagram"))
            content.append(Graphs.draw_text(all_pattern_group_image))
            img = Image(all_pattern_group_image)
            img.drawHeight = (5*60) * mm
            img.drawWidth = 260 * mm
            img.hAlign = TA_CENTER
            content.append(img)

        # 分组图
        PatternGroupPngUrl = '{}{}_patterngroup.png'.format(PATTERN_ATTRIBUTE_OUTPUT_FOLDER, self.search_id)
        if os.path.exists(PatternGroupPngUrl):
            content.append(Graphs.draw_text("The best group is group {}".format(int(self.group_id)+1)))
            content.append(Graphs.draw_text('{}{}_patterngroup.csv'.format(HOTEL_PATTERN_OUTPUT_FOLDER, self.search_id)))
            content.append(Graphs.draw_text(PatternGroupPngUrl))
            # content.append(Spacer(1, 10 * mm))
            img = Image(PatternGroupPngUrl)
            img.drawHeight = 60 * mm
            img.drawWidth = 260 * mm
            img.hAlign = TA_CENTER
            content.append(img)

        # 线性方程图
        LinearPngUrl = OUTPUT_LINEAR_FILE_NAME.format(PATTERN_ATTRIBUTE_OUTPUT_FOLDER, self.search_id)
        if os.path.exists(LinearPngUrl):
            content.append(Graphs.draw_text("Linear fitting equation diagram"))
            content.append(Graphs.draw_text(LinearPngUrl))
            # content.append(Spacer(1, 10 * mm))
            img = Image(LinearPngUrl)
            img.drawHeight = (6*50) * mm
            img.drawWidth = 260 * mm
            img.hAlign = TA_CENTER
            content.append(img)

        display_content.append("First we used the Mapping Function on Hotelid for {}.".format(self.search_id))
        display_content.append("In the personCnt={} case,".format(personCnt))
        display_content.append(" lengthOfStayCnt = {}".format(lengthOfStayDayCnt))
        display_content.append(" and ratePlanLevel = {}".format(ratePlanLevel))
        display_content.append(" and LOS = {}".format(LOS))
        display_content.append(" and Observe = {}".format(Observe))
        display_content.append(" and the best group is group {}".format((int(self.group_id)+1)))

        content.append(Graphs.draw_text("RatePlan Attribute Relationship Graph"))
        pic_url = '{}{}_{}_pic.png'.format(PATTERN_ATTRIBUTE_OUTPUT_FOLDER, 16639, 166628)
        if os.path.exists(pic_url):
            content.append(Spacer(1, 10 * mm))
            img = Image(pic_url)
            img.drawHeight = 60 * mm
            img.drawWidth = 200 * mm
            img.hAlign = TA_CENTER
            content.append(img)

        if not mappingFunctionResult.empty:
            childTotalSizes = sum(mappingFunctionResult['ChildSize'])
            mappingFunctionTotalSize = sum(mappingFunctionResult['MappingFunctionSize'])
            compressionRatio = (childTotalSizes - mappingFunctionTotalSize) / childTotalSizes
            # 添加表格数据
            data = [('BaseRatePlan', 'childTotalSizes', "mappingFunctionTotalSize", 'Validation', 'CompressionRatio'),
                    (mappingFunctionResult['BaseRP'][0], childTotalSizes, mappingFunctionTotalSize, all(mappingFunctionResult['Validation']), '{:.2%}'.format(compressionRatio))]
            display_content.append(" and Mapping Function compression ratio = {:.2%}.".format(compressionRatio))
            # 添加表
            content_text = "".join(display_content)
            content.append(Graphs.draw_text(content_text))
            content.append(Spacer(1, 10 * mm)) # 间隔区
            content.append(Graphs.draw_table(*data))
            # 添加bar装图
            max_value = max(childTotalSizes, mappingFunctionTotalSize)
            b_data = [[childTotalSizes, None], [None, mappingFunctionTotalSize]]
            ax_data = ['ChildSize', 'MappingFunctionSize']
            leg_items = [(colors.red, 'ChildTotalSize'), (colors.green, 'MappingFunctionTotalSize')]
            content.append(Spacer(1, 30 * mm))
            content.append(Graphs.draw_bar(b_data, ax_data, leg_items, max_value))

        # 生成pdf文件
        doc = SimpleDocTemplate(OUTPUT_RESULT_FILE_NAME.format(self.search_id), pagesize=elevenSeventeen, rightMargin=0.2 * inch,
                        leftMargin=0.2 * inch,
                        topMargin=10, bottomMargin=68)
        doc.build(content)

if __name__ == '__main__':
    LOS = 1
    group_id = 1
    PERSONCNT = 2
    hotel_id = 16639
    observe = 'CostAmt'
    RATEPLANLEVEL = 0
    lengthOfStayDayCnt = 1
    patternMappingInstance = PatternMapping(int(hotel_id), observe, int(group_id), RATEPLANLEVEL, lengthOfStayDayCnt, PERSONCNT)
    rate_plan_list_ids, read_data = patternMappingInstance.read_and_preprocess_csv_file()
    mappingFunctionResult = patternMappingInstance.linear_prediction(rate_plan_list_ids, read_data)
    patternMappingInstance.generate_report(mappingFunctionResult)