import re
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import scipy.spatial.distance as dis
from scipy.ndimage import filters
from scipy.sparse import csgraph
from scipy.sparse import csr_matrix
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt, seaborn
from settings import HOME_FOLDER
from io import StringIO
import cProfile
import os
from sklearn import preprocessing
from pandas.core.common import flatten
from tools import logger
from settings import HOTEL_PATTERN_INPUT_FOLDER, HOTEL_PATTERN_INPUT_FOLDER2, HOTEL_PATTERN_OUTPUT_FOLDER
from settings import HOTEL_PATTERN_RATEPLANLEVEL, HOTEL_PATTERN_LOS, HOTEL_PATTERN_PERSONCNT, PATTERN_ATTRIBUTE_OUTPUT_FOLDER
from settings import HOTEL_PATTERN_Observes
import glob
matplotlib.use('Agg')
logger = logger.Logger("debug")

class HotelPattern(object):

    def read_file_dbo_RoomType_NoIdent(self, hotel_id):
        read_data_rt = pd.read_csv(HOTEL_PATTERN_INPUT_FOLDER + 'dbo_RoomType_NoIdent.csv', encoding='utf-8', sep=',',
                                   engine='python',
                                   header=0).fillna(0)
        read_data_rt = read_data_rt[['SKUGroupID', 'RoomTypeID', 'ActiveStatusTypeID']]
        read_data_rt = read_data_rt.loc[read_data_rt['ActiveStatusTypeID'] == 2]
        read_data_rt.drop(['ActiveStatusTypeID'], axis=1, inplace=True)
        logger.debug(read_data_rt.head(10))
        read_data_rt = read_data_rt.loc[read_data_rt['SKUGroupID'].isin([hotel_id])]
        return read_data_rt


    def read_file_dboRatePlanNoIdent(self, read_data_rt):
        read_data_rp = pd.read_csv(HOTEL_PATTERN_INPUT_FOLDER + 'dbo_RatePlan_NoIdent.csv', encoding='utf-8', sep=',',
                                   engine='python',
                                   header=0).fillna(0)
        read_data_rp.drop(['UpdateTPID', 'ChangeRequestID', 'UpdateTUID'], axis=1, inplace=True)
        read_data_rp.drop(['UpdateDate', 'LastUpdatedBy', 'UpdateClientID', 'RatePlanLogID'], axis=1, inplace=True)
        read_data_rp = read_data_rp.loc[
            (read_data_rp['ActiveStatusTypeID'] == 2) & (read_data_rp['RoomTypeID'].isin(read_data_rt['RoomTypeID']))]

        # read_data_rp = pd.read_csv(HOTEL_PATTERN_INPUT_FOLDER + 'dbo_RatePlan_NoIdent.csv', sep=',', engine='python', header=0).fillna(0)
        # read_data_rp = read_data_rp.loc[(read_data_rp['ActiveStatusTypeID'] == 2) \
        #                                 & (read_data_rp['RoomTypeID'].isin(read_data_rt['RoomTypeID']))][['RatePlanID']]
        return read_data_rp


    def read_file_RatePlanLevelCostPrice(self, hotel_id, read_data_rp):
        read_data = pd.read_csv(HOTEL_PATTERN_INPUT_FOLDER2 + str(hotel_id) + '_RatePlanLevelCostPrice.csv.zip',
                                sep=',', engine='python',
                                header=0).fillna(0)
        read_data = read_data.loc[read_data['RatePlanID'].isin(read_data_rp['RatePlanID'])]
        logger.debug(read_data)

        #     RatePlanID,StayDate,RatePlanLevel,PersonCnt,LengthOfStayDayCnt,ActiveStatusTypeID,
        #     RatePlanLevelCostPriceLogSeqNbr,CostAmt,PriceAmt,CostCode,ChangeRequestIDOld,
        #     SupplierUpdateDate,SupplierUpdateTPID,SupplierUpdateTUID,UpdateDate,SupplierLogSeqNbr,
        #     ChangeRequestID,LARAmt,LARMarginAmt,LARTaxesAndFeesAmt

        read_data.drop(['ActiveStatusTypeID', 'RatePlanLevelCostPriceLogSeqNbr', 'ChangeRequestIDOld'], axis=1,
                       inplace=True)
        read_data.drop(['SupplierUpdateDate', 'SupplierUpdateTPID', 'SupplierUpdateTUID'], axis=1, inplace=True)
        read_data.drop(['UpdateDate', 'SupplierLogSeqNbr', 'ChangeRequestID'], axis=1, inplace=True)
        read_data = read_data.loc[(read_data['RatePlanLevel'] == HOTEL_PATTERN_RATEPLANLEVEL) & (
                    read_data['LengthOfStayDayCnt'] == HOTEL_PATTERN_LOS)
                                  & (read_data['PersonCnt'] == HOTEL_PATTERN_PERSONCNT)]
        read_data.drop(['RatePlanLevel', 'LengthOfStayDayCnt', 'PersonCnt'], axis=1, inplace=True)
        return read_data


    def read_csv_data_and_filter(self, hotel_id):
        '''读取CSV文件并进行过滤
            First step: 根据 HotelID 获取所有 RoomTypeId (read_data_rt)
                读取./Data/dbo_RoomType_NoIdent.csv文件 = read_data_rt
                取出 'SKUGroupID', 'RoomTypeID', 'ActiveStatusTypeID' 列
                选择 ActiveStatusTypeID == 2 并删除ActiveStatusTypeID列
                选择 SKUGroupID 为指定 HotelID 的数据 = read_data_rt

            Second step: 根据 RoomTypeId 获取对应的 RatePlanID (read_data_rp)
                读取./Data/dbo_RatePlan_NoIdent.csv 数据文件 = read_data_rp
                过滤 ActiveStatusTypeID == 2 并且 RoomTypeID == read_data_rt 的 RoomTypeID 的数据并取出它的 RatePlanID  = read_data_rp

            Third step: 根据RatePlanID 获取 CostPrice (read_data)
                从 ./Data2/HotelID_RatePlanLevelCostPrice.csv.zip 压缩文件中读取数据
                选出 RatePlanID 在 read_data_rp['RatePlanID'] 中的数据
                删除 'ActiveStatusTypeID', 'RatePlanLevelCostPriceLogSeqNbr', 'ChangeRequestIDOld', 'SupplierUpdateDate', 'SupplierUpdateTPID', 'SupplierUpdateTUID', 'UpdateDate', 'SupplierLogSeqNbr', 'ChangeRequestID' 列
                选出 'RatePlanLevel' == HOTEL_PATTERN_RATEPLANLEVEL 'LengthOfStayDayCnt' == HOTEL_PATTERN_LOS 'PersonCnt' == HOTEL_PATTERN_PERSONCNT 的数据并删除这几列  = read_data
            Fourth step:
        '''
        read_data_rt = self.read_file_dbo_RoomType_NoIdent(hotel_id)
        read_data_rp = self.read_file_dboRatePlanNoIdent(read_data_rt)
        read_data = self.read_file_RatePlanLevelCostPrice(hotel_id, read_data_rp)
        return read_data_rt, read_data_rp, read_data

    def get_connected_components(self, read_data, Observe):
        read_data['z_score'] = stats.zscore(read_data[Observe])
        # print(read_data.head(20))
        logger.debug(read_data.head(20))
        read_data = read_data.loc[read_data['z_score'].abs() <= 3]
        read_data_gp = read_data[['StayDate', Observe, 'RatePlanID']].groupby(['RatePlanID'], sort=False)
        df_corr = pd.DataFrame()
        for name, group in read_data_gp:
            group.reset_index(drop=True, inplace=True)
            df_corr[name] = group.set_index('StayDate')[Observe]

        # https://blog.csdn.net/walking_visitor/article/details/85128461
        # 默认使用 pearson 相关系数计算方法，但这种方式存在误判
        df_corr = df_corr.corr()
        np.fill_diagonal(df_corr.values, 0)
        # df_corr = df_corr.mask(df_corr<0.95)
        # plt.figure(figsize=(18, 7))
        # seaborn.heatmap(df_corr, center=0, annot=True, cmap='YlGnBu')
        graph = csr_matrix(df_corr >= 0.99)
        n, labels = csgraph.connected_components(graph)
        output_df = pd.DataFrame(columns=['GroupID', 'RatePlanID'])
        # print('{}/{}'.format(n, len(read_data_gp.ngroup())))
        logger.debug('The number of connected components: {} || The number of groupby[RatePlanID]:{}'.format(n,read_data_gp.ngroups))
        return n, labels, df_corr


    def generate_group_file_and_img(self, read_data, hotel_id):
        '''生成结果到CSV文件
            First Step: 对观测值的RatePlanID进行分组 (read_data_gp)
                创建一个空的 DataFrame 来存储数据 = df_cdist
                分别从read_data(CostPrice文件)对 'CostAmt', 'PriceAmt', 'LARAmt', 'LARMarginAmt', 'LARTaxesAndFeesAmt' 计算 zscore 分数并存储到 read_data 文件的 zscore 列
                过滤掉 read_data z_score 分数小于等于 3 的数据
                然后根据 RatePlanID 进行分组，分组后取出 'StayDate', Observe, 'RatePlanID' 这三列 = read_data_gp
                遍历分组后的数据，并对分组数据的索引重置为日期(StayDate) = df_corr
                对 df_corr 计算相关系数,矩阵对角填充为0 并选出相关系数大于95% 的图
            Second Step: 使用稀疏矩阵图对不同的观测值进行 RatePlanId 分组并保存结果到文件 (df_cdist)
                分析稀疏图的连通分量 获取 连通域个数 及 连接组件的标签长度
                log 打印 连通图分组个数 及 根据RatePlanID 的分组个数
                遍历分组结果,选取 read_data 的 RatePlanID 在稀疏矩阵图中的分组结果
                对 read_data 的 'StayDate', 'RatePlanID' 进行分组, 并对 Observe 的金额进行求和并画图
                将 df_cdist 分组结果 [观测值，分组id,分组的RatePlanId数组]  输出到 ./Result/MINE2/HotelID_patterngroup.csv 文件中
        '''

        # for Observe in HOTEL_PATTERN_Observes:
        #     if Observe != "CostAmt":
        #         continue
        n, labels, df_corr = self.get_connected_components(read_data, "CostAmt")
        df_cdist = pd.DataFrame()

        if n <= 0:
            return df_cdist, 0

        count = 0
        column_size = math.ceil(n ** 0.5)
        row_size = math.ceil(n / column_size)
        fig, axes = plt.subplots(row_size, column_size, figsize=(20, 30))
        # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0.1, wspace=0.1)
        plt.subplots_adjust(left=0.125, bottom=0.04, right=0.9, top=1, hspace=0.1, wspace=0.1)
        for i in range(row_size):
            for j in range(0, column_size):
                if count >= n:
                    continue
                nodes = df_corr.index[np.where(labels == count)]
                df_cdist = df_cdist.append([["CostAmt", count, nodes.values]], ignore_index=True)
                read_data.loc[(read_data['RatePlanID'].isin(nodes))].groupby(['StayDate', 'RatePlanID']).sum()[
                    "CostAmt"].unstack().plot(ax=axes[i][j])
                logger.debug("left compatue length: {}".format(n - count))
                count += 1
        plt.savefig('{}{}_all_pattern_group.png'.format(PATTERN_ATTRIBUTE_OUTPUT_FOLDER, hotel_id))
        # for i in range(n):
        #     nodes = df_corr.index[np.where(labels == i)]
        #     df_cdist = df_cdist.append([["CostAmt", i, nodes.values]], ignore_index=True)
            # fig, ax = plt.subplots(figsize=(18, 7))
            # read_data.loc[(read_data['RatePlanID'].isin(nodes))].groupby(['StayDate', 'RatePlanID']).sum()[
            #     "CostAmt"].unstack().plot(ax=ax)

        df_cdist.columns = ['Observe', 'GroupID', 'Group']
        df_cdist_copy = df_cdist.copy()
        df_cdist_copy["Group"] = df_cdist_copy.apply(lambda x: x["Group"].tolist(), axis=1)
        df_cdist_copy["RatePlanLen"] = df_cdist_copy.apply(lambda x: len(x["Group"]), axis=1)
        best_group_id = np.random.choice(df_cdist_copy["RatePlanLen"][df_cdist_copy["RatePlanLen"] == df_cdist_copy["RatePlanLen"].max()].index)
        logger.debug("The best group is group_{}".format(best_group_id))
        # df_cdist_copy.loc[:, ('Group')] = df_cdist['Group'].iloc[0].tolist()
        logger.debug("Generate {}'s grouping files".format(hotel_id))
        df_cdist_copy.to_csv('{}{}_patterngroup.csv'.format(HOTEL_PATTERN_OUTPUT_FOLDER, hotel_id), index=False)
        logger.debug("The generation of grouping files is complete")

        nodes = df_corr.index[np.where(labels == best_group_id)]
        self.save_sigle_pattern_group_img(read_data, nodes, hotel_id)
        # fig, ax = plt.subplots(figsize=(18, 7))
        # read_data.loc[(read_data['RatePlanID'].isin(nodes))].groupby(['StayDate', 'RatePlanID']).sum()[
        #     "CostAmt"].unstack().plot(ax=ax)
        # plt.savefig('{}{}_patterngroup.png'.format(PATTERN_ATTRIBUTE_OUTPUT_FOLDER, hotel_id))
        # plt.show()
        # plt.close()
        return df_cdist, best_group_id


    def save_sigle_pattern_group_img(self, read_data, nodes, hotel_id):
        fig, ax = plt.subplots(figsize=(18, 7))
        read_data.loc[(read_data['RatePlanID'].isin(nodes))].groupby(['StayDate', 'RatePlanID']).sum()[
            "CostAmt"].unstack().plot(ax=ax)
        plt.savefig('{}{}_patterngroup.png'.format(PATTERN_ATTRIBUTE_OUTPUT_FOLDER, hotel_id))
        # plt.show()
        # plt.close()

    def show_comparison_with_other_amt(self, df_cdist):
        ''' 对分组后Observe = CostAmt 的数据与其他费用数据做多标签二值化后计算距离
            取出 Observe=='CostAmt'分组后Group列的 ratePlanId 数组
            遍历剩余非 'CostAmt' 的Group列与 Observe = CostAmt 的 Group 做多标签二值化
            计算 CostAmt 与其他费用的距离
        :return:
        '''
        XA = df_cdist.loc[df_cdist['Observe'] == 'CostAmt']['Group'].to_numpy()
        for Observe in HOTEL_PATTERN_Observes:
            if Observe == 'CostAmt':
                continue
            XB = df_cdist.loc[df_cdist['Observe'] == Observe]['Group'].to_numpy()
            mlb = preprocessing.MultiLabelBinarizer()

            mlb.fit([flatten(XA), flatten(XB)])

            XA = mlb.transform(XA)
            XB = mlb.transform(XB)

            d = dis.cdist(XA, XB, 'cosine')

            df = pd.DataFrame(d)
            df = df.mask(df < 0.5, 0)

            # print(Observe)
            logger.debug("current observe is {} compare with CostAmt".format(Observe))
            # print(df)
            logger.debug(df)

            # plt.figure(figsize=(18, 7))
            # seaborn.heatmap(df, center=0, annot=True, cmap='YlGnBu')
            # XA = df_cdist.loc[df_cdist['Observe'] == 'CostAmt']['Group'].to_numpy()

        # plt.show()
        # print(mlb.inverse_transform(XA[0].reshape(1,-1)))
        # print(mlb.inverse_transform(XB[0].reshape(1,-1)))
        # print(mlb.inverse_transform(XA[1].reshape(1,-1)))
        # print(mlb.inverse_transform(XB[1].reshape(1,-1)))
        # print(df)
if __name__ == '__main__':
    hotelPattern = HotelPattern()
    read_data_rt, read_data_rp, read_data = hotelPattern.read_csv_data_and_filter(16639)
    df_cdist, best_group_id = hotelPattern.generate_group_file_and_img(read_data, 16639)
    hotelPattern.show_comparison_with_other_amt(df_cdist)

