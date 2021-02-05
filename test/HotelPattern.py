import pandas as pd
import numpy as np
from scipy.sparse import csgraph
from scipy.sparse import csr_matrix
from scipy import stats
import matplotlib.pyplot as plt
import os
import logging
from tools.cycle_graph import DFSFindCircle
logger = logging.getLogger()
#     CRITICAL
#     ERROR
#     WARNING
#     INFO
#     DEBUG
logging.disable(logging.DEBUG);
logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')

DATAVERSION = 2
HotelID = 16639
RATEPLANLEVEL = 0
LOS = 1
PERSONCNT = 2
from settings import HOME_FOLDER
os.chdir(HOME_FOLDER)
from settings import HOTEL_PATTERN_INPUT_FOLDER as INPUT_FOLDER
from settings import HOTEL_PATTERN_INPUT_FOLDER2 as INPUT_FOLDER2


class HotelPatternTest:
    def read_file(self):
        read_data_rt = pd.read_csv(INPUT_FOLDER + 'dbo_RoomType_NoIdent.csv', encoding='utf-8', sep=',', engine='python',
                                   header=0).fillna(0)
        read_data_rt = read_data_rt[['SKUGroupID', 'RoomTypeID', 'ActiveStatusTypeID']]
        read_data_rt = read_data_rt.loc[read_data_rt['ActiveStatusTypeID'] == 2]
        read_data_rt.drop(['ActiveStatusTypeID'], axis=1, inplace=True)
        logger.debug(read_data_rt.head(10))
        read_data_rt = read_data_rt.loc[read_data_rt['SKUGroupID'].isin([HotelID])]
        read_data_rp = pd.read_csv(INPUT_FOLDER + 'dbo_RatePlan_NoIdent.csv', sep=',', engine='python', header=0).fillna(0)
        read_data_rp = read_data_rp.loc[(read_data_rp['ActiveStatusTypeID'] == 2) \
                                        & (read_data_rp['RoomTypeID'].isin(read_data_rt['RoomTypeID']))][['RatePlanID']]
        read_data = pd.read_csv(INPUT_FOLDER2 + str(HotelID) + '_RatePlanLevelCostPrice.csv.zip', sep=',', engine='python',
                                header=0).fillna(0)
        read_data = read_data.loc[read_data['RatePlanID'].isin(read_data_rp['RatePlanID'])]
        logger.debug(read_data)

        #     RatePlanID,StayDate,RatePlanLevel,PersonCnt,LengthOfStayDayCnt,ActiveStatusTypeID,
        #     RatePlanLevelCostPriceLogSeqNbr,CostAmt,PriceAmt,CostCode,ChangeRequestIDOld,
        #     SupplierUpdateDate,SupplierUpdateTPID,SupplierUpdateTUID,UpdateDate,SupplierLogSeqNbr,
        #     ChangeRequestID,LARAmt,LARMarginAmt,LARTaxesAndFeesAmt

        read_data.drop(['ActiveStatusTypeID', 'RatePlanLevelCostPriceLogSeqNbr', 'ChangeRequestIDOld'], axis=1, inplace=True)
        read_data.drop(['SupplierUpdateDate', 'SupplierUpdateTPID', 'SupplierUpdateTUID'], axis=1, inplace=True)
        read_data.drop(['UpdateDate', 'SupplierLogSeqNbr', 'ChangeRequestID'], axis=1, inplace=True)
        read_data = read_data.loc[(read_data['RatePlanLevel'] == RATEPLANLEVEL) & (read_data['LengthOfStayDayCnt'] == LOS)
                                  & (read_data['PersonCnt'] == PERSONCNT)]
        read_data.drop(['RatePlanLevel', 'LengthOfStayDayCnt', 'PersonCnt'], axis=1, inplace=True)
        return read_data

    def data_process(self, read_data):
        df_cdist = pd.DataFrame()
        Observe = 'CostAmt'
        read_data['z_score'] = stats.zscore(read_data[Observe])
        print(read_data.head(20))
        read_data = read_data.loc[read_data['z_score'].abs() <= 3]
        # TODO
        read_data_gp = read_data[['StayDate', Observe, 'RatePlanID']].groupby(['RatePlanID'], sort=False)
        df_corr = pd.DataFrame()

        for name, group in read_data_gp:
            group.reset_index(drop=True, inplace=True)
            df_corr[name] = group.set_index('StayDate')[Observe]

        df_corr.fillna(0, inplace=True)
        # https://blog.csdn.net/walking_visitor/article/details/85128461
        # 默认使用 pearson 相关系数计算方法，但这种方式存在误判
        # 全量数据
        df_corr = df_corr[[260281798, 260281804, 260281808, 260281855, 260281860, 260281863, 260281880, 260281894, 260281904, 260281911, 260281920, 260281932, 260281983, 260281991, 260281994, 260281995, 260281999, 260282006, 260282033, 260282043, 260282050, 260282056, 260282062, 260282064, 260282110, 260282123, 260282124, 260282172, 260282183, 260282188, 260332873, 260332875, 260332876, 260332877, 260332879, 260332880, 260332881, 260332882, 260332884, 260332886, 260332889, 260332891, 260332892, 260332893, 260332895, 260332896, 260332897, 260332898, 260332900, 260332902]]
        # 不包含点
        # df_corr = df_corr[[260281880]]
        # df_corr = df_corr[
        #     [260281798, 260281804, 260281808, 260281855, 260281860, 260281863, 260281880, 260281894, 260281904, 260281911,
        #      260281920, 260281932, 260281983, 260281991, 260281994, 260281995, 260281999, 260282006, 260282033, 260282043,
        #      260282050, 260282056, 260282062, 260282064, 260282110, 260282123, 260282124, 260282172, 260282183, 260282188]]
        # 与非线性数据多一个相关性
        # df_corr = df_corr[
        #     [260281798, 260281804, 260281808, 260281855, 260281860, 260281863, 260281880, 260281894, 260281904, 260281911,
        #      260281920, 260281932, 260281983, 260281991, 260281994, 260281995, 260281999, 260282006, 260282033, 260282043,
        #      260282050, 260282056, 260282062, 260282064, 260282110, 260282123, 260282124, 260282172, 260282183, 260282188, 260332873]]
        # 剩余集合
        # df_corr = df_corr[[260332873,260332875, 260332876, 260332877, 260332879, 260332880, 260332881, 260332882, 260332884, 260332886, 260332889, 260332891, 260332892, 260332893, 260332895, 260332896, 260332897, 260332898, 260332900, 260332902]]
        # cycle graph
        # df_corr = df_corr[[260282183, 260332873, 260332876, 260332877, 260282188, 260332879, 260332880, 260332881, 260332882, 260332884, 260332886, 260332889, 260332891, 260332892, 260332893, 260332895, 260332896, 260332897, 260332898, 260332900, 260332902]]
        df_corr = df_corr.corr()
        np.fill_diagonal(df_corr.values, 0)

        # df_corr = df_corr.mask(df_corr<0.99,other=-1)
        # plt.figure(figsize=(18, 7))
        # seaborn.heatmap(df_corr, center=0, annot=True, cmap='YlGnBu')
        print("draw graph !!")
        import networkx as nx
        # C = nx.from_numpy_matrix(np.array(df_corr >= 0.99)).degree
        graph = nx.Graph(df_corr)
        nx.draw(graph, with_labels = True)
        # import matplotlib.pyplot as plt
        # plt.pause(0)
        # 创建图的边(Edge)
        # graph.add_edges_from(C)
        # pos = nx.spring_layout(graph)
        # nx.draw(graph, pos, node_size=len(C), node_color=range(len(C)))
        #
        # edge_labels = nx.get_edge_attributes(graph, 'weight')
        # nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
        #
        # nx.draw_networkx_labels(graph,pos,alpha=0.5)
        # import matplotlib.pyplot as plt
        # plt.pause(0)

        # dfs = DFSFindCircle(inputDataFrame=df_corr >= 0.99)
        # has_circle = dfs.findcircle()
        # if has_circle:
        #     print(dfs.trace)
        graph = csr_matrix(df_corr >= 0.99)
        n, labels = csgraph.connected_components(graph, connection='strong')

        output_df = pd.DataFrame(columns=['GroupID', 'RatePlanID'])

        print('{}/{}'.format(n, len(read_data_gp.ngroup())))

        for i in range(n):
            nodes = df_corr.index[np.where(labels == i)]
            # if set(nodes.values.tolist())> set([260281880,ƒ,260281904]):
            #     temp_data = read_data.loc[(read_data['RatePlanID'].isin(np.array([260281880,260281894,260281904])))]
            #     temp_data.groupby(['StayDate', 'RatePlanID']).sum()[Observe].unstack().plot()
            #     plt.show()
            #     print("哈哈")
            df_cdist = df_cdist.append([[Observe, i, nodes.values]], ignore_index=True)

            fig, ax = plt.subplots(figsize=(18, 7))

            read_data.loc[(read_data['RatePlanID'].isin(nodes))].groupby(['StayDate', 'RatePlanID']).sum()[
                Observe].unstack().plot(ax=ax)

        df_cdist.columns = ['Observe', 'GroupID', 'Group']

        # df_cdist.to_csv('{}{}_patterngroup.csv'.format(OUTPUT_FOLDER, HotelID), index=False)

        plt.show()
        plt.close()

if __name__ == '__main__':
    hotelPatternTest = HotelPatternTest()
    read_data = hotelPatternTest.read_file()
    hotelPatternTest.data_process(read_data)