import pandas as pd
from settings import HOTEL_PATTERN_INPUT_FOLDER, PATTERN_ATTRIBUTE_OUTPUT_FOLDER, HOTEL_PATTERN_INPUT_FOLDER2

hotel_id_list = [16639,862,1797,6362,12079,12160,12800,14388,15144,15212,19692,19961,21327,22673,24662,27740,42448,42526,50886,60561,67394,67970,197433,208594,215262,281521,281536,281920,281939,297146,328177,424868,426205,454849,519720,519726,531180,531190,538638,787571,890490,909717,973778,1074382,1155964,1172342,1191979,1228332,1246232,1321942,1545120,1601018,1636325,1781224,1784929,1793627,1844848,2009510,2009515,2046329,2058961,2147324,2150460,2163016,2191649,2239837,2246517,2270665,2270677,2292826,2351486,2406330,2418923,2419417,2597717,2602846,2774577,2918077,3508863,3859038,5977476,6257605,6282047,6384898,6502660,6828402,7362466,7714604,7757367,8150608,8362336,8474909,8731496,8745457,9261405,18109739,23251275,23379342,23830678,23977136,27041847,27041849,27238090,27238095,27373038,29098521,30073675,30435488,30449392,30473613,31356803,32749220,32911452,33213151,35129611,35521694,35562140,36447066,36501393,38318741,41979240,42839826,45970429,45976006,48251410,50706495,51263687,53635112,54845071,55259840,55573268]
# hotel_id_list = [16639]
# read_data_rt = pd.read_csv(HOTEL_PATTERN_INPUT_FOLDER + 'dbo_RoomType_NoIdent.csv', encoding='utf-8', sep=',',
#                                    engine='python',
#                                    header=0).fillna(0)
# read_data_rt = read_data_rt[['SKUGroupID', 'RoomTypeID', 'ActiveStatusTypeID']]
# read_data_rt = read_data_rt.loc[read_data_rt['ActiveStatusTypeID'] == 2]
# read_data_rt.drop(['ActiveStatusTypeID'], axis=1, inplace=True)
# read_data_rt = read_data_rt.loc[read_data_rt['SKUGroupID'].isin(hotel_id_list)]
#
# read_data_rp = pd.read_csv(HOTEL_PATTERN_INPUT_FOLDER + 'dbo_RatePlan_NoIdent.csv', encoding='utf-8', sep=',',
#                                    engine='python',
#                                    header=0).fillna(0)
# read_data_rp.drop(['UpdateTPID', 'ChangeRequestID', 'UpdateTUID'], axis=1, inplace=True)
# read_data_rp.drop(['UpdateDate', 'LastUpdatedBy', 'UpdateClientID', 'RatePlanLogID'], axis=1, inplace=True)
# read_data_rp = read_data_rp.loc[(read_data_rp['ActiveStatusTypeID'] == 2) & (read_data_rp['RoomTypeID'].isin(read_data_rt['RoomTypeID'].values.tolist()))]
#
# room_type_id_list = read_data_rp.groupby(['RoomTypeID'], sort=False).count().sort_values(by=['RatePlanID'], ascending=False).head(30).index.tolist()
# print(room_type_id_list)
# read_data_rp[["RatePlanID","RoomTypeID"]].groupby(['RoomTypeID'], sort=False).count().sort_values(by=['RatePlanID'], ascending=False).head(30).to_csv('{}{}.csv'.format(PATTERN_ATTRIBUTE_OUTPUT_FOLDER, "RoomRatePlanStatistics"))


def read_file_dbo_RoomType_NoIdent_by_room_id(room_type_id):
    print("read file dbo_RoomType_NoIdent.csv")
    read_data_rt = pd.read_csv(HOTEL_PATTERN_INPUT_FOLDER + 'dbo_RoomType_NoIdent.csv', encoding='utf-8', sep=',',
                               engine='python',
                               header=0).fillna(0)
    read_data_rt = read_data_rt[['SKUGroupID', 'RoomTypeID', 'ActiveStatusTypeID']]
    read_data_rt = read_data_rt.loc[read_data_rt['ActiveStatusTypeID'] == 2]
    read_data_rt.drop(['ActiveStatusTypeID'], axis=1, inplace=True)
    read_data_rt = read_data_rt.loc[read_data_rt['RoomTypeID'].isin([room_type_id])]
    return read_data_rt



# if __name__ == '__main__':
#     read_data_rt = read_file_dbo_RoomType_NoIdent_by_room_id(193413)
#     hotel_id = read_data_rt['SKUGroupID'].values.tolist()[0]
#     print(hotel_id)

# 倒着找
# for hotel_id in hotel_id_list:
#     read_data = pd.read_csv(HOTEL_PATTERN_INPUT_FOLDER2 + str(hotel_id) + '_RatePlanLevelCostPrice.csv.zip',
#                                 sep=',', engine='python',
#                                 header=0).fillna(0)
#     read_data = read_data.loc[read_data['ActiveStatusTypeID'] == 2]
#     read_data = read_data[["RoomTypeID", "RatePlanID"]]
#     read_data_gp = read_data.drop_duplicates("RatePlanID").groupby(['RoomTypeID'], sort=False)
#     df_rate_plan = pd.DataFrame()
#     for name, group in read_data_gp:
#         print("group_name:{}".format(name))
#         print(group['RatePlanID'].tolist())
#         df_rate_plan = df_rate_plan.append(pd.DataFrame([[name, len(group['RatePlanID'].tolist()), group['RatePlanID'].tolist()]]))
#     read_data_group_df = read_data_gp.count().sort_values(by=['RatePlanID'], ascending=False).head(30)
#     df_rate_plan.columns = ['RoomTypeId', 'Count', 'RatePlanIDList']
#     df_rate_plan.sort_values(by='Count', ascending=False, inplace=True)
#     df_rate_plan.to_csv('{}{}_{}.csv'.format(PATTERN_ATTRIBUTE_OUTPUT_FOLDER, hotel_id, "RoomRatePlanStatistics"))


# 取交集
# hotel_id = 16639
# RATEPLANLEVEL = 0
# LOS = 1
# PERSONCNT = 2
# read_data = pd.read_csv(HOTEL_PATTERN_INPUT_FOLDER2 + str(hotel_id) + '_RatePlanLevelCostPrice.csv.zip',
#                                 sep=',', engine='python',
#                                 header=0).fillna(0)
#
# read_data = read_data.loc[(read_data['RatePlanLevel'] == RATEPLANLEVEL) & (read_data['LengthOfStayDayCnt'] == LOS) \
#                           & (read_data['PersonCnt'] == PERSONCNT)]
# # read_data = read_data[['StayDate', self.observe, 'RatePlanID']]
# read_data = read_data[['RatePlanID', 'RoomTypeID']]
# read_data.drop_duplicates(["RatePlanID"],inplace=True)
#
# read_data_rp = pd.read_csv(HOTEL_PATTERN_INPUT_FOLDER + 'dbo_RatePlan_NoIdent.csv', encoding='utf-8', sep=',',engine='python',header=0).fillna(0)
# read_data_rp.drop(['UpdateTPID', 'ChangeRequestID', 'UpdateTUID'], axis=1, inplace=True)
# read_data_rp.drop(['UpdateDate', 'LastUpdatedBy', 'UpdateClientID', 'RatePlanLogID'], axis=1, inplace=True)
# read_data_rp = read_data_rp.loc[read_data_rp['ActiveStatusTypeID'] == 2]
# read_data_rp = read_data_rp[["RatePlanID", "RoomTypeID"]]
# read_data_gp = pd.merge(read_data, read_data_rp, how='inner', on=["RoomTypeID","RatePlanID"]).groupby("RoomTypeID")
#
# df_rate_plan = pd.DataFrame()
# for name, group in read_data_gp:
#     print("group_name:{}".format(name))
#     print(group['RatePlanID'].tolist())
#     df_rate_plan = df_rate_plan.append(pd.DataFrame([[name, len(group['RatePlanID'].tolist()), group['RatePlanID'].tolist()]]))
#
# df_rate_plan.columns = ['RoomTypeId', 'Count', 'RatePlanIDList']
# df_rate_plan.sort_values(by='Count', ascending=False, inplace=True)
# df_rate_plan.to_csv('{}{}_{}.csv'.format(PATTERN_ATTRIBUTE_OUTPUT_FOLDER, hotel_id, "RoomRatePlanStatistics"))



# for hotel_id in hotel_id_list:
#     # 通过Hotel_ID 获取所有Active的RoomTypeID
#     read_data = pd.read_csv(HOTEL_PATTERN_INPUT_FOLDER2 + str(hotel_id) + '_RatePlanLevelCostPrice.csv.zip',
#                                 sep=',', engine='python',
#                                 header=0).fillna(0)
#     read_data = read_data.loc[read_data['ActiveStatusTypeID'] == 2]
#     read_data = read_data[["RoomTypeID", "RatePlanID"]]
#     read_data_gp = read_data.drop_duplicates("RatePlanID").groupby(['RoomTypeID'], sort=False)
#     df_rate_plan = pd.DataFrame()
#     for name, group in read_data_gp:
#         print("group_name:{}".format(name))
#         print(group['RatePlanID'].tolist())
#         df_rate_plan = df_rate_plan.append(pd.DataFrame([[name, len(group['RatePlanID'].tolist()), group['RatePlanID'].tolist()]]))
#     read_data_group_df = read_data_gp.count().sort_values(by=['RatePlanID'], ascending=False).head(30)
#     df_rate_plan.columns = ['RoomTypeId', 'Count', 'RatePlanIDList']
#     df_rate_plan.sort_values(by='Count', ascending=False, inplace=True)
#     df_rate_plan.to_csv('{}{}_{}.csv'.format(PATTERN_ATTRIBUTE_OUTPUT_FOLDER, hotel_id, "RoomRatePlanStatistics"))




# for hotel_id in hotel_id_list:
# HOTEL_PATTERN_LOS = 1
# HOTEL_PATTERN_PERSONCNT = 2
# HOTEL_PATTERN_RATEPLANLEVEL = 0
# hotel_id = 16639
# read_data = pd.read_csv(HOTEL_PATTERN_INPUT_FOLDER2 + str(hotel_id) + '_RatePlanLevelCostPrice.csv.zip',
#                             sep=',', engine='python',
#                             header=0).fillna(0)
# read_data = read_data.loc[read_data['ActiveStatusTypeID'] == 2]
# read_data.drop(['ActiveStatusTypeID', 'RatePlanLevelCostPriceLogSeqNbr', 'ChangeRequestIDOld'], axis=1,
#                    inplace=True)
# read_data.drop(['SupplierUpdateDate', 'SupplierUpdateTPID', 'SupplierUpdateTUID'], axis=1, inplace=True)
# read_data.drop(['UpdateDate', 'SupplierLogSeqNbr', 'ChangeRequestID'], axis=1, inplace=True)
# read_data = read_data.loc[(read_data['RatePlanLevel'] == HOTEL_PATTERN_RATEPLANLEVEL) & (
#                 read_data['LengthOfStayDayCnt'] == HOTEL_PATTERN_LOS)
#                               & (read_data['PersonCnt'] == HOTEL_PATTERN_PERSONCNT)]
#
# read_data.drop(['RatePlanLevel', 'LengthOfStayDayCnt', 'PersonCnt'], axis=1, inplace=True)
# read_data = read_data[["StayDate", "CostAmt", "RoomTypeID", "RatePlanID"]]
#
# read_data_gp = read_data[['StayDate', "CostAmt", 'RatePlanID']].groupby(['RatePlanID'], sort=False)
# df_corr = pd.DataFrame()
# for name, group in read_data_gp:
#     group.reset_index(drop=True, inplace=True)
#     df_corr[name] = group.set_index('StayDate')["CostAmt"]
# df_corr.fillna(0, inplace=True)
# 删除缺失值比例大于30%
# s1 = (df_corr.isnull().sum() / df_corr.shape[0]) >= 100  # 得到缺失值的比例大于30%
# df_corr_copy = df_corr[s1[s1 == False].index.tolist()]
# print("llll")
# df_rate_plan.to_csv('{}{}_{}.csv'.format(PATTERN_ATTRIBUTE_OUTPUT_FOLDER, hotel_id, "RoomRatePlanStatistics"))

[260281795, 260281796, 260281837, 260281842, 260281852, 260281853, 260281917, 260281918, 260281973, 260281977, 260282002, 260282003, 260282028, 260282030, 260282055, 260282057, 260282083, 260282084, 260282115, 260282118, 260282169, 260282170, 260282226, 260282228]