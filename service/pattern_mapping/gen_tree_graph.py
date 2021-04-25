import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import scipy.spatial.distance as dis
from scipy.ndimage import filters
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.feature_extraction.text import CountVectorizer
from graphviz import Digraph
import cProfile
import os
import logging
import glob
import seaborn as sns
from matplotlib import cm
from settings import HOME_FOLDER, COL_CFG, Observe, GEN_TREE_GRAPH_INPUT_FOLDER, GEN_TREE_GRAPH_OUTPUT_FOLDER, GEN_TREE_GRAPH_INPUT_FOLDER2
logger = logging.getLogger()
logging.disable(logging.DEBUG)


class GenTreeGraph:

    def gaussian_filter(self, x):
        filtered_data = filters.gaussian_filter(x, sigma=20)
        return filtered_data

    def get_distance(self, abp_df, col_cfg):
        try:
            d_index = col_cfg[col_cfg['algo'] != 'None']['name'].tolist()
            d_weight = col_cfg[col_cfg['algo'] != 'None']['weight'].values.astype(np.float)
            level = d_weight.min() / d_weight.sum()

            rows = abp_df.copy()

            d_list = []
            for c in d_index:

                algo = col_cfg[col_cfg['name'] == c]['algo'].iloc[0]
                if algo == 'Dice':
                    one_hot = MultiLabelBinarizer()
                    d_list.append(pd.DataFrame(
                        dis.pdist(one_hot.fit_transform(rows[c].apply(lambda x: tuple(str(x).split(',')))), algo)))
                elif algo == 'cityblock':
                    ud = dis.pdist(rows[c].values.reshape(-1, 1), algo).reshape(-1, 1)
                    scaler = MinMaxScaler()
                    scaler.fit(ud)
                    d_list.append(pd.DataFrame(scaler.transform(ud)))
                elif algo == 'ngram':
                    corpus = rows[c]
                    v = CountVectorizer(ngram_range=(1, 3), binary=True, lowercase=True)
                    d_list.append(pd.DataFrame(dis.pdist(v.fit_transform(corpus).toarray(), 'Dice')))
                elif algo == 'None':
                    continue
                else:
                    print('error')

            dm = pd.concat(d_list, ignore_index=True, axis=1)
            dm.columns = d_index

            ag1 = (dm.values * d_weight).mean(axis=1)
            ag1_sq = dis.squareform(ag1)
            self.gaussian_filter(ag1_sq)
            np.fill_diagonal(ag1_sq, 1)

            # ag1_sq[ag1_sq==0] = 1
            distance_df = pd.DataFrame(ag1_sq)
            #     print(abp_df['RoomTypeID'].tolist())
            #         print(np.array(distance_df).tolist())
            result = []
            #         print('level',level)
            for row_index, row in distance_df.iterrows():
                for col_index, distance in row.iteritems():
                    rootid = str(abp_df.iloc[row_index].RatePlanID)
                    childid = str(abp_df.iloc[col_index].RatePlanID)
                    if distance <= level:
                        if self.check_oneroom(abp_df, rootid, childid) == True:
                            result.append([rootid, childid, distance])
            result_df = pd.DataFrame(np.array(result), columns=['root', 'child', 'distance'])
        except ValueError:
            result_df = pd.DataFrame()
        finally:
            return result_df

    # check 2 rateplan into 1 room,
    def check_oneroom(self, abp_df, rootid, childid):
        if len(set(abp_df[abp_df['RatePlanID'].isin([rootid, childid])]['RoomTypeID'])) > 1:
            return False
        else:
            return True

    def translation_offer(self, root_no, child_no, abp_df, comp_columns):
        root = abp_df[abp_df['RatePlanID'] == root_no].reset_index(drop=True)
        child = abp_df[abp_df['RatePlanID'] == child_no].reset_index(drop=True)

        # price = child.loc[0]['Price'].astype(int) - root.loc[0]['Price'].astype(int)

        offer = ''

        #  if price >= 0:
        #      offer += '+$' + str(price)
        #  else:
        #      offer += '-$' + str(abs(price))

        root_title = str(root.loc[0]['RatePlanID'])
        child_title = str(child.loc[0]['RatePlanID'])

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
                                     root['Value'] + ' => ' + child['Value'])

        root = root.loc[root['ValueMatch'] == False]
        root.drop(['Value', 'Value2', 'ValueMatch'], axis=1, inplace=True)

        offer = root['ValueDiff'].to_string()

        return [root_no, child_no, root_title, child_title, offer, 0, 0, 0]
        # root.loc[0]['Price'].astype(int), child.loc[0]['Price'].astype(int), price]


    def get_offer_list(self, abp_df, col_cfg, level):
        distance_df = self.get_distance(abp_df, col_cfg, level)
        if len(distance_df) == 0:
            level += col_cfg['weight'].astype(float).min() / col_cfg['weight'].astype(float).sum()
            offer_df, level = self.get_offer_list(abp_df, col_cfg, level)
        # get offer
        offer_list = []
        cols = col_cfg['name'].tolist()
        for index, row in distance_df.iterrows():
            offer_list.append(self.translation_offer(int(row['root']), int(row['child']), abp_df, cols))
        offer_df = pd.DataFrame(offer_list,
                                columns=['root', 'child', 'root_roomname', 'child_roomname', 'detail', 'root_price',
                                         'child_price', 'price'])
        offer_df = self.clean_offer(offer_df)

        if len(self.check_root(abp_df, offer_df)) > 1:
            level += col_cfg['weight'].astype(float).min() / col_cfg['weight'].astype(float).sum()
            offer_df, level = self.get_offer_list(abp_df, col_cfg, level)
        return offer_df, level

    # choose base
    def choose_base(self, abp_df, cols):

        c = cols.copy()
        c.append('RatePlanID')

        base_df = abp_df.copy()
        base_df = base_df.sort_values(by=['RatePlanID']).reset_index(drop=True)

        #     candidate_df = base_df[(base_df['Price']==base_df['Price'].min())]
        #     candidate = pd.DataFrame()
        #     # choose base room if it is a base room class, only hotel level
        #     if (len(candidate_df[candidate_df['RoomClass'].isin(['Basic', 'Standard'])]) > 0) & (rule == 'Hotel'):
        #         candidate = candidate_df[candidate_df['RoomClass'].isin(['Basic', 'Standard'])]
        #     else:
        #         candidate = candidate_df
        #     print(candidate)
        #     candidate = candidate[c].astype(str)connect_base
        #     # choose minimum options
        #     r = candidate.iloc[:,2:9].apply(lambda x: x.str.contains('No|-1|Non|False|Basic|Standard', regex=True))
        #     print('r:',(r == True).sum(axis=1).idxmax())
        #     index = (r == True).sum(axis=1).idxmax()
        #     print('baseIds0:',base_df.iloc[index]['RatePlanID']   )
        #     return base_df.iloc[index]['RatePlanID']

        return base_df.iloc[0]['RatePlanID']

    def check_root(self, abp_df, offer_df):
        rateplan_list = np.array(list(set(abp_df['RatePlanID'])))
        child_list = np.array(list(set(offer_df['child'])))
        mask_before = np.isin(rateplan_list, child_list)
        root_list = rateplan_list[~mask_before]
        return root_list

    def connect_base(self, abp_df, offer_df, cols):
        root_list = self.check_root(abp_df, offer_df)
        if len(root_list) > 1:
            base_df = abp_df[abp_df['RatePlanID'].isin(root_list)]
            baseIds = int(self.choose_base(base_df, cols))
            print('baseIds:', baseIds)
            base_df = base_df[base_df['RatePlanID'] != baseIds]
            s = base_df.apply(lambda x: pd.Series(self.translation_offer(baseIds, int(x['RatePlanID']), abp_df, cols)
                                                  , index=['root', 'child', 'root_roomname', 'child_roomname', 'detail',
                                                           'root_price', 'child_price', 'price'])
                              , axis=1)
            offer_df = offer_df.append(s, ignore_index=True)
        return offer_df

    def clean_offer(self, abp_df, cols, offer_df):
        # get base rateplans ids, base on roomtype id
        #     baseIds = choose_base(abp_df,cols,['RoomTypeId'])
        #     offer_df = offer_df.drop(offer_df[offer_df['child'].isin(baseIds)].index)
        out_offer = offer_df
        # Remove duplicate connections
        for index, row in offer_df.iterrows():
            rootid = int(row['root'])
            childid = int(row['child'])
            out_offer = out_offer.drop(offer_df[(offer_df['root'] == childid) & (offer_df['child'] == rootid)].index)
            offer_df = offer_df.drop(offer_df[(offer_df['root'] == childid) & (offer_df['child'] == rootid)].index)
            offer_df = offer_df.drop(offer_df[(offer_df['child'] == childid) & (offer_df['root'] == rootid)].index)

        # remove connection
        out_offer = out_offer.sort_values(by=['price'])
        out_offer = out_offer.drop_duplicates(subset=['child'], keep='first')
        return out_offer

    def multi_base(self, input_data_final, col_cfg):
        hotelid = input_data_final.iloc[0]['HotelId']
        roomtypeid = input_data_final.iloc[0]['RoomTypeID']

        file_name = str(hotelid) + '_' + str(roomtypeid)

        print('------', file_name, '------')
        cols = col_cfg['name'].tolist()
        # got distance
        distance_df = self.get_distance(input_data_final, col_cfg)

        if len(distance_df) > 1:
            # get offer
            offer_list = []
            for index, row in distance_df.iterrows():
                offer_list.append(self.translation_offer(int(row['root']), int(row['child']), input_data_final, cols))
            offer_df = pd.DataFrame(offer_list,
                                    columns=['root', 'child', 'root_roomname', 'child_roomname', 'detail', 'root_price',
                                             'child_price', 'price'])
            offer_df = self.clean_offer(input_data_final, cols, offer_df)
        else:
            offer_df = pd.DataFrame(
                columns=['root', 'child', 'root_roomname', 'child_roomname', 'detail', 'root_price', 'child_price',
                         'price'])

        # connect base if there has multi base
        offer_df = self.connect_base(input_data_final, offer_df, cols)
        # draw graph
        self.outputView(input_data_final, offer_df, hotelid, roomtypeid)
        return 'success'

    def check_offer(self, offer_df):
        if len(offer_df) < 1:
            return False
        else:
            return True

    def outputView(self, abp_df, offer_df, hotelid, roomtypeid):
        # output result
        offer_df.to_csv('{}{}_{}_raw.csv'.format(GEN_TREE_GRAPH_OUTPUT_FOLDER, hotelid, roomtypeid))

        gpfile = '{}{}_{}_gp.csv'.format(GEN_TREE_GRAPH_INPUT_FOLDER2, hotelid, roomtypeid)
        hasGP = False

        gp = pd.DataFrame(columns=['GroupID', 'RatePlanID'])

        if os.path.exists(gpfile):
            gp = pd.read_csv(gpfile, encoding='utf-8', sep=',', engine='python', header=0).fillna(0)

        dot = Digraph(comment='Product Graph')

        # add node

        palette = sns.light_palette("blue", 8)

        #     for RatePlanID in abp_df['RatePlanID'].values:
        #         dot.attr('node', shape='ellipse', style='filled', color='lightgrey')
        #         if RatePlanID in gp['RatePlanID'].values:
        #             groupID = int(gp.loc[gp['RatePlanID']==RatePlanID,['GroupID']].values[0])
        #             print(groupID)
        #             dot.attr('node', style='filled', color=','.join(map(str,palette[groupID])))
        #         dot.node(str(RatePlanID))

        gp_gp = gp.groupby(['GroupID'], sort=False)

        gp_gp_sort = gp.groupby(['GroupID']).count()

        print(gp_gp_sort)

        for name, group in gp_gp:
            with dot.subgraph(name=str(name)) as c:
                groupID = name
                palID = np.clip(gp_gp_sort.iloc[name][0], 0, 5)
                c.attr(color='blue')
                if len(group.index) > 1:
                    c.attr('node', style='filled', color=palette.as_hex()[palID])
                else:
                    c.attr('node', style='filled', fillcolor='white')
                for RatePlanID in group['RatePlanID'].values:
                    c.node(str(RatePlanID))

        for RatePlanID in abp_df[~abp_df.RatePlanID.isin(gp['RatePlanID'])]['RatePlanID'].values:
            dot.attr('node', style='filled', color='lightgrey')
            dot.node(str(RatePlanID))

        # abp_df.apply(lambda x : dot.node(str(x.RatePlanID),str(x.RatePlanID)), axis = 1)

        if self.check_offer(offer_df):
            offer_df.apply(lambda x: dot.edge(str(x.root), str(x.child), label=x.detail), axis=1)

            # draw
        dot.render(('{}{}_tree_pic'.format(GEN_TREE_GRAPH_OUTPUT_FOLDER, roomtypeid)), view=False, format='png')


    def genGraph(self, input_data):
        input_data_group = input_data.groupby(['HotelId', 'RoomTypeID'], sort=False)
        # with cProfile.Profile() as pr:
        i = 0
        for name, group in input_data_group:
            i += 1
            print('{}/{}'.format(i, input_data_group.ngroups))
            self.multi_base(group, COL_CFG)
        # pr.print_stats()


if __name__ == '__main__':
    os.chdir(HOME_FOLDER)
    HotelID = 16639
    RoomTypeID = 166628
    CLEANUP_OUTPUT = True
    if CLEANUP_OUTPUT == True:
        files = glob.glob(GEN_TREE_GRAPH_OUTPUT_FOLDER + '*.csv')
        files.extend(glob.glob(GEN_TREE_GRAPH_OUTPUT_FOLDER + '*_pic'))
        for f in files:
            os.remove(f)
    read_data_rt = pd.read_csv(GEN_TREE_GRAPH_INPUT_FOLDER + 'dbo_RoomType_NoIdent.csv', encoding='utf-8', sep=',', engine='python',
                               header=0).fillna(0)
    read_data_rt = read_data_rt[['SKUGroupID', 'RoomTypeID', 'ActiveStatusTypeID']]
    read_data_rt = read_data_rt.loc[
        (read_data_rt['ActiveStatusTypeID'] == 2) & (read_data_rt['RoomTypeID'] == RoomTypeID)]
    read_data_rt.drop(['ActiveStatusTypeID'], axis=1, inplace=True)
    read_data_rt = read_data_rt.loc[read_data_rt['SKUGroupID'].isin([HotelID])]

    #     892034 (Millennium Hilton New York Downtown)
    #     14411 (Embassy Suites by Hilton Seattle Bellevue)
    #     2612 (Hilton London Euston)
    #     442954 (Hilton Tokyo Narita Airport)
    #     49148638 (Hampton Inn & Suites Deptford, NJ)
    #     Hotel 9
    #     wikkiki hilton village

    #         558979
    #         ,1155964
    #         ,14388
    #         ,19692

    logger.debug(read_data_rt)
    read_data_rp = pd.read_csv(GEN_TREE_GRAPH_INPUT_FOLDER + 'dbo_RatePlan_NoIdent.csv', encoding='utf-8', sep=',', engine='python',
                               header=0).fillna(0)
    # "RatePlanID","RatePlanTypeID","RoomTypeID","ActiveStatusTypeID","RatePlanCodeSupplier","PersonCntIncluded",
    # "ManageOnExtranetBool","UpdateDate","UpdateTPID","UpdateTUID","CostCodeDefault","AllowInventoryLimitEditBool",
    # "RatePlanIDOriginal","ARIEnabledBool","WaiveTaxesBool","SKUGroupFeeSetID","SKUGroupCancelPolicySetID",
    # "SuppressionOverrideBool","RatePlanIDOriginalDC","SKUGroupMarginRuleSetID","ARIRolloutBool","RatePlanCostPriceTypeID",
    # "DOACostPriceBool","LOSCostPriceBool","RatePlanLogID","ChangeRequestID","SpecialDiscountPercent","BusinessModelMask",
    # "CostCodeDefaultAgency","SKUGroupMarginRuleSetIDAgency","DepositRequiredBool","SyncBookingOverrideBool","LastUpdatedBy","UpdateClientID"
    read_data_rp.drop(['UpdateTPID', 'ChangeRequestID', 'UpdateTUID'], axis=1, inplace=True)
    read_data_rp.drop(['UpdateDate', 'LastUpdatedBy', 'UpdateClientID', 'RatePlanLogID'], axis=1, inplace=True)
    # read_data_rp.drop(['RatePlanTypeID', 'ActiveStatusTypeID','CostCodeDefaultAgency'], axis=1, inplace=True)
    # read_data_rp = read_data_rp.set_index('RatePlanID').rename_axis(None)
    read_data_rp = read_data_rp.loc[read_data_rp['ActiveStatusTypeID'] == 2]
    read_data_hilton = pd.merge(read_data_rt, read_data_rp, how='inner', left_on='RoomTypeID', right_on='RoomTypeID')
    read_data_hilton.rename(columns={'SKUGroupID': 'HotelId'}, inplace=True)
    logger.debug(read_data_hilton)
    # Get hotel's information
    # Load configuration (input column names / roomtype and roomclass dict)
    # Fill missing columns
    # Group rooms (base on some attribute to group rooms, such as bedtype/roomtype/roomview etc.)
    # Get distance
    # Transform offer (base on distance from #5)
    # Clean offer (delete cyclic)
    # Connect base (connect base rooms for each grouping)
    # Package result
    input_data = read_data_hilton
    genTreeGraph = GenTreeGraph()
    genTreeGraph.genGraph(input_data)