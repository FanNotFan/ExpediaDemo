import pandas as pd
import numpy as np
import os
import glob
from settings import PATTERN_ATTRIBUTE_CLEANUP_OUTPUT
from settings import PATTERN_ATTRIBUTE_OUTPUT_FOLDER, PATTERN_ATTRIBUTE_INPUT_FOLDER
from tools import logger
from service.pattern_mapping.hotel_pattern import HotelPattern
logger = logger.Logger("debug")

class PatternAttribute:

    def main(self, group_rate_plan_ids, read_data_rt, read_data_rp):
        read_data_hilton = pd.merge(read_data_rt, read_data_rp, how='inner', left_on='RoomTypeID', right_on='RoomTypeID')
        read_data_hilton.rename(columns={'SKUGroupID': 'HotelId'}, inplace=True)
        logger.debug(read_data_hilton)
        input_data = read_data_hilton.loc[read_data_hilton['RatePlanID'].isin(group_rate_plan_ids)]
        self.translation_offer(260281798, 260281994, input_data)


    def translation_offer(self, root_no, child_no, abp_df):
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
                                     root['Value'] + ' → ' + child['Value'])
        root = root.loc[root['ValueMatch'] == False]
        root.drop(['Value', 'Value2', 'ValueMatch'], axis=1, inplace=True)
        delta = root['ValueDiff'].to_string()
        print("delta:{}".format(delta))


if __name__ == '__main__':
    hotel_id = 16639
    group_rate_plan_ids = [260281798, 260281804, 260281808, 260281823, 260281846, 260281848, 260281855
                                                 , 260281860, 260281863, 260281880, 260281894, 260281904, 260281911,
                                              260281920
                                                 , 260281932, 260281941, 260281950, 260281956, 260281983, 260281991,
                                              260281994
                                                 , 260281995, 260281999, 260282006, 260282033, 260282043, 260282050,
                                              260282056
                                                 , 260282062, 260282064, 260282087, 260282088, 260282099, 260282110,
                                              260282123
                                                 , 260282124, 260282172, 260282183, 260282188, 260282230, 260282243,
                                              260282262
                                                 , 260332873, 260332874, 260332875, 260332876, 260332877, 260332879,
                                              260332880
                                                 , 260332881, 260332882, 260332883, 260332884, 260332878, 260332886,
                                              260332888
                                                 , 260332889, 260332890, 260332891, 260332892, 260332893, 260332894,
                                              260332895
                                                 , 260332896, 260332897, 260332898, 260332899, 260332900, 260332904,
                                              260332902]
    hotelPattern = HotelPattern()
    read_data_rt, read_data_rp, read_data = hotelPattern.read_csv_data_and_filter(hotel_id)
    # df_cdist, best_group_id = hotelPattern.generate_group_file_and_img(read_data, hotel_id)
    # group_rate_plan_ids = df_cdist.loc[(df_cdist['GroupID'] == 1) & (df_cdist['Observe'] == 'CostAmt')]['Group'].iloc[0].tolist()
    patternAttribute = PatternAttribute()
    patternAttribute.main(group_rate_plan_ids, read_data_rt, read_data_rp)

