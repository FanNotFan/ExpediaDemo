#!/usr/bin/env python
# encoding: utf-8
'''
@author: lennon
@license: (C) Copyright 2019-2020, Node Supply Chain Manager Corporation Limited.
@contact: v-lefan@expedia.com
@software: pycharm
@file: settings.py.py
@time: 2019-06-25 15:52
@desc:
'''

import os
import time
import platform
import pandas as pd
import numpy as np
from multiprocessing import cpu_count

# __file__ refers to the file settings.py

t = time.localtime(time.time())
foldername = str(t.__getattribute__("tm_year")) + "-" + str(t.__getattribute__("tm_mon")) + "-" + \
             str(t.__getattribute__("tm_mday"))

APP_ROOT = os.path.dirname(os.path.abspath(__file__))  # refers to application_top
APP_STATIC = os.path.join(APP_ROOT, 'static')
print("APP_STATIC: {}".format(APP_STATIC))

# get system info
PLATFORM_SYSTEM = platform.system()
print("PLATFORM_SYSTEM: {}".format(PLATFORM_SYSTEM))

LOCAL_CHROME_DRIVER = '/Users/hiCore/Software/WebDrivers/chromedriver_83'
GOOGLE_LOCAL_IMAGE_STORAGE_PATH = os.path.join(APP_STATIC, 'GoogleImage', foldername)
BAIDU_LOCAL_IMAGE_STORAGE_PATH = os.path.join(APP_STATIC, 'BaiduImage', foldername)
BING_LOCAL_IMAGE_STORAGE_PATH = os.path.join(APP_STATIC, 'BingImage', foldername)

# Number of processes opened 开启的进程数
NUMBER_OF_PROCESSES = cpu_count()

# HotelID = 16639
HOTEL_PATTERN_LOS = 1
HOTEL_PATTERN_PERSONCNT = 2
HOTEL_PATTERN_DATAVERSION = 2
HOTEL_PATTERN_RATEPLANLEVEL = 0

HOTEL_PATTERN_INPUT_FOLDER2 = './Data{}/'.format(HOTEL_PATTERN_DATAVERSION)
HOTEL_PATTERN_INPUT_FOLDER = './Data/'
HOTEL_PATTERN_OUTPUT_FOLDER = './Result/MINE2/'
HOTEL_PATTERN_Observes = ['CostAmt', 'PriceAmt', 'LARAmt', 'LARMarginAmt', 'LARTaxesAndFeesAmt']

PATTERN_ATTRIBUTE_CLEANUP_OUTPUT = True
PATTERN_ATTRIBUTE_OUTPUT_FOLDER = './Result/DAG.nosync/'
PATTERN_ATTRIBUTE_INPUT_FOLDER = './Data/'
PATTERN_ATTRIBUTE_INPUT_FOLDER2 = './Result/MINE2/'

PATTERN_MAPPING_INPUT_FOLDER = './Data2/'
PATTERN_MAPPING_OUTPUT_FOLDER = './Result/MINE2/'
PATTERN_MAPPING_INPUT_FOLDER2 = './Result/MINE2/'

COL_CFG = pd.DataFrame(
    np.array(
        [['RatePlanID', 'None', 1],
         ['RatePlanTypeID', 'Dice', 1],
         ['RoomTypeID', 'None', 1],
         ['ActiveStatusTypeID', 'None', 1],
         ['RatePlanCodeSupplier', 'Dice', 1],
         ['ManageOnExtranetBool', 'cityblock', 1],
         ['CostCodeDefault', 'Dice', 1],
         ['AllowInventoryLimitEditBool', 'cityblock', 1],
         ['RatePlanIDOriginal', 'None', 1],
         ['ARIEnabledBool', 'cityblock', 1],
         ['WaiveTaxesBool', 'cityblock', 1],
         ['SKUGroupFeeSetID', 'Dice', 1],
         ['SKUGroupCancelPolicySetID', 'Dice', 1],
         ['SuppressionOverrideBool', 'cityblock', 1],
         ['RatePlanIDOriginalDC', 'None', 1],
         ['SKUGroupMarginRuleSetID', 'Dice', 1],
         ['ARIRolloutBool', 'cityblock', 1],
         ['RatePlanCostPriceTypeID', 'Dice', 1],
         ['DOACostPriceBool', 'cityblock', 1],
         ['LOSCostPriceBool', 'cityblock', 1],
         ['SpecialDiscountPercent', 'cityblock', 1],
         ['SuppressionOverrideBool', 'None', 1],
         ['BusinessModelMask', 'Dice', 1],
         ['CostCodeDefaultAgency', 'Dice', 1],
         ['SKUGroupMarginRuleSetIDAgency', 'Dice', 1],
         ['DepositRequiredBool', 'cityblock', 1],
         ['SyncBookingOverrideBool', 'cityblock', 1],
         ]),
    columns=['name', 'algo', 'weight'])

# HOME_FOLDER = './'
# HOME_FOLDER = '/Users/xyao/Library/Mobile Documents/com~apple~CloudDocs/JupyterHome/Simplification/'
HOME_FOLDER = '/Users/lefan/Develope/Workspace_Pycharm/Simplification'
OUTPUT_RESULT_FILE_NAME = "{}_mapping_result.pdf"
OUTPUT_LINEAR_FILE_NAME = "{}{}_linear_result.jpg"
os.chdir(HOME_FOLDER)
DEBUG_LOG_PATH = os.path.join(APP_ROOT, "Log")
CONFIG_FILE_PATH = os.path.join(APP_ROOT, "config")
DOWNLOAD_PIC_VERSION = 'V1.0.0'
PATTERN_MAPPING_VERSION = 'V2.0.0'