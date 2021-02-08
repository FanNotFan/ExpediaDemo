import os
import pandas as pd
import numpy as np
from flask import Flask, send_file, make_response, url_for
from threading import Lock
from flask import request, redirect
from datetime import timedelta
from tools import logger
# 方法来渲染模板
# 将模板名和你想作为关键字的参数传入模板的变量
from flask import render_template, flash
from settings import HOME_FOLDER, OUTPUT_RESULT_FILE_NAME, OUTPUT_LINEAR_FILE_NAME, DEBUG_LOG_PATH, HOTEL_PATTERN_OUTPUT_FOLDER, PATTERN_ATTRIBUTE_OUTPUT_FOLDER, PATTERN_MAPPING_OUTPUT_FOLDER
from settings import PATTERN_MAPPING_VERSION, DOWNLOAD_PIC_VERSION
from service.image_download.googleimagedownload import GoogleCrawler
from service.image_download.baiduimagedownload import BaiduCrawler
from service.image_download.bingimagedownload import BingCrawler
from service.pattern_mapping.pattern_mapping import PatternMapping
from service.pattern_mapping.hotel_pattern import HotelPattern

logger = logger.Logger("debug")
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
# 设置缓存时间为1s
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)
thread = None
thread_lock = Lock()

# async_mode = None
# socketio = SocketIO(app)

# 打开调试模式：启用了调试支持，服务器会在代码修改后自动重新载入，并在发生错误时提供一个相当有用的调试器
# app.run(debug=True)

@app.route("/", methods=['GET', 'POST'])
@app.route('/mapping/')
def search_temple(name=None):
    return render_template('room.html', name=name, page_version=PATTERN_MAPPING_VERSION)


@app.route('/download_pic/')
def download_pic(name=None):
    return render_template('search.html', name=name, page_version=DOWNLOAD_PIC_VERSION)

@app.route("/export",methods = ['GET'])
def export():
    content = "long text"
    response = make_response(content)
    response.headers["Content-Disposition"] = "p_w_upload; filename=myfilename.txt"
    return response

@app.route("/export_mapping_result/?<string:hotelId>",methods = ['GET'])
def export_mapping_result(hotelId):
    response = make_response(send_file(os.path.join(HOME_FOLDER, OUTPUT_RESULT_FILE_NAME.format(hotelId))))
    response.headers["Content-Disposition"] = "p_w_upload; filename={};".format(OUTPUT_RESULT_FILE_NAME.format(hotelId))
    return response

@app.route("/pattern_mapping", methods=['GET', 'POST'])
def pattern_mapping():
    logger.debug("REQUEST_METHOD : {}".format(request.method))
    searchLevel = request.form.get("searchLevel")
    hotel_id = request.form.get('hotelId')
    room_id = request.form.get('roomId')
    observe = request.form.get('observe')
    enableCache = request.form.get('enableCache')
    group_id = request.form.get('groupID')
    message = request.form.get('message')
    personCnt = request.form.get('personCnt')
    ratePlanLevel = request.form.get('ratePlanLevel')
    lengthOfStayDayCnt = request.form.get('lengthOfStayDayCnt')
    hotel_id = hotel_id.strip()
    personCnt = personCnt.strip()
    ratePlanLevel = ratePlanLevel.strip()
    lengthOfStayDayCnt = lengthOfStayDayCnt.strip()
    logger.debug("hotel_id::" + str(hotel_id))
    if searchLevel is None or searchLevel == '':
        logger.debug("search level is none")
        flash('please input search level')
        return render_template("room.html", page_version=PATTERN_MAPPING_VERSION)

    if searchLevel =="Hotel" and (hotel_id is None or hotel_id == ''):
        logger.debug("hotelId is none")
        flash('please input hotelId')
        return render_template("room.html", page_version=PATTERN_MAPPING_VERSION)

    if searchLevel =="Room" and (room_id is None or room_id == ''):
        logger.debug("room id is none")
        flash('please input room id')
        return render_template("room.html", page_version=PATTERN_MAPPING_VERSION)

    if searchLevel =="Room":
        search_id = int(room_id)
    if searchLevel =="Hotel":
        search_id = int(hotel_id)

    if personCnt is None or personCnt == '' or int(personCnt) <= 0:
        logger.debug("personCnt value is none, default is 2")
        personCnt = 2
    if lengthOfStayDayCnt is None or lengthOfStayDayCnt == '' or int(lengthOfStayDayCnt) <= 0:
        logger.debug("lengthOfStayDayCnt value is none, default is 1")
        lengthOfStayDayCnt = 1
    logger.debug("todo")
    try:
        hotelPattern = HotelPattern()
        if searchLevel == "Room":
            read_data_rt = hotelPattern.read_file_dbo_RoomType_NoIdent_by_room_id(search_id)
            hotel_id = read_data_rt['SKUGroupID'].values.tolist()[0]

        gpfile = '{}{}_patterngroup.csv'.format(HOTEL_PATTERN_OUTPUT_FOLDER, search_id)
        all_pattern_group_image = '{}{}_all_pattern_group.png'.format(PATTERN_ATTRIBUTE_OUTPUT_FOLDER, search_id)
        gfimg = '{}{}_patterngroup.png'.format(PATTERN_ATTRIBUTE_OUTPUT_FOLDER, search_id)
        if os.path.exists(gpfile) and os.path.exists(gfimg) and os.path.exists(all_pattern_group_image) and enableCache=='true':
            logger.debug("Grouping file already exist!!")
            gp_df = pd.read_csv(gpfile, encoding='utf-8', sep=',', engine='python', header=0).fillna(0)
            best_group_id = np.random.choice(
                gp_df["RatePlanLen"][gp_df["RatePlanLen"] == gp_df["RatePlanLen"].max()].index)
            logger.debug("The best group index is group_{}".format(int(best_group_id)+1))
            global_bast_group_id = best_group_id
        else:
            # Room级别: 在 dbo_RoomType_Noident.csv 文件中通过 RoomId 查询 HotelId
            if searchLevel == "Room":
                read_data_rp, read_data = hotelPattern.read_csv_data_and_filter_by_room_type_id(search_id, hotel_id)
            if searchLevel == "Hotel":
                read_data_rt, read_data_rp, read_data = hotelPattern.read_csv_data_and_filter(search_id)
            df_cdist, best_group_id = hotelPattern.generate_group_file_and_img(read_data, search_id)
            global_bast_group_id = best_group_id

        linear_relationship_image = OUTPUT_LINEAR_FILE_NAME.format(PATTERN_ATTRIBUTE_OUTPUT_FOLDER, search_id)
        mapping_function_file = '{}{}_{}_{}_mappingFunction.csv'.format(PATTERN_MAPPING_OUTPUT_FOLDER, search_id, global_bast_group_id,observe)
        if os.path.exists(linear_relationship_image) and os.path.exists(mapping_function_file) and enableCache=='true':
            logger.debug("Mapping function file already exist!!")
            patternMappingInstance = PatternMapping(search_id, int(hotel_id), searchLevel, observe, int(global_bast_group_id),
                                                    int(ratePlanLevel), int(lengthOfStayDayCnt), int(personCnt))
            mappingFunctionResult = pd.read_csv(mapping_function_file, encoding='utf-8', sep=',', engine='python', header=0).fillna(0)
            patternMappingInstance.generate_report(mappingFunctionResult)
        else:
            patternMappingInstance = PatternMapping(search_id, int(hotel_id), searchLevel, observe, int(global_bast_group_id), int(ratePlanLevel),int(lengthOfStayDayCnt), int(personCnt))
            rate_plan_list_ids, read_data = patternMappingInstance.read_and_preprocess_csv_file()
            mappingFunctionResult = patternMappingInstance.linear_prediction(rate_plan_list_ids, read_data)
            patternMappingInstance.generate_report(mappingFunctionResult)
        flash('Pattern Mapping Had Finished !!!')
        return redirect(url_for('export_mapping_result',hotelId=search_id))
    except Exception as e:
        print('Error: ' + str(e))
        flash(str(e))
        return render_template("room.html", page_version=PATTERN_MAPPING_VERSION)


@app.route("/download", methods=['GET', 'POST'])
def download():
    logger.debug("REQUEST_METHOD : {}".format(request.method))
    searchEngine = request.form.get('searchEngine')
    if searchEngine is None or searchEngine == '':
        searchEngine = "Google"
        logger.debug("searchEngine : {}".format(searchEngine))
    else:
        logger.debug("searchEngine : {}".format(searchEngine))
    keyword = request.form.get('keyword')
    numberOfImages = request.form.get('numberOfImages')
    keyword = keyword.strip()
    numberOfImages = numberOfImages.strip()
    print("keyword::" + str(keyword))
    if keyword is None or keyword == '':
        print("keyword is none")
        flash('please input keyword')
        return redirect("/search")
    if numberOfImages is None or numberOfImages == '':
        print("keyword is none")
        numberOfImages = 10
    print("todo")
    try:
        if searchEngine == "Google":
            print("Google 执行中")
            craw = GoogleCrawler()
            craw.run(keyword)
        if searchEngine == "Baidu":
            print("Baidu 执行中")
            craw = BaiduCrawler()
            craw.run(keyword)
        if searchEngine == "Bing":
            print("Bing 执行中")
            craw = BingCrawler()
            craw.run(keyword, numberOfImages)
        else:
            flash('engine error')
            return redirect("/search")
        flash('Download has finished!!!')
        return redirect("/search")
    except Exception as e:
        print('Error: ' + str(e))
        return redirect("/search")


@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404


@app.errorhandler(500)
def page_not_found(error):
    return render_template('500.html'), 500


# Get data from queue and push to front
# def background_thread():
#     while True:
#         if logF.not_empty:
#             socketio.emit('server_response', logF.pop(), namespace='/log')


# @socketio.on('connect', namespace='/log')
# def log_socket():
#     global thread
#     with thread_lock:
#         if thread is None:
#             thread = socketio.start_background_task(target=background_thread)


if __name__ == '__main__':
    app.run(debug=True)
