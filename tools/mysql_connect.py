import os
import json
import yaml
import pymysql
from settings import CONFIG_FILE_PATH

class ConDb:
    def read_yaml(self):
        yamlPath = os.path.join(CONFIG_FILE_PATH, "mysql.yml")
        with open(yamlPath, encoding='utf-8') as f:
            file_content = f.read()
        # print("config:{}".format(file_content))
        data = yaml.load(file_content, Loader=yaml.FullLoader)
        return data

    def openClose(fun):
        def run(self, sql=None):
            # 读取配置文件
            config = self.read_yaml()
            # 创建数据库连接
            db = pymysql.connect(host=config['mysql']['config']['host'], port=config['mysql']['config']['port'], user=config['mysql']['config']['user'], password=config['mysql']['config']['password'], db=config['mysql']['config']['database'], charset=config['mysql']['parameters']['charset'])
            # 创建游标
            cursor = db.cursor()
            try:
                # 运行sql语句
                cursor.execute(fun(self, sql))
                # 得到返回值
                li = cursor.fetchall()
                # 提交事务
                db.commit()
            except Exception as e:
                # 如果出现错误，回滚事务
                db.rollback()
                # 打印报错信息
                print('运行', str(fun), '方法时出现错误，错误代码：', e)
            finally:
                # 关闭游标和数据库连接
                cursor.close()
                db.close()
            try:
                # 返回sql执行信息
                return li
            except:
                print('没有得到返回值，请检查代码，该信息出现在ConDb类中的装饰器方法')

        return run

    @openClose
    def runSql(self, sql=None):
        if sql is None:
            sql = 'select * from demo'
        return sql

    @openClose
    def runSql1(self, sql=None):
        return sql

if __name__ == '__main__':
    conDb = ConDb()
    exec_sql = 'select * from sys_depart'
    result = conDb.runSql(exec_sql)
    print(result)