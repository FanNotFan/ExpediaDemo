import pymysql

class ConDb:
    def openClose(fun):
        def run(self, sql=None):
            # 创建数据库连接
            db = pymysql.connect(host='localhost', port=3306, user='root', password='Fanfanbh.89757', db='jeecg-boot', charset='utf8')
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