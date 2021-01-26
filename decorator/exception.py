import traceback
from tools import logger
from functools import wraps
from datetime import datetime
logger = logger.Logger("debug")

# 异常输出
def except_output(msg='Exception Message'):
    # msg用于自定义函数的提示信息
    def except_execute(func):
        @wraps(func)
        def execept_print(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                sign = '=' * 60 + '\n'
                # print(f'{sign}>>>Time：\t{datetime.now()}\n>>>Method Name：\t{func.__name__}\n>>>{msg}：\t{e}')
                # print(f'{sign}{traceback.format_exc()}{sign}')
                logger.debug(f'{sign}>>>Time：\t{datetime.now()}\n>>>Method Name：\t{func.__name__}\n>>>{msg}：\t{e}')
                logger.debug(f'{sign}{traceback.format_exc()}{sign}')
        return execept_print
    return except_execute

if __name__ == '__main__':
    @except_output()
    def lig(a=5, b=0):
        print(a / b)
    lig()