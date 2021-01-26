import os
import sys
import time
import logging
from settings import DEBUG_LOG_PATH
from decorator.singleton import Singleton

@Singleton  # 如需打印不同路径的日志（运行日志、审计日志），则不能使用单例模式（注释或删除此行）。此外，还需设定参数name。
class Logger:
    def __init__(self, set_level="INFO",
                 name=os.path.split(os.path.splitext(sys.argv[0])[0])[-1],
                 log_name=time.strftime("%Y-%m-%d.log", time.localtime()),
                 # log_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "log"),
                 log_path=DEBUG_LOG_PATH,
                 use_console=True):
        """
        :param set_level: 日志级别["NOTSET"|"DEBUG"|"INFO"|"WARNING"|"ERROR"|"CRITICAL"]，默认为INFO
        :param name: 日志中打印的name，默认为运行程序的name
        :param log_name: 日志文件的名字，默认为当前时间（年-月-日.log）
        :param log_path: 日志文件夹的路径，默认为logger.py同级目录中的log文件夹
        :param use_console: 是否在控制台打印，默认为True
        """
        if not set_level:
            set_level = self._exec_type()  # 设置set_level为None，自动获取当前运行模式
        self.__logger = logging.getLogger(name)
        self.setLevel(
            getattr(logging, set_level.upper()) if hasattr(logging, set_level.upper()) else logging.INFO)  # 设置日志级别
        if not os.path.exists(log_path):  # 创建日志目录
            os.makedirs(log_path)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler_list = list()
        handler_list.append(logging.FileHandler(os.path.join(log_path, log_name), mode='w', encoding="utf-8"))
        if use_console:
            handler_list.append(logging.StreamHandler())
        for handler in handler_list:
            handler.setFormatter(formatter)
            self.addHandler(handler)

    def __getattr__(self, item):
        return getattr(self.logger, item)

    @property
    def logger(self):
        return self.__logger

    @logger.setter
    def logger(self, func):
        self.__logger = func

    def _exec_type(self):
        return "DEBUG" if os.environ.get("IPYTHONENABLE") else "INFO"

if __name__ == '__main__':
    x = Logger("debug")
    x.critical("这是一个 critical 级别的问题！")
    x.error("这是一个 error 级别的问题！")
    x.warning("这是一个 warning 级别的问题！")
    x.info("这是一个 info 级别的问题！")
    x.debug("这是一个 debug 级别的问题！")

    x.log(50, "这是一个 critical 级别的问题的另一种写法！")
    x.log(40, "这是一个 error 级别的问题的另一种写法！")
    x.log(30, "这是一个 warning 级别的问题的另一种写法！")
    x.log(20, "这是一个 info 级别的问题的另一种写法！")
    x.log(10, "这是一个 debug 级别的问题的另一种写法！")

    x.log(51, "这是一个 Level 51 级别的问题！")
    x.log(11, "这是一个 Level 11 级别的问题！")
    x.log(9, "这条日志等级低于 debug，不会被打印")
    x.log(0, "这条日志同样不会被打印")

    """
    运行结果：
    2018-10-12 00:18:06,562 - demo - CRITICAL - 这是一个 critical 级别的问题！
    2018-10-12 00:18:06,562 - demo - ERROR - 这是一个 error 级别的问题！
    2018-10-12 00:18:06,562 - demo - WARNING - 这是一个 warning 级别的问题！
    2018-10-12 00:18:06,562 - demo - INFO - 这是一个 info 级别的问题！
    2018-10-12 00:18:06,562 - demo - DEBUG - 这是一个 debug 级别的问题！
    2018-10-12 00:18:06,562 - demo - CRITICAL - 这是一个 critical 级别的问题的另一种写法！
    2018-10-12 00:18:06,562 - demo - ERROR - 这是一个 error 级别的问题的另一种写法！
    2018-10-12 00:18:06,562 - demo - WARNING - 这是一个 warning 级别的问题的另一种写法！
    2018-10-12 00:18:06,562 - demo - INFO - 这是一个 info 级别的问题的另一种写法！
    2018-10-12 00:18:06,562 - demo - DEBUG - 这是一个 debug 级别的问题的另一种写法！
    2018-10-12 00:18:06,562 - demo - Level 51 - 这是一个 Level 51 级别的问题！
    2018-10-12 00:18:06,562 - demo - Level 11 - 这是一个 Level 11 级别的问题！
    """