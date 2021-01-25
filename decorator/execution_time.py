import time
import functools
from tools import logger
logger = logger.Logger("debug")

def execute_time(arg):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            if arg and isinstance(arg, str):
                print('Decorator Pass Parameters：%s' % arg)
            start_time = time.time()
            time_local = time.localtime(int(start_time))
            show_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
            logger.debug("[METHOD_NAME: {}] invocation time: {}".format(func.__name__, show_start_time))
            res = func(*args, **kw)
            # logger.debug("[METHOD_NAME: {}] end execute".format(func.__name__))
            exec_time = time.time() - start_time
            logger.debug("[METHOD_NAME: {}] takes {:.2f}s".format(func.__name__, exec_time))
            return res
        return wrapper
    if callable(arg):
        return decorator(arg)
    return decorator


# without return
def decorator_calc_exec_time(arg):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            logger.debug("[METHOD_NAME: " + func.__name__ + "] start execute")
            start_time = time.time()
            func(*args, **kw)
            logger.debug("[METHOD_NAME: " + func.__name__ + "] end execute")
            end_time = time.time()
            # print("["+func.__name__+'] time consuming：%ss' % int(end_time - start_time))
            logger.debug("[METHOD_NAME: " + func.__name__ + '] time consuming: %ss' % int(end_time - start_time))
        return wrapper
    if callable(arg):
        return decorator(arg)
    return decorator