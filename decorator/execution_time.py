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
            logger.debug("[METHOD_NAME: {}] invocation time:{:.2}".format(func.__name__, start_time))
            res = func(*args, **kw)
            # logger.debug("[METHOD_NAME: {}] end execute".format(func.__name__))
            exec_time = time.time() - start_time
            logger.debug("[METHOD_NAME: {}] takes {:.2}s".format(func.__name__, exec_time))
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