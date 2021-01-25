import json
from functools import wraps

class Error1(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


def json_output(func):
    @wraps(func)
    def inner(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
        except Error1 as ex:
            result = {"status": "error", "msg": str(ex)}
        return json.dumps(result)

    return inner


if __name__ == '__main__':
    # 使用方法
    @json_output
    def error():
        raise Error1("该条异常会被捕获并按JSON格式输出")