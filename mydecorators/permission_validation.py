from functools import wraps


class User(object):
    def __init__(self, username, email):
        self.username = username
        self.email = email


class AnonymousUser(object):
    def __init__(self):
        self.username = self.email = None

    def __nonzero__(self):  # 将对象转换为bool类型时调用
        return False


def requires_user(func):
    @wraps(func)
    def inner(user, *args, **kwargs):  # 由于第一个参数无法支持self, 该装饰器不支持装饰类
        if user and isinstance(user, User):
            return func(user, *args, **kwargs)
        else:
            raise ValueError("非合法用户")

    return inner