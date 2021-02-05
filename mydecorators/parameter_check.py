from functools import wraps

def require_ints(func):
    @wraps(func)  # 将func的信息复制给inner
    def inner(*args, **kwargs):
        for arg in list(args) + list(kwargs.values()):
            if not isinstance(arg, int) : raise TypeError("{} 只接受int类型参数".format(func.__name__))
        return func(*args, **kwargs)
    return inner

if __name__ == '__main__':
    @require_ints
    def test(paramter):
        return paramter

    test("hello")