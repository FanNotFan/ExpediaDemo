funcs = []

def register(func):
    funcs.append(func)
    return func

if __name__ == '__main__':
    @register
    def a():
        return 3

    @register
    def b():
        return 5

    # 访问结果
    result = [func() for func in funcs]
    print(result)