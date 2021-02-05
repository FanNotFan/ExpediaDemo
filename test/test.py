def flatten(a):
    for each in a:
        if not isinstance(each, tuple):
            yield each
        else:
            yield from flatten(each)


list_temp = [(260332881, 260332880), (260332882, 260332880), (260332884, 260332880), (260332895, 260332880), (260332896, 260332880), (260332897, 260332880), (260332898, 260332880), (260332900, 260332880), (260332881, 260332879), (260332882, 260332879), (260332886, 260332879), (260332889, 260332879), (260332891, 260332879), (260332892, 260332879), (260332893, 260332879), (260332895, 260332879), (260332897, 260332879), (260332898, 260332879), (260332902, 260332879), (260332881, 260332877), (260332882, 260332877), (260332886, 260332877), (260332889, 260332877), (260332891, 260332877), (260332892, 260332877), (260332893, 260332877), (260332895, 260332877), (260332897, 260332877), (260332898, 260332877), (260332902, 260332877), (260332879, 260332876), (260332881, 260332876), (260332882, 260332876), (260332886, 260332876), (260332889, 260332876), (260332891, 260332876), (260332892, 260332876), (260332893, 260332876), (260332895, 260332876), (260332897, 260332876), (260332902, 260332876), (260332877, 260282188), (260332879, 260282188), (260332881, 260282188), (260332882, 260282188), (260332886, 260282188), (260332889, 260282188), (260332891, 260282188), (260332892, 260282188), (260332893, 260282188), (260332895, 260282188), (260332897, 260282188), (260332898, 260282188), (260332902, 260282188), (260332873, 260282183)]
print(list(set(flatten(list_temp))))
print(list(flatten(list_temp)))