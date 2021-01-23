import datetime
from tools.time_tool import TimeToolObject

class MappingFunction:
    def __init__(self):
        self.A = 0
        self.b = 0
        self.precision = 0
        self.exceptionPoint = {}

    def calculate(self, x, date):
        timeToolObject = TimeToolObject()
        specicDate = timeToolObject.convert_date_to_int(datetime.datetime.strptime(date, '%Y-%m-%d'))
        if x in self.exceptionPoint.keys():
            for cost, dates in self.exceptionPoint[x].items():
                if len(dates) == 0:
                    return cost

                if specicDate in dates:
                    return cost

        return round(self.A * x + self.b, self.precision)

    def validation(self, x, y, date):
        i = 0
        result = True
        while i < len(x):
            calculate_y = self.calculate(x[i][0], date[i])
            if calculate_y != y[i][0]:
                result = False
                # print('{}, {}, {}'.format(x[i][0], y[i][0], calculate_y))
            i += 1
        return result

    def __str__(self):
        return '{:.4f} * x {:+.4f}'.format(self.A, self.b)