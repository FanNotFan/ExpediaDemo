import pandas as pd
import numpy as np
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Table, SimpleDocTemplate, Paragraph
from reportlab.lib.pagesizes import letter, elevenSeventeen
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.shapes import Drawing
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.units import mm, inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from settings import PATTERN_MAPPING_OUTPUT_FOLDER,PATTERN_ATTRIBUTE_OUTPUT_FOLDER

# 注册字体
# 支持中文下载SimSun.ttf字体，并把它放在/ python3.7/site-packages/reportlab/fonts文件夹下
# https://github.com/StellarCN/scp_zh/tree/master/fonts
pdfmetrics.registerFont(TTFont('SimSun', 'SimSun.ttf'))


class Graphs:
    def __init__(self):
        pass

    # 绘制标题
    @staticmethod
    def draw_title():
        style = getSampleStyleSheet()
        ct = style['Title']
        ct.fontName = 'SimSun'
        ct.fontSize = 20
        # 设置行距
        ct.leading = 50
        # 颜色
        ct.textColor = colors.grey
        # 居中
        ct.alignment = 1
        # 添加标题并居中
        title = Paragraph('Mapping Function Research Report', ct)
        return title

    # 绘制内容
    @staticmethod
    def draw_text(content):
        style = getSampleStyleSheet()
        # 常规字体(非粗体或斜体)
        ct = style['BodyText']
        # 使用的字体s
        ct.fontName = 'SimSun'
        ct.fontSize = 14
        # 设置自动换行
        ct.wordWrap = 'CJK'
        # 居左对齐
        ct.alignment = 0
        # 第一行开头空格
        ct.firstLineIndent = 32
        # 设置行距
        ct.leading = 30
        text = Paragraph(content, ct)
        return text

    # 绘制表格
    @staticmethod
    def draw_table(*args):
        col_width = 120
        style = [
            ('FONTNAME', (0, 0), (-1, -1), 'SimSun'),  # 字体
            ('BACKGROUND', (0, 0), (-1, 0), '#d5dae6'),  # 设置第一行背景颜色
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # 对齐
            ('VALIGN', (-1, 0), (-2, 0), 'MIDDLE'),  # 对齐
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),  # 设置表格框线为grey色，线宽为0.5
        ]
        table = Table(args, colWidths=col_width, style=style)
        return table

    # 创建图表
    @staticmethod
    def draw_bar(bar_data=[], ax=[], items=[], max_value=100):
        drawing = Drawing(440, 200)
        drawing.hAlign = 'CENTRE'
        drawing.vAlign = 'TOP'
        bc = VerticalBarChart()
        bc.x = 35
        bc.y = 40
        # bc.y = y_axis
        bc.height = 200
        bc.width = 440
        bc.data = bar_data
        bc.strokeColor = colors.black
        bc.valueAxis.valueMin = 0
        bc.valueAxis.valueMax = max_value
        bc.valueAxis.valueStep = round(max_value / 10)
        bc.valueAxis.valueMax = max_value + bc.valueAxis.valueStep
        bc.categoryAxis.labels.dx = 8
        bc.categoryAxis.labels.dy = -10
        bc.categoryAxis.labels.angle = 20
        bc.categoryAxis.categoryNames = ax
        # 图示
        leg = Legend()
        leg.fontName = 'SimSun'
        leg.alignment = 'right'
        leg.boxAnchor = 'ne'
        leg.x = 465
        leg.y = 220
        leg.dxTextSpace = 10
        leg.columnMaximum = 3
        leg.colorNamePairs = items
        drawing.add(leg)
        drawing.add(bc)
        return drawing

if __name__ == "__main__":
    SPACER = Spacer(0, 10)
    hotelId = 862
    groupId = 1
    # compressionRatio = '99%'
    personCnt = 2
    lengthOfStayDayCnt = 1
    LOS = 1
    Observe = 'CostAmt'
    ratePlanLevel = 0
    content = list()
    # 添加标题
    content.append(Graphs.draw_title())
    # 添加段落
    display_content = list()
    display_content.append("First we used the Mapping Function on Hotelid for {}.".format(hotelId))
    display_content.append("In the personCnt={} case,".format(personCnt))
    display_content.append(" lengthOfStayCnt = {}".format(lengthOfStayDayCnt))
    display_content.append(" and ratePlanLevel = {}".format(ratePlanLevel))
    display_content.append(" and LOS = {}".format(LOS))
    display_content.append(" and Observe = {}".format(Observe))

    mappingFunctionFile = '{}{}_{}_{}_mappingFunction.csv'.format(PATTERN_MAPPING_OUTPUT_FOLDER, hotelId, groupId,Observe)
    mappingFunctionFileDF = pd.read_csv(mappingFunctionFile, encoding='utf-8', sep=',', engine='python', header=0).fillna(0)

    childTotalSizes = sum(mappingFunctionFileDF['ChildSize'])
    mappingFunctionTotalSize = sum(mappingFunctionFileDF['MappingFunctionSize'])

    compressionRatio = mappingFunctionTotalSize / childTotalSizes
    # 添加表格数据
    # data = [('BaseRatePlan', mappingFunctionFileDF['ChildRP'][0]),
    #         ('childTotalSizes', childTotalSizes),
    #         ("mappingFunctionTotalSize", mappingFunctionTotalSize),
    #         ('CompressionRatio', '{:.2%}'.format(compressionRatio))
    #         ]

    data = [('BaseRatePlan', 'childTotalSizes',"mappingFunctionTotalSize", 'Validation' ,'CompressionRatio'),
            (mappingFunctionFileDF['ChildRP'][0], childTotalSizes, mappingFunctionTotalSize, all(mappingFunctionFileDF['Validation']), '{:.2%}'.format(compressionRatio))
            ]

    display_content.append(" and Mapping Function compression ratio = {:.2%}.".format(compressionRatio))
    content_text = "".join(display_content)
    content.append(Spacer(1, 10 * mm))
    content.append(Graphs.draw_text("RatePlan Attribute Relationship Graph"))
    pic_url = '{}{}_{}_pic.png'.format(PATTERN_ATTRIBUTE_OUTPUT_FOLDER, 16639, 166628)
    img = Image(pic_url)
    img.drawHeight = 40 * mm
    img.drawWidth = 200 * mm
    img.hAlign = TA_CENTER
    content.append(img)

    content.append(Spacer(1, 10 * mm))
    content.append(Graphs.draw_text(content_text))
    content.append(Spacer(1, 10 * mm))
    content.append(Graphs.draw_table(*data))

    # 添加图表
    max_value = max(childTotalSizes,mappingFunctionTotalSize)
    b_data = [[childTotalSizes,None], [None,mappingFunctionTotalSize]]
    ax_data = ['ChildSize','MappingFunctionSize']
    leg_items = [(colors.red, 'ChildTotalSize'), (colors.green, 'MappingFunctionTotalSize')]
    content.append(Spacer(1, 30 * mm))
    content.append(Graphs.draw_bar(b_data, ax_data, leg_items, max_value))

    # 生成pdf文件
    doc = SimpleDocTemplate('{}_report.pdf'.format(hotelId), pagesize=elevenSeventeen, rightMargin=0.2 * inch,
                        leftMargin=0.2 * inch,
                        topMargin=100, bottomMargin=68)
    content.append(SPACER)
    doc.build(content)