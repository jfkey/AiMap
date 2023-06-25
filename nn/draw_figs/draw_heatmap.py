# import re
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm
#
# def parseQoR(filename):
#     # max: Before tuning: (Area:2312.56, Delay:3809.03)
#     # Effective After Strategy 3 Gain:0.28, (Area:2270.00, Delay:2796.24) with  (randType:5, blockSize:11)
#     # Effective After Strategy 2 Gain:0.28, (Area:2270.22, Delay:2806.23) with  (beta:1.50, gamma:1.10, tau:10.00, constF:1.10)
#     # Effective After Strategy 1 Gain:0.19, (Area:2328.33, Delay:3049.55) with  (alpha:1.40, gain:350.00, slew:50.00)
#
#     data_arr = []  # 存储每行内容的数组
#     with open(filename, 'r') as file:
#         for line in file:
#             line = line.strip()  # 去除行尾的换行符和空白字符
#             data_arr.append(line)
#
#     area_values = []  # 存储Area值的数组
#     delay_values = []  # 存储Delay值的数组
#
#     # 通过正则表达式提取Area和Delay值
#     area_pattern = r"Area =\s+(\d+\.\d+)"
#     delay_pattern = r"Delay =\s+(\d+\.\d+)"
#
#     for line_str in data_arr:
#         # 查找匹配的结果
#         area_matches = re.findall(area_pattern, line_str)
#         delay_matches = re.findall(delay_pattern, line_str)
#         for area_ma, delay_ma in zip(area_matches, delay_matches):
#             if (float(delay_ma)>5600):continue
#             area_values.append(float(area_ma))
#             delay_values.append(float(delay_ma))
#
#         # 将结果添加到数组中
#         # area_values.extend([float(match) for match in area_matches])
#         # delay_values.extend([float(match) for match in delay_matches])
#
#
#     min_area, max_area = int(min(area_values)/ 30), int(max(area_values)/30)
#     area_tick = []
#     for i in range(min_area, max_area):
#         area_tick.append(i * 30)
#
#     min_delay, max_delay = int(min(delay_values)/100), int(max(delay_values)/100)
#     delay_tick = []
#     for i in range(min_delay, max_delay):
#         delay_tick.append(i * 100)
#     print(area_tick)
#     print(delay_tick)
#
#     area_values = np.array(area_values)
#     delay_values = np.array(delay_values)
#     heatQoR = np.zeros((len(area_tick), len(delay_tick)))
#     for area, delay in zip(area_values, delay_values):
#         row_idx = np.abs(area_tick - area).argmin()
#         col_idx = np.abs(delay_tick - delay).argmin()
#         if row_idx < len(area_tick) and col_idx < len(delay_tick):
#             heatQoR[row_idx, col_idx] += 1
#
#     print("{}, {} ".format(np.sum(heatQoR), heatQoR))
#
#     min_area, max_area = int(min(area_values) / 150), int(max(area_values) / 150)
#     area_tick = []
#     for i in range(min_area, max_area):
#         area_tick.append(i * 150)
#
#     min_delay, max_delay = int(min(delay_values) / 500), int(max(delay_values) / 500)
#     delay_tick = []
#     for i in range(min_delay, max_delay):
#         delay_tick.append(i * 500)
#     # print(area_tick)
#     # print(delay_tick)
#
#     print("Area values:{}, {}",len(area_tick),  area_tick)
#     print("Delay values:{}, {}", len(delay_tick), delay_tick)
#
#
#     return area_tick, delay_tick, heatQoR
#
# def draw_heatmap(area_tick, delay_tick, heatQoR):
#     # 将Area和Delay的值转换为矩阵形式
#     # data = np.array([area_values, delay_values])
#
#     left = 2100
#     right = 2850
#     bottom = 2500
#     top = 5000
#
#
#
#     extent = [left, right, bottom, top]#
#
#
#     fig, ax = plt.subplots(  figsize=(6, 6)) # YlOrRd  Spectral   rainbow jet
#     im = ax.imshow(heatQoR, cmap='jet', interpolation='none', norm=LogNorm(vmin=1, vmax=100), extent=extent, origin='lower')
#
#     # Show all ticks and label them with the respective list entries
#     # ax.set_xticks(np.arange(len(area_tick)), labels=area_tick)
#     # ax.set_yticks(np.arange(len(delay_tick)), labels=delay_tick)
#     # plt.xlim((2100, 2850))
#     # plt.ylim((2500, 5000))
#     # plt.xticks(np.arange(2500, 5000, 500))
#     # plt.yticks(np.arange(2100, 1850, 150))
#
#
#     # ax.invert_yaxis()
#     fig.colorbar(im, ax=ax, location='right', )
#
#
#     # 绘制热图
#     # plt.imshow(data, cmap='hot')
#     # 添加颜色条
#     # plt.colorbar()
#     # 设置坐标轴标签
#     plt.xlabel('Index')
#     plt.ylabel('Category')
#     # 显示图形
#     plt.show()
#
# def draw_exm():
#     vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
#                   "potato", "wheat", "barley"]
#     farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
#                "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]
#
#     harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
#                         [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
#                         [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
#                         [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
#                         [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
#                         [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
#                         [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])
#
#     fig, ax = plt.subplots()
#     im = ax.imshow(harvest)
#
#     # Show all ticks and label them with the respective list entries
#     ax.set_xticks(np.arange(len(farmers)), labels=farmers)
#     ax.set_yticks(np.arange(len(vegetables)), labels=vegetables)
#
#     # Rotate the tick labels and set their alignment.
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#              rotation_mode="anchor")
#
#     # Loop over data dimensions and create text annotations.
#     for i in range(len(vegetables)):
#         for j in range(len(farmers)):
#             text = ax.text(j, i, harvest[i, j],
#                            ha="center", va="center", color="w")
#
#     ax.set_title("Harvest of local farmers (in tons/year)")
#     fig.tight_layout()
#     plt.show()
#
# if __name__ == '__main__':
#     area_tick, delay_tick, heatQoR = parseQoR("max_train.log")
#     draw_heatmap(area_tick, delay_tick, heatQoR)
#     #draw_exm()
#
#
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def parseQoR(filename):
    data_arr = []  # 存储每行内容的数组
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()  # 去除行尾的换行符和空白字符
            data_arr.append(line)

    area_values = []  # 存储Area值的数组
    delay_values = []  # 存储Delay值的数组

    # 通过正则表达式提取Area和Delay值
    area_pattern = r"Area =\s+(\d+\.\d+)"
    delay_pattern = r"Delay =\s+(\d+\.\d+)"

    for line_str in data_arr:
        # 查找匹配的结果
        area_matches = re.findall(area_pattern, line_str)
        delay_matches = re.findall(delay_pattern, line_str)
        for area_ma, delay_ma in zip(area_matches, delay_matches):
            if (float(delay_ma)>5600):continue
            area_values.append(float(area_ma))
            delay_values.append(float(delay_ma))

    min_area, max_area = int(min(area_values)/ 30), int(max(area_values)/30)
    area_tick = []
    for i in range(min_area, max_area):
        area_tick.append(i * 30)

    min_delay, max_delay = int(min(delay_values)/100), int(max(delay_values)/100)
    delay_tick = []
    for i in range(min_delay, max_delay):
        delay_tick.append(i * 100)
    print(area_tick)
    print(delay_tick)

    area_values = np.array(area_values)
    delay_values = np.array(delay_values)
    heatQoR = np.zeros((len(area_tick), len(delay_tick)))
    for area, delay in zip(area_values, delay_values):
        row_idx = np.abs(area_tick - area).argmin()
        col_idx = np.abs(delay_tick - delay).argmin()
        if row_idx < len(area_tick) and col_idx < len(delay_tick):
            heatQoR[row_idx, col_idx] += 1

    print("{}, {} ".format(np.sum(heatQoR), heatQoR))

    min_area, max_area = int(min(area_values) / 150), int(max(area_values) / 150)
    area_tick = []
    for i in range(min_area, max_area):
        area_tick.append(i * 150)

    min_delay, max_delay = int(min(delay_values) / 500), int(max(delay_values) / 500)
    delay_tick = []
    for i in range(min_delay, max_delay):
        delay_tick.append(i * 500)

    print("Area values:{}, {}",len(area_tick),  area_tick)
    print("Delay values:{}, {}", len(delay_tick), delay_tick)

    return area_tick, delay_tick, heatQoR


def draw_heatmap(area_tick, delay_tick, heatQoR):
    fontsize = 14
    fig, ax = plt.subplots(1, 1, figsize=(7, 6)) # YlOrRd  Spectral   rainbow jet
    # heatQoR = heatQoR / np.sum(heatQoR)
    im = ax.imshow(heatQoR, cmap='jet', interpolation='none', norm=LogNorm(vmin=1, vmax=100), extent=None, origin='lower', aspect='auto')
    fig.colorbar(im, ax=ax, location='right', )


    x_ticks = np.linspace(0, heatQoR.shape[0], len(delay_tick))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(delay_tick, fontsize=fontsize)

    y_ticks = np.linspace(0, heatQoR.shape[0], len(area_tick))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(area_tick, fontsize=fontsize)

    plt.xlabel(r'Delay($ps$)', fontsize=fontsize)
    plt.ylabel(r'Area($\mu m^2$)', fontsize=fontsize)

    plt.tight_layout()
    plt.savefig('exp-heatmap.pdf', dpi=800)
    plt.show()


def draw_exm():
    vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
                  "potato", "wheat", "barley"]
    farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
               "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

    harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                        [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                        [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                        [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                        [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                        [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                        [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])

    fig, ax = plt.subplots()
    im = ax.imshow(harvest)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(farmers)), labels=farmers)
    ax.set_yticks(np.arange(len(vegetables)), labels=vegetables)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(vegetables)):
        for j in range(len(farmers)):
            text = ax.text(j, i, harvest[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("Harvest of local farmers (in tons/year)")
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    area_tick, delay_tick, heatQoR = parseQoR("max_train.log")
    draw_heatmap(area_tick, delay_tick, heatQoR)
    #draw_exm()


