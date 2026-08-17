import platform

import matplotlib.pyplot as plt


def setup_chinese_font():
    """
    设置 matplotlib 中文字体，兼容 macOS / Windows / Linux
    """
    system_name = platform.system()
    if system_name == 'Darwin':  # macOS
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang HK', 'STHeiti']
    elif system_name == 'Windows':
        plt.rcParams['font.sans-serif'] = ['SimHei']
    else:  # Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
    # 解决坐标轴负号显示问题
    plt.rcParams['axes.unicode_minus'] = False
