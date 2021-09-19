import matplotlib.pyplot as plt


def plot():
    fig, ax = plt.subplots()
    for m_name, m_time, m_error, m_marker, m_color in zip(models, times, Errors, markers, colors):
        if m_name == "ROPNet":
            ax.scatter(x=m_time, y=m_error, s=88, c=m_color, marker=m_marker)
        else:
            ax.scatter(x=m_time, y=m_error, s=64, c=m_color, marker=m_marker)
            # ax.scatter(x=m_time, y=m_error, c=m_color, marker=m_marker)

    ax.set_xlim(left=0, right=0.15)
    ax.set_ylim(bottom=0, top=30)

    # https://stackoverflow.com/questions/7965743/how-can-i-set-the-aspect-ratio-in-matplotlib
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))

    # ax.grid(b=True, which='major', color='k', linestyle='--')
    ax.grid(linestyle='-.')

    # https://stackoverflow.com/questions/19677963/matplotlib-keep-grid-lines-behind-the-graph-but-the-y-and-x-axis-above
    ax.set_axisbelow(True)

    # https://blog.csdn.net/lanluyug/article/details/80002273
    # plt.legend(labels=models, loc="lower right", fontsize='x-large')
    plt.legend(labels=models, loc=(0.075/0.15, 3/30), fontsize=12)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Error (R) (degree)', fontsize=12)
    fig.savefig('/home/lifa/data/cmp_error_time.pdf', dpi=1000)
    plt.close(fig)
    plt.show()


colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#17becf', '#9467bd', '#8c564b', '#e377c2', '#d62728']
markers = [',', 'o', '^', 'D', '+', 'X', '1', '*']
models = ['ICP', 'FGR', 'PCRNet', 'DCP-v2', 'IDAM-GNN', 'DeepGMR', 'RPMNet', 'ROPNet(ours)']
times = [0.0493, 0.1286, 0.0145, 0.0196, 0.0263, 0.0063, 0.0812, 0.0387]
Errors = [25.5219, 29.2813, 21.7476, 11.5672, 16.9805, 18.8075, 1.7569, 1.4656]
plot()
