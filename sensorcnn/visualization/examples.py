from matplotlib.pylab import figure, subplot, legend


def plot_examples(examples):
    fig = figure(figsize=(20,10))
    ax1 = subplot(311)
    ax1.set_ylabel('X - Acceleration')
    ax1.plot(examples[:,0])

    ax2 = subplot(312, sharex=ax1)
    ax2.set_ylabel('Y - Acceleration')
    ax2.plot(examples[:,1])

    ax3 = subplot(313, sharex=ax1)
    ax3.set_ylabel('Z - Acceleration')
    ax3.plot(examples[:,2])

    legend(examples)
    return fig
