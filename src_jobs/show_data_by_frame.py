import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.text import Text

real_data_df = pd.read_csv("./data/real/normalized_deviation_updated.csv")
dataset_name = "normalized_deviation_updated"

real_data_df = real_data_df.rename(
    columns={"Schedule deviation": "data"}, errors="raise"
)

artifacts_df = pd.DataFrame(columns=["start", "end", "width", "speed"])


def onpick1(event):
    if isinstance(event.artist, Line2D):
        thisline = event.artist
        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()
        ind = event.ind
        print("onpick1 line:", np.column_stack([xdata[ind], ydata[ind]]))
        return [xdata[0], ydata[0]]
    elif isinstance(event.artist, Text):
        text = event.artist
        print("onpick1 text:", text.get_text())


i = 0
# 104400 - frame with 3 good artifacts
# 147000 - frame where I stopped
width = 300

while i < real_data_df.shape[0]:
    print(i)
    fig, ax1 = plt.subplots()
    ax1.set_title("click on points, rectangles or text", picker=True)
    plt.xticks(range(i, i + width, 15), rotation=90)
    # ax1.set_yticks(real_data_df["data"])
    (line,) = ax1.plot(
        range(i, i + width),
        list(real_data_df["data"].iloc[i : i + width]),
        "-",
        picker=1,
    )

    fig.canvas.mpl_connect("pick_event", onpick1)
    plt.show()

    i = i + width
