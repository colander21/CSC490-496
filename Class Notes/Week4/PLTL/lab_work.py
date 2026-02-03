import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal import savgol_filter

df = pd.read_csv("step_responses.csv")

t_ss = 1.45
idx = (df["time"] - t_ss).abs().idxmin()
v_ss = df.loc[idx, "500 rpm step response"]
print(v_ss)
print(v_ss * 0.63)


idx_tau = (df["500 rpm step response"] - v_ss*0.63).abs().idxmin()
tau = df.loc[idx_tau, "time"]
print(tau)

print("tau:", tau)
print("response at tau:", df.loc[idx_tau, "500 rpm step response"])
print("expected 63% value:", v_ss * 0.63)




# print(df)
df_smoothed = pd.DataFrame(columns=["time", "500 rpm step response (avg)"])
plt.plot(df["time"], df["500 rpm step response"])
plt.show()

step = 50
for i in range(0, len(df) - step + 1, step):
    avg = 0
    for j in range(step):
        avg += df.loc[i+j, "500 rpm step response"]

    df_smoothed.loc[i, "time"] = df.loc[i, "time"]

    df_smoothed.loc[i, "500 rpm step response (avg)"] = avg / step

plt.plot(df_smoothed["time"], df_smoothed["500 rpm step response (avg)"])

plt.show()


scipy.signal.savgol_filter()


# best_fit = np.polyfit(df["time"], df["500 rpm step response"], 1)
# print(best_fit)