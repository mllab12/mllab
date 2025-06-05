import numpy as np, pandas as pd
data = pd.read_csv("output.csv")
concepts, target = np.array(data.iloc[:, :-1]), np.array(data.iloc[:, -1])

def learn(concepts, target):
    s, g = concepts[0].copy(), [['?' for _ in s] for _ in s]
    for i, h in enumerate(concepts):
        if target[i] == "Yes":
            for j in range(len(s)):
                if h[j] != s[j]: s[j], g[j][j] = '?', '?'
        else:
            for j in range(len(s)):
                if h[j] != s[j]: g[j][j] = s[j]
    g = [h for h in g if any(v != '?' for v in h)]
    return s, g

s_final, g_final = learn(concepts, target)
print("Specific:", s_final, "\nGeneral:", g_final)
