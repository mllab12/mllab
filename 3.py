import numpy as np
import pandas as pd

data = pd.read_csv("home/output.csv")
print("Dataset:\n", data)

concepts = np.array(data.iloc[:, 0:-1])
print("\nConcepts:\n", concepts)

target = np.array(data.iloc[:, -1])
print("\nTarget:\n", target)

def learn(concepts, target):
    specific_h = concepts[0].copy()
    print("\nInitialization of specific_h and general_h")
    print("specific_h:", specific_h)

    general_h = [['?' for _ in range(len(specific_h))] for _ in range(len(specific_h))]
    print("general_h:", general_h)

    for i, h in enumerate(concepts):
        if target[i] == "Yes":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = "?"
                    general_h[x][x] = "?"
        elif target[i] == "No":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = "?"

    print("\nSteps of Candidate Elimination Algorithm:")
    print("Specific Hypothesis after training:\n", specific_h)
    print("General Hypothesis after training:\n", general_h)

    general_h = [h for h in general_h if any(val != '?' for val in h)]
    return specific_h, general_h

s_final, g_final = learn(concepts, target)
print("\nFinal Specific_h:\n", s_final)
print("\nFinal General_h:\n", g_final)
