import pandas as pd
import numpy as np
import streamlit as st
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules

# Test First Step Of FPGrowth

transactions = []
dataset = pd.read_csv("Market_Basket_Optimisation.csv")
for i in range(0, dataset.shape[0]):
    for j in range(0, dataset.shape[1]):
        transactions.append(dataset.values[i, j])

transactions = np.array(transactions)
# print(transactions)

df = pd.DataFrame(transactions, columns=["items"])
df["incident_count"] = 1
indexNames = df[df['items'] == "nan"].index
df.drop(indexNames, inplace=True)
df_table = df.groupby("items").sum().sort_values("incident_count", ascending=False).reset_index()
# minSupport = 150
#
# minIndex = df_table[df_table["incident_count"] < 150].index
# df_table.drop(minIndex, inplace=True)
# print(df_table)

# Test FPGrowth Full

transactions = []
dataset = pd.read_csv("Market_Basket_Optimisation.csv")
for i in range(0, dataset.shape[0]):
    transactions.append([str(dataset.values[i, j]) for j in range(dataset.shape[1])])

transactions = np.array(transactions)
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)

dataset = pd.DataFrame(te_array, columns=te.columns_)
# print(dataset)

topSupport = df_table["items"].head(30).values
# print(topSupport)

dataset = dataset.loc[:, topSupport]
# print(dataset)

res = fpgrowth(dataset, min_support=0.05, use_colnames=True)
# print(res.head(10))

res = association_rules(res, metric="lift", min_threshold=1)
res.sort_values("confidence",ascending=False)
# print(res)

st.write("### FPGrowth Test Page With MLXTEND Library")
st.table(res)
