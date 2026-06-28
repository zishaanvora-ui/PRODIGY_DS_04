import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import filediSalog


print("Select TRAINING file...")
train_file = filedialog.askopenfilename()
train_df = pd.read_csv(train_file, header=None, names=['ID','Entity','Sentiment','Text'])

print("Select VALIDATION file...")
val_file = filedialog.askopenfilename()
val_df = pd.read_csv(val_file, header=None, names=['ID','Entity','Sentiment','Text'])

COLORS = {'Positive':'#2ecc71', 'Negative':'#e74c3c', 'Neutral':'#3498db', 'Irrelevant':'#95a5a6'}

plt.figure(figsize=(8,5))
sns.countplot(x='Sentiment', data=train_df, palette=COLORS)
plt.title('Sentiment Distribution — Training')
plt.tight_layout()
plt.show()

top10 = train_df['Entity'].value_counts().head(10).index
filtered = train_df[train_df['Entity'].isin(top10)]

plt.figure(figsize=(10,6))
sns.countplot(y='Entity', hue='Sentiment', data=filtered, palette=COLORS)
plt.title('Top 10 Entities by Sentiment')
plt.tight_layout()
plt.show()

train_pct = train_df['Sentiment'].value_counts(normalize=True) * 100
val_pct   = val_df['Sentiment'].value_counts(normalize=True) * 100

compare = pd.DataFrame({'Training': train_pct, 'Validation': val_pct})
compare.plot(kind='bar', figsize=(8,5), color=['#3498db','#e67e22'])
plt.title('Training vs Validation Sentiment %')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
