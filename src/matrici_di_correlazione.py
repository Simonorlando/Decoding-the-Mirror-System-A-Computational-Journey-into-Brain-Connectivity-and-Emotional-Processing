# Matrice sistema specchio

data = pd.DataFrame(region_series)

correlation_matrix = data.corr()

plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Matrice di Correlazione tra Regioni")
plt.show()


# In[61]:


correlations = correlation_matrix.values[np.triu_indices_from(correlation_matrix, k=1)]

mean_correlation = np.mean(correlations)
std_correlation = np.std(correlations)

print(f"Media delle correlazioni: {mean_correlation:.2f}")
print(f"Deviazione standard delle correlazioni: {std_correlation:.2f}")


# In[62]:


plt.figure(figsize=(10, 6))
plt.hist(correlations, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
plt.title("Distribuzione delle Correlazioni del sistema spcchio", fontsize=14)
plt.xlabel("Valore di Correlazione", fontsize=12)
plt.ylabel("Frequenza", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()


# In[63]:


high_correlation = correlations[correlations >= 0.8]
medium_correlation = correlations[(correlations >= 0.6) & (correlations < 0.8)]
low_correlation = correlations[correlations < 0.6]

print(f"Numero di correlazioni alte (>= 0.8): {len(high_correlation)}")
print(f"Numero di correlazioni medie (0.6 <= correlazione < 0.8): {len(medium_correlation)}")
print(f"Numero di correlazioni basse (< 0.6): {len(low_correlation)}")

print("\nValori di correlazioni alte:")
print(high_correlation)

print("\nValori di correlazioni medie:")
print(medium_correlation)

print("\nValori di correlazioni basse:")
print(low_correlation)


# In[64]:


# Array dei valori di correlazione alta
high_correlation_values = [
    0.89404553, 0.97272186, 0.82883924, 0.82546828, 0.85494661, 0.82784274,
    0.94520269, 0.95467602, 0.91076766, 0.80426981, 0.90218872, 0.9314653,
    0.89358233, 0.87377463, 0.91308369, 0.87726736, 0.87064944, 0.91785733,
    0.80455723, 0.88972108, 0.86500814, 0.87792647, 0.85652017, 0.94029924,
    0.91237502, 0.98733023, 0.86802573, 0.97020143, 0.83642003, 0.95812753,
    0.85055743, 0.80660543, 0.84382865, 0.90662419, 0.80412009, 0.95614309,
    0.93112507, 0.91178911, 0.98490108, 0.89534455, 0.96789304, 0.86176856,
    0.87280773, 0.89225193, 0.92655279, 0.95205635, 0.95446649, 0.95829521,
    0.97895061, 0.92374675, 0.87864733, 0.98378299, 0.95739739, 0.89721096,
    0.93096552, 0.92137279, 0.9835467, 0.92997755, 0.96800462, 0.91377201,
    0.92219494, 0.93597964, 0.82353475, 0.96332293, 0.84892656, 0.98182632,
    0.98790977, 0.92203429, 0.83625536, 0.96769867, 0.90860599, 0.87005921,
    0.87572072, 0.87007711, 0.94320405, 0.80383608, 0.86686995, 0.8663305,
    0.82583833, 0.86796894, 0.92431566, 0.95156516, 0.87128551, 0.97661771,
    0.80476436, 0.86495191, 0.87943957, 0.97037921, 0.94825212, 0.95365678,
    0.89819524, 0.97370487, 0.91736042, 0.92325422, 0.88037922, 0.95557319,
    0.92520611, 0.95658425, 0.91787197, 0.96581473, 0.88330561, 0.87358448,
    0.97805558, 0.96064854, 0.90844783, 0.81715146, 0.9668109
]

top_3_correlation = sorted(high_correlation_values, reverse=True)[:3]

print("Top 3 valori di correlazione:")
for i, value in enumerate(top_3_correlation, 1):
    print(f"{i}. {value:.2f}")


# In[65]:


correlation_matrix = data.corr()

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

filtered_matrix = correlation_matrix.where((correlation_matrix >= 0.98) | (correlation_matrix <= 0.56))

filtered_matrix = np.ma.masked_where(mask, filtered_matrix)

plt.figure(figsize=(12, 10))
sns.heatmap(filtered_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, linewidths=1, linecolor='black', mask=mask, xticklabels=correlation_matrix.columns, yticklabels=correlation_matrix.columns)

plt.title("Heatmap delle Correlazioni ", fontsize=14)
plt.xlabel("Aree Cerebrali", fontsize=12)
plt.ylabel("Aree Cerebrali", fontsize=12)
plt.xticks(rotation=90, fontsize=10)
plt.yticks(rotation=0, fontsize=10)

plt.show()


# In[66]:


high_correlation_indices = np.where(correlation_matrix >= 0.98)
low_correlation_indices = np.where(correlation_matrix <= 0.55)

high_correlation_pairs = []
low_correlation_pairs = []

for i, j in zip(*high_correlation_indices):
    if i < j:  # Solo la parte superiore (escludendo i duplicati)
        high_correlation_pairs.append((data.columns[i], data.columns[j], correlation_matrix.iloc[i, j]))

for i, j in zip(*low_correlation_indices):
    if i < j:  # Solo la parte superiore (escludendo i duplicati)
        low_correlation_pairs.append((data.columns[i], data.columns[j], correlation_matrix.iloc[i, j]))


print("Coppie di regioni con correlazione >= 0.98:")
for pair in high_correlation_pairs:
    print(f"{pair[0]} e {pair[1]}: correlazione = {pair[2]:.2f}")

print("\nCoppie di regioni con correlazione <= 0.55:")
for pair in low_correlation_pairs:
    print(f"{pair[0]} e {pair[1]}: correlazione = {pair[2]:.2f}")


# correlazioni globali
masker = NiftiLabelsMasker(labels_img=resampled_full_mask_img, standardize=True)
time_series = masker.fit_transform(fmri_img)

correlation_matrix = np.corrcoef(time_series.T)
num_areas = correlation_matrix.shape[0]

correlation_values = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]

plt.figure(figsize=(10, 6))
plt.hist(correlation_values, bins=50, color='skyblue', edgecolor='black')
plt.title("Distribuzione delle Correlazioni tra le tutte Aree Cerebrali")
plt.xlabel("Valori di Correlazione")
plt.ylabel("Frequenza")
plt.grid(axis='y', alpha=0.75)
plt.show()

juelich_labels = juelich_atlas.labels

region_index_to_name = {index: name for index, name in enumerate(juelich_labels)}

pairs_named = []
for i in range(correlation_matrix.shape[0]):
    for j in range(i + 1, correlation_matrix.shape[1]):  # Parte superiore della matrice
        correlation = correlation_matrix[i, j]
        if 0.0 <= correlation <= 0.40:
            region1 = region_index_to_name.get(all_regions[i], f"Region {all_regions[i]}")
            region2 = region_index_to_name.get(all_regions[j], f"Region {all_regions[j]}")
            pairs_named.append((region1, region2, correlation))


pairs_named_df = pd.DataFrame(pairs_named, columns=["Regione 1", "Regione 2", "Correlazione"])

pairs_named_df = pairs_named_df.sort_values(by="Correlazione", ascending=True)

pairs_named_df

pairs_named_df.describe()


mean_correlation = pairs_named_df["Correlazione"].mean()
std_correlation = pairs_named_df["Correlazione"].std()

threshold = mean_correlation - 2 * std_correlation

low_correlation_df = pairs_named_df[pairs_named_df["Correlazione"] < threshold]

print(f"Media delle correlazioni: {mean_correlation:.2f}")
print(f"Deviazione standard delle correlazioni: {std_correlation:.2f}")
print(f"Soglia (media - 2*std): {threshold:.2f}")

display(low_correlation_df.style.set_table_styles(
    [{'selector': 'th', 'props': [('text-align', 'center')]},
     {'selector': 'td', 'props': [('text-align', 'left')]}]
).set_caption("Aree con Correlazioni Inferiori a Due Deviazioni Standard"))