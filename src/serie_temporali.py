# ## Calcolo e visualizzazione media varianza segnale Bold per finestre temporali

window_step = 10  # Passo tra i volumi temporali
selected_volumes = list(range(0, fmri_data.shape[3], window_step))
num_windows = len(selected_volumes)


variance_per_window = []
for vol in selected_volumes:
    # Seleziona un singolo volume
    window_data = fmri_data[..., vol]
    variance_per_window.append(window_data)


variance_per_window = np.stack(variance_per_window, axis=3)

mean_variance = [np.mean(variance_per_window[..., i]) for i in range(num_windows)]

plt.figure(figsize=(10, 6))
plt.plot(selected_volumes, mean_variance, marker='o', linestyle='-', color='b')  # Volumi sull'asse x
plt.title("Media della Varianza Segnale Bold per Finestre Temporali")
plt.xlabel("Indice Volume Temporale")
plt.ylabel("Media della Varianza BOLD")
plt.grid(True)
plt.show()


window_size = 25  # Numero di volumi per finestra
num_windows = fmri_data.shape[3] // window_size  # Numero di finestre


variance_per_window = []


for i in range(num_windows):
    start = i * window_size
    end = start + window_size
    window_data = fmri_data[..., start:end]
    window_variance = np.var(window_data, axis=3)
    variance_per_window.append(window_variance)


variance_per_window = np.stack(variance_per_window, axis=3)

mean_variance = [np.mean(variance_per_window[..., i]) for i in range(num_windows)]


plt.figure(figsize=(10, 6))
plt.plot(range(num_windows), mean_variance, marker='o', linestyle='-', color='b')  # Indici delle finestre sull'asse x
plt.title("Media della Varianza per Finestre Temporali")
plt.xlabel("Indice Finestra Temporale")
plt.ylabel("Media della Varianza BOLD")
plt.grid(True)
plt.show()


normalized_data = (variance_per_window[..., 0] - np.min(variance_per_window[..., 0])) / (
    np.max(variance_per_window[..., 0]) - np.min(variance_per_window[..., 0])
)

num_windows = 6
window_indices = np.linspace(0, variance_per_window.shape[3] - 1, num_windows, dtype=int)
grouped_data = variance_per_window[..., window_indices]


z_slice = grouped_data.shape[2] // 2
images = [grouped_data[:, :, z_slice, i] for i in range(num_windows)]


images = [(img - np.min(img)) / (np.max(img) - np.min(img)) for img in images]


fig = go.Figure()


for i, img in enumerate(images):
    fig.add_trace(go.Heatmap(
        z=img,
        colorscale='Viridis',
        visible=(i == 0)
    ))


steps = []
for i in range(num_windows):
    step = dict(
        method="update",
        args=[{"visible": [j == i for j in range(num_windows)]}],
        label=f"Finestra {i+1}"
    )
    steps.append(step)

sliders = [dict(
    active=0,
    currentvalue={"prefix": "Finestra: "},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    title="Segnale BOLD nelle Diverse Finestre Temporali",
    sliders=sliders,
    xaxis_title="Pixel X",
    yaxis_title="Pixel Y"
)


fig.write_html("bold_signal_interactive.html")

print("Il grafico è stato salvato come 'bold_signal_interactive.html'. Aprilo nel browser per visualizzarlo.")


z_center = fmri_data.shape[2] // 2

plt.figure(figsize=(18, 10))

for i in range(min(num_windows, 6)):
    plt.subplot(2, 3, i + 1)  # Griglia 2x3
    plt.imshow(variance_per_window[:, :, z_center, i].T, cmap='hot', origin='lower')
    plt.title(f"Varianza - Finestra {i}")
    plt.colorbar(label="Varianza BOLD")
    plt.axis("off")

plt.tight_layout()
plt.show()


# ## Serie temporale segnale globale vs maschera

mask = resampled_mask_data.astype(bool)

if not np.any(mask):
    raise ValueError("La maschera è vuota. Controlla i valori in 'resampled_mask_data'.")

global_bold = [np.mean(variance_per_window[..., i]) for i in range(variance_per_window.shape[3])]

masked_bold = [np.mean(variance_per_window[..., i][mask]) for i in range(variance_per_window.shape[3])]

plt.figure(figsize=(10, 6))
plt.plot(global_bold, label='Segnale Globale', marker='o')
plt.plot(masked_bold, label='Segnale nella Maschera', marker='x')
plt.title("Serie Temporale del Segnale BOLD")
plt.xlabel("Finestra Temporale")
plt.ylabel("Intensità Media BOLD")
plt.legend()
plt.grid(True)
plt.show()


adf_global = adfuller(global_bold)
adf_masked = adfuller(masked_bold)


kpss_global = kpss(global_bold, regression='c', nlags="legacy")
kpss_masked = kpss(masked_bold, regression='c', nlags="legacy")


ks_stat, ks_p_value = ks_2samp(global_bold, masked_bold)

anderson_stat, _, _ = anderson_ksamp([global_bold, masked_bold])

t_stat, t_p_value = ttest_ind(global_bold, masked_bold, equal_var=False)

mw_stat, mw_p_value = mannwhitneyu(global_bold, masked_bold, alternative='two-sided')

levene_stat, levene_p_value = levene(global_bold, masked_bold)

bartlett_stat, bartlett_p_value = bartlett(global_bold, masked_bold)

mean_diff = np.mean(masked_bold) - np.mean(global_bold)
pooled_std = np.sqrt((np.std(global_bold, ddof=1)**2 + np.std(masked_bold, ddof=1)**2) / 2)
cohens_d = mean_diff / pooled_std


pearson_corr, pearson_p_value = pearsonr(global_bold, masked_bold)

spearman_corr, spearman_p_value = spearmanr(global_bold, masked_bold)

print(f"T-test: p-value = {t_p_value}")
print(f"Mann-Whitney U test: p-value = {mw_p_value}")


print(f"ADF Test (Globale): p-value = {adf_global[1]}")
print(f"ADF Test (Maschera): p-value = {adf_masked[1]}")
print(f"KPSS Test (Globale): p-value = {kpss_global[1]}")
print(f"KPSS Test (Maschera): p-value = {kpss_masked[1]}")

print(f"Correlazione di Pearson: coefficiente = {pearson_corr}, p-value = {pearson_p_value}")
print(f"Correlazione di Spearman: coefficiente = {spearman_corr}, p-value = {spearman_p_value}")


signal_df = pd.DataFrame({
    "Segnale": np.concatenate([global_bold, masked_bold]),
    "Gruppo": ["Globale"] * len(global_bold) + ["Maschera"] * len(masked_bold)
})

# Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x="Gruppo", y="Segnale", data=signal_df, palette="coolwarm")
plt.title("Confronto delle Intensità del Segnale BOLD")
plt.ylabel("Intensità Media del Segnale")
plt.xlabel("Gruppo")
plt.grid(axis='y', alpha=0.7)
plt.show()

# Violin Plot
plt.figure(figsize=(8, 6))
sns.violinplot(x="Gruppo", y="Segnale", data=signal_df, palette="coolwarm")
plt.title("Distribuzioni del Segnale BOLD")
plt.ylabel("Intensità Media del Segnale")
plt.xlabel("Gruppo")
plt.grid(axis='y', alpha=0.7)
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(global_bold, label="Globale", marker='o', linestyle='-', color='blue', alpha=0.7)
plt.plot(masked_bold, label="Maschera", marker='x', linestyle='--', color='orange', alpha=0.7)

plt.axhline(np.mean(global_bold), color='blue', linestyle=':', label="Media Globale")
plt.axhline(np.mean(masked_bold), color='orange', linestyle=':', label="Media Maschera")

plt.title("Serie Temporali e Stazionarietà")
plt.xlabel("Finestra Temporale")
plt.ylabel("Intensità Media del Segnale")
plt.legend()
plt.grid(True, alpha=0.5)
plt.show()

atlas = fetch_atlas_juelich("maxprob-thr0-1mm")
labels = atlas['labels']

label_dict = {i: labels[i] for i in range(len(labels)) if labels[i] != 'Background'}

regions = np.unique(resampled_mask_data[resampled_mask_data > 0])
regions_with_labels = {region: label_dict.get(region, f"Region {int(region)}") for region in regions}

print("Mapping delle regioni (valore -> label):")
for region, label in regions_with_labels.items():
    print(f"{region} -> {label}")

region_series = {}

for region, label in regions_with_labels.items():

    region_mask = resampled_mask_data == region
    series_for_region = []

    for t in range(variance_per_window.shape[3]):
        mean_value = np.mean(variance_per_window[..., t][region_mask])
        series_for_region.append(mean_value)

    region_series[label] = series_for_region


plt.figure(figsize=(12, 6))

for region, label in regions_with_labels.items():
    plt.plot(region_series[label], label=label)

plt.title("Serie Temporali di Tutte le Regioni del sistema specchio")
plt.xlabel("Finestra Temporale")
plt.ylabel("Intensità Media BOLD")
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1), fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.show()


num_regions = len(regions_with_labels)
cols = 3
rows = int(np.ceil(num_regions / cols))

fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), constrained_layout=True)
axes = axes.flatten()

for idx, (region, label) in enumerate(regions_with_labels.items()):
    ax = axes[idx]
    ax.plot(region_series[label], label=label)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("Finestra Temporale")
    ax.set_ylabel("Intensità Media BOLD")
    ax.grid(True)
    ax.legend(fontsize=8)

for ax in axes[num_regions:]:
    ax.axis('off')

plt.suptitle("Serie Temporali delle Regioni del sistema specchio", fontsize=16)
plt.show()


region_activation = {
    label: {
        "max": max(series),
        "mean": sum(series) / len(series)
    }
    for label, series in region_series.items()
}

sorted_regions = sorted(region_activation.items(), key=lambda x: x[1]['mean'], reverse=True)

print("Top 3 regioni con maggiore attivazione media:")
for region, values in sorted_regions[:3]:
    print(f"Regione: {region} - Attivazione Media: {values['mean']:.2f}, Attivazione Massima: {values['max']:.2f}")

# In[59]:


regions_sorted = [region for region, _ in sorted_regions]
means_sorted = [values['mean'] for _, values in sorted_regions]

plt.figure(figsize=(12, 6))
plt.bar(regions_sorted, means_sorted, color="blue")
plt.xticks(rotation=45, ha="right")
plt.xlabel("Regioni")
plt.ylabel("Attivazione Media")
plt.title("Attivazione Media di Tutte le Regioni del sistema specchio")
plt.show()
