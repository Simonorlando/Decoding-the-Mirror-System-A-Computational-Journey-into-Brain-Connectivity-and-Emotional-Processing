juelich_atlas = datasets.fetch_atlas_juelich('maxprob-thr25-1mm')
juelich_atlas_img = juelich_atlas.maps
juelich_data = juelich_atlas_img.get_fdata()
print(f"Dimensioni dell'atlante Juelich: {juelich_data.shape} (x, y, z)")

#Sistema specchio

juelich_areas_of_interest = [4, 5, 6, 7, 14, 15, 16, 17, 18, 19, 20, 27, 31, 32, 44, 45, 46, 47]

juelich_data = juelich_atlas_img.get_fdata()
print(f"Dimensioni dell'atlante Juelich: {juelich_data.shape}")

juelich_mask = np.zeros_like(juelich_data, dtype=np.int16)
for region in juelich_areas_of_interest:
    juelich_mask[juelich_data == region] = region

print(f"Valori unici nella maschera originale: {np.unique(juelich_mask)}")

juelich_mask_img = nib.Nifti1Image(juelich_mask, affine=juelich_atlas_img.affine)

resampled_mask_img = resample_to_img(
    source_img=juelich_mask_img,
    target_img=fmri_img,
    interpolation="nearest"
)

resampled_mask_data = resampled_mask_img.get_fdata().astype(np.int16)
print(f"Valori unici nella maschera resample-ata: {np.unique(resampled_mask_data)}")

matching_regions = [region for region in np.unique(resampled_mask_data) if region in juelich_areas_of_interest]
print(f"Regioni di interesse trovate nella maschera resample-ata: {matching_regions}")
if len(matching_regions) == 0:
    raise ValueError("Nessuna regione di interesse trovata nella maschera resample-ata.")

fmri_mean_data = np.mean(fmri_data, axis=3)  # Media lungo l'asse temporale
fmri_3d = nib.Nifti1Image(fmri_mean_data, affine=fmri_img.affine)

plotting.plot_roi(
    resampled_mask_img,
    bg_img=fmri_3d,
    title="Maschera del sistema specchio con atlante Juelich",
    cut_coords=(0, -32, 58),  # Coordinate di taglio
    display_mode='ortho',  # Vista ortogonale
    cmap="autumn",
    alpha=0.6  # Trasparenza
)
plotting.show()

fmri_3d = fmri_img.slicer[..., 0]  # Volume temporale 0

interactive_display = plotting.view_img(
    fmri_3d,
    threshold=None,
    title='fMRI con Maschera Sovrapposta',
    cmap='gray',
    annotate=True
)

interactive_display = plotting.view_img(
    resampled_mask_img,
    bg_img=fmri_3d,
    threshold=0.5,
    title='fMRI + Maschera (Volume Temporale 0)',
    cmap='hot',
    annotate=True
)

interactive_display

# Maschera con tutte le regioni

juelich_atlas = datasets.fetch_atlas_juelich('maxprob-thr25-1mm')
juelich_atlas_img = juelich_atlas.maps  # Oggetto NIfTI
juelich_data = juelich_atlas_img.get_fdata()
print(f"Dimensioni dell'atlante Juelich: {juelich_data.shape} (x, y, z)")

all_regions = np.unique(juelich_data)
all_regions = all_regions[all_regions > 0]
print(f"Regioni disponibili nell'atlante: {all_regions}")

juelich_full_mask = np.zeros_like(juelich_data, dtype=np.int16)


for region in all_regions:
    juelich_full_mask[juelich_data == region] = region
print(f"Valori unici nella maschera completa: {np.unique(juelich_full_mask)}")

juelich_full_mask_img = nib.Nifti1Image(juelich_full_mask, affine=juelich_atlas_img.affine)

resampled_full_mask_img = resample_to_img(
    source_img=juelich_full_mask_img,
    target_img=fmri_img,
    interpolation="nearest"
)

resampled_full_mask_data = resampled_full_mask_img.get_fdata().astype(np.int16)
print(f"Valori unici nella maschera resample-ata: {np.unique(resampled_full_mask_data)}")


if len(fmri_img.shape) == 4:
    fmri_img_display = image.mean_img(fmri_img)
else:
    fmri_img_display = fmri_img


plot_roi(
    resampled_full_mask_img,
    bg_img=fmri_img_display,
    title="Allineamento Maschera Resample-ata con fMRI",
    display_mode='ortho',
    cut_coords=(0, 0, 0),
    colorbar=True
)

plt.show()


resampled_regions = np.unique(resampled_full_mask_data)
resampled_regions = resampled_regions[resampled_regions > 0]

missing_regions = set(all_regions) - set(resampled_regions)

print(f"Numero totale di regioni nell'atlante originale: {len(all_regions)}")
print(f"Numero di regioni trovate nella maschera resample-ata: {len(resampled_regions)}")
if missing_regions:
    print(f"Regioni mancanti nella maschera resample-ata: {missing_regions}")
else:
    print("Tutte le regioni dell'atlante sono state mappate correttamente!")

# Maschera aree sistema specchio maggiormente rilevanti nel compito

juelich_atlas = datasets.fetch_atlas_juelich('maxprob-thr25-1mm')
juelich_atlas_img = juelich_atlas.maps  # Oggetto NIfTI
juelich_data = juelich_atlas_img.get_fdata()
print(f"Dimensioni dell'atlante Juelich: {juelich_data.shape} (x, y, z)")


modulo_0_labels = ["GM Broca's area BA44", "GM Inferior parietal lobule PGp", "GM Premotor cortex BA6"]


modulo_0_ids = [i for i, label in enumerate(labels) if label in modulo_0_labels]
print(f"IDs delle regioni nel Modulo 0: {modulo_0_ids}")

modulo_0_mask = np.isin(juelich_data, modulo_0_ids).astype(np.uint8)

resampled_modulo_0_mask_img = resample_to_img(
    source_img=nib.Nifti1Image(modulo_0_mask, juelich_atlas_img.affine),
    target_img=fmri_img,
    interpolation="nearest",
)


if len(fmri_img.shape) == 4:
    fmri_img_display = image.mean_img(fmri_img)  # Media nel tempo
else:
    fmri_img_display = fmri_img  # Caso 3D

coordinates = {
    "GM Broca's area BA44": (-48, 12, 20),
    "GM Inferior parietal lobule PGp": (-50, -55, 45),
    "GM Premotor cortex BA6": (-50, -5, 50)
}

vivid_cmap = LinearSegmentedColormap.from_list("vivid", ["#FF5733", "#33FF57", "#3357FF"], N=256)

for label, cut_coords in coordinates.items():
    plot_roi(
        resampled_modulo_0_mask_img,
        bg_img=fmri_img_display,
        title=f"Visualizzazione di {label}",
        display_mode='ortho',
        cut_coords=cut_coords,
        colorbar=True,
        cmap=vivid_cmap
    )
    plt.show()

cut_coords_combined = (-48, -30, 35)
plot_roi(
    resampled_modulo_0_mask_img,
    bg_img=fmri_img_display,
    title="Visualizzazione combinata delle regioni",
    display_mode='ortho',
    cut_coords=cut_coords_combined,
    colorbar=True,
    cmap=vivid_cmap
)
plt.show()


