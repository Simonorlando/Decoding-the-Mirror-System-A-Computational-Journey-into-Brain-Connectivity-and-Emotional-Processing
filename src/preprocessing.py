file_path = ''
fmri_img = nib.load(file_path)
fmri_data = fmri_img.get_fdata()

timepoint = 10
fmri_volume = fmri_data[:, :, :, timepoint]
fmri_volume_mean = np.mean(fmri_data, axis=3)  # Media su tutte le scansioni temporali

x_center = fmri_volume.shape[0] // 2
y_center = fmri_volume.shape[1] // 2
z_center = fmri_volume.shape[2] // 2

plt.figure(figsize=(15, 5))

# Slice lungo l'asse sagittale (x)
plt.subplot(1, 3, 1)
plt.imshow(fmri_volume[x_center, :, :].T, cmap="gray", origin="lower")
plt.title(f"Slice sagittale (x={x_center})")
plt.axis("off")

# Slice lungo l'asse coronale (y)
plt.subplot(1, 3, 2)
plt.imshow(fmri_volume[:, y_center, :].T, cmap="gray", origin="lower")
plt.title(f"Slice coronale (y={y_center})")
plt.axis("off")

# Slice lungo l'asse assiale (z)
plt.subplot(1, 3, 3)
plt.imshow(fmri_volume[:, :, z_center].T, cmap="gray", origin="lower")
plt.title(f"Slice assiale (z={z_center})")
plt.axis("off")

plt.tight_layout()
plt.show()

dim_x, dim_y, dim_z, dim_t = fmri_data.shape
print(f"Dimensioni fMRI: {fmri_data.shape} (x, y, z, tempo)")

fmri_3d = fmri_img.slicer[..., 0]  # Primo volume temporale (indice 0)
interactive_display = plotting.view_img(fmri_3d, threshold=None, title='Volume Temporale 0')
interactive_display
