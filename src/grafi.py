#Grafo maschera

G = nx.Graph()

for area in correlation_matrix.columns:
    G.add_node(area)

for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        correlation_value = correlation_matrix.iloc[i, j]
        region1 = correlation_matrix.columns[i]
        region2 = correlation_matrix.columns[j]
        G.add_edge(region1, region2, weight=correlation_value)

pos = nx.spring_layout(G, seed=42, k=0.15)

plt.figure(figsize=(16, 10))

nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightgreen', alpha=0.8)

high_corr_edges = []
low_corr_edges = []
medium_corr_edges = []

high_corr_weights = []
low_corr_weights = []
medium_corr_weights = []

for edge in G.edges(data=True):
    weight = edge[2]['weight']
    if weight > 0.97:
        high_corr_edges.append(edge)
        high_corr_weights.append(weight)
    elif weight < 0.56:
        low_corr_edges.append(edge)
        low_corr_weights.append(weight)
    else:
        medium_corr_edges.append(edge)
        medium_corr_weights.append(weight)

nx.draw_networkx_edges(G, pos, edgelist=low_corr_edges, width=1, alpha=0.7, edge_color='blue')

nx.draw_networkx_edges(G, pos, edgelist=high_corr_edges, width=1, alpha=0.8, edge_color='red')

nx.draw_networkx_edges(G, pos, edgelist=medium_corr_edges, width=0.5, alpha=0.3, edge_color='gray')

nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', font_color='black')

legend_elements = [
    Line2D([0], [0], color='red', lw=1, label='Correlazioni alte (> 0.97)'),
    Line2D([0], [0], color='blue', lw=1, label='Correlazioni basse (< 0.56)'),
    Line2D([0], [0], color='gray', lw=0.5, label='Correlazioni medie (0.57 - 0.97)'),
]

plt.legend(handles=legend_elements, loc='upper left', fontsize=12)

plt.title("Grafo sistema specchio con Correlazioni Pesate", fontsize=16)

plt.axis('off')

plt.show()



areas_in_mask = np.unique(resampled_mask_data[resampled_mask_data > 0])

coordinates_3d = {}

affine = juelich_atlas_img.affine

for area in areas_in_mask:
    area_voxels = np.where(resampled_mask_data == area)

    x_voxels, y_voxels, z_voxels = area_voxels

    coordinates_3d_area = np.vstack((x_voxels, y_voxels, z_voxels, np.ones_like(x_voxels)))
    real_coordinates = np.dot(affine, coordinates_3d_area)[:3].T

    mean_coordinates = np.mean(real_coordinates, axis=0)

    coordinates_3d[area] = mean_coordinates

for area, coord in coordinates_3d.items():
    print(f"Area {area}: Coordinate 3D = {coord}")



# Dati delle coordinate (x, y) delle aree cerebrali (estratte da 3D)
coordinates = {
    "GM Anterior intra-parietal sulcus hIP1": [-48.0738342, -94.31606218],
    "GM Anterior intra-parietal sulcus hIP2": [-44.19390582, -89.8365651],
    "GM Anterior intra-parietal sulcus hIP3": [-52.99664992, -95.64489112],
    "GM Broca's area BA44": [-48.19812383, -63.89193246],
    "GM Inferior parietal lobule PF": [-51.82032478, -88.06914615],
    "GM Inferior parietal lobule PFcm": [-51.01643836, -87.69726027],
    "GM Inferior parietal lobule PFm": [-59.88517442, -93.7994186],
    "GM Inferior parietal lobule PFop": [-50.68587361, -81.69702602],
    "GM Inferior parietal lobule PFt": [-53.11292346, -82.48933501],
    "GM Inferior parietal lobule PGp": [-52.92454998, -106.18690157],
    "GM Inferior parietal lobule Pga": [-57.40584246, -98.75273865],
    "GM Premotor cortex BA6": [-52.35521628, -73.02249364],
    "GM Primary motor cortex BA4a": [-52.13435115, -84.23816794],
    "GM Primary motor cortex BA4p": [-50.72327965, -80.61639824],
    "GM Superior parietal lobule 7A": [-50.61768489, -100.24790997],
    "GM Superior parietal lobule 7M": [-53.26349206, -105.66666667],
    "GM Superior parietal lobule 7P": [-53.87253521, -106.66056338],
    "GM Superior parietal lobule 7PC": [-57.18971631, -94.2677305]
}

G = nx.Graph()

for region_name, coord in coordinates.items():
    G.add_node(region_name, pos=coord)

for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        correlation_value = correlation_matrix.iloc[i, j]
        region1 = correlation_matrix.columns[i]
        region2 = correlation_matrix.columns[j]
        G.add_edge(region1, region2, weight=correlation_value)

pos = nx.get_node_attributes(G, 'pos')

high_corr_edges = []
low_corr_edges = []
medium_corr_edges = []
max_corr_edges = []

for edge in G.edges(data=True):
    weight = edge[2]['weight']
    if weight > 0.98:
        max_corr_edges.append(edge)
    elif weight > 0.90:
        high_corr_edges.append(edge)
    elif weight < 0.56:
        low_corr_edges.append(edge)
    else:
        medium_corr_edges.append(edge)

plt.figure(figsize=(26, 18))

nx.draw_networkx_nodes(G, pos, node_size=1500, node_color='lightgreen', alpha=0.8)

nx.draw_networkx_edges(G, pos, edgelist=low_corr_edges, width=2, alpha=0.7, edge_color='blue')

nx.draw_networkx_edges(G, pos, edgelist=high_corr_edges, width=2, alpha=0.7, edge_color='black')

nx.draw_networkx_edges(G, pos, edgelist=medium_corr_edges, width=1, alpha=0.3, edge_color='gray')

nx.draw_networkx_edges(G, pos, edgelist=max_corr_edges, width=3, alpha=0.9, edge_color='red')

region_numbers = {region: str(i) for i, region in enumerate(coordinates.keys(), 1)}
nx.draw_networkx_labels(G, pos, labels=region_numbers, font_size=14, font_weight='bold', font_color='black')

legend_elements = [
    Line2D([0], [0], color='red', lw=3, label='Correlazioni massime (> 0.98)'),
    Line2D([0], [0], color='black', lw=2, label='Correlazioni alte (0.90 - 0.97)'),
    Line2D([0], [0], color='blue', lw=1.5, label='Correlazioni basse (< 0.56)'),
    Line2D([0], [0], color='gray', lw=1, label='Correlazioni medie (0.57 - 0.90)'),
]

plt.legend(handles=legend_elements, loc='upper left', fontsize=18)

area_labels = {i: area for i, area in enumerate(coordinates.keys(), 1)}
plt.figtext(0.04, 0.45, "Legenda delle aree cerebrali:\n" + "\n".join([f"{k}: {v}" for k, v in area_labels.items()]),
            ha="left", fontsize=18)

plt.title("Grafo sistema specchio con Correlazioni Pesate e Posizione Anatomica", fontsize=16)

plt.axis('off')

plt.show()



# ## Analisi del grafo maschera

num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
density = nx.density(G)
average_degree = np.mean([deg for node, deg in G.degree()])
assortativity = nx.degree_assortativity_coefficient(G)
clustering_coefficient = nx.average_clustering(G)
diameter = nx.diameter(G) if nx.is_connected(G) else "Inf"  # Diametro se il grafo è connesso
avg_shortest_path = nx.average_shortest_path_length(G) if nx.is_connected(G) else "Inf"  # Lunghezza media del cammino più breve

print("Grafico delle statistiche di base:")
print(f"Numero di nodi: {num_nodes}")
print(f"Numero di archi: {num_edges}")
print(f"Densità del grafo: {density}")
print(f"Assortatività del grafo: {assortativity}")
print(f"Grado medio: {average_degree}")
print(f"Coefficiente di clustering medio: {clustering_coefficient}")
print(f"Diametro del grafo: {diameter}")
print(f"Lunghezza media del cammino più breve: {avg_shortest_path}")


degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)

sorted_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
sorted_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)

# I primi 5 hubs (nodi con il grado maggiore)
print("Top 5 Hubs (centralità di grado):")
for node, value in sorted_degree[:5]:
    print(f"{node}: {value}")

# I primi 5 influencer (nodi con la maggiore centralità di intermediazione)
print("\nTop 5 Influencer (centralità di intermediazione):")
for node, value in sorted_betweenness[:5]:
    print(f"{node}: {value}")



node_corr_high = {node: 0 for node in G.nodes}
for edge in high_corr_edges:
    node_corr_high[edge[0]] += 1
    node_corr_high[edge[1]] += 1

top_high_corr_nodes = sorted(node_corr_high.items(), key=lambda x: x[1], reverse=True)

print("\nTop 5 nodi con più archi con correlazioni alte (> 0.90):")
for i, (node, count) in enumerate(top_high_corr_nodes[:5]):
    print(f"{i+1}. {node}: {count} archi")

# ## Grafo con tutte le aree cerebrali

juelich_atlas = datasets.fetch_atlas_juelich('maxprob-thr25-1mm')
juelich_atlas_img = juelich_atlas.maps
juelich_data = juelich_atlas_img.get_fdata()
labels = juelich_atlas['labels']

regions_with_labels = {
    i: label for i, label in enumerate(labels) if label != 'Background'
}

areas_in_mask = np.unique(juelich_data[juelich_data > 0])

coordinates_3d = {}
affine = juelich_atlas_img.affine

for area in areas_in_mask:
    # Trova i voxel della regione
    area_voxels = np.where(juelich_data == area)

    voxel_coords = np.vstack((area_voxels[0], area_voxels[1], area_voxels[2], np.ones_like(area_voxels[0])))
    real_coords = np.dot(affine, voxel_coords)[:3].T  # Coordinate 3D
    mean_coordinates = np.mean(real_coords, axis=0)

    coordinates_3d[regions_with_labels.get(area, f"Region {int(area)}")] = mean_coordinates

coordinates_2d = {label: (coord[0], coord[1]) for label, coord in coordinates_3d.items()}

G2 = nx.Graph()

for label, coord in coordinates_2d.items():
    G2.add_node(label, pos=coord)

num_areas = len(coordinates_2d)
correlation_matrix = np.random.rand(num_areas, num_areas)
region_names = list(coordinates_2d.keys())

for i in range(num_areas):
    for j in range(i + 1, num_areas):
        weight = correlation_matrix[i, j]
        G2.add_edge(region_names[i], region_names[j], weight=weight)

pos = nx.get_node_attributes(G2, 'pos')

high_corr_edges = [(u, v) for u, v, d in G2.edges(data=True) if 0.80 <= d['weight'] <= 1.0]
medium_corr_edges = [(u, v) for u, v, d in G2.edges(data=True) if 0.50 <= d['weight'] < 0.80]
low_corr_edges = [(u, v) for u, v, d in G2.edges(data=True) if 0.0 <= d['weight'] < 0.50]

plt.figure(figsize=(16, 10))

nx.draw_networkx_nodes(G2, pos, node_size=50, node_color="black", alpha=0.6)

nx.draw_networkx_edges(G2, pos, edgelist=low_corr_edges, width=0.2, edge_color="blue", alpha=0.2)

nx.draw_networkx_edges(G2, pos, edgelist=medium_corr_edges, width=0.4, edge_color="gray", alpha=0.5)

nx.draw_networkx_edges(G2, pos, edgelist=high_corr_edges, width=0.2, edge_color="red", alpha=0.5)

legend_elements = [
    Line2D([0], [0], color='red', lw=2, label='Correlazioni alte (0.80 - 1.0)'),
    Line2D([0], [0], color='gray', lw=2, label='Correlazioni medie (0.50 - 0.79)'),
    Line2D([0], [0], color='blue', lw=2, label='Correlazioni basse (0.0 - 0.49)')
]
plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

plt.title("Grafo con tutte le aree cerebrali", fontsize=16)
plt.axis('off')
plt.show()


num_nodes = G2.number_of_nodes()
num_edges = G2.number_of_edges()
density = nx.density(G2)
average_degree = np.mean([deg for node, deg in G2.degree()])
assortativity = nx.degree_assortativity_coefficient(G2)
clustering_coefficient = nx.average_clustering(G2)
diameter = nx.diameter(G2) if nx.is_connected(G2) else "Inf"  # Diametro se il grafo è connesso
avg_shortest_path = nx.average_shortest_path_length(G2) if nx.is_connected(
    G2) else "Inf"  # Lunghezza media del cammino più breve

print("Grafico delle statistiche di base:")
print(f"Numero di nodi: {num_nodes}")
print(f"Numero di archi: {num_edges}")
print(f"Densità del grafo: {density}")
print(f"Assortatività del grafo: {assortativity}")
print(f"Grado medio: {average_degree}")
print(f"Coefficiente di clustering medio: {clustering_coefficient}")
print(f"Diametro del grafo: {diameter}")
print(f"Lunghezza media del cammino più breve: {avg_shortest_path}")



degree_centrality = nx.degree_centrality(G2)
betweenness_centrality = nx.betweenness_centrality(G2)

sorted_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
sorted_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)

# I primi 5 hubs (nodi con il grado maggiore)
print("Top 5 Hubs (centralità di grado):")
for node, value in sorted_degree[:5]:
    print(f"{node}: {value}")

# I primi 5 influencer (nodi con la maggiore centralità di intermediazione)
print("\nTop 5 Influencer (centralità di intermediazione):")
for node, value in sorted_betweenness[:5]:
    print(f"{node}: {value}")


node_corr_high = {node: 0 for node in G2.nodes}
for edge in high_corr_edges:
    node_corr_high[edge[0]] += 1
    node_corr_high[edge[1]] += 1

top_high_corr_nodes = sorted(node_corr_high.items(), key=lambda x: x[1], reverse=True)

print("\nTop 5 nodi con più archi con correlazioni alte (> 0.90):")
for i, (node, count) in enumerate(top_high_corr_nodes[:5]):
    print(f"{i + 1}. {node}: {count} archi")

