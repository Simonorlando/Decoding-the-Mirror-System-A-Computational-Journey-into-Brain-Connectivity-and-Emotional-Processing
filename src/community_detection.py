# Partition maschera
partition = community_louvain.best_partition(G)


pos = nx.spring_layout(G, seed=42, k=0.2)

plt.figure(figsize=(18, 12))

node_colors = [partition[node] for node in G.nodes()]
cmap = plt.cm.get_cmap('rainbow', max(node_colors) + 1)


nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors, cmap=cmap, alpha=0.8)


nx.draw_networkx_edges(G, pos, alpha=0.7, width=2, edge_color='gray')


nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', font_color='black')


plt.title("Identificazione dei Moduli con algoritmo di Louvain", fontsize=16)


plt.axis('off')

legend_elements = [Line2D([0], [0], marker='o', color='w', label=f"Modulo {i}",
                          markerfacecolor=cmap(i), markersize=10) for i in range(max(node_colors) + 1)]

plt.legend(handles=legend_elements, loc='upper left', fontsize=12)

plt.show()

print("Divisione in moduli:")
for community, nodes in partition.items():
    print(f"Comunità {community}: {nodes}")


# Partition globale
partition2 = community_louvain.best_partition(G2)


# In[102]:


if not partition2:
    print("Errore: partition2 non contiene dati. Assicurati che i moduli siano stati calcolati correttamente.")
else:
    plt.figure(figsize=(18, 12))

    node_colors = [partition2[node] for node in G2.nodes()]
    cmap = plt.cm.get_cmap('rainbow', max(node_colors) + 1)

    nx.draw_networkx_nodes(G2, pos, node_size=500, node_color=node_colors, cmap=cmap, alpha=0.8)

    plt.title("Visualizzazione dei Moduli (Louvain)", fontsize=16)

    plt.axis('off')

    legend_elements = [Line2D([0], [0], marker='o', color='w', label=f"Modulo {i}",
                              markerfacecolor=cmap(i), markersize=10) for i in range(max(node_colors) + 1)]

    plt.legend(handles=legend_elements, loc='upper left', fontsize=12, frameon=True)

    plt.show()

    communities = {}
    for node, module in partition2.items():
        if module not in communities:
            communities[module] = []
        communities[module].append(node)

    community_data = []
    for module, nodes in communities.items():
        for node in nodes:
            community_data.append({"Modulo": module, "Nodo": node})

    community_df = pd.DataFrame(community_data)


# In[103]:


community_df


# In[104]:


communities = {}
for node, module in partition2.items():
    if module not in communities:
        communities[module] = []
    communities[module].append(node)

community_statistics = []

for module, nodes in communities.items():

    num_nodes = len(nodes)

    # Grado medio dei nodi nella comunità
    avg_degree = np.mean([G2.degree[node] for node in nodes])

    subgraph = G2.subgraph(nodes)
    if nx.is_connected(subgraph):
        diameter = nx.diameter(subgraph)
    else:
        diameter = None

    community_statistics.append({
        "Modulo": module,
        "Numero di Nodi": num_nodes,
        "Grado Medio": avg_degree,
        "Diametro (se connesso)": diameter
    })

statistics_df = pd.DataFrame(community_statistics)


# In[105]:


statistics_df


# In[106]:


x = np.arange(len(statistics_df["Modulo"]))

width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(x - width/2, statistics_df["Numero di Nodi"], width, label="Numero di Nodi", color="skyblue")

plt.bar(x + width/2, statistics_df["Grado Medio"], width, label="Grado Medio", color="orange")

plt.title("Statistiche Contestuali per Modulo", fontsize=16)
plt.xlabel("Modulo", fontsize=12)
plt.ylabel("Valori", fontsize=12)
plt.xticks(x, statistics_df["Modulo"])
plt.legend(bbox_to_anchor=(0.71, 1), loc='upper left', fontsize=8)

plt.tight_layout()
plt.show()


# In[107]:


grouped_communities = community_df.groupby("Modulo")["Nodo"].apply(list).reset_index()

for index, row in grouped_communities.iterrows():
    print(f"Modulo {row['Modulo']}:")
    print(", ".join(row["Nodo"]))
    print("\n" + "-"*50 + "\n")


# ### valutiamo se la maschera del sistema specchio viene raggruppata in uno di questi moduli

# In[108]:


labels_to_check = [
    "GM Anterior intra-parietal sulcus hIP1",
    "GM Anterior intra-parietal sulcus hIP2",
    "GM Anterior intra-parietal sulcus hIP3",
    "GM Broca's area BA44",
    "GM Inferior parietal lobule PF",
    "GM Inferior parietal lobule PFcm",
    "GM Inferior parietal lobule PFm",
    "GM Inferior parietal lobule PFop",
    "GM Inferior parietal lobule PFt",
    "GM Inferior parietal lobule PGp",
    "GM Inferior parietal lobule Pga",
    "GM Premotor cortex BA6",
    "GM Primary motor cortex BA4a",
    "GM Primary motor cortex BA4p",
    "GM Superior parietal lobule 7A",
    "GM Superior parietal lobule 7M",
    "GM Superior parietal lobule 7P",
    "GM Superior parietal lobule 7PC",
]

# Dizionario con i moduli (modules)
modules = {
    0: [
        "Background", "GM Amygdala_centromedial group", "GM Amygdala_laterobasal group", "GM Broca's area BA44",
        "GM Broca's area BA45", "GM Hippocampus cornu ammonis", "GM Hippocampus dentate gyrus",
        "GM Hippocampus entorhinal cortex", "GM Hippocampus hippocampal-amygdaloid transition area",
        "GM Inferior parietal lobule PGp", "GM Insula Id1", "GM Insula Ig1", "GM Lateral geniculate body",
        "GM Premotor cortex BA6", "GM Primary auditory cortex TE1.1",
        "GM Secondary somatosensory cortex / Parietal operculum OP2",
        "GM Secondary somatosensory cortex / Parietal operculum OP3",
        "GM Superior parietal lobule 5Ci", "GM Superior parietal lobule 5M", "GM Visual cortex V5",
        "WM Acoustic radiation", "WM Callosal body", "WM Corticospinal tract", "WM Fornix",
        "WM Superior occipito-frontal fascicle",
    ],
    1: ["GM Superior parietal lobule 7PC", "GM Visual cortex V1 BA17", "GM Visual cortex V2 BA18", "GM Visual cortex V3V"],
    2: [
        "GM Amygdala_superficial group", "GM Anterior intra-parietal sulcus hIP1", "GM Anterior intra-parietal sulcus hIP2",
        "GM Anterior intra-parietal sulcus hIP3", "GM Hippocampus subiculum", "GM Inferior parietal lobule PF",
        "GM Inferior parietal lobule PFcm", "GM Inferior parietal lobule PFm", "GM Inferior parietal lobule PFop",
        "GM Inferior parietal lobule PFt", "GM Inferior parietal lobule Pga", "GM Insula Ig2", "GM Mamillary body",
        "GM Medial geniculate body", "GM Primary auditory cortex TE1.0", "GM Primary auditory cortex TE1.2",
        "GM Primary motor cortex BA4a", "GM Primary motor cortex BA4p", "GM Primary somatosensory cortex BA1",
        "GM Primary somatosensory cortex BA2", "GM Primary somatosensory cortex BA3a", "GM Primary somatosensory cortex BA3b",
        "GM Secondary somatosensory cortex / Parietal operculum OP1", "GM Secondary somatosensory cortex / Parietal operculum OP4",
        "GM Superior parietal lobule 5L", "GM Superior parietal lobule 7A", "GM Superior parietal lobule 7M",
        "GM Superior parietal lobule 7P", "GM Visual cortex V4", "WM Cingulum", "WM Inferior occipital fascicle",
    ],
}

for label in labels_to_check:
    found_in_modules = [module for module, regions in modules.items() if label in regions]
    if found_in_modules:
        print(f"'{label}' si trova nei moduli: {found_in_modules}")
    else:
        print(f"'{label}' non è stato trovato in nessun modulo.")


region_module_data = []

for label in labels_to_check:
    found_in_modules = [module for module, regions in modules.items() if label in regions]
    region_module_data.append({"Regione": label, "Modulo": found_in_modules[0] if found_in_modules else None})

region_module_df = pd.DataFrame(region_module_data)

grouped_regions = region_module_df.groupby("Modulo")["Regione"].apply(list).reset_index()

print("Regioni raggruppate per modulo:")
for _, row in grouped_regions.iterrows():
    print(f"Modulo {row['Modulo']}:")
    print(", ".join(row["Regione"]))
    print("\n" + "-" * 50 + "\n")

