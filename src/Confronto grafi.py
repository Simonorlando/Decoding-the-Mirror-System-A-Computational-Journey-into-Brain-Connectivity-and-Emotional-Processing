is_subgraph = nx.is_isomorphic(G, G2.subgraph(G.nodes))
print(f"Is G a subgraph of G2? {is_subgraph}")

node_overlap = len(set(G2.nodes) & set(G.nodes)) / len(G2.nodes)
print(f"Percentuale di nodi di G in G2: {node_overlap * 100:.2f}%")

edge_overlap = len(set(G2.edges) & set(G.edges)) / len(G2.edges)
print(f"Percentuale di archi di G in G2: {edge_overlap * 100:.2f}%")

pos = nx.spring_layout(G2, seed=42)  # Seed per layout stabile

plt.figure(figsize=(12, 8))
nx.draw(
    G2, pos, node_color="lightgray", edge_color="gray", with_labels=False, node_size=100, alpha=0.9
)

nx.draw(
    G, pos, node_color="red", edge_color="red", with_labels=False, node_size=100, alpha=0.5
)

plt.title("Grafo specchio sovrapposto", fontsize=16)

plt.show()


def calculate_graph_metrics(graph):
    metrics = {}
    metrics["Numero di nodi"] = graph.number_of_nodes()
    metrics["Numero di archi"] = graph.number_of_edges()
    metrics["Densità"] = nx.density(graph)
    metrics["Diametro"] = nx.diameter(graph) if nx.is_connected(graph) else None
    metrics["Grado medio"] = sum(dict(graph.degree()).values()) / graph.number_of_nodes()
    metrics["Coefficiente di clustering medio"] = nx.average_clustering(graph)
    return metrics

metrics_G = calculate_graph_metrics(G)
metrics_G2 = calculate_graph_metrics(G2)

comparison_df = pd.DataFrame([metrics_G, metrics_G2], index=["G", "G2"])

print("Confronto tra le metriche dei grafi:")
display(comparison_df)
