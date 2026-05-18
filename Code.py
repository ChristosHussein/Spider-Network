import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import networkx as nx
import sknw
import os

def analyze_spider_web(image_filename, title_prefix):
    print(f"\n{'='*50}")
    print(f"ΕΝΑΡΞΗ ΑΝΑΛΥΣΗΣ: {image_filename}")
    print(f"{'='*50}")
    
    # 1. ΦΟΡΤΩΣΗ ΕΙΚΟΝΑΣ
    current_folder = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_folder, image_filename)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"ΣΦΑΛΜΑ: Η εικόνα '{image_filename}' δεν βρέθηκε! Βεβαιώσου ότι είναι στον ίδιο φάκελο.")
        return

    # 2. ΕΞΥΠΝΗ ΕΠΕΞΕΡΓΑΣΙΑ (Otsu's Method)
    print("Αυτόματος διαχωρισμός φόντου/ιστού (Otsu)...")
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    skeleton = skeletonize(thresh > 0)

    print("Εξαγωγή του Γράφου...")
    graph = sknw.build_sknw(skeleton)
    G = nx.Graph(graph)

    # 3. ΤΟΠΟΛΟΓΙΚΟΣ ΚΑΘΑΡΙΣΜΟΣ
    print("Καθαρισμός δικτύου...")
    for comp in list(nx.connected_components(G)):
        if len(comp) < 100:
            G.remove_nodes_from(comp)

    for _ in range(3):
        dead_ends = [node for node, degree in dict(G.degree()).items() if degree <= 1]
        G.remove_nodes_from(dead_ends)

    while True:
        deg2_nodes = [n for n, d in dict(G.degree()).items() if d == 2]
        if not deg2_nodes: break 
        for n in deg2_nodes:
            if G.has_node(n):
                neighbors = list(G.neighbors(n))
                if len(neighbors) == 2:
                    u, v = neighbors
                    if u != v:
                        pts_u = G[u][n].get('pts', [])
                        pts_v = G[n][v].get('pts', [])
                        new_pts = np.vstack((pts_u, pts_v)) if len(pts_u)>0 and len(pts_v)>0 else (pts_u if len(pts_u)>0 else pts_v)
                        G.add_edge(u, v, pts=new_pts)
                G.remove_node(n)

    # 4. ΥΠΟΛΟΓΙΣΜΟΣ ΜΕΤΡΙΚΩΝ
    nodes_count = G.number_of_nodes()
    edges_count = G.number_of_edges()
    avg_degree = (2 * edges_count) / nodes_count if nodes_count > 0 else 0

    try:
        largest_cc = max(nx.connected_components(G), key=len)
        G_main = G.subgraph(largest_cc)
        C = nx.average_clustering(G_main)
        L = nx.average_shortest_path_length(G_main)
    except ValueError:
        C = 0
        L = 0

    print("\n--- ΑΠΟΤΕΛΕΣΜΑΤΑ ---")
    print(f"Αριθμός Κόμβων (N): {nodes_count}")
    print(f"Αριθμός Ακμών: {edges_count}")
    print(f"Μέσος Βαθμός <k>: {avg_degree:.2f}")
    print(f"Συντελεστής Ομαδοποίησης <C>: {C:.4f}")
    print(f"Μέσο Μήκος Μονοπατιού <l>: {L:.4f}")
    print("--------------------\n")

    # 5. ΟΠΤΙΚΟΠΟΙΗΣΗ
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap='gray')

    for s, e, data in G.edges(data=True):
        if 'pts' in data and len(data['pts']) > 0:
            ps = data['pts']
            plt.plot(ps[:, 1], ps[:, 0], 'green', linewidth=1.2)
        else:
            try:
                node_s = G.nodes[s]['o']
                node_e = G.nodes[e]['o']
                plt.plot([node_s[1], node_e[1]], [node_s[0], node_e[0]], 'green', linewidth=1.2)
            except KeyError:
                pass

    valid_nodes = [data['o'] for n, data in G.nodes(data=True) if 'o' in data]
    if valid_nodes:
        ps = np.array(valid_nodes)
        plt.plot(ps[:, 1], ps[:, 0], 'ro', markersize=2)

    plt.title(f"{title_prefix}\nN:{nodes_count} | <k>:{avg_degree:.2f} | <C>:{C:.4f} | <l>:{L:.2f}")
    plt.axis('off')
    plt.tight_layout()

# ==========================================
# ΕΚΤΕΛΕΣΗ ΚΑΙ ΓΙΑ ΤΙΣ ΔΥΟ ΕΙΚΟΝΕΣ
# ==========================================
# Βεβαιώσου ότι έχεις και τα δύο αρχεία (web.jpg και cartoon.jpg) στον ίδιο φάκελο με το Code.py
analyze_spider_web('web.jpg', 'Ρεαλιστικός Ιστός (Με Θόρυβο)')
analyze_spider_web('cartoon.jpg', 'Ιδεατός Ιστός (Synthetic Data)')

# Εμφανίζει και τα δύο παράθυρα γραφικών ταυτόχρονα στο τέλος
plt.show()