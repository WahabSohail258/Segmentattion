def find_root(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(parent, x, y):
    root_x = find_root(parent, x)
    root_y = find_root(parent, y)
    if root_x != root_y:
        parent[root_y] = root_x

def pad_image(image):
    return np.pad(image, pad_width=1, mode='constant', constant_values=0)

def connected_component_labeling(image, vset):
    
    parent = {}
    padded_image = pad_image(image)
    rows, cols = padded_image.shape
    labels = np.zeros((rows, cols), dtype=int)
    current_label = 1
    neighbors = [(-1, -1), (0, -1), (-1,1), (-1, 0)] 

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if padded_image[i, j] in vset:
                neighbor_labels = []
                for dx, dy in neighbors:
                    neighbor_label = labels[i+dx, j+dy]
                    if neighbor_label > 0:
                        neighbor_labels.append(neighbor_label)
                
                if not neighbor_labels:
                    labels[i, j] = current_label
                    parent[current_label] = current_label
                    current_label += 1
                else:
                    min_label = min(neighbor_labels)
                    labels[i, j] = min_label
                    for label in neighbor_labels:
                        union(parent, label, min_label)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if labels[i, j] > 0:
                labels[i, j] = find_root(parent, labels[i, j])

    unique_labels = np.unique(labels[labels > 0])
    
    label_map = {}
    new_label = 1
    for old in unique_labels:
        if old > 0:
            label_map[old] = new_label
            new_label += 1

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if labels[i, j] > 0:
                labels[i, j] = label_map[labels[i, j]]

    return labels[1:-1, 1:-1]

def get_largest_component(mask, vset):
    
    labels = connected_component_labeling(mask, vset)
    unique_labels, counts = np.unique(labels[labels > 0], return_counts=True)
    
    if len(unique_labels) > 0:
        largest_label = unique_labels[np.argmax(counts)]
        return (labels == largest_label).astype(np.uint8) * 255
    
    return np.zeros_like(mask)
