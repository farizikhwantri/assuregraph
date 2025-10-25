import os
import sys
import json
import ast
from collections import OrderedDict
import torch

from torch_geometric.data import Data, InMemoryDataset
import re

# -- Build per-document graph data object --
def create_document_graph(sentences, edge_index, edge_pairs, labels, node_labels=None):
    output = {
        'sentences': sentences,
        'edge_index': edge_index,
        'edge_pairs': edge_pairs,
        'labels': labels,
    }
    if node_labels is not None:
        output['node_labels'] = node_labels
    return output

def transform_document(doc: dict) -> dict:
    """
    Transforms a document (loaded from JSON) into a graph-data dictionary.
    
    The function extracts nodes from the "1_hop" field (using meta information
    to map node identifiers to sentence texts) and uses the "parent_child"
    field to define edge connectivity.
    
    Args:
        doc (dict): A document dictionary that includes fields like 
            "requirement", "model_name", "parent_child", and a "1_hop" list.
    
    Returns:
        dict: A dictionary with keys 'sentences', 'edge_index', 'edge_pairs', 
              and 'labels' (compatible with create_document_graph).
    """
    # --- Extract node information from the '1_hop' field ---
    # We'll collect for each hop its parent's node and current-node.
    sentence_dict = OrderedDict()  # to store sentences indexed by node ID
    node_labels = OrderedDict()  # to store node labels
    direct_edges = []  # list to store (parent, child) pairs
    if "nodes" in doc:
        # If '1_hop' is not present, check for 'nodes' field.
        # print("Processing 'nodes' field in the document.")
        # print(doc)
        nodes = doc["nodes"]
        # print(nodes)
        for node_id, node in nodes.items():
            # print(node_id, node)
            description = node.get("description", "")
            if node_id and description:
                sentence_dict[node_id] = description
            label = node.get('type', "")
            if label:
                node_labels[node_id] = label

    elif "1_hop" in doc:
        # print("Processing '1_hop' field in the document.")
        one_hop = doc["1_hop"]
        for hop in one_hop:
            meta = hop.get("meta", {})
            parent = meta.get("parent_node")
            current = meta.get("current_node")
            # get type of parent and current by removing the '{type}-{num}' or {type}_num suffix
            def extract_type_and_num(node_name):
                """
                Extracts the type and numeric suffix from a node name using regex.
                Example: 'req-1' -> ('req', '1'), 'goal_2' -> ('goal', '2')
                """
                match = re.match(r"([a-zA-Z]+)[-_]?(\d+)?", node_name or "")
                if match:
                    node_type = match.group(1)
                    node_num = match.group(2) if match.group(2) else None
                    return node_type, node_num
                return 'Unknown', None

            parent_type, _ = extract_type_and_num(parent)
            current_type, _ = extract_type_and_num(current)
            # print(f"Parent: {parent}, Current: {current}, Parent Type: {parent_type}, Current Type: {current_type}")
            # Store the node type in node_labels
            if parent:
                node_labels[parent] = parent_type
            if current:
                node_labels[current] = current_type

            # For parent's node, use the 'premise'; for current node, use the 'hypothesis'
            if parent and parent not in sentence_dict:
                sentence_dict[parent] = hop.get("premise", "")
            if current and current not in sentence_dict:
                sentence_dict[current] = hop.get("hypothesis", "")

            if parent and current:
                direct_edges.append((parent, current))

    if not sentence_dict:
        raise ValueError("No node information extracted from '1_hop' field.")
    
    # assert that node_labels and sentence_dict have the same keys
    if set(node_labels.keys()) != set(sentence_dict.keys()):
        print(f"Node labels: {node_labels.keys(), len(node_labels)}")
        print(f"Sentence dict: {sentence_dict.keys(), len(sentence_dict)}")
        raise ValueError("Node labels and sentence dictionary keys do not match.")
    
    # For reproducibility, sort the node keys and build a mapping: node_name -> index.
    # node_keys = sorted(sentence_dict.keys())
    node_keys = sentence_dict
    node_mapping = {node: idx for idx, node in enumerate(node_keys)}
    sentences = [sentence_dict[node] for node in node_keys]
    # sentences = [f"{node}: {sentence_dict[node]}" for node in node_keys]
    
    # --- Parse the parent_child field to obtain connectivity ---
    # if parent is a string
    parent_child = {}
    parent_child_str = doc.get("parent_child", "{}")
    if isinstance(parent_child_str, str):
        try:
            parent_child = ast.literal_eval(parent_child_str)
        except Exception as e:
            raise ValueError(f"Error parsing parent_child field: {e}")
    elif isinstance(parent_child_str, dict):
        parent_child = parent_child_str

    # print(f"Parent-Child Mapping: {parent_child}")

    if len(direct_edges) == 0:
        for parent, children in parent_child.items():
            if parent in node_mapping and isinstance(children, list):
                # print(f"Parent: {parent}, Children: {children}")
                for child in children:
                    if child in node_mapping:
                        direct_edges.append((parent, child))
    
    # print(f"Direct Edges: {direct_edges}")

    # # assert edges pc equals direct_edges
    # if len(edges_pc) != len(direct_edges):
    #     print(f"Edges from parent_child: {edges_pc}")
    #     print(f"Direct edges extracted from '1_hop': {direct_edges}")
    #     raise ValueError("Edges not match.")
    
    # if not edges:
    #     raise ValueError("No valid edges found for the document graph.")

    # Build edge list using only the direct_edges extracted from "1_hop"
    edges = []
    for parent, child in direct_edges:
        if parent in node_mapping and child in node_mapping:
            edges.append((node_mapping[parent], node_mapping[child]))

    if len(edges) == 0:
        raise ValueError(f"No valid direct edges found in the document!, \
                         {doc, edges, direct_edges}")
    
    
    # Convert edge list into PyTorch tensors compatible with your functions
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_pairs = torch.tensor(edges, dtype=torch.long)
    labels = torch.ones(edge_pairs.shape[0], dtype=torch.float)
    
    # print(node_labels)
    # Create and return the document graph dictionary.
    output = {}
    if node_labels:
        output = create_document_graph(sentences, edge_index, edge_pairs, labels, node_labels=node_labels)
    else:
        output = create_document_graph(sentences, edge_index, edge_pairs, labels)
    # output['model_name'] = doc.get('model_name', '')
    return output

class DocumentGraphDataset(InMemoryDataset):
    def __init__(self, root, filename=None, transform=None, pre_transform=None):
        """
        Args:
            root (str): Root directory where the dataset should be saved.
            filename (str): The raw JSON filename.
        """
        self.filename = filename
        self.root = root
        self.raw_dir = os.path.join(root, "raw")
        super(DocumentGraphDataset, self).__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        # Expecting the raw file to be in the 'raw' subfolder.
        return [self.filename]
    
    @property
    def processed_file_names(self):
        return ["data.pt"]
    
    def download(self):
        # If needed, code to download the raw file could go here.
        pass
    
    def process(self):
        data_list = []
        raw_path = os.path.join(self.raw_dir, self.filename)
        # Load the JSON document list
        with open(raw_path, "r") as f:
            docs = json.load(f)
        
        # Process each document using your transform_document() function.
        for doc in docs:
            try:
                graph_dict = transform_document(doc)
                # Convert the graph dictionary into a torch_geometric Data object.
                # Note: we put sentences as an attribute (string list) so that later you can use your encoder.
                data_obj = Data(
                    edge_index=graph_dict["edge_index"],
                    edge_pairs=graph_dict["edge_pairs"],
                    y=graph_dict["labels"],  # assuming labels is a 1D tensor
                )
                # Attach the sentences for later tokenization (this is not used as numerical node features,
                # but can be used in your model's encoder)
                data_obj.sentences = graph_dict["sentences"]
                data_list.append(data_obj)
            except Exception as e:
                print(f"Error processing document: {e}")
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def main():
    default_folder = "/Users/<username>/repositories/REng-xai-cert/data/sac_gdpr/merged_traced"
    default_file_path = "test_data_docname.json"
    file_path = None
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = os.path.join(default_folder, default_file_path)

    hop_file_path = file_path
    with open(hop_file_path, "r") as f:
        hop_data = json.load(f)

    print(f"Loaded {len(hop_data)} hops from {hop_file_path}")

    for i, hop in enumerate(hop_data):
        print(hop.keys())
        # You can further process each hop as needed
        hop_graph = transform_document(hop)
        print("Sentences:")
        for idx, sent in enumerate(hop_graph["sentences"]):
            print(f"{idx}: {sent}")
        print("\nEdge Index:")
        print(hop_graph["edge_index"])
        print("\nEdge Pairs:")
        print(hop_graph["edge_pairs"])
        print("\nLabels:")
        print(hop_graph["labels"])
        if 'node_labels' in hop_graph:
            print("\nNode Labels:")
            print(hop_graph["node_labels"])

    # dataset_root = "/Users/<username>/repositories/REng-xai-cert/data/sac_gdpr/merged_traced"
    # dataset = DocumentGraphDataset(root=dataset_root, filename="test_data_docname.json")

    # print(f"Number of graphs: {len(dataset)}")
    # data = dataset[0]
    # print("Sentences:", data.sentences)
    # print("Edge Index:", data.edge_index)

if __name__ == "__main__":
    main()