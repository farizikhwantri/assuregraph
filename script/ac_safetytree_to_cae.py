import re
import json
import networkx as nx
import os
import matplotlib.pyplot as plt
from pydot import graph_from_dot_data
from collections import defaultdict

# Map GSN node types to CAE roles
TYPE_MAPPING = {
    'Hazard': 'Claim',
    'SafetyRequirement': 'Claim',
    'Requirement': 'Claim',
    'ProcessRequirement': 'Argument',
    'DesignDefinition': 'Argument',
    'Delegated': 'Argument',
    'Package': 'Evidence',
    'Code with': 'Evidence',
    'EnvironmentalAssumption': 'Context',
    'Context': 'Context',
    'Strategy': 'Evidence',
    'WARNING': 'Evidence',
    'SafetyAnalysis': 'Evidence',
    'Assumption': 'Context',
    'Acceptance': 'Evidence',
    'FormalReview': 'Evidence',
    'Simulation': 'Evidence',
    'Test': 'Evidence',
}

def parse_node_label(label):
    parts = label.strip().split('\\n')
    # print("parts:", parts, "label:", label)
    for p in parts:
        for key in TYPE_MAPPING:
            if key in p:
                # print(f"Matched {key} in {p}")
                # remove \" from p
                p = p.replace('"', '').strip()
                description = '\n'.join(parts[2:])
                description = description.replace('"', '').strip()
                # remove enter from description
                description = re.sub(r'\n+', ' ', description)
                # remove two spaces from description
                description = re.sub(r'\s{2,}', ' ', description)
                return TYPE_MAPPING[key], description, p
    print("No matching type found in label:", label)
    # If no type matches, return 'Unknown' type with the full label
    return 'Unknown', label, 'Unknown'

def build_cae(dot_string):
    graphs = graph_from_dot_data(dot_string)
    graph = nx.DiGraph(nx.drawing.nx_pydot.from_pydot(graphs[0]))

    cae_output = {
        'nodes': {},
        'parent_child': defaultdict(list),
        # 'claims': [],
        # 'arguments': [],
        # 'evidences': [],
        # 'contexts': [],
        # 'links': []
    }

    id_to_node = {}

    for node_id, attrs in graph.nodes(data=True):
        label = attrs.get('label', '')
        node_type, description, old_type = parse_node_label(label)
        if node_type == 'Unknown' and not description.strip():
            # Skip nodes with no description
            continue
        # print(f"Processing node {node_id}: type={node_type}, description={description}")
        entry = {
            'id': node_id,
            'type': node_type,
            'old_type': old_type,
            'description': description.strip()
        }
        id_to_node[node_id] = entry
        # print(f"Node {node_id} parsed as {node_type} with description: {description}")
        # lower_type = node_type.lower() + 's'
        # if lower_type in cae_output:
        #     print (f"Adding {node_type}")
        #     cae_output[lower_type].append(entry)
        cae_output['nodes'][node_id] = entry        

    for source, target in graph.edges():
        # cae_output['links'].append({'from': source, 'to': target})
        if source in id_to_node and target in id_to_node:
            cae_output['parent_child'][source].append(target)
            # cae_output['links'].append({
            #     'from': source,
            #     'to': target
            # })
        else:
            print(f"Warning: Edge from {source} to {target} has missing nodes.")

    # Attach contexts to relevant elements
    # for context in cae_output['contexts']:
    #     for link in cae_output['links']:
    #         if link['from'] == context['id']:
    #             target = id_to_node[link['to']]
    #             target.setdefault('context', []).append(context['description'])

    return cae_output

def visualize_cae_graph(cae_result):
    """
    Visualize the CAE graph based on the 'links' and node elements in cae_result,
    arranged in a hierarchical tree structure.
    
    Args:
        cae_result (dict): Output dictionary from build_cae containing keys such as
                           'claims', 'arguments', 'evidences', 'contexts', and 'links'.
    """
    import networkx as nx
    import matplotlib.pyplot as plt

    # Create a directed graph
    G = nx.DiGraph()
    
    # Combine nodes from every category.
    all_nodes = []
    for key in ['claims', 'arguments', 'evidences', 'contexts']:
        all_nodes.extend(cae_result.get(key, []))
    
    # Add nodes with labels.
    for node in all_nodes:
        label = f"{node['id']} ({node['type']})"
        G.add_node(node['id'], label=label)
    
    # Add edges from the 'links' list.
    for link in cae_result.get('links', []):
        G.add_edge(link['from'], link['to'])
    
    # Setting Graphviz graph attribute to force tree (top-to-bottom) layout.
    G.graph['graph'] = {'rankdir': 'TB'}
    
    # Create a hierarchical layout using Graphviz 'dot'. Requires pydot to be installed.
    try:
        pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
    except Exception as e:
        print("graphviz_layout failed, falling back to spring layout:", e)
        pos = nx.spring_layout(G, seed=42)
    
    # Draw the graph.
    node_labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1500)
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color="gray", width=2)
    nx.draw_networkx_labels(G, pos, node_labels, font_size=10)
    
    plt.title("CAE Graph Visualization (Tree Layout)", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def process_one_document(dot_file):
    with open(dot_file, "r") as f:
        dot_data = f.read()

    cae_result = build_cae(dot_data)
    # print(cae_result)
    # add         "num_nodes": 24,
        # "num_edges": ,
        # "docname": ,
        # "requirement": ,
        # "model_name": "human",
        # "generation": "ground_truth"
    cae_result["num_nodes"] = len(cae_result["nodes"])
    cae_result["num_edges"] = sum(len(children) for children in cae_result["parent_child"].values())
    cae_result["docname"] = os.path.basename(dot_file)
    cae_result["requirement"] = os.path.splitext(os.path.basename(dot_file))[0]
    cae_result["model_name"] = "human"
    cae_result["generation"] = "ground_truth"

    return cae_result


# Example usage
if __name__ == "__main__":
    # input the argument for the dot file
    import argparse
    parser = argparse.ArgumentParser(description="Convert a Graphviz assurance case into a CAE model.")
    parser.add_argument("--input_dots", type=str, required=True, help="Path to the Graphviz dot file representing the assurance case")
    # output file
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the CAE model output")
    args = parser.parse_args()

    results = []

    # if input is a directory, process all dot files in the directory
    if os.path.isdir(args.input_dots):
        import glob
        dot_files = glob.glob(os.path.join(args.input_dots, "*.gv"))
        for dot_file in dot_files:
            print(f"Processing file: {dot_file}")
            cae_result = process_one_document(dot_file)
            results.append(cae_result)
    else:
        cae_result = process_one_document(args.input_dots)
        # add docname, 
        results.append(cae_result)

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)

    # visualize_cae_graph(cae_result)
