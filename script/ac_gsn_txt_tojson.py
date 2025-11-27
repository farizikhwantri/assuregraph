#!/usr/bin/env python3

import argparse
import sys
import re
import json
from collections import defaultdict

def extract_model_and_generation(filename):
    """
    Extract model_name and generation_num from filenames matching the pattern {model_name}_*_{num}.txt.

    Args:
        filename (str): The filename to extract information from.

    Returns:
        tuple: (model_name, generation_num) if the pattern matches, otherwise (None, None).
    """
    # Regular expression to match the filename pattern with one or two numbers
    pattern_two_numbers = re.compile(r"^(.*?)_.*_(\d+_\d+)\.txt$")
    pattern_one_number = re.compile(r"^(.*?)_.*_(\d+)\.txt$")

    # Try to match the two-number pattern first
    match = pattern_two_numbers.match(filename)
    if match:
        model_name = match.group(1)  # Extract the model name
        generation_num = match.group(2)  # Extract the generation number (e.g., "0_0")
        return model_name, generation_num

    # If no match, try the one-number pattern
    match = pattern_one_number.match(filename)
    if match:
        model_name = match.group(1)  # Extract the model name
        generation_num = match.group(2)  # Extract the generation number (e.g., "0")
        return model_name, generation_num

    # If neither pattern matches, return None
    return None, None

def get_node_type(node_id):
    if node_id.startswith("G"):
        return "Goal"
    elif node_id.startswith("S"):
        return "Strategy"
    elif node_id.startswith("C"):
        return "Context"
    elif node_id.startswith("A"):
        return "Assumption"
    elif node_id.startswith("J"):
        return "Justification"
    elif node_id.startswith("Sn"):
        return "Solution"
    elif node_id.startswith("E"):
        return "Evidence"
    else:
        return "Unknown"
    
def substitute_special(parent_id, nodes):
    """
    For InContextOf relationships, if the parent id is "G0.X", try to replace it
    with a node id that matches the pattern "G0.[digits]".
    """
    if parent_id == "G0.X":
        matches = [nid for nid in nodes.keys() if re.match(r"^G0\.\d+$", nid)]
        if matches:
            return matches[0]
    return parent_id


def adjust_dot_zero_nodes(nodes, relationships):
    """
    Adjust relationships to treat nodes with `.0` suffix as the root of subsequent nodes.
    For example, `G3.0` becomes `G3` and is treated as the parent of `G3.1`, `G3.2`, etc.
    """
    updated_nodes = {}
    updated_relationships = []

    for node_id in list(nodes.keys()):
        if node_id.endswith(".0"):
            # Remove `.0` suffix to create the new root node
            root_id = node_id[:-2]
            updated_nodes[root_id] = nodes.pop(node_id)
            updated_nodes[root_id]["id"] = root_id

            # Find all subsequent nodes with the same prefix
            prefix = root_id
            for other_node in list(nodes.keys()):
                if other_node.startswith(prefix + ".") and other_node != root_id:
                    updated_relationships.append((root_id, other_node))
        else:
            updated_nodes[node_id] = nodes[node_id]

    # Add existing relationships to the updated relationships
    updated_relationships.extend(relationships)

    return updated_nodes, updated_relationships

def handle_has_multiplicity(nodes, relationships):
    """
    Handle `HasMultiplicity` relationships where a node like `G.X` has children `G.X.{1,2,...}`.
    Connect these children to the parent of `G.X`.
    """
    multiplicity_pattern = re.compile(r"HasMultiplicity\s*\(\s*([\w.\-]+)\s*,\s*([\w.\-]+)\s*,\s*(\d+)\s*of\s*\*\)")
    for relationship in relationships:
        if isinstance(relationship, str) and multiplicity_pattern.match(relationship):
            match = multiplicity_pattern.match(relationship)
            parent = match.group(1).strip()
            child_prefix = match.group(2).strip()
            count = int(match.group(3).strip())

            # Connect children `child_prefix.{1,2,...}` to the parent
            for i in range(1, count + 1):
                child_id = f"{child_prefix}.{i}"
                if child_id in nodes:
                    relationships.append((parent, child_id))

def adjust_undeclared_nodes(nodes, relationships, filename=None, warn=True, raise_error=False):
    """
    Adjust relationships to handle nodes that are referenced but not declared.
    If a node is referenced but not declared, connect it to its declared sub-nodes (e.g., G3 -> G3.1).
    If no sub-nodes exist, throw an error.
    """
    for parent, child in relationships:
        if parent not in nodes:
            # Check for sub-nodes with the same prefix
            sub_nodes = [nid for nid in nodes.keys() if nid.startswith(parent + ".")]
            if sub_nodes:
                # Connect parent to its sub-nodes
                for sub_node in sub_nodes:
                    relationships.append((sub_node, child))
            else:
                if raise_error:
                    raise ValueError(f"Error: Node '{parent}' is referenced but not declared. {filename}")
                elif warn:
                    print(f"Warning: Node '{parent}' is referenced but not declared. {filename}")
        if child not in nodes:
            # Check for sub-nodes with the same prefix
            sub_nodes = [nid for nid in nodes.keys() if nid.startswith(child + ".")]
            if sub_nodes:
                # Connect child to its sub-nodes
                for sub_node in sub_nodes:
                    relationships.append((parent, sub_node))
            else:
                if raise_error:
                    raise ValueError(f"Error: Node '{child}' is referenced but not declared. {filename}")
                elif warn:
                    print(f"Warning: Node '{child}' is referenced but not declared. {filename}")

def parse_gsn_file(filename, warn=False, raise_error=True):
    """
    Parses an ACAS Xu GSN text file.
    Expected format lines:
      - Node definitions: "NODE_ID: text of the node"
      - SupportedBy lines: "SupportedBy (ParentID, ChildID, weight)"
        or "SupportedBy (ParentID, [ChildID1, ChildID2, ...], weight)"
      - InContextOf lines: "IncontextOf (ChildID, ParentID, weight)" or
                           "IncontextOf (ChildID, [ParentID1, ParentID2, ...], weight)"
    Returns a tuple: (nodes, relationships)
    Each node is set with a type based on its identifier and an empty description if not provided.
    """
    node_pattern = re.compile(r"^([\w.\-]+):\s*(.*)$")
    element_pattern = re.compile(r"^(Goal|Strategy|Solution|Context|Assumption|Justification)\(([\w.\-]+),\s*\"(.*)\"\)$")  # New pattern for elements
    rel_pattern_sup = re.compile(
        r"^SupportedBy\s*\(\s*([\w.\-]+)\s*,\s*(\[[^\]]+\]|[\w.\-]+)\s*,\s*\d+\s*\)"
    )
    rel_pattern_in_one = re.compile(
        r"^IncontextOf\s*\(\s*([\w.\-]+)\s*,\s*([\w.\-]+)\s*,\s*\d+\s*\)",
        re.IGNORECASE,
    )
    rel_pattern_in_multi = re.compile(
        r"^IncontextOf\s*\(\s*([\w.\-]+)\s*,\s*\[([^\]]+)\]\s*,\s*\d+\s*\)",
        re.IGNORECASE,
    )
    
    nodes = {}
    relationships = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # if line start with \t or hypen - or space, remove it
            line = re.sub(r'^[\t -]+', '', line)
            # print("line:", line, "file:", filename)
            if not line:
                continue

            if line.startswith("SupportedBy"):
                m_rel = rel_pattern_sup.match(line)
                if m_rel:
                    parent = m_rel.group(1).strip()
                    child_str = m_rel.group(2).strip()
                    if child_str.startswith("[") and child_str.endswith("]"):
                        child_list = [c.strip() for c in child_str.strip("[]").split(",")]
                        for child in child_list:
                            relationships.append((parent, child))
                    else:
                        relationships.append((parent, child_str))
            # Parse InContextOf: first element is child, second element is parent (or list of parents)
            elif line.startswith("IncontextOf") or line.startswith("InContextOf"):
                m_in_multi = rel_pattern_in_multi.match(line)
                if m_in_multi:
                    child = m_in_multi.group(1).strip()
                    parent_list = [p.strip() for p in m_in_multi.group(2).split(",")]
                    for parent in parent_list:
                        parent = substitute_special(parent, nodes)
                        # relationships.append((parent, child))
                        relationships.append((child, parent))
                else:
                    m_in_one = rel_pattern_in_one.match(line)
                    if m_in_one:
                        child = m_in_one.group(1).strip()
                        parent = substitute_special(m_in_one.group(2).strip(), nodes)
                        # relationships.append((parent, child))
                        relationships.append((child, parent))
            elif line.startswith("HasMultiplicity"):
                # Parse HasMultiplicity relationships
                multiplicity_pattern = re.compile(r"HasMultiplicity\s*\(\s*([\w.\-]+)\s*,\s*([\w.\-]+)\s*,\s*(\d+)\s*of\s*\*\)")
                m_mult = multiplicity_pattern.match(line)
                # print("mult:", m_mult, "file:", filename)
                if m_mult:
                    parent = m_mult.group(1).strip()
                    child_prefix = m_mult.group(2).strip()
                    count = int(m_mult.group(3).strip())
                    # Generate and connect children `child_prefix.{1,2,...}` to the parent
                    for i in range(1, count + 1):
                        child_id = f"{child_prefix}.{i}"
                        if child_id not in nodes:
                            # Create the child node if it doesn't exist
                            nodes[child_id] = {
                                "id": child_id,
                                "type": get_node_type(child_prefix),
                                "description": f"Automatically generated child node {child_id}",
                                "parents": [parent]
                            }
                        relationships.append((parent, child_id))
            else:

                m_node = node_pattern.match(line)
                element_match = element_pattern.match(line)
                if element_match:
                    # Handle elements like Goal, Strategy, etc.
                    node_id = element_match.group(2).strip()
                    text = element_match.group(3).strip()
                    nodes[node_id] = {
                        "id": node_id,
                        "type": element_match.group(1).strip(),
                        "description": text,
                        "parents": []  # list of parents for multiple incoming relationships
                    }
                elif m_node:
                    node_id = m_node.group(1).strip()
                    text = m_node.group(2).strip()
                    nodes[node_id] = {
                        "id": node_id,
                        "type": get_node_type(node_id),
                        "description": text,
                        "parents": []  # list of parents for multiple incoming relationships
                    }

    adjust_dot_zero_nodes(nodes, relationships)
    # handle_has_multiplicity(nodes, relationships)

    adjust_undeclared_nodes(nodes, relationships, filename=filename, warn=warn, raise_error=raise_error)

    return nodes, relationships

def contextual_decomposition_with_hierarchy(nodes, parent_child):
    """
    Decomposes a node with multiple parent contexts into subclaims,
    while preserving the hierarchical structure with the main claim as the root.

    Args:
        nodes (dict): Dictionary of nodes with their details.
        parent_child (dict): Dictionary mapping parent nodes to their child nodes.

    Returns:
        tuple: Updated nodes and parent_child dictionaries.
    """
    updated_nodes = {}
    updated_parent_child = defaultdict(list)

    for node_id, node_data in nodes.items():
        if len(node_data["parents"]) > 1:
            # Create subclaims for each parent context
            for parent in node_data["parents"]:
                subclaim_id = f"{node_id}-{parent}"
                updated_nodes[subclaim_id] = {
                    "id": subclaim_id,
                    "type": node_data["type"],
                    "description": f"{node_data['description']} (context: {parent})",
                    "parents": [node_id]  # Connect subclaim to the main claim
                }
                # Copy child relationships to the subclaim
                if node_id in parent_child:
                    for child in parent_child[node_id]:
                        updated_parent_child[subclaim_id].append(child)
                        # Ensure the child node is connected to the subclaim
                        if child in nodes:
                            nodes[child]["parents"].append(subclaim_id)
            # Preserve the main claim as a standalone node
            updated_nodes[node_id] = {
                "id": node_id,
                "type": node_data["type"],
                "description": node_data["description"],
                "parents": []
            }
        else:
            # Keep the original node if it has only one parent
            updated_nodes[node_id] = node_data
            if node_id in parent_child:
                updated_parent_child[node_id].extend(parent_child[node_id])

    return updated_nodes, dict(updated_parent_child)

def infer_parent_child(nodes, relationships):
    """
    Build a mapping of parent -> list of children.
    For every (parent, child) relationship, add the parent to child's "parents" list.
    """
    parent_child = defaultdict(list)
    for parent, child in relationships:
        # if parent not in nodes:
        #     # nodes[parent] = {"id": parent, "type": get_node_type(parent), "description": "", "parents": []}
        #     # ignore this parent if it is not in nodes
        #     print(f"Warning: Parent {parent} not found in nodes, skipping relationship with child {child}")
        #     continue
        # if child not in nodes:
        #     # nodes[child] = {"id": child, "type": get_node_type(child), "description": "", "parents": [parent]}
        #     # ignore this child if it is not in nodes
        #     print(f"Warning: Child {child} not found in nodes, skipping relationship with parent {parent}")
        #     continue
        # else:
        if child in nodes:
            # Ensure the child node has a "parents" list
            if parent not in nodes[child]["parents"]:
                # print(f"Adding parent {parent} to child {child}")
                nodes[child]["parents"].append(parent)
        else:
            print(f"Warning: Child {child} not found in nodes, skipping relationship with parent {parent}")
            continue
        
        parent_child[parent].append(child)
    # ensure that all parents child is unique
    for parent, children in parent_child.items():
        parent_child[parent] = list(set(children))
    return nodes, dict(parent_child)

def build_multihop(nodes, parent_child, 
                   docname=None, requirement=None, 
                   model_name=None, generation=None):
    """
    Build the final multi-hop dataset.
    In this version, we do not choose a single node as the root.
    The "parent" field is set to null.
    """
    num_nodes = len(nodes)
    num_edges = sum(len(childs) for childs in parent_child.values())
    
    result = {
        "nodes": nodes,
        "parent_child": parent_child,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "docname": docname,
        "requirement": requirement,
        "model_name": model_name,
        "generation": generation
    }
    return result

def check_unconnected_nodes(nodes, parent_child, filename=None):
    """
    Check for isolated nodes that never appear as a parent nor as a child.
    Prints a warning and returns a list of such node ids.
    """
    # All nodes from the nodes dict.
    all_nodes = set(nodes.keys())
    # Nodes that appear in relationships as a parent or child.
    connected_nodes = set(parent_child.keys())
    for children in parent_child.values():
        connected_nodes.update(children)
    # The nodes that are never connected:
    unconnected = list(all_nodes - connected_nodes)
    if unconnected:
        print(f"Warning: Unconnected nodes found in {filename}:")
        for nid in unconnected:
            print("  ", nid)
    return unconnected

def process_file(input_file, warn=True, raise_error=True, model_name="human", generation="ground_truth"):
    nodes, relationships = parse_gsn_file(input_file, warn=warn, raise_error=raise_error)
    docname = input_file.split('/')[-1]  # Get the file name from the path
    print(f"Processing file: {docname}")
    nodes, parent_child = infer_parent_child(nodes, relationships)
    check_unconnected_nodes(nodes, parent_child, filename=input_file)
    # remove the unconnected nodes from the nodes dict
    # nodes = {nid: nodes[nid] for nid in nodes if nid in parent_child
    #         or any(nid in children for children in parent_child.values())}
    result = build_multihop(nodes, parent_child, docname=docname, requirement="Safety", 
                            model_name=model_name, generation=generation)

    return result

def main():
    # use input_directory and output_directory from command line arguments
    parser = argparse.ArgumentParser(description="Convert GSN text file to multi-hop JSON dataset.")
    # add input argument, if file process only one, if directory process all files
    parser.add_argument("--input", type=str, required=True, 
                        help="Path to the input GSN text file or directory containing GSN files")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save the output multi-hop JSON dataset")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--error_handling", type=str, choices=["warn", "raise_error", "none"], default="warn",
                        help="Set error handling mode: 'warn' to show warnings, 'raise_error' to raise errors, 'none' to ignore issues")
    parser.add_argument("--extract_model_generation", action="store_true",
                        help="Extract model_name and generation_num from the filename")
    parser.set_defaults(extract_model_generation=False)
    args = parser.parse_args()

    results = []

    input_path = args.input
    output_path = args.output
    
    if input_path.endswith('.txt'):
        # Process a single file
        result = process_file(input_path)
        results.append(result)
    else:
        # Process all .txt files in the directory
        import os
        if not os.path.isdir(input_path):
            print(f"Error: {input_path} is not a directory.")
            sys.exit(1)
        
        for filename in os.listdir(input_path):
            if filename.endswith('.txt'):
                model_name, generation = "human", "ground_truth"
                if args.extract_model_generation:
                    model_name, generation = extract_model_and_generation(filename)
                    if model_name is None or generation is None:
                        print(f"Warning: Could not extract model and generation from {filename}. Using defaults.")
                        model_name, generation = None, None
                input_file = os.path.join(input_path, filename)
                result = process_file(input_file, warn=args.error_handling == "warn",
                                      raise_error=args.error_handling == "raise_error",
                                      model_name=model_name, 
                                      generation=generation)
                results.append(result)

    # Save all results to a single output file
    print(len(results), "results to save")
    with open(output_path, 'w') as out:
        json.dump(results, out, indent=4)
    print(f"Multi-hop dataset written to {output_path}")

if __name__ == "__main__":
    main()