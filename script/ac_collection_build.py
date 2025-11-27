import json
import argparse
# import os
import re
import math
import random
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union


def extract_docname_from_content(content: List[Dict], docname: str) -> Union[Dict, None]:
    """
    Extract the 'docname' from a list of dictionaries.
    
    Args:
        content: List of dictionaries containing document information.
        docname: The original document name to extract.

    Returns:
        The dictionary containing the specified 'docname', or None if not found.
    """
    # print(f"Extracting docname from content... {content}")
    return next((item for item in content if item.get('docname') == docname), None)


def extract_content_from_split(split_config: Dict, experiment_type: str) -> Dict[str, List[str]]:
    """
    Extract file paths from split configuration based on experiment type.
    
    Args:
        split_config: The split configuration dictionary
        experiment_type: Type of experiment ('source_based', 'task_based', or 'document_aware')
    
    Returns:
        Dictionary with 'train', 'val', 'test' keys containing lists of file paths
    """
    
    if experiment_type == 'source_based_document_aware':
        # Source-based has separate human and llm sections
        results = {
            'human': {'train': [], 'val': [], 'test': []},
            'llm': {'train': [], 'val': [], 'test': []}
        }
        for source_type in ['human', 'llm']:
            if source_type in split_config:
                for split_name in ['train', 'val', 'test']:
                    if split_name in split_config[source_type]:
                        file_paths = [doc['file_path'] for doc in split_config[source_type][split_name]]
                        # result[split_name].extend(file_paths)
                        # open the file and read the content find the original docname
                        for i, file_path in enumerate(file_paths):
                            with open(file_path, 'r') as f:
                                # load json content
                                content = json.load(f)
                                # get the original docname
                                # iterate through the content and find the 'docname' key
                                docname = split_config[source_type][split_name][i].get('original_docname', None)
                                # print(docname)
                                doc_content = extract_docname_from_content(content, docname)
                                doc_content['group_docname'] = split_config[source_type][split_name][i].get('docname', None)
                                if doc_content:
                                    results[source_type][split_name].append(doc_content)
        return results

    elif experiment_type == 'task_based_document_aware':
        # Task-based has separate ac_safety_cases and safety_tree sections
        print(split_config.keys())
        results = {
            'ac_safety_cases': {'train': [], 'val': [], 'test': []},
            'safety_tree': {'train': [], 'val': [], 'test': []}
        }

        for task_type in ['ac_safety_cases', 'safety_tree']:
            if task_type in split_config:
                for split_name in ['train', 'val', 'test']:
                    if split_name in split_config[task_type]:
                        file_paths = [doc['file_path'] for doc in split_config[task_type][split_name]]
                        for i, file_path in enumerate(file_paths):
                            with open(file_path, 'r') as f:
                                content = json.load(f)
                                docname = split_config[task_type][split_name][i].get('original_docname', None)
                                doc_content = extract_docname_from_content(content, docname)
                                # add the group_docname
                                doc_content['group_docname'] = split_config[task_type][split_name][i].get('docname', None)
                                if doc_content:
                                    results[task_type][split_name].append(doc_content)

        return results

    
    elif experiment_type == 'document_aware':
        # Document-aware has direct train/val/test sections
        result = {
            'train': [],
            'val': [],
            'test': []
        }
        print(split_config.keys())
        for split_name in ['train', 'val', 'test']:
            if split_name in split_config:
                file_paths = [doc['file_path'] for doc in split_config[split_name]]
                for i, file_path in enumerate(file_paths):
                    with open(file_path, 'r') as f:
                        content = json.load(f)
                        docname = split_config[split_name][i].get('original_docname', None)
                        doc_content = extract_docname_from_content(content, docname)
                        doc_content['group_docname'] = split_config[split_name][i].get('docname', None)
                        if doc_content:
                            result[split_name].append(doc_content)
    
        return result
    
def save_to_json(data: Dict, output_file: str):
    """
    Save a dictionary to a JSON file.
    
    Args:
        data: Dictionary to save
        output_file: Path to the output JSON file
    """
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to {output_file}")

def process_split_configs(config_dir: str, output_dir: str = None) -> Dict[str, Dict[str, List[str]]]:
    """
    Process all split configuration files and extract file paths.
    
    Args:
        config_dir: Directory containing split configuration JSON files
        output_dir: Directory to save output JSON files (optional)
    
    Returns:
        Dictionary mapping experiment names to their train/val/test file paths
    """
    config_dir = Path(config_dir)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define expected config files and their experiment types
    config_files = {
        'source_based_document_aware_split.json': 'source_based_document_aware',
        'task_based_document_aware_split.json': 'task_based_document_aware', 
        'document_aware_split.json': 'document_aware'
    }
    
    results = {}
    
    for config_file, experiment_type in config_files.items():
        config_path = config_dir / config_file
        
        if not config_path.exists():
            print(f"Warning: {config_file} not found in {config_dir}")
            continue
        
        print(f"Processing {config_file}...")
        
        # Load the split configuration
        with open(config_path, 'r') as f:
            split_config = json.load(f)
        
            # Extract document content
            doc_content = extract_content_from_split(split_config, experiment_type)

            # print(doc_content)
        
            # Store results
            experiment_name = config_file.replace('_split.json', '')
            results[experiment_name] = doc_content
        
            # Print summary
            print(f"  {experiment_name}:")
            saved = False
            for split_name, docs in doc_content.items():
                print(f"    {split_name}: {len(docs)} files")
                # if not train test and validation keys iterate through the docs
                if isinstance(docs, dict):
                    for sub_split, sub_docs in docs.items():
                        print(f"      {sub_split}: {len(sub_docs)} files")
                    if output_dir:
                        output_file = output_dir / f"{experiment_name}_{split_name}.json"
                        save_to_json(docs, output_file)
                        # convert the docs to csv
                        saved = True

            if output_dir and not saved:
                output_file = output_dir / f"{experiment_name}.json"
                print(f"Saving {experiment_name} to {output_file}")
                save_to_json(doc_content, output_file)

        # # Save individual output file if output directory specified
        # if output_dir:
        #     output_file = output_dir / f"{experiment_name}.json"
        #     save_to_json(doc_content, output_file)
    
    return results

def main():
    """Main function to process split configurations."""

    parser = argparse.ArgumentParser(description="Process AC collection split configurations")
    parser.add_argument('--config_dir', type=str, required=True, help="Directory containing split configuration JSON files")
    parser.add_argument('--output_dir', type=str, default=None, help="Directory to save output JSON files")
    parser.add_argument('--base_dir', type=str, default=None, help="Base directory to resolve relative paths")
    args = parser.parse_args()
    
    print("=== Processing Split Configurations ===")
    
    # Process all split configurations
    results = process_split_configs(args.config_dir, args.output_dir)

    assert results, "No results found. Check your split configuration files."

    # assert all json in output_dir are valid json
    if args.output_dir:
        output_dir = Path(args.output_dir)
        for output_file in output_dir.glob("*.json"):
            try:
                with open(output_file, 'r') as f:
                    json.load(f)
                print(f"Valid JSON: {output_file}")
            except json.JSONDecodeError as e:
                print(f"Invalid JSON in {output_file}: {e}")


if __name__ == "__main__":
    results = main()

