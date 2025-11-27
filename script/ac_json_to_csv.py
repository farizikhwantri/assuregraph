import argparse
import re
import math
import random
import json
import pandas as pd

def extract_numeric_suffix(s):
    # Use regex to match the non-numeric part and the numeric suffix
    match = re.match(r'(\D*)(\d*)$', s)
    print(f"Extracting numeric suffix from {s}: {match}")
    
    if match:
        non_numeric = match.group(1)
        numeric = match.group(2)
        
        # Convert the numeric part to an integer if it is not empty, otherwise return nan
        if numeric:
            numeric = int(numeric)
        else:
            numeric = math.nan
    
    return (non_numeric, numeric)

def dict_to_csv(data, with_trace=False):
    """flatten list of requirement dict contain list of instances hop
    to list of dict with requirement, model_name, one_hop, two_hop, three_hop
    example:
    {
        "requirement": "CR1.7",
        "model_name": "chatGPT4o",
        "generation": "1",
        "docname": "iec62443_4.2",
        "num_nodes": 19,
        "num_edges": 19,
        "parent_child": "{'MainClaim-0': ['SubClaim1-3', 'SubClaim2-35'], 
                          'SubClaim1-3': ['ArgumentClaim-6'], 
                          'ArgumentClaim-6': ['ArgumentSubClaim1-9', 'ArgumentSubClaim2-22'], 
                          'ArgumentSubClaim1-9': ['Evidence1-12', 'Evidence2-17'], 
                          'Evidence1-12': [], 'Evidence2-17': [], 
                          'ArgumentSubClaim2-22': ['Evidence1-25', 'Evidence2-30'], 
                          'Evidence1-25': [], 'Evidence2-30': [], 
                          'SubClaim2-35': ['ArgumentClaim-38'], 
                          'ArgumentClaim-38': ['ArgumentSubClaim1-41', 'ArgumentSubClaim2-64'], 
                          'ArgumentSubClaim1-41': ['Evidence1-44', 'Evidence2-49', 'Evidence3-54', 'Evidence4-59'], 
                          'Evidence1-44': [], 'Evidence2-49': [], 'Evidence3-54': [], 'Evidence4-59': [], 
                          'ArgumentSubClaim2-64': ['Evidence1-67', 'Evidence2-72'], 'Evidence1-67': [], 'Evidence2-72': []
                        }",
        "1_hop": [
            {
                "premise": "The system enforces configurable password strength in compliance with internationally recognized guidelines.",
                "hypothesis": "The system provides the ability to enforce password strength through configurable policies.",
                "meta": {
                    "current_node": "SubClaim1-3",
                    "parent_node": "MainClaim-0",
                    "hop": 1
                },
            }, ...
            ...
        ],
    }
    """
    new_data = []
    for instance in data:
        # print(instance)
        req = instance['requirement']
        model_name = instance['model_name']
        docname = instance['docname']
        # parent_child = instance['parent_child']
        # get the hop inferences by iterating the hops that contains list as value
        for key, value in instance.items():
            if isinstance(value, list) and 'hop' in key:
                for hop_instance in value:
                    new_instance = {}
                    new_instance['requirement'] = req
                    new_instance['model_name'] = model_name
                    new_instance['docname'] = docname
                    # new_instance['parent_child'] = parent_child
                    new_instance['premise'] = str(hop_instance['premise'])
                    new_instance['hypothesis'] = str(hop_instance['hypothesis'])
                    new_instance['group_docname'] = instance.get('group_docname', None)
                    # new_instance['meta'] = hop_instance['meta']
                    # unroll meta
                    for meta_key, meta_value in hop_instance['meta'].items():
                        new_instance[meta_key] = meta_value

                    # print(new_instance.keys())

                    if 'trace' in hop_instance and with_trace:
                        for i, trace_key in enumerate(hop_instance['trace']):
                            # get the trace dict from nodes
                            trace_dict = {trace_key: instance['nodes'][trace_key]["description"]}
                            for trace_key, trace_value in trace_dict.items():
                                # check if there is a . in the new_instance premise last character
                                # if new_instance['premise'][-1] == '.':
                                #     new_instance['premise'] += f" {trace_value}"
                                # else:
                                #     new_instance['premise'] += f". {trace_value}"
                                new_instance['premise'] += f"|| {trace_value}"

                    new_instance['label'] = 'entailment'
                    new_data.append(new_instance)
        
        # get the negative samples
        # for negative_instance in instance['negative_samples?']:
        #     new_instance = {}
        #     new_instance['requirement'] = req
        #     new_instance['model_name'] = model_name
        #     new_instance['docname'] = docname
        #     # new_instance['parent_child'] = parent_child
        #     new_instance['premise'] = str(negative_instance['premise'])
        #     new_instance['hypothesis'] = str(negative_instance['hypothesis'])
        #     # new_instance['meta'] = negative_instance['meta']
        #     # unroll meta
        #     for meta_key, meta_value in negative_instance['meta'].items():
        #         new_instance[meta_key] = meta_value

        #     # new_instance['label'] = 'not_entailment'
        #     new_instance['label'] = 'neutral'
        #     new_data.append(new_instance)

    # negative samples from different requirement 
    negative_data = [] 
    for instance in new_data:
        anchor_key = 'group_docname'
        instance_anchor = instance[anchor_key]
        # get the basename without number
        instance_type = instance['current_node'].split('-')[0]
        # instance_type = extract_numeric_suffix(instance_type)[0]
        # sample new_data, select 10% of the data
        new_data_sample = random.sample(new_data, math.ceil(len(new_data)*0.1))
        for negative_instance in new_data_sample:
            # pick negative samples from different docname with the same level
            negative_instance_type = negative_instance['current_node'].split('-')[0]
            # negative_instance_type = extract_numeric_suffix(negative_instance_type)[0]
            if random.random() < 0.1 and (negative_instance[anchor_key] != instance_anchor) and negative_instance_type == instance_type:
                # print(instance_type, negative_instance_type)
                new_instance = {}
                new_instance['requirement'] = instance['requirement']
                new_instance['model_name'] = instance['model_name']
                new_instance['docname'] = instance['docname']
                new_instance['premise'] = str(negative_instance['premise'])
                new_instance['hypothesis'] = str(instance['hypothesis'])
                new_instance['current_node'] = instance['current_node']
                new_instance['parent_node'] = negative_instance['current_node']
                new_instance['hop'] = -1
                new_instance['target'] = 'negative'
                # new_instance['target'] = 'negative'
                new_instance['label'] = 'not_entailment'
                negative_data.append(new_instance)
    
    new_data.extend(negative_data)

    
    print(len(new_data))
    # new_data to df csv
    new_data = pd.DataFrame(new_data)
    print("Number of instances in new_data:", len(new_data))
    # remove duplicates
    new_data = new_data.drop_duplicates()
    print("Number of instances after removing duplicates:", len(new_data))

    # fix premise and hypothesis to string format in pandas column
    new_data['premise'] = new_data['premise'].astype(str)
    new_data['hypothesis'] = new_data['hypothesis'].astype(str)
    return new_data

def main(args):
    """Main function to convert JSON data to CSV."""
    with open(args.input_file, 'r') as f:
        data = json.load(f)

    # Convert the data to CSV format
    # iterate over dict train, val and test, create a DataFrame for each
    
    for split in ['train', 'val', 'test']:
        if split in data:
            df = dict_to_csv(data[split], with_trace=args.with_trace)

            # Save the DataFrame to a CSV file
            df.to_csv(f"{args.output_file}_{split}.csv", index=False)
            print(f"Data saved to {args.output_file}_{split}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON to CSV")
    parser.add_argument("input_file", help="Path to the input JSON file")
    parser.add_argument("output_file", help="Path to the output CSV file")
    parser.add_argument("--with_trace", action="store_true", help="Include trace information")
    # set default values for with trace
    parser.set_defaults(with_trace=False)
    args = parser.parse_args()
    main(args)
