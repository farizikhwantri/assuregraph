import json
import os
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set
from sklearn.model_selection import StratifiedKFold, train_test_split
import argparse
import logging
from datetime import datetime
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetSplitter:
    """
    Comprehensive dataset splitter for assurance case collections.
    Groups by docname first, then performs various splitting strategies.
    """
    
    def __init__(self, data_dir: str, output_dir: str = "splits", seed: int = 42, overwrite: bool = False):
        """
        Initialize the dataset splitter.
        
        Args:
            data_dir: Path to the ac_collection directory
            output_dir: Directory to save split configurations
            seed: Random seed for reproducibility
            overwrite: Whether to overwrite existing splits
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.overwrite = overwrite
        self.output_dir.mkdir(exist_ok=True)
        
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        
        # Manual mappings for docname normalization (reverse key-value)
        self.ac_gsn_mapping = {
            'ACAS XU': 'acas_xu',
            'BLUEROV2': 'bluerov2',
            'DeepMind': 'deepmind',
            'Deepmind': 'deepmind',
            'GPCA': 'gpca',
            'IM_Software': 'im_software'
        }
        
        self.safety_cases_mapping = {
            'ML': 'ml',
            'X-ray': 'x-ray',
            'X-Ray': 'x-ray'
        }
        
        # Discover files and group by docname
        self.files = self._discover_files()
        self.docname_groups = self._group_by_docname()
        
        # Define cross-experiment test allocation schema
        self.test_allocation_schema = self._create_cross_experiment_allocation_schema()
        
        logger.info(f"Discovered {len(self.files['human']) + len(self.files['llm'])} files")
        logger.info(f"Grouped into {len(self.docname_groups)} unique docnames")
        logger.info(f"Cross-experiment allocation schema: {self.test_allocation_schema}")
    
    def _discover_files(self) -> Dict[str, List[str]]:
        """Discover all JSON files in the dataset."""
        files = {'human': [], 'llm': []}
        
        logger.info(f"Looking for files in: {self.data_dir}")
        
        # Check different possible directory structures
        structures = [
            # Structure 1: data_dir/human/*.json, data_dir/llm/*.json
            {'human': self.data_dir / 'human', 'llm': self.data_dir / 'llm'},
        ]
        
        for structure in structures:
            found_files = False
            temp_files = {'human': [], 'llm': []}
            
            for category, category_path in structure.items():
                if category_path.exists():
                    json_files = list(category_path.glob("*.json"))
                    temp_files[category] = [str(f) for f in json_files]
                    if json_files:
                        found_files = True
                        logger.info(f"Found {len(json_files)} files in {category_path}")
            
            if found_files:
                files = temp_files
                break
        
        total_files = len(files['human']) + len(files['llm'])
        if total_files == 0:
            logger.error("No JSON files found! Check directory structure.")

        return files
    
    # def _normalize_docname(self, docname: str, file_path: str) -> str:
    #     """Normalize docname based on file type and manual mappings."""
    #     # Remove file extension
    #     if '.' in docname:
    #         docname_base = docname.rsplit('.', 1)[0]
    #     else:
    #         docname_base = docname
        
    #     # Determine file type from path
    #     filename = Path(file_path).name.lower()
        
    #     if 'ac_gsn' in filename:
    #         # For ac_gsn files, apply manual mapping
    #         docname_lower = docname_base.lower()
    #         for key, value in self.ac_gsn_mapping.items():
    #             if key in docname_lower:
    #                 return value
    #         return docname_base
            
    #     elif 'safety_cases' in filename:
    #         # For safety_cases files, apply manual mapping
    #         for key, value in self.safety_cases_mapping.items():
    #             if key in docname_base:
    #                 return value
    #         return docname_base
            
    #     else:
    #         # For other files (safety_tree), just remove extension
    #         return docname_base
    
    # def _group_by_docname(self) -> Dict[str, Dict]:
    #     """
    #     Group all documents by their normalized docname.
    #     This is the core function that ensures no data leakage.
    #     """
    #     logger.info("Grouping documents by normalized docname...")
        
    #     docname_groups = defaultdict(lambda: {
    #         'files': set(),
    #         'documents': [],
    #         'source_types': set(),
    #         'requirements': set(),
    #         'original_docnames': set(),
    #         'file_types': set()
    #     })
        
    #     all_files = self.files['human'] + self.files['llm']
        
    #     for file_path in all_files:
    #         try:
    #             with open(file_path, 'r') as f:
    #                 data = json.load(f)
                
    #             # Handle different data structures
    #             if isinstance(data, list):
    #                 documents = data
    #             elif isinstance(data, dict):
    #                 documents = [data]
    #             else:
    #                 logger.warning(f"Unexpected data structure in {file_path}")
    #                 continue
                
    #             # Determine source type and file type
    #             source_type = 'human' if 'human' in str(file_path) else 'llm'
    #             filename = Path(file_path).name.lower()
                
    #             if 'ac_gsn' in filename:
    #                 file_type = 'ac_gsn'
    #             elif 'safety_cases' in filename:
    #                 file_type = 'safety_cases'
    #             elif 'safety_tree' in filename:
    #                 file_type = 'safety_tree'
    #             else:
    #                 file_type = 'other'
                
    #             for doc in documents:
    #                 if 'docname' in doc:
    #                     # Normalize docname
    #                     normalized_docname = self._normalize_docname(doc['docname'], file_path)
                        
    #                     # Add to group
    #                     group = docname_groups[normalized_docname]
    #                     group['files'].add(file_path)
    #                     group['documents'].append({
    #                         'doc': doc,
    #                         'file_path': file_path,
    #                         'source_type': source_type,
    #                         'file_type': file_type
    #                     })
    #                     group['source_types'].add(source_type)
    #                     group['file_types'].add(file_type)
    #                     group['original_docnames'].add(doc['docname'])
                        
    #                     if 'requirement' in doc:
    #                         group['requirements'].add(doc['requirement'])
                            
    #         except Exception as e:
    #             logger.warning(f"Could not process {file_path}: {e}")
        
    #     # Convert defaultdict to regular dict and convert sets to lists for JSON serialization
    #     result = {}
    #     for docname, group in docname_groups.items():
    #         result[docname] = {
    #             'files': list(group['files']),
    #             'documents': group['documents'],
    #             'source_types': list(group['source_types']),
    #             'requirements': list(group['requirements']),
    #             'original_docnames': list(group['original_docnames']),
    #             'file_types': list(group['file_types']),
    #             'document_count': len(group['documents']),
    #             'file_count': len(group['files'])
    #         }
        
    #     logger.info(f"Grouped {sum(len(g['documents']) for g in result.values())} documents into {len(result)} docname groups")
        
    #     # Log group statistics
    #     mixed_groups = sum(1 for g in result.values() if len(g['source_types']) > 1)
    #     human_only = sum(1 for g in result.values() if g['source_types'] == ['human'])
    #     llm_only = sum(1 for g in result.values() if g['source_types'] == ['llm'])
        
    #     logger.info(f"Docname groups: {human_only} human-only, {llm_only} llm-only, {mixed_groups} mixed")
        
    #     return result
    
    def _categorize_docnames_by_source(self) -> Tuple[List[str], List[str], List[str]]:
        """Categorize docnames by source type availability."""
        human_only_docnames = []
        llm_only_docnames = []
        mixed_docnames = []
        
        for docname, group in self.docname_groups.items():
            source_types = set(group['source_types'])
            if source_types == {'human'}:
                human_only_docnames.append(docname)
            elif source_types == {'llm'}:
                llm_only_docnames.append(docname)
            else:
                mixed_docnames.append(docname)
        
        return human_only_docnames, llm_only_docnames, mixed_docnames
    
    def _save_split_config(self, split_config: Dict, experiment_name: str):
        """Save split configuration to JSON file."""
        output_file = self.output_dir / f"{experiment_name}_split.json"
        
        # Handle overwrite behavior
        if output_file.exists() and not self.overwrite:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.output_dir / f"{experiment_name}_split_backup_{timestamp}.json"
            output_file.rename(backup_file)
            logger.info(f"Backed up existing split to {backup_file}")
        
        # Add metadata
        split_config['metadata'] = {
            'created_at': datetime.now().isoformat(),
            'seed': self.seed,
            'total_files': len(self.files['human']) + len(self.files['llm']),
            'total_docnames': len(self.docname_groups),
        }
        
        with open(output_file, 'w') as f:
            json.dump(split_config, f, indent=2)
        
        logger.info(f"Saved split configuration to {output_file}")
        return output_file
    
    # def validate_split_integrity(self, split_config: Dict) -> Dict:
    #     """
    #     Validate that no docnames are split across different splits.
    #     """
    #     logger.info("Validating split integrity...")
        
    #     def get_docnames_from_documents(doc_list):
    #         """Extract docnames from document list."""
    #         docnames = set()
    #         for doc in doc_list:
    #             if isinstance(doc, dict) and 'docname' in doc:
    #                 docnames.add(doc['docname'])
    #         return docnames
        
    #     # Handle task-based split with nested structure
    #     # Handle source-based split with separate human/llm structure
    #     if (split_config.get('experiment_type') == 'source_based_document_aware' and 
    #         'human' in split_config and 'llm' in split_config):
            
    #         all_overlaps = set()
    #         source_validation = {}
            
    #         # Validate each source type individually
    #         for source_type in ['human', 'llm']:
    #             if source_type in split_config:
    #                 source_data = split_config[source_type]
                    
    #                 train_docnames = get_docnames_from_documents(source_data.get('train', []))
    #                 val_docnames = get_docnames_from_documents(source_data.get('val', []))
    #                 test_docnames = get_docnames_from_documents(source_data.get('test', []))
                    
    #                 # Check for overlaps within this source type
    #                 train_val_overlap = train_docnames & val_docnames
    #                 train_test_overlap = train_docnames & test_docnames
    #                 val_test_overlap = val_docnames & test_docnames
                    
    #                 source_overlaps = train_val_overlap | train_test_overlap | val_test_overlap
    #                 all_overlaps.update(source_overlaps)
                    
    #                 source_validation[source_type] = {
    #                     'is_valid': len(source_overlaps) == 0,
    #                     'train_docnames': len(train_docnames),
    #                     'val_docnames': len(val_docnames),
    #                     'test_docnames': len(test_docnames),
    #                     'overlaps': {
    #                         'train_val': list(train_val_overlap),
    #                         'train_test': list(train_test_overlap),
    #                         'val_test': list(val_test_overlap)
    #                     }
    #                 }
            
    #         # Check if test sets are aligned (same docnames)
    #         test_alignment_valid = True
    #         test_alignment_info = {}
            
    #         if 'human' in split_config and 'llm' in split_config:
    #             human_test_docnames = get_docnames_from_documents(split_config['human'].get('test', []))
    #             llm_test_docnames = get_docnames_from_documents(split_config['llm'].get('test', []))
                
    #             # test_alignment_valid = human_test_docnames == llm_test_docnames
    #             # the valid test should be all llm test docnames are in human test docnames
    #             test_alignment_valid = llm_test_docnames.issubset(human_test_docnames)
    #             test_alignment_info = {
    #                 'human_test_docnames': list(human_test_docnames),
    #                 'llm_test_docnames': list(llm_test_docnames),
    #                 'shared_test_docnames': list(human_test_docnames & llm_test_docnames),
    #                 'human_only_test': list(human_test_docnames - llm_test_docnames),
    #                 'llm_only_test': list(llm_test_docnames - human_test_docnames)
    #             }

    #             # check that train and val docnames are not in test docnames
    #             train_val_docnames = get_docnames_from_documents(
    #                 split_config['human'].get('train', []) + split_config['human'].get('val', [])
    #             )
    #             if not train_val_docnames.isdisjoint(human_test_docnames):
    #                 test_alignment_valid = False
    #                 logger.warning("Train/val docnames found in human test docnames, alignment invalid")
    #                 test_alignment_info['train_val_overlap'] = list(train_val_docnames & human_test_docnames)   

    #         validation_report = {
    #             'is_valid': len(all_overlaps) == 0 and test_alignment_valid,
    #             'source_validation': source_validation,
    #             'test_alignment': {
    #                 'is_aligned': test_alignment_valid,
    #                 'details': test_alignment_info
    #             },
    #             'all_overlapping_docnames': list(all_overlaps),
    #             'summary': {
    #                 'total_sources': len(source_validation),
    #                 'valid_sources': sum(1 for sv in source_validation.values() if sv['is_valid']),
    #                 'test_sets_aligned': test_alignment_valid
    #             }
    #         }

    #     elif split_config.get('experiment_type') == 'task_based_document_aware':
    #         all_overlaps = set()
    #         task_validation = {}
            
    #         # Validate each task type individually
    #         for task_name in ['ac_safety_cases', 'safety_tree']:
    #             if task_name in split_config:
    #                 task_data = split_config[task_name]
                    
    #                 train_docnames = get_docnames_from_documents(task_data.get('train', []))
    #                 val_docnames = get_docnames_from_documents(task_data.get('val', []))
    #                 test_docnames = get_docnames_from_documents(task_data.get('test', []))
                    
    #                 # Check for overlaps within this task
    #                 train_val_overlap = train_docnames & val_docnames
    #                 train_test_overlap = train_docnames & test_docnames
    #                 val_test_overlap = val_docnames & test_docnames
                    
    #                 task_overlaps = train_val_overlap | train_test_overlap | val_test_overlap
    #                 all_overlaps.update(task_overlaps)
                    
    #                 task_validation[task_name] = {
    #                     'is_valid': len(task_overlaps) == 0,
    #                     'train_docnames': len(train_docnames),
    #                     'val_docnames': len(val_docnames),
    #                     'test_docnames': len(test_docnames),
    #                     'overlaps': {
    #                         'train_val': list(train_val_overlap),
    #                         'train_test': list(train_test_overlap),
    #                         'val_test': list(val_test_overlap)
    #                     }
    #                 }
            
    #         # Check for overlaps between different tasks (this should not happen)
    #         cross_task_overlaps = set()
    #         all_task_docnames = {}
            
    #         for task_name in ['ac_safety_cases', 'safety_tree']:
    #             if task_name in split_config:
    #                 task_data = split_config[task_name]
    #                 task_all_docnames = set()
    #                 for split_type in ['train', 'val', 'test']:
    #                     task_all_docnames.update(get_docnames_from_documents(task_data.get(split_type, [])))
    #                 all_task_docnames[task_name] = task_all_docnames
            
    #         # Check cross-task overlaps
    #         task_names = list(all_task_docnames.keys())
    #         for i in range(len(task_names)):
    #             for j in range(i + 1, len(task_names)):
    #                 task1, task2 = task_names[i], task_names[j]
    #                 overlap = all_task_docnames[task1] & all_task_docnames[task2]
    #                 if overlap:
    #                     cross_task_overlaps.update(overlap)
            
    #         validation_report = {
    #             'is_valid': len(all_overlaps) == 0 and len(cross_task_overlaps) == 0,
    #             'task_validation': task_validation,
    #             'cross_task_overlaps': list(cross_task_overlaps),
    #             'all_overlapping_docnames': list(all_overlaps | cross_task_overlaps),
    #             'summary': {
    #                 'total_tasks': len(task_validation),
    #                 'valid_tasks': sum(1 for tv in task_validation.values() if tv['is_valid']),
    #                 'total_cross_task_overlaps': len(cross_task_overlaps)
    #             }
    #         }
    #     else:
    #         # Handle traditional train/val/test splits (existing code)
    #         # ... (keep existing validation logic for other split types)
    #         # Traditional train/val/test split
    #         if 'train' in split_config and isinstance(split_config['train'], list):
    #             # Document-based split
    #             train_docnames = get_docnames_from_documents(split_config.get('train', []))
    #             val_docnames = get_docnames_from_documents(split_config.get('val', []))
    #             test_docnames = get_docnames_from_documents(split_config.get('test', []))
    #         else:
    #             train_docnames = set()
    #             val_docnames = set()
    #             test_docnames = set()
            
    #         # Check for overlaps
    #         train_val_overlap = train_docnames & val_docnames
    #         train_test_overlap = train_docnames & test_docnames
    #         val_test_overlap = val_docnames & test_docnames
            
    #         all_overlaps = train_val_overlap | train_test_overlap | val_test_overlap
            
    #         validation_report = {
    #             'is_valid': len(all_overlaps) == 0,
    #             'total_docnames': len(train_docnames | val_docnames | test_docnames),
    #             'split_sizes': {
    #                 'train_docnames': len(train_docnames),
    #                 'val_docnames': len(val_docnames),
    #                 'test_docnames': len(test_docnames)
    #             },
    #             'overlaps': {
    #                 'train_val': list(train_val_overlap),
    #                 'train_test': list(train_test_overlap),
    #                 'val_test': list(val_test_overlap)
    #             },
    #             'all_overlapping_docnames': list(all_overlaps),
    #             'docname_distribution': {
    #                 'train': list(train_docnames),
    #                 'val': list(val_docnames),
    #                 'test': list(test_docnames)
    #             }
    #         }

    #     if validation_report['is_valid']:
    #         logger.info("✅ Split validation passed - no docname overlaps found")
    #         if 'test_alignment' in validation_report and validation_report['test_alignment']['is_aligned']:
    #             logger.info("✅ Test sets are properly aligned between human and LLM")
    #     else:
    #         logger.error(f"❌ Split validation failed")
    #         if validation_report.get('all_overlapping_docnames'):
    #             for overlap in validation_report['all_overlapping_docnames']:
    #                 logger.error(f"  Overlapping docname: {overlap}")
    #         if 'test_alignment' in validation_report and not validation_report['test_alignment']['is_aligned']:
    #             logger.error("❌ Test sets are not aligned between human and LLM")
        
    #     return validation_report

    def _get_documents_for_docnames(self, docnames: List[str]) -> List[Dict]:
        """Get specific documents (not just files) for given docnames."""
        documents = []
        for docname in docnames:
            if docname in self.docname_groups:
                group = self.docname_groups[docname]
                for doc_info in group['documents']:
                    documents.append({
                        'docname': docname,
                        'file_path': doc_info['file_path'],
                        'source_type': doc_info['source_type'],
                        'model_name': doc_info['model_name'],
                        # get the original document content if available
                        'original_docname': doc_info['doc'].get('docname', ''),
                        # 'document': doc_info['doc']  # The actual document content
                    })
        return documents

    def _create_cross_experiment_allocation_schema(self) -> Dict:
        """
        Create cross-experiment allocation where:
        1. Each experiment has EXCLUSIVE test docnames (no overlap)
        2. Other experiments can use those test docnames for validation/testing ONLY
        3. NEVER use other experiments' test docnames for training
        """
        logger.info("Creating cross-experiment allocation schema...")
        
        # Categorize docnames by task type and source availability
        ac_safety_cases_docnames = []
        safety_tree_docnames = []
        
        for docname, group in self.docname_groups.items():
            file_types = set(group['file_types'])
            
            if 'ac_gsn' in file_types or 'safety_cases' in file_types:
                ac_safety_cases_docnames.append(docname)
            elif 'safety_tree' in file_types:
                safety_tree_docnames.append(docname)
        
        # Categorize by source availability
        human_only, llm_only, mixed = self._categorize_docnames_by_source()
        
        # # Define EXCLUSIVE test allocations (no overlap between experiments)
        # source_based_test_docnames = ['ACAS_XU', 'ML']  # Must be mixed AC/Safety Cases
        # task_based_test_docnames = ['UAV-906_SafetyTree', 'UAV-945_SafetyTree', 'UAV-1006_SafetyTree']  # Safety Tree
        # define the exclusive test docnames based on the categorized docnames
        # pick from [doc for doc in ac_safety_cases_docnames if doc in mixed
        source_based_test_docnames = random.sample(
            [doc for doc in ac_safety_cases_docnames if doc in mixed], 
            min(2, len(ac_safety_cases_docnames)*30//100)  # At least 2 mixed AC/Safety Cases
        )
        # task_based_test_docnames = [doc for doc in safety_tree_docnames if doc in human_only]
        task_based_test_docnames = random.sample(
            [doc for doc in safety_tree_docnames if doc in human_only], 
            min(3, len(safety_tree_docnames)*30//100)  # At least 3 human-only Safety Tree
        )
        
        # Document-aware gets BOTH (can test on everything)
        document_aware_test_docnames = source_based_test_docnames + task_based_test_docnames
        
        schema = {
            'allocation_strategy': 'non_overlapping_cross_experiment',
            'experiment_allocations': {
                'source_based_document_aware': {
                    'exclusive_test_docnames': source_based_test_docnames,
                    'can_use_for_validation': task_based_test_docnames,  # Can use task test docnames for validation ONLY
                    'description': 'Mixed AC/Safety Cases for Human vs LLM comparison'
                },
                'task_based_document_aware': {
                    'exclusive_test_docnames': task_based_test_docnames,
                    'can_use_for_validation': source_based_test_docnames,  # Can use source test docnames for validation ONLY
                    'description': 'Safety Tree docnames for task-specific evaluation'  
                },
                'document_aware': {
                    'exclusive_test_docnames': document_aware_test_docnames,  # Can test on all
                    'can_use_for_validation': [],  # Uses remaining docnames for validation
                    'description': 'General pool - can test on all docnames from other experiments'
                }
            },
            'validation_rules': {
                'exclusive_test_sets': True,
                'cross_experiment_validation_only': True,  # Can use others' test for validation, NOT training
                'no_cross_experiment_training': True  # NEVER use others' test docnames for training
            }
        }
        
        return schema
    
    def document_aware_split(self, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Dict:
        """Document-aware split with cross-experiment allocation."""
        logger.info("Creating document-aware split with cross-experiment allocation")
        
        human_only, llm_only, mixed = self._categorize_docnames_by_source()
        
        # Get allocation from schema
        allocation = self.test_allocation_schema['experiment_allocations']['document_aware']
        exclusive_test_docnames = allocation['exclusive_test_docnames']
        can_use_for_validation = allocation['can_use_for_validation']

        logger.info(f"Document-aware exclusive test docnames: {exclusive_test_docnames}")
        logger.info(f"Can use for validation (from task-based): {can_use_for_validation }")
        # Get all docnames available for training/validation
        all_docnames = set(human_only + llm_only + mixed)
        # Remove exclusive test docnames from train/val pool
        available_for_trainval = [d for d in all_docnames if d not in exclusive_test_docnames]
        # Split available docnames between train and validation
        random.shuffle(available_for_trainval)
        n_trainval = len(available_for_trainval)
        train_end = int(n_trainval * train_ratio / (train_ratio + val_ratio))
        train_docnames = available_for_trainval[:train_end]
        base_val_docnames = available_for_trainval[train_end:]
        # Add cross-experiment docnames to validation ONLY (not training)
        val_docnames = base_val_docnames + [d for d in can_use_for_validation if d in all_docnames]
        test_docnames = exclusive_test_docnames 
        logger.info(f"Train docnames: {train_docnames}")
        logger.info(f"Val docnames (including cross-validation): {val_docnames}")
        logger.info(f"Test docnames (exclusive): {test_docnames}")  

        split_config = {
            'experiment_type': 'document_aware',
            'description': 'Document-aware split with cross-experiment allocation',
            
            'train': self._get_documents_for_docnames(train_docnames),
            'val': self._get_documents_for_docnames(val_docnames),
            'test': self._get_documents_for_docnames(test_docnames),

            'docnames': {
                'train': train_docnames,
                'val': val_docnames,
                'test': test_docnames
            },
            'counts': {
                'train_docs': len(train_docnames),
                'val_docs': len(val_docnames),
                'test_docs': len(test_docnames)
            },
            'objective': 'Compare human vs LLM on all available AC/Safety Cases',
            'cross_experiment_allocation': {
                'exclusive_test_docnames': exclusive_test_docnames,
                'validation_from_other_experiments': [d for d in val_docnames if d in can_use_for_validation]
            }   
        }

        # Save the split configuration
        output_file = self._save_split_config(split_config, 'document_aware')   
        
        return split_config

    def source_based_document_aware_split(self, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Dict:
        """Source-based split with exclusive test allocation and cross-experiment validation ONLY."""
        logger.info("Creating source-based document-aware split with cross-experiment allocation")
        
        human_only, llm_only, mixed = self._categorize_docnames_by_source()
        
        # Get allocation from schema
        allocation = self.test_allocation_schema['experiment_allocations']['source_based_document_aware']
        exclusive_test_docnames = allocation['exclusive_test_docnames']
        can_use_for_validation = allocation['can_use_for_validation']  # Other experiments' test docnames
        
        logger.info(f"Source-based exclusive test docnames: {exclusive_test_docnames}")
        logger.info(f"Can use for validation (from task-based): {can_use_for_validation}")
        
        # Get all docnames available for training/validation
        all_docnames = set(human_only + llm_only + mixed)
        
        # ONLY remove our exclusive test docnames from train/val pool
        # NEVER remove other experiments' test docnames from training pool
        available_for_trainval = [d for d in all_docnames if d not in exclusive_test_docnames]
        
        # Split available docnames between train and validation
        random.shuffle(available_for_trainval)
        n_trainval = len(available_for_trainval)
        train_end = int(n_trainval * train_ratio / (train_ratio + val_ratio))
        
        train_docnames = available_for_trainval[:train_end]
        base_val_docnames = available_for_trainval[train_end:]
        
        # Add cross-experiment docnames to validation ONLY (not training)
        val_docnames = base_val_docnames + [d for d in can_use_for_validation if d in all_docnames]
        test_docnames = exclusive_test_docnames
        
        logger.info(f"Train docnames: {train_docnames}")
        logger.info(f"Val docnames (including cross-validation): {val_docnames}")
        logger.info(f"Test docnames (exclusive): {test_docnames}")
        
        # Helper function to get documents by source type
        def get_documents_by_source(docnames, source_type):
            documents = []
            for docname in docnames:
                if docname in self.docname_groups:
                    group = self.docname_groups[docname]
                    for doc_info in group['documents']:
                        if doc_info['source_type'] == source_type:
                            doc_entry = {
                                'docname': docname,
                                'file_path': doc_info['file_path'],
                                'source_type': doc_info['source_type'],
                                'model_name': doc_info['model_name'],
                                'original_docname': doc_info['doc'].get('docname', '')
                            }
                            documents.append(doc_entry)
            return documents
        
        # Create splits for human data
        train_human = get_documents_by_source(train_docnames, 'human')
        val_human = get_documents_by_source(val_docnames, 'human')
        test_human = get_documents_by_source(test_docnames, 'human')
        
        # Create splits for LLM data
        train_llm = get_documents_by_source(train_docnames, 'llm')
        val_llm = get_documents_by_source(val_docnames, 'llm')
        test_llm = get_documents_by_source(test_docnames, 'llm')

        def filter_docnames_by_source(docnames, source_type):
            result = []
            for docname in docnames:
                if docname in self.docname_groups:
                    group = self.docname_groups[docname]
                    if any(doc['source_type'] == source_type for doc in group['documents']):
                        result.append(docname)
            return result
        
        split_config = {
            'experiment_type': 'source_based_document_aware',
            'description': 'Source-based split with exclusive test allocation and cross-experiment validation ONLY',
            
            'human': {
                'train': train_human,
                'val': val_human,
                'test': test_human,
                'docnames': {
                    'train': filter_docnames_by_source(train_docnames, 'human'),
                    'val': filter_docnames_by_source(val_docnames, 'human'),
                    'test': filter_docnames_by_source(test_docnames, 'human')
                },
                'counts': {
                    'train_docs': len(train_human),
                    'val_docs': len(val_human),
                    'test_docs': len(test_human)
                }
            },
            
            'llm': {
                'train': train_llm,
                'val': val_llm,
                'test': test_llm,
                'docnames': {
                    'train': filter_docnames_by_source(train_docnames, 'llm'),
                    'val': filter_docnames_by_source(val_docnames, 'llm'),
                    'test': filter_docnames_by_source(test_docnames, 'llm')
                },
                'counts': {
                    'train_docs': len(train_llm),
                    'val_docs': len(val_llm),
                    'test_docs': len(test_llm)
                }
            },
            
            'objective': 'Compare human vs LLM on exclusive AC/Safety Cases with cross-experiment validation',
            'cross_experiment_allocation': {
                'exclusive_test_docnames': exclusive_test_docnames,
                'validation_from_other_experiments': [d for d in val_docnames if d in can_use_for_validation],
                'own_validation_docnames': [d for d in val_docnames if d not in can_use_for_validation],
                'training_docnames': train_docnames  # NEVER includes other experiments' test docnames
            },
            'split_ratios': {
                'train_ratio': train_ratio,
                'val_ratio': val_ratio,
                'test_ratio': f"Exclusive test allocation: {len(test_docnames)} docnames"
            },
            'split_details': {
                'train_docnames': train_docnames,
                'val_docnames': val_docnames,
                'test_docnames': test_docnames,
                'all_docnames': list(all_docnames),
                'human_only': human_only,
                'llm_only': llm_only,
                'mixed': mixed
            }
        }

        self._save_split_config(split_config, 'source_based_document_aware')
        return split_config
    
    def task_based_document_aware_split(self, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Dict:
        """Task-based split with exclusive test allocation and cross-experiment validation ONLY."""
        logger.info("Creating task-based document-aware split with cross-experiment allocation")
        
        # Get allocation from schema
        allocation = self.test_allocation_schema['experiment_allocations']['task_based_document_aware']
        exclusive_test_docnames = allocation['exclusive_test_docnames']
        can_use_for_validation = allocation['can_use_for_validation']  # Other experiments' test docnames
        
        logger.info(f"Task-based exclusive test docnames: {exclusive_test_docnames}")
        logger.info(f"Can use for validation (from source-based): {can_use_for_validation}")
        
        # Categorize docnames by file types
        ac_safety_cases_docnames = []
        safety_tree_docnames = []
        
        for docname, group in self.docname_groups.items():
            file_types = set(group['file_types'])
            
            if 'ac_gsn' in file_types or 'safety_cases' in file_types:
                ac_safety_cases_docnames.append(docname)
            elif 'safety_tree' in file_types:
                safety_tree_docnames.append(docname)
        
        # CRITICAL FIX: Remove BOTH exclusive test docnames AND cross-validation docnames from train/val pools
        # Cross-validation docnames can ONLY be used for validation, NOT for training
        all_reserved_docnames = set(exclusive_test_docnames + can_use_for_validation)
        
        ac_safety_cases_trainval = [d for d in ac_safety_cases_docnames if d not in all_reserved_docnames]
        safety_tree_trainval = [d for d in safety_tree_docnames if d not in all_reserved_docnames]
        
        logger.info(f"Reserved docnames (test + cross-val): {all_reserved_docnames}")
        logger.info(f"AC/Safety Cases trainval pool (after removing reserved): {ac_safety_cases_trainval}")
        logger.info(f"Safety Tree trainval pool (after removing reserved): {safety_tree_trainval}")
        
        # Determine which task type the exclusive test docnames belong to
        ac_safety_cases_test = [d for d in exclusive_test_docnames if d in ac_safety_cases_docnames]
        safety_tree_test = [d for d in exclusive_test_docnames if d in safety_tree_docnames]
        
        # Split train/val for each task type
        def split_docnames_clean(docnames, cross_val_docnames, task_type, train_ratio, val_ratio):
            if not docnames:
                # If no docnames available for training, only use cross-validation for validation
                cross_val_for_task = []
                if task_type == 'ac_safety_cases':
                    cross_val_for_task = [d for d in cross_val_docnames if d in ac_safety_cases_docnames]
                else:
                    cross_val_for_task = [d for d in cross_val_docnames if d in safety_tree_docnames]
                
                return [], cross_val_for_task
            
            random.shuffle(docnames)
            n = len(docnames)
            train_end = int(n * train_ratio / (train_ratio + val_ratio))
            
            train_docnames = docnames[:train_end]
            base_val_docnames = docnames[train_end:]
            
            # Add cross-experiment docnames to validation ONLY (for the appropriate task type)
            if task_type == 'ac_safety_cases':
                cross_val_for_task = [d for d in cross_val_docnames if d in ac_safety_cases_docnames]
            else:
                cross_val_for_task = [d for d in cross_val_docnames if d in safety_tree_docnames]
            
            val_docnames = base_val_docnames + cross_val_for_task
            
            return train_docnames, val_docnames
        
        # Split each task type's docnames
        ac_safety_cases_train, ac_safety_cases_val = split_docnames_clean(
            ac_safety_cases_trainval, can_use_for_validation, 'ac_safety_cases', train_ratio, val_ratio
        )
        safety_tree_train, safety_tree_val = split_docnames_clean(
            safety_tree_trainval, can_use_for_validation, 'safety_tree', train_ratio, val_ratio
        )
        
        # VALIDATION CHECK: Ensure no overlap between train and val within each task
        ac_train_val_overlap = set(ac_safety_cases_train) & set(ac_safety_cases_val)
        safety_train_val_overlap = set(safety_tree_train) & set(safety_tree_val)
        
        # CRITICAL VALIDATION: Ensure no cross-contamination with reserved docnames
        ac_train_reserved_overlap = set(ac_safety_cases_train) & all_reserved_docnames
        safety_train_reserved_overlap = set(safety_tree_train) & all_reserved_docnames
        
        if ac_train_val_overlap:
            logger.error(f"❌ AC/Safety Cases train-val overlap detected: {ac_train_val_overlap}")
            raise ValueError(f"Document-aware integrity violation in AC/Safety Cases: {ac_train_val_overlap}")
        
        if safety_train_val_overlap:
            logger.error(f"❌ Safety Tree train-val overlap detected: {safety_train_val_overlap}")
            raise ValueError(f"Document-aware integrity violation in Safety Tree: {safety_train_val_overlap}")
        
        if ac_train_reserved_overlap:
            logger.error(f"❌ AC/Safety Cases training contains reserved docnames: {ac_train_reserved_overlap}")
            raise ValueError(f"Cross-experiment contamination in AC/Safety Cases: {ac_train_reserved_overlap}")
        
        if safety_train_reserved_overlap:
            logger.error(f"❌ Safety Tree training contains reserved docnames: {safety_train_reserved_overlap}")
            raise ValueError(f"Cross-experiment contamination in Safety Tree: {safety_train_reserved_overlap}")
        
        logger.info("✅ No train-val overlaps detected within tasks")
        logger.info("✅ No cross-experiment contamination detected")
        
        split_config = {
            'experiment_type': 'task_based_document_aware',
            'description': 'Task-based organization with exclusive test allocation and cross-experiment validation ONLY',
            
            'ac_safety_cases': {
                'train': self._get_documents_for_docnames(ac_safety_cases_train),
                'val': self._get_documents_for_docnames(ac_safety_cases_val),
                'test': self._get_documents_for_docnames(ac_safety_cases_test),
                'docnames': {
                    'train': ac_safety_cases_train,
                    'val': ac_safety_cases_val,
                    'test': ac_safety_cases_test
                },
                'cross_experiment_validation': {
                    'validation_from_other_experiments': [d for d in ac_safety_cases_val if d in can_use_for_validation],
                    'own_validation_docnames': [d for d in ac_safety_cases_val if d not in can_use_for_validation],
                    'training_docnames': ac_safety_cases_train,  # NEVER includes reserved docnames
                    'reserved_docnames_excluded': list(all_reserved_docnames & set(ac_safety_cases_docnames))
                },
                'validation_integrity': {
                    'train_val_overlap': list(set(ac_safety_cases_train) & set(ac_safety_cases_val)),
                    'train_test_overlap': list(set(ac_safety_cases_train) & set(ac_safety_cases_test)),
                    'val_test_overlap': list(set(ac_safety_cases_val) & set(ac_safety_cases_test)),
                    'train_reserved_overlap': list(set(ac_safety_cases_train) & all_reserved_docnames)
                },
                'counts': {
                    'train_docs': len(self._get_documents_for_docnames(ac_safety_cases_train)),
                    'val_docs': len(self._get_documents_for_docnames(ac_safety_cases_val)),
                    'test_docs': len(self._get_documents_for_docnames(ac_safety_cases_test)),
                    'total_docnames': len(ac_safety_cases_docnames),
                    'available_for_trainval': len(ac_safety_cases_trainval),
                    'reserved_docnames': len(all_reserved_docnames & set(ac_safety_cases_docnames))
                }
            },
            
            'safety_tree': {
                'train': self._get_documents_for_docnames(safety_tree_train),
                'val': self._get_documents_for_docnames(safety_tree_val),
                'test': self._get_documents_for_docnames(safety_tree_test),
                'docnames': {
                    'train': safety_tree_train,
                    'val': safety_tree_val,
                    'test': safety_tree_test
                },
                'cross_experiment_validation': {
                    'validation_from_other_experiments': [d for d in safety_tree_val if d in can_use_for_validation],
                    'own_validation_docnames': [d for d in safety_tree_val if d not in can_use_for_validation],
                    'training_docnames': safety_tree_train,  # NEVER includes reserved docnames
                    'reserved_docnames_excluded': list(all_reserved_docnames & set(safety_tree_docnames))
                },
                'validation_integrity': {
                    'train_val_overlap': list(set(safety_tree_train) & set(safety_tree_val)),
                    'train_test_overlap': list(set(safety_tree_train) & set(safety_tree_test)),
                    'val_test_overlap': list(set(safety_tree_val) & set(safety_tree_test)),
                    'train_reserved_overlap': list(set(safety_tree_train) & all_reserved_docnames)
                },
                'counts': {
                    'train_docs': len(self._get_documents_for_docnames(safety_tree_train)),
                    'val_docs': len(self._get_documents_for_docnames(safety_tree_val)),
                    'test_docs': len(self._get_documents_for_docnames(safety_tree_test)),
                    'total_docnames': len(safety_tree_docnames),
                    'available_for_trainval': len(safety_tree_trainval),
                    'reserved_docnames': len(all_reserved_docnames & set(safety_tree_docnames))
                }
            },
            
            'objective': 'Organize by task type with exclusive test allocation and cross-experiment validation ONLY',
            'cross_experiment_allocation': {
                'exclusive_test_docnames': exclusive_test_docnames,
                'can_use_for_validation': can_use_for_validation,
                'all_reserved_docnames': list(all_reserved_docnames)
            },
            'task_summary': {
                'ac_safety_cases_docnames': len(ac_safety_cases_docnames),
                'safety_tree_docnames': len(safety_tree_docnames),
                'total_docnames': len(ac_safety_cases_docnames) + len(safety_tree_docnames)
            },
            'split_ratios': {
                'train_ratio': train_ratio,
                'val_ratio': val_ratio,
                'test_ratio': f"Exclusive test allocation: {len(exclusive_test_docnames)} docnames"
            }
        }

        self._save_split_config(split_config, 'task_based_document_aware')
        return split_config
    
    def stratified_document_aware_split(self, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Dict:
        """
        Create stratified document-aware split maintaining source distribution.
        """
        logger.info("Creating stratified document-aware split")
        
        human_only, llm_only, mixed = self._categorize_docnames_by_source()
        
        # Shuffle each category
        random.shuffle(human_only)
        random.shuffle(llm_only)
        random.shuffle(mixed)
        
        def split_list(lst, train_r, val_r):
            """Split a list proportionally."""
            n = len(lst)
            train_end = int(n * train_r)
            val_end = int(n * (train_r + val_r))
            return lst[:train_end], lst[train_end:val_end], lst[val_end:]
        
        # Split each category proportionally
        train_docnames = []
        val_docnames = []
        test_docnames = []
        
        for docname_list in [human_only, llm_only, mixed]:
            if docname_list:  # Only process non-empty lists
                train_part, val_part, test_part = split_list(docname_list, train_ratio, val_ratio)
                train_docnames.extend(train_part)
                val_docnames.extend(val_part)
                test_docnames.extend(test_part)
        
        # Get files for each split
        train_files = self._get_documents_for_docnames(train_docnames)
        val_files = self._get_documents_for_docnames(val_docnames)
        test_files = self._get_documents_for_docnames(test_docnames)
        
        split_config = {
            'experiment_type': 'stratified_document_aware',
            'description': 'Stratified document-aware split maintaining source distribution',
            'train': train_files,
            'val': val_files,
            'test': test_files,
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': 1 - train_ratio - val_ratio,
            'objective': 'Prevent data leakage while maintaining source distribution',
            'source_distribution': {
                'human_only_docnames': len(human_only),
                'llm_only_docnames': len(llm_only),
                'mixed_docnames': len(mixed)
            },
            'split_details': {
                'train_docnames': train_docnames,
                'val_docnames': val_docnames,
                'test_docnames': test_docnames
            }
        }
        
        self._save_split_config(split_config, 'stratified_document_aware')
        return split_config
    
    def create_all_experiments(self) -> Dict[str, Any]:
        """
        Create all experimental splits.
        """
        logger.info("Creating all document-aware experimental splits...")
        
        experiments = {}
        
        # 1. Basic document-aware split
        experiments['document_aware'] = self.document_aware_split()

        # 2. Source-based document-aware split
        experiments['source_based_document_aware'] = self.source_based_document_aware_split()

        # 3. Task-based document-aware split
        experiments['task_based_document_aware'] = self.task_based_document_aware_split()
        
        # Validate all splits
        # validation_results = {}
        # for exp_name, exp_config in experiments.items():
        #     validation_results[exp_name] = self.validate_split_integrity(exp_config)
        
        # Print validation summary
        # print("\n" + "="*70)
        # print("DOCUMENT-AWARE SPLIT VALIDATION SUMMARY")
        # print("="*70)
        
        # for exp_name, validation in validation_results.items():
        #     status = "✅ VALID" if validation['is_valid'] else "❌ INVALID"
        #     print(f"{exp_name:35s}: {status}")
        #     if not validation['is_valid']:
        #         print(f"  Overlapping docnames: {len(validation['all_overlapping_docnames'])}")
        #         for overlap in validation['all_overlapping_docnames']:
        #             print(f"    - {overlap}")
        #     print()
        
        # Save comprehensive experiment summary
        summary = {
            'total_experiments': len(experiments),
            'experiment_types': list(experiments.keys()),
            # 'validation_results': validation_results,
            'total_docnames': len(self.docname_groups),
            'docname_statistics': {
                'human_only': len([d for d, g in self.docname_groups.items() if g['source_types'] == ['human']]),
                'llm_only': len([d for d, g in self.docname_groups.items() if g['source_types'] == ['llm']]),
                'mixed': len([d for d, g in self.docname_groups.items() if len(g['source_types']) > 1])
            },
            'created_at': datetime.now().isoformat(),
            'seed': self.seed
        }
        
        with open(self.output_dir / 'experiment_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Created {len(experiments)} experiment configurations")
        return experiments
    
    def debug_docname_groups(self):
        """Debug function to analyze docname grouping."""
        print(f"\n{'='*70}")
        print("DOCNAME GROUPING DEBUG")
        print(f"{'='*70}")
        
        print(f"Total docname groups: {len(self.docname_groups)}")
        
        # Show sample groups
        for i, (docname, group) in enumerate(list(self.docname_groups.items())[:10]):
            print(f"\nGroup {i+1}: {docname}")
            print(f"  Source types: {group['source_types']}")
            print(f"  File types: {group['file_types']}")
            print(f"  Documents: {group['document_count']}")
            print(f"  Files: {group['file_count']}")
            print(f"  Original docnames: {group['original_docnames']}")
        
        if len(self.docname_groups) > 10:
            print(f"\n... and {len(self.docname_groups) - 10} more groups")
        
        # Source distribution
        human_only, llm_only, mixed = self._categorize_docnames_by_source()
        print(f"\nSource distribution:")
        print(f"  Human-only: {len(human_only)}")
        print(f"  LLM-only: {len(llm_only)}")
        print(f"  Mixed: {len(mixed)}")
        
        # File type distribution
        file_type_dist = defaultdict(int)
        for group in self.docname_groups.values():
            for file_type in group['file_types']:
                file_type_dist[file_type] += 1
        
        print(f"\nFile type distribution:")
        for file_type, count in file_type_dist.items():
            print(f"  {file_type}: {count}")

    def _normalize_docname(self, docname: str, file_path: str) -> str:
        """Normalize docname using simple mapping rules."""
        # Remove file extension
        if '.' in docname:
            docname_base = docname.rsplit('.', 1)[0]
        else:
            docname_base = docname
        
        # Convert to lowercase for comparison
        docname_lower = docname_base.lower()
        
        # Simple exact matching for known patterns
        # AC GSN mappings
        if 'acas' in docname_lower and 'xu' in docname_lower:
            return 'ACAS_XU'
        elif 'bluerov2' in docname_lower or 'bluerov' in docname_lower:
            return 'BLUEROV2'
        elif 'gpca' in docname_lower:
            return 'GPCA'
        elif 'im_software' in docname_lower or ('im' in docname_lower and 'software' in docname_lower):
            return 'IM_SOFTWARE'
        elif 'deepmind' in docname_lower:
            return 'DEEPMIND'
        
        # Safety cases mappings
        elif docname_lower == 'ml' or 'ml-safety' in docname_lower:
            return 'ML'
        elif 'x-ray' in docname_lower or 'xray' in docname_lower:
            return 'X-Ray'
        
        # For everything else, return the base docname
        else:
            return docname_base

    def _group_by_docname(self) -> Dict[str, Dict]:
        """
        Group all documents by their normalized docname.
        Uses model_name field to determine source type.
        """
        logger.info("Grouping documents by normalized docname...")
        
        docname_groups = defaultdict(lambda: {
            'files': set(),
            'documents': [],
            'source_types': set(),
            'requirements': set(),
            'original_docnames': set(),
            'file_types': set()
        })
        
        all_files = self.files['human'] + self.files['llm']
        
        for file_path in all_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Handle different data structures
                if isinstance(data, list):
                    documents = data
                elif isinstance(data, dict):
                    documents = [data]
                else:
                    logger.warning(f"Unexpected data structure in {file_path}")
                    continue
                
                # Determine file type from filename
                filename = Path(file_path).name.lower()
                if 'ac_gsn' in filename:
                    file_type = 'ac_gsn'
                elif 'safety_cases' in filename:
                    file_type = 'safety_cases'
                elif 'safety_tree' in filename:
                    file_type = 'safety_tree'
                else:
                    file_type = 'other'
                
                # Track which docnames this file contributes to
                file_docnames = set()
                
                for doc in documents:
                    if 'docname' in doc:
                        # Normalize docname
                        normalized_docname = self._normalize_docname(doc['docname'], file_path)
                        file_docnames.add(normalized_docname)
                        
                        # Determine source type from model_name field
                        model_name = doc.get('model_name', 'unknown')
                        if model_name == 'human':
                            source_type = 'human'
                        else:
                            source_type = 'llm'  # gpt-4o, claude, etc.
                        
                        # Add to group
                        group = docname_groups[normalized_docname]
                        group['documents'].append({
                            'doc': doc,
                            'file_path': file_path,
                            'source_type': source_type,
                            'file_type': file_type,
                            'model_name': model_name
                        })
                        group['source_types'].add(source_type)
                        group['file_types'].add(file_type)
                        group['original_docnames'].add(doc['docname'])
                        
                        if 'requirement' in doc:
                            group['requirements'].add(doc['requirement'])
                
                # Add this file to each docname group it contributes to (only once per group)
                for docname in file_docnames:
                    docname_groups[docname]['files'].add(file_path)
                            
            except Exception as e:
                logger.warning(f"Could not process {file_path}: {e}")
        
        # Convert defaultdict to regular dict and convert sets to lists for JSON serialization
        result = {}
        for docname, group in docname_groups.items():
            result[docname] = {
                'files': list(group['files']),
                'documents': group['documents'],
                'source_types': list(group['source_types']),
                'requirements': list(group['requirements']),
                'original_docnames': list(group['original_docnames']),
                'file_types': list(group['file_types']),
                'document_count': len(group['documents']),
                'file_count': len(group['files'])
            }
        
        logger.info(f"Grouped {sum(len(g['documents']) for g in result.values())} documents into {len(result)} docname groups")
        
        # Log group statistics
        mixed_groups = sum(1 for g in result.values() if len(g['source_types']) > 1)
        human_only = sum(1 for g in result.values() if g['source_types'] == ['human'])
        llm_only = sum(1 for g in result.values() if g['source_types'] == ['llm'])
        
        logger.info(f"Docname groups: {human_only} human-only, {llm_only} llm-only, {mixed_groups} mixed")
        
        return result

    def debug_model_names(self):
        """Debug function to see what model_name values we have."""
        print(f"\n{'='*70}")
        print("MODEL NAME DEBUG")
        print(f"{'='*70}")
        
        model_name_counts = defaultdict(int)
        sample_docnames = defaultdict(list)
        
        all_files = self.files['human'] + self.files['llm']
        
        for file_path in all_files[:5]:  # Sample first 5 files
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                documents = data if isinstance(data, list) else [data]
                
                for doc in documents[:3]:  # Sample first 3 docs per file
                    if 'model_name' in doc and 'docname' in doc:
                        model_name = doc['model_name']
                        docname = doc['docname']
                        
                        model_name_counts[model_name] += 1
                        
                        if len(sample_docnames[model_name]) < 3:
                            sample_docnames[model_name].append(docname)
                            
            except Exception as e:
                continue
        
        print("Model name distribution:")
        for model_name, count in model_name_counts.items():
            print(f"  {model_name}: {count}")
            print(f"    Sample docnames: {sample_docnames[model_name]}")
            print()

    def analyze_statistics(self) -> Dict:
        """
        Analyze collection statistics by docname: nodes/edges for human data and averages for LLM data.
        """
        logger.info("Analyzing collection statistics by docname...")
        
        statistics = {}
        
        for docname, group in self.docname_groups.items():
            docname_stats = {
                'docname': docname,
                'total_documents': group['document_count'],
                'file_types': group['file_types'],
                'source_types': group['source_types'],
                'human_data': {},
                'llm_data': {},
                'summary': {}
            }
            
            human_docs = []
            llm_docs = []
            
            # Separate human and LLM documents
            for doc_info in group['documents']:
                if doc_info['source_type'] == 'human':
                    human_docs.append(doc_info)
                else:
                    llm_docs.append(doc_info)
            
            # Analyze human data (detailed per document)
            if human_docs:
                human_stats = []
                for doc_info in human_docs:
                    doc = doc_info['doc']
                    stats = self._extract_document_stats(doc, doc_info['file_path'])
                    if stats:
                        stats['model_name'] = doc_info['model_name']
                        stats['file_path'] = Path(doc_info['file_path']).name
                        human_stats.append(stats)
                
                docname_stats['human_data'] = {
                    'count': len(human_stats),
                    'documents': human_stats
                }
            
            # Analyze LLM data (aggregated statistics)
            if llm_docs:
                llm_stats = []
                model_stats = defaultdict(list)
                
                for doc_info in llm_docs:
                    doc = doc_info['doc']
                    stats = self._extract_document_stats(doc, doc_info['file_path'])
                    if stats:
                        stats['model_name'] = doc_info['model_name']
                        stats['file_path'] = Path(doc_info['file_path']).name
                        llm_stats.append(stats)
                        model_stats[doc_info['model_name']].append(stats)
                
                # Calculate averages by model
                model_averages = {}
                for model_name, model_docs in model_stats.items():
                    if model_docs:
                        avg_stats = self._calculate_average_stats(model_docs)
                        avg_stats['count'] = len(model_docs)
                        model_averages[model_name] = avg_stats
                
                docname_stats['llm_data'] = {
                    'total_count': len(llm_stats),
                    'by_model': model_averages,
                    'all_documents': llm_stats  # Keep individual docs for reference
                }
            
            # Calculate summary statistics
            all_docs = human_docs + llm_docs
            if all_docs:
                all_stats = []
                for doc_info in all_docs:
                    doc = doc_info['doc']
                    stats = self._extract_document_stats(doc, doc_info['file_path'])
                    if stats:
                        all_stats.append(stats)
                
                if all_stats:
                    summary_stats = self._calculate_average_stats(all_stats)
                    summary_stats['total_documents'] = len(all_stats)
                    summary_stats['human_count'] = len(human_docs)
                    summary_stats['llm_count'] = len(llm_docs)
                    docname_stats['summary'] = summary_stats
            
            statistics[docname] = docname_stats
        
        # Save statistics
        stats_file = self.output_dir / 'collection_statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(statistics, f, indent=2)
        
        logger.info(f"Saved collection statistics to {stats_file}")
        
        # Print summary table
        self._print_statistics_table(statistics)
        
        return statistics

    def _extract_document_stats(self, doc: Dict, file_path: str) -> Dict:
        """Extract nodes and edges statistics from a document."""
        stats = {
            'nodes': 0,
            'edges': 0,
            'file_type': self._get_file_type(file_path)
        }
        
        try:
            # Just use the existing num_nodes and num_edges fields
            if 'num_nodes' in doc:
                stats['nodes'] = doc['num_nodes']
            
            if 'num_edges' in doc:
                stats['edges'] = doc['num_edges']
                
        except Exception as e:
            logger.warning(f"Could not extract stats from {file_path}: {e}")
        
        return stats

    def _get_file_type(self, file_path: str) -> str:
        """Determine file type from path."""
        filename = Path(file_path).name.lower()
        if 'ac_gsn' in filename:
            return 'ac_gsn'
        elif 'safety_cases' in filename:
            return 'safety_cases'
        elif 'safety_tree' in filename:
            return 'safety_tree'
        else:
            return 'other'

    def _calculate_average_stats(self, stats_list: List[Dict]) -> Dict:
        """Calculate average statistics from a list of document stats."""
        if not stats_list:
            return {}
        
        # Filter out documents with no nodes/edges
        valid_stats = [s for s in stats_list if s['nodes'] > 0 or s['edges'] > 0]
        
        if not valid_stats:
            return {'avg_nodes': 0, 'avg_edges': 0, 'valid_documents': 0}
        
        avg_nodes = sum(s['nodes'] for s in valid_stats) / len(valid_stats)
        avg_edges = sum(s['edges'] for s in valid_stats) / len(valid_stats)
        
        # Calculate hop distribution averages if present
        hop_avg = {}
        hop_docs = [s for s in valid_stats if s.get('hop_distribution')]
        if hop_docs:
            all_hops = set()
            for doc in hop_docs:
                all_hops.update(doc['hop_distribution'].keys())
            
            for hop in all_hops:
                hop_values = [doc['hop_distribution'].get(hop, 0) for doc in hop_docs]
                hop_avg[hop] = sum(hop_values) / len(hop_values)
        
        result = {
            'avg_nodes': round(avg_nodes, 2),
            'avg_edges': round(avg_edges, 2),
            'valid_documents': len(valid_stats),
            'total_documents': len(stats_list)
        }
        
        if hop_avg:
            result['avg_hop_distribution'] = hop_avg
        
        return result

    def _print_statistics_table(self, statistics: Dict):
        """Print a formatted table with each docname-model combination as separate rows, grouped by file type."""
        print(f"\n{'='*90}")
        print("COLLECTION STATISTICS BY DOCNAME AND MODEL")
        print(f"{'='*90}")
        
        # Group by file type based on file paths
        file_type_groups = {
            'multihop_ac_gsn': [],
            'multihop_safety_cases': [],
            'multihop_safety_tree': []
        }
        
        # Collect all docname-model combinations
        for docname, stats in statistics.items():
            # Determine file group from file paths
            file_group = 'other'
            if stats.get('documents'):
                sample_file = stats['documents'][0]['file_path']
                if 'ac_gsn' in sample_file:
                    file_group = 'multihop_ac_gsn'
                elif 'safety_cases' in sample_file:
                    file_group = 'multihop_safety_cases'
                elif 'safety_tree' in sample_file:
                    file_group = 'multihop_safety_tree'
            
            if file_group not in file_type_groups:
                file_type_groups[file_group] = []
            
            # Add human row if exists
            if stats.get('human_data') and stats['human_data']['documents']:
                human_docs = stats['human_data']['documents']
                if human_docs:
                    # Calculate average if multiple documents
                    if len(human_docs) > 1:
                        avg_nodes = sum(doc['nodes'] for doc in human_docs) / len(human_docs)
                        avg_edges = sum(doc['edges'] for doc in human_docs) / len(human_docs)
                        nodes_edges = f"{avg_nodes:.1f}/{avg_edges:.1f}"
                    else:
                        first_doc = human_docs[0]
                        nodes_edges = f"{first_doc['nodes']}/{first_doc['edges']}"
                    
                    file_type_groups[file_group].append({
                        'docname': docname,
                        'model': 'human',
                        'nodes_edges': nodes_edges,
                        'docs': len(human_docs)
                    })
            
            # Add LLM rows if exist
            if stats.get('llm_data') and stats['llm_data']['by_model']:
                for model_name, model_stats in stats['llm_data']['by_model'].items():
                    avg_nodes = model_stats.get('avg_nodes', 0)
                    avg_edges = model_stats.get('avg_edges', 0)
                    count = model_stats.get('count', 0)
                    
                    nodes_edges = f"{avg_nodes:.1f}/{avg_edges:.1f}"
                    
                    file_type_groups[file_group].append({
                        'docname': docname,
                        'model': model_name,
                        'nodes_edges': nodes_edges,
                        'docs': count
                    })
        
        # Print each group
        for group_name, rows in file_type_groups.items():
            if rows:
                print(f"\n{group_name.upper().replace('_', ' ')}:")
                print(f"{'Docname':<25} {'Model':<10} {'Avg Nodes/Edges':<18} {'Count':<8}")
                print("-" * 65)
                
                # Sort rows by docname, then by model (human first)
                def sort_key(row):
                    model_priority = 0 if row['model'] == 'human' else 1
                    return (row['docname'], model_priority, row['model'])
                
                rows.sort(key=sort_key)
                
                # Print rows
                for row in rows:
                    print(f"{row['docname']:<25} {row['model']:<10} {row['nodes_edges']:<18} {row['docs']:<8}")
        
        print(f"\n{'='*90}")
        
        # Summary statistics
        all_rows = []
        for rows in file_type_groups.values():
            all_rows.extend(rows)
        
        total_rows = len(all_rows)
        human_rows = sum(1 for row in all_rows if row['model'] == 'human')
        llm_rows = total_rows - human_rows
        # unique_docnames = len(set(row['original_docname'] for row in all_rows))
        unique_docnames = len(list(row['docname'] for row in all_rows))
        total_documents = sum(row['docs'] for row in all_rows)
        # print docs
        # for row in all_rows:
        #     print(f"  {row['docname']} ({row['model']}): {row['docs']} docs")

        print(f"SUMMARY:")
        print(f"  Total docname-model combinations: {total_rows}")
        print(f"  Human entries: {human_rows}")
        print(f"  LLM entries: {llm_rows}")
        print(f"  Unique docnames: {unique_docnames}")
        print(f"  Total documents: {total_documents}")
        
        # Group statistics
        print(f"\nBy file type:")
        for group_name, rows in file_type_groups.items():
            if rows:
                group_docs = sum(row['docs'] for row in rows)
                print(f"  {group_name}: {len(rows)} entries, {group_docs} documents")

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Document-Aware Dataset Splitter for Assurance Case Collections')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to ac_collection directory')
    parser.add_argument('--output_dir', type=str, default='splits',
                       help='Output directory for split configurations')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--experiment', type=str, 
                       choices=['document_aware', 'task_based_document_aware', 'all'],
                       default='all', help='Type of experiment to create')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing split files')
    parser.add_argument('--debug', action='store_true',
                       help='Debug docname grouping')
    parser.add_argument('--stats', action='store_true',
                       help='Analyze collection statistics')
    
    args = parser.parse_args()
    
    # Create splitter
    splitter = DatasetSplitter(args.data_dir, args.output_dir, args.seed, args.overwrite)
    
    if args.debug:
        splitter.debug_docname_groups()
        return

    if args.stats:
        splitter.analyze_statistics()
    
    # Run requested experiment
    if args.experiment == 'all':
        experiments = splitter.create_all_experiments()
    elif args.experiment == 'document_aware':
        config = splitter.document_aware_split()
        validation = splitter.validate_split_integrity(config)
        print(f"Document-aware split created: {'✅ Valid' if validation['is_valid'] else '❌ Invalid'}")

    elif args.experiment == 'source_based_document_aware':
        config = splitter.source_based_document_aware_split()
        validation = splitter.validate_split_integrity(config)
        print(f"Source-based document-aware split created: {'✅ Valid' if validation['is_valid'] else '❌ Invalid'}")
        
    elif args.experiment == 'task_based_document_aware':
        config = splitter.task_based_document_aware_split()
        validation = splitter.validate_split_integrity(config)
        print(f"Task-based document-aware split created: {'✅ Valid' if validation['is_valid'] else '❌ Invalid'}")

if __name__ == "__main__":
    main()