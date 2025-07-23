import json
from typing import Dict, List, Tuple, Set
from tqdm import tqdm
import time

# ===================PRE-PROCESSING===================
def extract_all_trigger_tokens(trigger: Dict) -> Set[str]:
    tokens = set(trigger["text"].split())
    extra_spans = trigger.get("extra_trigger_spans", [])
    for extra in extra_spans:
        if isinstance(extra, str):
            tokens.update(extra.split())
        elif isinstance(extra, dict):  # nested discontiguous
            tokens.update(extract_all_trigger_tokens(extra)) 
    return tokens

def sample_data(json_data: Dict, sample_size: int = 500) -> Dict:
    """Sample first N documents from JSON data"""
    if len(json_data) <= sample_size:
        return json_data
    doc_keys = list(json_data.keys())[:sample_size]
    sampled_data = {key: json_data[key] for key in doc_keys}
    print(f"Sampled {len(sampled_data)} documents from {len(json_data)} total documents")
    return sampled_data


def parse_events(json_data: Dict) -> List[Dict]:
    """Parse events from JSON data matching your format"""
    events = []
    print("Parsing events from JSON...")
    total_docs = len(json_data)
    with tqdm(total=total_docs, desc="Processing documents", unit="doc") as pbar:
        for doc_id, doc in json_data.items():
            doc_events = 0
            for e in doc["event_mentions"]:
                tokens = extract_all_trigger_tokens(e["trigger"])
                events.append({
                    "doc_id": doc_id,  
                    "event_id": e["id"],  
                    "tokens": tokens,
                    "type": e["event_type"],
                    "subtype": e["event_subtype"],
                    "modality": e["factuality"]["modality"],
                    "polarity": e["factuality"]["polarity"]
                })
                doc_events += 1
            pbar.set_postfix(events=doc_events)
            pbar.update(1)
    print(f"Parsed {len(events)} events from {total_docs} documents")
    return events


# ===================METRICS===================
def dice_coefficient(set1: Set[str], set2: Set[str]) -> float:
    intersection = len(set1 & set2)
    return 2 * intersection / (len(set1) + len(set2)) if (set1 or set2) else 0.0

def mention_mapping(gold: List[Dict], system: List[Dict], threshold) -> Dict[int, List[Tuple[int, float]]]:
    """
    Implementation of Algorithm 1 with ID matching constraint:
    Only match events that have the same doc_id and event_id
    """
    print("Computing mention mapping with ID constraint...")
    
    # Step 1: Create index mappings for faster lookup
    system_index = {}
    for sid, s in enumerate(system):
        key = (s["doc_id"], s["event_id"])
        if key not in system_index:
            system_index[key] = []
        system_index[key].append(sid)
    print(f"System index created with {len(system_index)} unique (doc_id, event_id) pairs")
    
    # Step 2: Compute Dice scores only for matching IDs
    score_list = []
    zero_score_pairs = []
    matching_pairs = 0
    total_possible_pairs = 0
    print("Computing Dice scores for matching ID pairs...")
    with tqdm(total=len(gold), desc="Processing gold events", unit="event") as pbar:
        for gid, g in enumerate(gold):
            gold_key = (g["doc_id"], g["event_id"])
            total_possible_pairs += 1
            if gold_key in system_index:
                for sid in system_index[gold_key]:
                    s = system[sid]
                    score = dice_coefficient(g["tokens"], s["tokens"])
                    matching_pairs += 1
                    if score > 0:
                        score_list.append((gid, sid, score))
                    else:
                        zero_score_pairs.append((gid, sid, g["tokens"], s["tokens"]))
            pbar.update(1)
    
    print(f"Found {matching_pairs} matching ID pairs from {total_possible_pairs} gold events")
    print(f"Found {len(score_list)} pairs with Dice score > 0")
    print(f"Found {len(zero_score_pairs)} pairs with Dice score = 0")
    
    # Show some examples of zero score pairs
    if zero_score_pairs:
        print("\nExamples of matching ID pairs with Dice score = 0:")
        print("-" * 60)
        for i, (gid, sid, gold_tokens, sys_tokens) in enumerate(zero_score_pairs[:5]):  # Show first 5
            print(f"   Pair {i+1}:")
            print(f"     Gold {gid} ({gold[gid]['doc_id']}, {gold[gid]['event_id']}): {gold_tokens}")
            print(f"     System {sid} ({system[sid]['doc_id']}, {system[sid]['event_id']}): {sys_tokens}")
            print(f"     Intersection: {gold_tokens & sys_tokens}")
            print()
        if len(zero_score_pairs) > 5:
            print(f"   ... and {len(zero_score_pairs) - 5} more pairs")
    
    # Step 3: Initialize mapping and used system mentions
    mapping = {}  # gold_id -> [(system_id, score), ...]
    used_sys = set()  # system mentions already used
    
    # Step 4: Algorithm 1 main loop
    print("Finding optimal mappings...")
    with tqdm(total=len(score_list), desc="Processing score pairs", unit="pair") as pbar:
        while score_list:
            # Find the pair with highest Dice score
            best_idx = max(range(len(score_list)), key=lambda i: score_list[i][2])
            gm, sn, best_score = score_list[best_idx]
            
            # Check if system mention not used and score >= threshold
            if sn not in used_sys and best_score >= threshold:
                # Add to mapping
                if gm not in mapping:
                    mapping[gm] = []
                mapping[gm].append((sn, best_score))
                # Mark system mention as used
                used_sys.add(sn)
            score_list.pop(best_idx)
            pbar.update(1)
    mapped_gold = len(mapping)
    mapped_system = sum(len(sys_list) for sys_list in mapping.values())
    print(f"Mapping complete: {mapped_gold}/{len(gold)} gold mentions mapped to {mapped_system}/{len(system)} system mentions")
    return mapping

# ----- Span F1 (Algorithm 2) -----
def compute_span_f1(gold, system, mapping):
    """
    Implementation of Algorithm 2: Compute TP and FP for span-level F1
    """
    print("Computing Span F1...")
    TP = 0.0
    FP = 0.0
    with tqdm(total=len(gold), desc="Computing Span F1", unit="mention") as pbar:
        for gid in range(len(gold)):
            if gid not in mapping or len(mapping[gid]) == 0:
                FP += 1
            else:
                best_dice = max(dice_score for _, dice_score in mapping[gid])
                TP += best_dice
            pbar.update(1)
    
    NS = len(system)  # Number of system mentions
    NG = len(gold)    # Number of gold mentions
    
    # According to the paper's formula
    precision = TP / NS if NS > 0 else 0.0
    recall = TP / NG if NG > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print(f"✅ Span F1 computed: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
    return round(precision * 100, 1), round(recall * 100, 1), round(f1 * 100, 1)

# ----- Attribute accuracy (Algorithm 3) -----
def compute_attribute_acc(gold, system, mapping, attr):
    """
    Implementation of Algorithm 3: Compute Attribute Accuracy
    """
    if not mapping:
        return 0.0
    
    print(f"🏷️  Computing {attr} accuracy...")
    
    total_accuracy = 0.0
    
    # For each gold mention that has mappings
    with tqdm(total=len(mapping), desc=f"Computing {attr} accuracy", unit="mention") as pbar:
        for gid, mapped_systems in mapping.items():
            mg_size = len(mapped_systems)
            if mg_size == 0:
                pbar.update(1)
                continue
            
            # Get gold attribute value
            gold_attr_value = gold[gid][attr]
            accuracy_for_this_gold = 0.0
            
            # For each system mention mapped to this gold mention
            for sid, dice_score in mapped_systems:
                system_attr_value = system[sid][attr]
                
                # If attributes match
                if system_attr_value == gold_attr_value:
                    accuracy_for_this_gold += 1.0 / mg_size
            
            total_accuracy += accuracy_for_this_gold
            pbar.update(1)
    
    # Average accuracy across all gold mentions with mappings
    num_gold_with_mapping = len(mapping)
    result = round(total_accuracy / num_gold_with_mapping * 100, 1) if num_gold_with_mapping > 0 else 0.0
    print(f"✅ {attr} accuracy: {result}%")
    return result

# ----- Realis accuracy -----
def compute_realis_acc(gold, system, mapping):
    """
    Compute Realis accuracy (modality + polarity must both match)
    """
    if not mapping:
        return 0.0
    
    print("🔄 Computing Realis accuracy...")
    
    total_accuracy = 0.0
    
    # For each gold mention that has mappings
    with tqdm(total=len(mapping), desc="Computing Realis accuracy", unit="mention") as pbar:
        for gid, mapped_systems in mapping.items():
            mg_size = len(mapped_systems)
            if mg_size == 0:
                pbar.update(1)
                continue
            
            # Get gold realis values
            gold_modality = gold[gid]["modality"]
            gold_polarity = gold[gid]["polarity"]
            accuracy_for_this_gold = 0.0
            
            # For each system mention mapped to this gold mention
            for sid, dice_score in mapped_systems:
                system_modality = system[sid]["modality"]
                system_polarity = system[sid]["polarity"]
                
                # Both modality and polarity must match
                if system_modality == gold_modality and system_polarity == gold_polarity:
                    accuracy_for_this_gold += 1.0 / mg_size
            
            total_accuracy += accuracy_for_this_gold
            pbar.update(1)
    
    # Average accuracy across all gold mentions with mappings
    num_gold_with_mapping = len(mapping)
    result = round(total_accuracy / num_gold_with_mapping * 100, 1) if num_gold_with_mapping > 0 else 0.0
    print(f"✅ Realis accuracy: {result}%")
    return result

# ----- Combined F1 -----
def compute_combined_f1(gold, system, mapping, attributes: List[str]) -> Tuple[float, float, float]:
    """
    Compute Combined F1 where TP requires both span overlap AND attribute match
    """
    print("🎯 Computing Combined F1...")
    
    total_tp = 0.0
    
    # For each gold mention
    with tqdm(total=len(gold), desc="Computing Combined F1", unit="mention") as pbar:
        for gid in range(len(gold)):
            if gid not in mapping or len(mapping[gid]) == 0:
                pbar.update(1)
                continue
            
            # Get all system mentions mapped to this gold mention
            mapped_systems = mapping[gid]
            mg_size = len(mapped_systems)
            
            # Calculate attribute-based TP for this gold mention
            tp_for_this_gold = 0.0
            
            # For each system mention mapped to this gold mention
            for sid, dice_score in mapped_systems:
                # Check if all attributes match
                all_attrs_match = all(gold[gid][attr] == system[sid][attr] for attr in attributes)
                
                if all_attrs_match:
                    # Use dice score weighted by 1/|MG|
                    tp_for_this_gold += dice_score / mg_size
            
            total_tp += tp_for_this_gold
            pbar.update(1)
    
    # Calculate combined F1
    NS = len(system)
    NG = len(gold)
    
    precision = total_tp / NS if NS > 0 else 0.0
    recall = total_tp / NG if NG > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print(f"✅ Combined F1 computed: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
    return round(precision * 100, 1), round(recall * 100, 1), round(f1 * 100, 1)

# ----- ID matching statistics -----
def print_id_matching_stats(gold, system):
    """Print statistics about ID matching"""
    print("\n📊 ID Matching Statistics:")
    print("-" * 40)
    
    # Count unique doc_ids and event_ids
    gold_docs = set(g["doc_id"] for g in gold)
    system_docs = set(s["doc_id"] for s in system)
    
    gold_pairs = set((g["doc_id"], g["event_id"]) for g in gold)
    system_pairs = set((s["doc_id"], s["event_id"]) for s in system)
    
    common_docs = gold_docs & system_docs
    common_pairs = gold_pairs & system_pairs
    
    print(f"   Gold documents: {len(gold_docs)}")
    print(f"   System documents: {len(system_docs)}")
    print(f"   Common documents: {len(common_docs)}")
    print(f"   Gold (doc_id, event_id) pairs: {len(gold_pairs)}")
    print(f"   System (doc_id, event_id) pairs: {len(system_pairs)}")
    print(f"   Common (doc_id, event_id) pairs: {len(common_pairs)}")
    print(f"   Coverage: {len(common_pairs)/len(gold_pairs)*100:.1f}% of gold events have matching system events")

# ----- Main evaluation function -----
def evaluate(gold_path: str, system_path: str, threshold: float = 0.0) -> Dict[str, float]:
    """
    Main evaluation function following the paper's methodology with ID matching
    """
    start_time = time.time()
    
    print("🚀 Starting evaluation with ID matching...")
    print(f"📁 Gold file: {gold_path}")
    print(f"📁 System file: {system_path}")
    print(f"🎯 Threshold: {threshold}")
    # print(f"📊 Sample size: {sample_size}")
    print("-" * 80)
    
    # Load data
    print("📥 Loading data files...")
    with open(gold_path, "r", encoding="utf-8") as f:
        gold_json = json.load(f)
    with open(system_path, "r", encoding="utf-8") as f:
        sys_json = json.load(f)
    print("Files loaded successfully")
    
    # # Sample data
    # print("\n" + "="*50)
    # print("PHASE 0: SAMPLING DATA")
    # print("="*50)
    
    # print("🎲 Sampling gold data...")
    # gold_json = sample_data(gold_json, sample_size)
    
    # print("🎲 Sampling system data...")
    # sys_json = sample_data(sys_json, sample_size)
              
    # Parse events
    print("\n" + "="*50)
    print("PHASE 1: PARSING EVENTS")
    print("="*50)
    
    print("🔍 Parsing gold events...")
    gold = parse_events(gold_json)
    
    print("🔍 Parsing system events...")
    system = parse_events(sys_json)
    
    print(f"\n📊 Summary:")
    print(f"   Gold events: {len(gold):,}")
    print(f"   System events: {len(system):,}")
    
    # Print ID matching statistics
    print_id_matching_stats(gold, system)
    
    # Compute mapping
    print("\n" + "="*50)
    print("PHASE 2: MENTION MAPPING (WITH ID CONSTRAINT)")
    print("="*50)
    
    mapping = mention_mapping(gold, system, threshold=threshold)
    
    # Print mapping statistics
    mapped_gold = len(mapping)
    mapped_system = sum(len(sys_list) for sys_list in mapping.values())
    print(f"\n📊 Mapping Summary:")
    print(f"   Mapped gold mentions: {mapped_gold:,}/{len(gold):,} ({mapped_gold/len(gold)*100:.1f}%)")
    print(f"   Mapped system mentions: {mapped_system:,}/{len(system):,} ({mapped_system/len(system)*100:.1f}%)")
    
    # Compute metrics
    print("\n" + "="*50)
    print("PHASE 3: COMPUTING METRICS")
    print("="*50)
    
    # Compute span F1
    span_p, span_r, span_f1 = compute_span_f1(gold, system, mapping)
    
    # Compute attribute accuracies
    type_acc = compute_attribute_acc(gold, system, mapping, "type")
    subtype_acc = compute_attribute_acc(gold, system, mapping, "subtype")
    modality_acc = compute_attribute_acc(gold, system, mapping, "modality")
    polarity_acc = compute_attribute_acc(gold, system, mapping, "polarity")
    realis_acc = compute_realis_acc(gold, system, mapping)
    
    # Compute combined F1
    all_attrs = ["type", "subtype", "modality", "polarity"]
    comb_p, comb_r, comb_f1 = compute_combined_f1(gold, system, mapping, all_attrs)
    
    # Calculate total time
    total_time = time.time() - start_time
    
    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("="*50)
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    
    return {
        "Span_Precision": span_p,
        "Span_Recall": span_r,
        "Span_F1": span_f1,
        "Type_Accuracy": type_acc,
        "Subtype_Accuracy": subtype_acc,
        "Modality_Accuracy": modality_acc,
        "Polarity_Accuracy": polarity_acc,
        "Realis_Accuracy": realis_acc,
        "Combined_Precision": comb_p,
        "Combined_Recall": comb_r,
        "Combined_F1": comb_f1
    }

# ----- Print results in a nice format -----
def print_results(results: Dict[str, float]):
    """Print results in a nicely formatted table"""
    print("\n" + "="*60)
    print("FINAL EVALUATION RESULTS")
    print("="*60)
    
    # Group metrics
    span_metrics = ["Span_Precision", "Span_Recall", "Span_F1"]
    attr_metrics = ["Type_Accuracy", "Subtype_Accuracy", "Modality_Accuracy", "Polarity_Accuracy", "Realis_Accuracy"]
    combined_metrics = ["Combined_Precision", "Combined_Recall", "Combined_F1"]
    
    print("\n📏 SPAN METRICS:")
    print("-" * 30)
    for metric in span_metrics:
        if metric in results:
            print(f"   {metric.replace('_', ' '):15s}: {results[metric]:6.1f}%")
    
    print("\n🏷️  ATTRIBUTE ACCURACY:")
    print("-" * 30)
    for metric in attr_metrics:
        if metric in results:
            print(f"   {metric.replace('_', ' '):15s}: {results[metric]:6.1f}%")
    
    print("\n🎯 COMBINED METRICS:")
    print("-" * 30)
    for metric in combined_metrics:
        if metric in results:
            print(f"   {metric.replace('_', ' '):15s}: {results[metric]:6.1f}%")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    try:
        # Test with sample data
        # test_with_sample()
        
        # Use with actual files - now with sample_size parameter
        gold_path = "./data/final/tokenized_data_500_accepted.json"
        # verified_path = "./data/processed/agentB/tokenized_data_500.json"
        verified_path = "./data/processed/agentA/tokenized_data_500.json"
        
        # Run evaluation with sample of 100 documents
        results = evaluate(gold_path, verified_path)
        print_results(results)
        
    except FileNotFoundError as e:
        print(f"❌ Error: File not found - {e}")
        print("Please check the file paths and make sure the files exist.")
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        print("Please check your data format and try again.")