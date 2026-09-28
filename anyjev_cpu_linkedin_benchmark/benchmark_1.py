import time
from typing import List, Dict, Any
import torch
import numpy as np

# Import AnyJev components
from anyjev import Decider, Question
from anyjev.backends.hf import HFBackend

AVAILABLE_TOOLS = [
    "calculator",
    "customer_lookup",
    "order_lookup",
    "web_search",
    "none"
]

def create_routing_question() -> Question:
    """
    Creates an AnyJev choice question for tool routing.
    """
    return Question.choice(
        "Which tool should be called to answer the user's request?",
        AVAILABLE_TOOLS,
        name="tool_router"
    )

import re
import numpy as np
from typing import Any, Tuple

def extract_choice_and_confidence(decision_output: Any) -> Tuple[str, float]:
    """
    Safely unpacks AnyJev DecisionSet and Decision objects to extract 
    the selected option string (predicted_str) and confidence value.
    """
    if decision_output is None:
        return "none", 0.0

    target_decision = decision_output

    # 1. Unpack AnyJev DecisionSet to get the inner Decision object
    if hasattr(decision_output, "decisions"):
        decisions = decision_output.decisions
        if isinstance(decisions, dict):
            target_decision = decisions.get("tool_router", next(iter(decisions.values()), None))
        elif isinstance(decisions, (list, tuple)) and len(decisions) > 0:
            target_decision = decisions[0]
    elif isinstance(decision_output, dict):
        target_decision = decision_output.get("tool_router", next(iter(decision_output.values()), None))

    if target_decision is None:
        return "none", 0.0

    # 2. Extract choice string (predicted_str)
    predicted_val = None

    if hasattr(target_decision, "choice_name") and target_decision.choice_name is not None:
        predicted_val = target_decision.choice_name
    elif hasattr(target_decision, "choice") and target_decision.choice is not None:
        predicted_val = target_decision.choice
    elif hasattr(target_decision, "choice_idx") and hasattr(target_decision, "options"):
        try:
            predicted_val = target_decision.options[target_decision.choice_idx]
        except (IndexError, TypeError):
            predicted_val = None

    raw_repr = str(target_decision)

    # Fallback string parsing for choice if attributes are missing
    if predicted_val is None:
        if ":" in raw_repr:
            val_part = raw_repr.split(":")[1].split(",")[0].strip()
            predicted_val = val_part.replace("'", "").replace('"', "")
        else:
            predicted_val = raw_repr

    predicted_str = str(predicted_val).strip().lower()
    if predicted_str in ["none", "null", "none_type", "nonetype", ""]:
        predicted_str = "none"

    # 3. Extract confidence float across all known AnyJev internal attributes
    confidence = 0.0

    # Direct attribute checks
    for attr_name in ["conf", "confidence", "prob", "_conf", "max_prob", "score"]:
        if hasattr(target_decision, attr_name):
            val = getattr(target_decision, attr_name)
            if val is not None and isinstance(val, (int, float, np.number)):
                confidence = float(val)
                break

    # Distribution vector max checks (distribution, dist, probs, _dist)
    if confidence == 0.0:
        for dist_attr in ["distribution", "dist", "probs", "probabilities", "_dist", "_probs"]:
            if hasattr(target_decision, dist_attr):
                arr = getattr(target_decision, dist_attr)
                if arr is not None and len(arr) > 0:
                    try:
                        confidence = float(np.max(arr))
                        break
                    except (ValueError, TypeError):
                        pass

    # Inspection of internal __dict__ keys
    if confidence == 0.0 and hasattr(target_decision, "__dict__"):
        for k, v in target_decision.__dict__.items():
            if "conf" in k.lower() or "prob" in k.lower():
                if isinstance(v, (int, float, np.number)):
                    confidence = float(v)
                    break

    # Regex extraction from __repr__ string e.g., Decision(..., conf=0.844)
    if confidence == 0.0:
        match = re.search(r"conf\s*=\s*([0-9\.]+)", raw_repr)
        if match:
            try:
                confidence = float(match.group(1))
            except ValueError:
                confidence = 0.0

    return predicted_str, confidence

def extract_choice_and_confidence2(decision_output: Any) -> tuple[str, float]:
    """
    Safely unpacks AnyJev DecisionSet and Decision objects to extract 
    the selected option string (predicted_str) and confidence value.
    """
    if decision_output is None:
        return "none", 0.0

    target_decision = decision_output

    # 1. Unpack AnyJev DecisionSet to get the inner Decision object
    if hasattr(decision_output, "decisions"):
        decisions = decision_output.decisions
        if isinstance(decisions, dict):
            target_decision = decisions.get("tool_router", next(iter(decisions.values()), None))
        elif isinstance(decisions, (list, tuple)) and len(decisions) > 0:
            target_decision = decisions[0]
    elif isinstance(decision_output, dict):
        target_decision = decision_output.get("tool_router", next(iter(decision_output.values()), None))

    if target_decision is None:
        return "none", 0.0

    # 2. Extract choice string (predicted_str)
    predicted_val = None

    # AnyJev choice properties
    if hasattr(target_decision, "choice_name") and target_decision.choice_name is not None:
        predicted_val = target_decision.choice_name
    elif hasattr(target_decision, "choice") and target_decision.choice is not None:
        predicted_val = target_decision.choice
    elif hasattr(target_decision, "choice_idx") and hasattr(target_decision, "options"):
        try:
            predicted_val = target_decision.options[target_decision.choice_idx]
        except (IndexError, TypeError):
            predicted_val = None

    # Fallback parsing if choice_name/choice_idx aren't exposed directly
    if predicted_val is None:
        raw_repr = str(target_decision)
        # Parse 'tool_router: \'none\'' or 'Decision(...)' string if needed
        if ":" in raw_repr:
            # Extracts 'none' from "Decision(tool_router: 'none', conf=...)"
            val_part = raw_repr.split(":")[1].split(",")[0].strip()
            predicted_val = val_part.replace("'", "").replace('"', "")
        else:
            predicted_val = raw_repr

    # Normalize extracted string
    predicted_str = str(predicted_val).strip().lower()
    if predicted_str in ["none", "null", "none_type", "nonetype", ""]:
        predicted_str = "none"

    # 3. Extract confidence float
    confidence = 0.0
    for conf_attr in ["conf", "confidence", "prob"]:
        if hasattr(target_decision, conf_attr):
            val = getattr(target_decision, conf_attr)
            if val is not None:
                try:
                    confidence = float(val)
                    break
                except (ValueError, TypeError):
                    pass

    if confidence == 0.0:
        if hasattr(target_decision, "distribution") and target_decision.distribution is not None:
            try:
                confidence = float(np.max(target_decision.distribution))
            except (ValueError, TypeError):
                confidence = 0.0
        elif hasattr(target_decision, "probs") and target_decision.probs is not None:
            try:
                confidence = float(np.max(target_decision.probs))
            except (ValueError, TypeError):
                confidence = 0.0

    return predicted_str, confidence

def extract_choice_and_confidence1(decision_output: Any) -> tuple[str, float]:
    """
    Safely unpacks AnyJev DecisionSet and Decision objects to extract 
    the selected option string and confidence value.
    """
    if decision_output is None:
        return "none", 0.0

    target_decision = decision_output

    # 1. Handle AnyJev DecisionSet
    if hasattr(decision_output, "decisions"):
        decisions = decision_output.decisions
        if isinstance(decisions, dict):
            target_decision = decisions.get("tool_router", next(iter(decisions.values()), None))
        elif isinstance(decisions, (list, tuple)) and len(decisions) > 0:
            target_decision = decisions[0]
    elif isinstance(decision_output, dict):
        target_decision = decision_output.get("tool_router", next(iter(decision_output.values()), None))

    if target_decision is None:
        return "none", 0.0

    # 2. Extract choice label
    predicted_val = None
    for attr in ["choice_name", "selected_option", "choice", "value"]:
        if hasattr(target_decision, attr):
            val = getattr(target_decision, attr)
            if val is not None and not callable(val):
                predicted_val = val
                break

    if predicted_val is None and hasattr(target_decision, "choice_idx") and hasattr(target_decision, "options"):
        try:
            predicted_val = target_decision.options[target_decision.choice_idx]
        except (IndexError, TypeError):
            predicted_val = "none"

    if predicted_val is None:
        predicted_val = str(target_decision)

    # Clean predicted string representation
    predicted_str = str(predicted_val).strip().lower()
    if predicted_str in ["none", "null", "none_type", "nonetype", ""]:
        predicted_str = "none"

    # 3. Extract confidence
    confidence = 0.0
    for conf_attr in ["conf", "confidence", "prob"]:
        if hasattr(target_decision, conf_attr):
            val = getattr(target_decision, conf_attr)
            if val is not None:
                try:
                    confidence = float(val)
                    break
                except (ValueError, TypeError):
                    pass

    if confidence == 0.0:
        if hasattr(target_decision, "distribution") and target_decision.distribution is not None:
            try:
                confidence = float(np.max(target_decision.distribution))
            except (ValueError, TypeError):
                confidence = 0.0
        elif hasattr(target_decision, "probs") and target_decision.probs is not None:
            try:
                confidence = float(np.max(target_decision.probs))
            except (ValueError, TypeError):
                confidence = 0.0

    return predicted_str, confidence

def normalize_expected_target(raw_expected: Any) -> str:
    """
    Normalizes dataset expected target to match tool router string keys.
    """
    if raw_expected is None:
        return "none"
    
    exp_str = str(raw_expected).strip().lower()
    if exp_str in ["none", "null", "none_type", "nonetype", ""]:
        return "none"
    return exp_str

def run_benchmark(dataset: List[Dict[str, Any]], decider: Decider, level: str = "L0") -> List[Dict[str, Any]]:
    """
    Runs evaluation using AnyJev on CPU.
    """
    question = create_routing_question()
    results = []

    for idx, item in enumerate(dataset):
        start_time = time.time()
        user_query = item.get("query", "")
        
        # Normalize expected label ('none', 'calculator', 'customer_lookup', etc.)
        expected = normalize_expected_target(item.get("expected"))

        context = {"query": user_query}

        # Obtain decision set from AnyJev decider
        decision = decider.decide(
            context, 
            [question], 
            level=level
        )

        predicted, confidence = extract_choice_and_confidence(decision)
        latency = (time.time() - start_time) * 1000
        #print(predicted)
        #print(expected)

        correct = (predicted == expected)

        res = {
            "id": f"{idx + 1:02d}",
            "expected": expected,
            "predicted": predicted,
            "correct": correct,
            "confidence": confidence,
            "latency": latency
        }
        results.append(res)

        print(f"[1/1] {res['id']} expected={res['expected']:<16} predicted={res['predicted']:<16} correct={str(res['correct']):<5} confidence={res['confidence']:.3f} latency={res['latency']:.1f}ms")

    return results

if __name__ == "__main__":
    device = "cpu"
    
    # Initialize HFBackend on CPU
    backend = HFBackend(
        model_name="Qwen/Qwen3-0.6B",
        device_map=device,
        torch_dtype=torch.float32
    )
    
    decider = Decider(backend)

    # Load dataset
    from dataset import DATASET

    print("Running AnyJev Benchmark on CPU...")
    run_benchmark(DATASET, decider, level="L0")