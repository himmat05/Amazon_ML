"""
Metric evaluation module for ML Challenge 2026: Business Entity Resolution.
Evaluates Macro F_0.5 score across all Source 1 entities with proper singleton handling.
"""

from typing import Dict, Iterable, Set, Union


def compute_entity_f05(
    true_ids: Set[str],
    pred_ids: Set[str],
    beta: float = 0.5,
) -> float:
    """Compute F_beta (default beta=0.5) for a single Source 1 entity.

    Rules:
    - If true_ids is empty (singleton):
        - Score = 1.0 if pred_ids is empty
        - Score = 0.0 if pred_ids is not empty (false merge penalized)
    - If true_ids is not empty:
        - Score = 0.0 if pred_ids is empty (missed matches)
        - Otherwise, compute precision and recall:
            P = |true_ids & pred_ids| / |pred_ids|
            R = |true_ids & pred_ids| / |true_ids|
            F_beta = (1 + beta^2) * (P * R) / (beta^2 * P + R)
    """
    is_true_empty = len(true_ids) == 0
    is_pred_empty = len(pred_ids) == 0

    if is_true_empty:
        return 1.0 if is_pred_empty else 0.0

    if is_pred_empty:
        return 0.0

    overlap = len(true_ids & pred_ids)
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_ids)
    recall = overlap / len(true_ids)

    beta_sq = beta ** 2
    denominator = (beta_sq * precision) + recall
    if denominator == 0.0:
        return 0.0

    f_beta = (1.0 + beta_sq) * (precision * recall) / denominator
    return f_beta


def compute_macro_f05(
    ground_truth: Dict[str, Union[Set[str], Iterable[str]]],
    predictions: Dict[str, Union[Set[str], Iterable[str]]],
    beta: float = 0.5,
) -> Dict[str, float]:
    """Compute Macro F_beta across all entities in ground_truth.

    Args:
        ground_truth: Mapping of source1_entity_id -> set of true matching IDs
        predictions: Mapping of source1_entity_id -> set of predicted matching IDs
        beta: Beta parameter (default 0.5 for precision-heavy scoring)

    Returns:
        dict containing:
            - macro_f05: Average score across all entities
            - singleton_accuracy: Accuracy specifically on singletons (0 true matches)
            - non_singleton_f05: Average score on entities with >= 1 true matches
            - num_entities: Total number of entities evaluated
            - num_singletons: Count of singletons in ground truth
    """
    scores = []
    singleton_scores = []
    non_singleton_scores = []

    for s1_id, true_set in ground_truth.items():
        true_s = set(true_set)
        pred_s = set(predictions.get(s1_id, []))

        score = compute_entity_f05(true_s, pred_s, beta=beta)
        scores.append(score)

        if len(true_s) == 0:
            singleton_scores.append(score)
        else:
            non_singleton_scores.append(score)

    total_count = len(scores)
    macro_score = sum(scores) / total_count if total_count > 0 else 0.0
    singleton_acc = (
        sum(singleton_scores) / len(singleton_scores)
        if len(singleton_scores) > 0
        else 0.0
    )
    non_singleton_macro = (
        sum(non_singleton_scores) / len(non_singleton_scores)
        if len(non_singleton_scores) > 0
        else 0.0
    )

    return {
        "macro_f05": macro_score,
        "singleton_accuracy": singleton_acc,
        "non_singleton_f05": non_singleton_macro,
        "num_entities": total_count,
        "num_singletons": len(singleton_scores),
    }


if __name__ == "__main__":
    # Test example from challenge specification:
    # Pred: [S2-00047, S2-00193, S3-00812]
    # Truth: [S2-00047, S3-00812]
    # P = 2/3, R = 1.0 => F_0.5 = (1.25 * 2/3 * 1.0) / (0.25 * 2/3 + 1.0) = 0.7142857
    spec_score = compute_entity_f05(
        true_ids={"S2-00047", "S3-00812"},
        pred_ids={"S2-00047", "S2-00193", "S3-00812"},
    )
    print(f"Spec example test: {spec_score:.3f} (Expected: 0.714)")
    assert abs(spec_score - 0.7142857) < 1e-4

    # Test singleton:
    assert compute_entity_f05(set(), set()) == 1.0
    assert compute_entity_f05(set(), {"S2-00001"}) == 0.0

    print("All metric unit tests passed successfully!")
