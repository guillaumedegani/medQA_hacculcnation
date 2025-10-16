import json
import pandas as pd
from pathlib import Path


"""
normal behavior
"Schizoaffective disorder"
Abusive Bruise
Autism Spectrum Delay
Generalized anxiety disorder
myocardial infarction

"""
def load_medqa_jsonl(path: str, limit: int = None) -> pd.DataFrame:
    """
    Efficiently load a MedQA .jsonl dataset into a pandas DataFrame.
    Each line must be a valid JSON object.
    
    Parameters
    ----------
    path : str
        Path to the .jsonl file.
    limit : int, optional
        Number of lines to read (for testing).

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                obj = json.loads(line)
                rows.append({
                    "question": obj.get("question", ""),
                    "answer": obj.get("answer", ""),
                    "answer_idx": obj.get("answer_idx", ""),
                    "options": obj.get("options", {}),
                    "meta_info": obj.get("meta_info", ""),
                    "metamap_phrases": obj.get("metamap_phrases", [])
                })
            except json.JSONDecodeError as e:
                print(f"⚠️ Skipping malformed line {i}: {e}")

    df = pd.DataFrame(rows)
    print(f"✅ Loaded {len(df)} questions from {path.name}")
    return df
def find_questions_with_answer(df: pd.DataFrame, answer_text: str) -> pd.DataFrame:
    """
    Return all MedQA questions that include the given answer_text
    in their multiple-choice options.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded MedQA dataframe.
    answer_text : str
        Text of the answer to search for (case-insensitive).

    Returns
    -------
    pd.DataFrame
        Subset of df where the answer_text appears in options.
    """
    answer_text_lower = answer_text.lower()

    def has_answer(options):
        if isinstance(options, dict):
            return any(answer_text_lower in str(v).lower() for v in options.values())
        return False

    mask = df["options"].apply(has_answer)
    result = df[mask].copy()
    print(f"✅ Found {len(result)} questions containing '{answer_text}' in their options.")
    return result

# Example usage:
# Replace this path with your actual file
file_path = "data/phrases_no_exclude_train.jsonl"

df = load_medqa_jsonl(file_path)

    # Example: view a few rows
print(find_questions_with_answer(df,"pulmonary embolism"))
# Example: search for a keyword
"""
keyword = "normal behavior"
subset = df[df["answer"].str.contains(keyword, case=False, na=False)]
print(f"\nFound {len(subset)} questions mentioning '{keyword}':")
print(subset[["question", "answer"]].head(5))
"""