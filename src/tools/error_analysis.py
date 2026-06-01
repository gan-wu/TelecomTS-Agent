import os
import re

import pandas as pd


def generate_bad_cases(predictions_path="results/predictions.csv", output_path="results/bad_cases.csv", mae_threshold=0.5):
    """Extract high-error samples from a pipeline prediction CSV.

    The output can be used for prompt/routing review without changing the
    benchmark data or model outputs.
    """
    if not os.path.exists(predictions_path):
        print(f"[Error] Prediction file not found: {predictions_path}. Run src/main_pipeline.py first.")
        return

    df = pd.read_csv(predictions_path)
    bad_cases = []

    for _, row in df.iterrows():
        q_type = row.get("type")
        pred = str(row.get("final_answer", ""))
        truth = str(row.get("ground_truth", ""))

        is_bad = False
        error_reason = ""

        if q_type == "network":
            t_lower = truth.lower()
            p_lower = pred.lower()

            if "yes" in t_lower and "yes" not in p_lower:
                is_bad, error_reason = True, "Network: missed 'Yes' classification"
            elif "no" in t_lower and ("yes" in p_lower or "no" not in p_lower):
                is_bad, error_reason = True, "Network: missed 'No' classification"
            elif "stationary" in t_lower and "stationary" not in p_lower:
                is_bad, error_reason = True, "Network: missed 'Stationary' state"

        elif q_type == "timeseries":
            p_num = _extract_number(pred)
            t_num = _extract_number(truth)

            if p_num is None:
                is_bad, error_reason = True, "TimeSeries: failed to parse a numeric answer"
            elif t_num is not None:
                err = abs(p_num - t_num)
                if err > mae_threshold:
                    is_bad, error_reason = True, f"TimeSeries: MAE too high ({err:.3f} > {mae_threshold})"

        if is_bad:
            row_dict = row.to_dict()
            row_dict["error_reason"] = error_reason
            bad_cases.append(row_dict)

    if not bad_cases:
        print("No bad cases found under the current thresholds.")
        return

    bad_df = pd.DataFrame(bad_cases)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    bad_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved {len(bad_df)} bad cases to: {output_path}")
    print("Review these samples to improve prompts, routing rules, or tool coverage.")


def _extract_number(text: str) -> float | None:
    matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text.replace(",", ""))
    return float(matches[-1]) if matches else None


if __name__ == "__main__":
    generate_bad_cases()
