import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

REPORT_PATH = Path("data/evaluation/evaluation_report.json")
HISTORY_PATH = Path("data/evaluation/evaluation_history.json")
OUTPUT_PLOT_PATH = Path("data/evaluation/evaluation_history_trend.png")

def update_and_plot_history():
    if not REPORT_PATH.exists():
        print(f"Evaluation report not found at {REPORT_PATH}. Please run evaluate_rag.py first.")
        return

    # 1. Load latest evaluation summary
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        latest_data = json.load(f)

    summary = latest_data.get("summary", {})
    
    # Create history entry with a timestamp
    current_run = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_cases": summary.get("total_test_cases", 0),
        "retrieval_hit_rate_pct": summary.get("retrieval_hit_rate_pct", 0),
        "generation_accuracy_pct": summary.get("generation_accuracy_pct", 0)
    }

    # 2. Append to historical log file
    history = []
    if HISTORY_PATH.exists():
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    history.append(current_run)

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)

    print(f"Historical run logged successfully. Total recorded runs: {len(history)}")

    # 3. Convert history to DataFrame for trend plotting
    df = pd.DataFrame(history)
    df["run_index"] = range(1, len(df) + 1) # Run sequence numbers

    # 4. Plot historical trend lines
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(df["run_index"], df["retrieval_hit_rate_pct"], marker='o', color='#2ECC71', linewidth=2.5, label='Retrieval Hit Rate (%)')
    ax.plot(df["run_index"], df["generation_accuracy_pct"], marker='s', color='#3498DB', linewidth=2.5, label='Generation Accuracy (%)')

    ax.set_title("RAG Pipeline Performance Trend Across Evaluation Runs", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Evaluation Run Sequence", fontsize=11)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_xticks(df["run_index"])
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower right', fontsize=11)

    # Annotate points with timestamps or run numbers
    for i, row in df.iterrows():
        ax.annotate(f"{row['retrieval_hit_rate_pct']}%", 
                    (row["run_index"], row["retrieval_hit_rate_pct"] + 2), 
                    textcoords="offset points", xytext=(0,0), ha='center', fontsize=9, color='#27AE60', fontweight='bold')
        ax.annotate(f"{row['generation_accuracy_pct']}%", 
                    (row["run_index"], row["generation_accuracy_pct"] - 6), 
                    textcoords="offset points", xytext=(0,0), ha='center', fontsize=9, color='#2980B9', fontweight='bold')

    plt.tight_layout()

    # Save figure
    plt.savefig(OUTPUT_PLOT_PATH, dpi=300, bbox_inches='tight')
    print(f"Trend plot successfully updated and saved to: {OUTPUT_PLOT_PATH}")

    # Display plot window
    plt.show()

if __name__ == "__main__":
    update_and_plot_history()