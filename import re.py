import re
import time
import ollama
import pandas as pd
from tqdm import tqdm

# ==========================================
# 1. BENCHMARK DATASET
# ==========================================
benchmark_data = [
    {
        "id": "Q1",
        "subject": "Physics",
        "question": "If a spaceship travels at 0.8c relative to Earth, how does time pass on the spaceship compared to Earth?",
        "options": "A) Faster than on Earth\nB) Slower than on Earth\nC) At the exact same rate\nD) Time stops completely",
        "correct_answer": "B"
    },
    {
        "id": "Q2",
        "subject": "Computer Science",
        "question": "What is the time complexity of searching for an element in a balanced Binary Search Tree (BST) in the worst case?",
        "options": "A) O(1)\nB) O(n)\nC) O(log n)\nD) O(n log n)",
        "correct_answer": "C"
    },
    {
        "id": "Q3",
        "subject": "Chemistry",
        "question": "Which type of chemical bond involves the unequal sharing of electrons between two atoms?",
        "options": "A) Nonpolar covalent bond\nB) Ionic bond\nC) Polar covalent bond\nD) Metallic bond",
        "correct_answer": "C"
    },
    {
        "id": "Q4",
        "subject": "Logic & Math",
        "question": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "options": "A) $0.10\nB) $0.05\nC) $0.01\nD) $0.15",
        "correct_answer": "B"
    }
]

df = pd.DataFrame(benchmark_data)

# ==========================================
# 2. ZERO-SHOT PROMPT & PARSER
# ==========================================
def build_zero_shot_prompt(row):
    return f"""Answer the following multiple-choice question. 
Think step-by-step if needed, and end your response with 'Final Answer: X' where X is A, B, C, or D.

Question: {row['question']}
Options:
{row['options']}
"""

def extract_answer(response_text):
    # Model ke jawab mein se A, B, C, ya D nikalne ke liye Regex
    match = re.search(r"Final Answer:\s*([A-D])", response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Fallback: Agar format follow na ho to aakhri A/B/C/D letter utha lein
    fallback = re.findall(r"\b([A-D])\b", response_text)
    return fallback[-1].upper() if fallback else "INVALID"

# ==========================================
# 3. BENCHMARK EVALUATION PIPELINE
# ==========================================
def run_benchmark(model_name):
    print(f"\n🚀 Starting Benchmark Evaluation for Model: {model_name}")
    results = []
    start_time = time.time()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        prompt = build_zero_shot_prompt(row)
        
        try:
            # Local Ollama model ko call karna
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0} # Deterministic results ke liye 0.0
            )
            raw_output = response["message"]["content"]
        except Exception as e:
            raw_output = f"ERROR: {str(e)}"

        predicted = extract_answer(raw_output)
        is_correct = (predicted == row["correct_answer"])

        results.append({
            "id": row["id"],
            "subject": row["subject"],
            "correct_answer": row["correct_answer"],
            "predicted_answer": predicted,
            "is_correct": is_correct
        })

    elapsed_time = round(time.time() - start_time, 2)
    results_df = pd.DataFrame(results)
    
    accuracy = (results_df["is_correct"].sum() / len(results_df)) * 100
    
    print("\n" + "="*40)
    print(f"📊 BENCHMARK REPORT: {model_name}")
    print("="*40)
    print(f"Total Questions : {len(results_df)}")
    print(f"Correct Answers : {results_df['is_correct'].sum()}")
    print(f"Accuracy Score  : {accuracy:.2f}%")
    print(f"Time Taken      : {elapsed_time} seconds")
    print("="*40)
    
    output_file = f"benchmark_{model_name.replace(':', '_')}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"✅ Detailed results saved to: {output_file}\n")
    
    return results_df

# ==========================================
# 4. RUN THE BENCHMARK
# ==========================================
if __name__ == "__main__":
    # Apne system par installed model ka naam likhein (e.g., 'deepseek-r1:1.5b' ya 'llama3.2')
    MODEL_TO_TEST = "deepseek-r1:1.5b"
    
    report = run_benchmark(MODEL_TO_TEST)
    print(report[["id", "subject", "correct_answer", "predicted_answer", "is_correct"]])
