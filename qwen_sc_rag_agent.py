import json
import re
import subprocess
from pathlib import Path
import chromadb
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

class ScanpyRAG:
    """Manages the local vector database for scanpy documentation."""
    def __init__(self, persist_directory: str = "./scanpy_chroma_db"):
        print("Initializing Scanpy RAG Vector Database...")
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Gets or creates a collection for our scanpy docs
        self.collection = self.client.get_or_create_collection(
            name="scanpy_knowledge"
        )
        
        # If the database is empty, seed it with some basic knowledge 
        # (In a real scenario, you would parse the scanpy readthedocs and bulk add them here)
        if self.collection.count() == 0:
            self._seed_basic_knowledge()

    def _seed_basic_knowledge(self):
        print("Seeding database with initial scanpy knowledge...")
        docs = [
            "To filter cells based on gene counts in scanpy: `sc.pp.filter_cells(adata, min_genes=200)`",
            "To filter genes based on cell counts in scanpy: `sc.pp.filter_genes(adata, min_cells=3)`",
            "To calculate QC metrics like mitochondrial percentage: `adata.var['mt'] = adata.var_names.str.startswith('MT-'); sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)`",
            "To normalize total counts per cell: `sc.pp.normalize_total(adata, target_sum=1e4)`",
            "To log-transform the data: `sc.pp.log1p(adata)`",
            "To find highly variable genes: `sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)`"
        ]
        ids = [f"doc_{i}" for i in range(len(docs))]
        self.collection.add(documents=docs, ids=ids)

    def retrieve_context(self, query: str, n_results: int = 2) -> str:
        """Searches the vector database for the most relevant scanpy docs."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Flatten the retrieved documents into a single string
        retrieved_docs = results['documents'][0]
        if not retrieved_docs:
            return ""
            
        context = "\n- ".join(retrieved_docs)
        return f"Helpful scanpy documentation for this task:\n- {context}\n"


def extract_python_code(text: str) -> str:
    """Extracts Python code from the LLM's markdown output."""
    match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1)
    return text.strip()

class QwenSingleCellAgent:
    def __init__(self, model_id=MODEL_ID, use_rag=True):
        print(f"Loading Qwen model: {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="auto",
            torch_dtype="auto"
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=1024,
            temperature=0.1,
        )
        
        self.use_rag = use_rag
        self.rag = ScanpyRAG() if use_rag else None

    def run_task(self, task_prompt: str, work_dir: Path, max_retries: int = 3) -> dict:
        work_dir = Path(work_dir)
        
        # --- RAG RETRIEVAL STEP ---
        rag_context = ""
        if self.use_rag:
            print("Retrieving scanpy context...")
            rag_context = self.rag.retrieve_context(task_prompt)
            print(f"Retrieved Context:\n{rag_context}")
        
        system_prompt = (
            "You are an expert computational biologist and AI agent for single-cell RNA-seq analysis. "
            "You use `scanpy`, `anndata`, and `pandas` to solve the user's task. "
            f"Your current working directory is '{work_dir.absolute()}'.\n\n"
            f"{rag_context}\n"
            "Write a self-contained Python script to complete the given task. "
            "CRITICAL: The script MUST save the final required answer as a JSON object into a file "
            f"named 'eval_answer.json' exactly in the directory '{work_dir.absolute()}'. "
            "Output ONLY the Python code in a Markdown ```python``` block. Do not provide explanations."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt}
        ]
        
        for attempt in range(max_retries):
            print(f"\n--- Attempt {attempt + 1}/{max_retries} ---")
            
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            outputs = self.pipe(prompt)
            response_text = outputs[0]["generated_text"][len(prompt):]
            code = extract_python_code(response_text)
            
            script_path = work_dir / f"agent_script_attempt_{attempt}.py"
            script_path.write_text(code)
            
            try:
                subprocess.run(
                    ["python", script_path.name], 
                    cwd=work_dir, 
                    check=True, 
                    capture_output=True, 
                    text=True
                )
                
                answer_path = work_dir / "eval_answer.json"
                if answer_path.exists():
                    with open(answer_path, "r") as f:
                        print("Task completed successfully!")
                        return json.load(f)
                else:
                    error_msg = "Script ran successfully but 'eval_answer.json' was not found. Please ensure your script dumps the result."
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({"role": "user", "content": error_msg})
                    print("Execution succeeded, but missing output file. Retrying...")
                    
            except subprocess.CalledProcessError as e:
                error_msg = f"Execution failed with error:\n{e.stderr}\nPlease fix the error and output the corrected script."
                print(f"Execution Error:\n{e.stderr.strip().splitlines()[-1]}")
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": error_msg})

        print("Agent failed to complete the task within the maximum number of retries.")
        return {}


# Initialize globally
sc_agent = QwenSingleCellAgent(use_rag=True)

def scbench_agent_wrapper(task_prompt: str, work_dir: Path):
    return sc_agent.run_task(task_prompt, work_dir)

if __name__ == "__main__":
    try:
        from scbench import EvalRunner
        canonical_eval_path = "evals_canonical/chromium/chromium_qc_4T1_filter_cells.json"
        
        print(f"Running scBench Evaluation: {canonical_eval_path}")
        runner = EvalRunner(canonical_eval_path)
        result = runner.run(agent_function=scbench_agent_wrapper)
        
        print("\n=== EVALUATION RESULTS ===")
        print(f"Passed: {result.get('passed', False)}")
        if 'error' in result:
            print(f"Benchmark Error: {result['error']}")
            
    except ImportError:
        print("scbench module not found.")