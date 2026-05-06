import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from prompts import ANSWER_PROMPT, ANSWER_PROMPT_GRAPH
from tqdm import tqdm

from .local_memory import build_local_memory

load_dotenv()


class MemorySearch:
    def __init__(self, output_path="results.json", top_k=10, filter_memories=False, is_graph=False):
        if is_graph:
            raise ValueError("Graph mode is disabled for local Mem0 runs.")

        self.memory = build_local_memory()
        self.top_k = top_k
        base_url = os.getenv("OPENAI_BASE_URL")
        api_key = os.getenv("OPENAI_API_KEY")
        self.openai_client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        self.results = defaultdict(list)
        self.output_path = output_path
        self.filter_memories = filter_memories
        self.is_graph = is_graph

        if self.is_graph:
            self.ANSWER_PROMPT = ANSWER_PROMPT_GRAPH
        else:
            self.ANSWER_PROMPT = ANSWER_PROMPT

    def search_memory(self, user_id, query, max_retries=3, retry_delay=1):
        start_time = time.time()
        retries = 0
        while retries < max_retries:
            try:
                search_results = self.memory.search(query, user_id=user_id, limit=self.top_k)
                break
            except Exception as e:
                print("Retrying...")
                retries += 1
                if retries >= max_retries:
                    raise e
                time.sleep(retry_delay)

        end_time = time.time()
        results = search_results.get("results", []) if isinstance(search_results, dict) else search_results
        query_time = search_results.get("query_time") if isinstance(search_results, dict) else None
        retrieval_time = search_results.get("retrieval_time") if isinstance(search_results, dict) else None

        semantic_memories = []
        for memory in results:
            metadata = memory.get("metadata", {}) or {}
            semantic_memories.append(
                {
                    "memory": memory.get("memory", ""),
                    "timestamp": metadata.get("timestamp"),
                    "score": round(memory.get("score", 0.0), 2),
                }
            )

        graph_memories = search_results.get("relations") if isinstance(search_results, dict) else None
        return semantic_memories, graph_memories, end_time - start_time, query_time, retrieval_time

    def answer_question(self, speaker_1_user_id, speaker_2_user_id, question, answer, category, prompt_question=None):
        if prompt_question is None:
            prompt_question = question
        (
            speaker_1_memories,
            speaker_1_graph_memories,
            speaker_1_memory_time,
            speaker_1_query_time,
            speaker_1_retrieval_time,
        ) = self.search_memory(speaker_1_user_id, question)
        (
            speaker_2_memories,
            speaker_2_graph_memories,
            speaker_2_memory_time,
            speaker_2_query_time,
            speaker_2_retrieval_time,
        ) = self.search_memory(speaker_2_user_id, question)

        search_1_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_1_memories]
        search_2_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_2_memories]

        template = Template(self.ANSWER_PROMPT)
        answer_prompt = template.render(
            speaker_1_user_id=speaker_1_user_id.split("_")[0],
            speaker_2_user_id=speaker_2_user_id.split("_")[0],
            speaker_1_memories=json.dumps(search_1_memory, indent=4),
            speaker_2_memories=json.dumps(search_2_memory, indent=4),
            speaker_1_graph_memories=json.dumps(speaker_1_graph_memories, indent=4),
            speaker_2_graph_memories=json.dumps(speaker_2_graph_memories, indent=4),
            question=prompt_question,
        )

        t1 = time.time()
        response = self.openai_client.chat.completions.create(
            model=os.getenv("MODEL"), messages=[{"role": "system", "content": answer_prompt}], temperature=0.0
        )
        t2 = time.time()
        response_time = t2 - t1
        return (
            response.choices[0].message.content,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            response_time,
            speaker_1_query_time,
            speaker_1_retrieval_time,
            speaker_2_query_time,
            speaker_2_retrieval_time,
        )

    def process_question(self, val, speaker_a_user_id, speaker_b_user_id):
        question = val.get("question", "")
        answer = val.get("answer", "")
        category = val.get("category", -1)
        evidence = val.get("evidence", [])
        adversarial_answer = val.get("adversarial_answer", "")
        prompt_question = question
        all_options = val.get("all_options", {})
        if isinstance(all_options, dict) and all_options:
            option_lines = []
            for key in ("(a)", "(b)", "(c)", "(d)"):
                if key in all_options:
                    option_lines.append(f"{key} {all_options[key]}")
            if option_lines:
                prompt_question = (
                    f"{question}\n\nOPTIONS:\n{chr(10).join(option_lines)}\n\n"
                    "IMPORTANT: This is a multiple-choice question. You MUST analyze the context and "
                    "select the BEST option. In your FINAL ANSWER, return ONLY the option letter "
                    "like (a), (b), (c), or (d), nothing else."
                )

        (
            response,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            response_time,
            speaker_1_query_time,
            speaker_1_retrieval_time,
            speaker_2_query_time,
            speaker_2_retrieval_time,
        ) = self.answer_question(
            speaker_a_user_id,
            speaker_b_user_id,
            question,
            answer,
            category,
            prompt_question=prompt_question,
        )

        result = {
            "question": question,
            "answer": answer,
            "category": category,
            "evidence": evidence,
            "response": response,
            "adversarial_answer": adversarial_answer,
            "speaker_1_memories": speaker_1_memories,
            "speaker_2_memories": speaker_2_memories,
            "num_speaker_1_memories": len(speaker_1_memories),
            "num_speaker_2_memories": len(speaker_2_memories),
            "speaker_1_memory_time": speaker_1_memory_time,
            "speaker_2_memory_time": speaker_2_memory_time,
            "speaker_1_graph_memories": speaker_1_graph_memories,
            "speaker_2_graph_memories": speaker_2_graph_memories,
            "response_time": response_time,
            "speaker_1_query_time": speaker_1_query_time,
            "speaker_1_retrieval_time": speaker_1_retrieval_time,
            "speaker_2_query_time": speaker_2_query_time,
            "speaker_2_retrieval_time": speaker_2_retrieval_time,
        }

        # Save results after each question is processed
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

        return result

    def process_data_file(self, file_path, start_idx: int = 0):
        with open(file_path, "r") as f:
            data = json.load(f)

        for idx, item in tqdm(enumerate(data, start=start_idx), total=len(data), desc="Processing conversations"):
            qa = item["qa"]
            conversation = item["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]

            speaker_a_user_id = f"{speaker_a}_{idx}"
            speaker_b_user_id = f"{speaker_b}_{idx}"

            for question_item in tqdm(
                qa, total=len(qa), desc=f"Processing questions for conversation {idx}", leave=False
            ):
                result = self.process_question(question_item, speaker_a_user_id, speaker_b_user_id)
                self.results[idx].append(result)

                # Save results after each question is processed
                with open(self.output_path, "w") as f:
                    json.dump(self.results, f, indent=4)

        # Final save at the end
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

    def process_questions_parallel(self, qa_list, speaker_a_user_id, speaker_b_user_id, max_workers=1):
        def process_single_question(val):
            result = self.process_question(val, speaker_a_user_id, speaker_b_user_id)
            # Save results after each question is processed
            with open(self.output_path, "w") as f:
                json.dump(self.results, f, indent=4)
            return result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                tqdm(executor.map(process_single_question, qa_list), total=len(qa_list), desc="Answering Questions")
            )

        # Final save at the end
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

        return results
