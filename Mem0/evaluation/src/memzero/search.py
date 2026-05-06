import json
import math
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

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
                search_results = self._search_with_anchor_time_rerank(user_id, query)
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

    @staticmethod
    def _round_half_up(value):
        return int(math.floor(float(value) + 0.5))

    @staticmethod
    def _parse_payload_time(payload):
        raw_ts = payload.get("timestamp")
        if isinstance(raw_ts, str) and raw_ts.strip():
            ts = re.sub(r"\b(am|pm)\b", lambda m: m.group(1).upper(), raw_ts.strip(), flags=re.IGNORECASE)
            for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
                try:
                    return datetime.strptime(ts, fmt).replace(tzinfo=timezone.utc)
                except ValueError:
                    continue

        raw_created = payload.get("created_at")
        if isinstance(raw_created, str) and raw_created.strip():
            try:
                dt = datetime.fromisoformat(raw_created.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except ValueError:
                pass
        return None

    def _search_with_anchor_time_rerank(self, user_id, query):
        try:
            vector_store = getattr(self.memory, "vector_store", None)
            embedding_model = getattr(self.memory, "embedding_model", None)
            client = getattr(vector_store, "client", None)
            collection_name = getattr(vector_store, "collection_name", None)
            create_filter = getattr(vector_store, "_create_filter", None)
            if (
                vector_store is None
                or embedding_model is None
                or client is None
                or not collection_name
                or not callable(create_filter)
            ):
                raise RuntimeError("Vector store direct access is unavailable.")

            t0 = time.time()
            query_vec = embedding_model.embed(query, "search")
            t1 = time.time()

            top_k = max(1, int(self.top_k))
            fetch_limit = max(top_k, top_k * 4)
            query_filter = create_filter({"user_id": user_id})
            hits = client.query_points(
                collection_name=collection_name,
                query=query_vec,
                query_filter=query_filter,
                limit=fetch_limit,
                with_payload=True,
                with_vectors=False,
            )
            t2 = time.time()

            points = list(hits.points)
            if not points:
                return {"results": [], "query_time": t1 - t0, "retrieval_time": t2 - t1}

            # 1) Anchors: the first 0.2 * top_k points (already sorted by query similarity).
            anchor_count = max(1, min(len(points), self._round_half_up(0.2 * top_k)))
            anchor_indices = list(range(anchor_count))

            # 2) Build timestamp positions (same timestamp -> same position).
            parsed_times = []
            raw_timestamps = []
            sort_keys = []
            group_keys = []
            for i, point in enumerate(points):
                payload = point.payload if isinstance(point.payload, dict) else {}
                dt = self._parse_payload_time(payload)
                raw_ts = payload.get("timestamp")
                raw_norm = raw_ts.strip() if isinstance(raw_ts, str) and raw_ts.strip() else ""
                parsed_times.append(dt)
                raw_timestamps.append(raw_norm)

                if dt is not None:
                    sort_keys.append((0, dt.timestamp(), raw_norm, i))
                    group_keys.append(("dt", dt.isoformat()))
                elif raw_norm:
                    sort_keys.append((1, 0.0, raw_norm, i))
                    group_keys.append(("raw", raw_norm))
                else:
                    sort_keys.append((2, 0.0, "", i))
                    group_keys.append(("idx", str(i)))

            ordered_indices = sorted(range(len(points)), key=lambda idx: sort_keys[idx])
            position_by_idx = {}
            current_pos = 0
            prev_group = None
            for idx in ordered_indices:
                group = group_keys[idx]
                if group != prev_group:
                    current_pos += 1
                    prev_group = group
                position_by_idx[idx] = current_pos

            # 3) Time-neighborhood score + final score.
            scored = []
            anchor_den = float(anchor_count)
            for i, point in enumerate(points):
                pos_i = position_by_idx.get(i, 0)
                time_sum = 0.0
                for a_idx in anchor_indices:
                    pos_a = position_by_idx.get(a_idx, 0)
                    dist = abs(pos_a - pos_i)
                    time_sum += math.exp(-1.0 * float(dist))
                time_adj_score = time_sum / anchor_den

                query_sim = float(point.score) if point.score is not None else 0.0
                final_score = 0.9 * query_sim + 0.1 * time_adj_score
                scored.append((i, final_score, query_sim, time_adj_score))

            # 4) Keep top_k by final score, then order returned memories by query similarity.
            scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
            selected = scored[:top_k]
            selected.sort(key=lambda x: x[2], reverse=True)

            formatted = []
            for idx, final_score, query_sim, time_adj_score in selected:
                point = points[idx]
                payload = point.payload if isinstance(point.payload, dict) else {}
                formatted.append(
                    {
                        "id": point.id,
                        "memory": payload.get("data", ""),
                        "score": query_sim,
                        "metadata": {
                            "timestamp": payload.get("timestamp"),
                            "time_adjacency_score": time_adj_score,
                            "final_score": final_score,
                        },
                    }
                )
            return {"results": formatted, "query_time": t1 - t0, "retrieval_time": t2 - t1}
        except Exception:
            return self.memory.search(query, user_id=user_id, limit=self.top_k)

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
