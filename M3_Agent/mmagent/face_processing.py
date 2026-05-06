# Copyright (2025) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import json
import os
import logging
from insightface.app import FaceAnalysis
from mmagent.src.face_extraction import extract_faces
from mmagent.src.face_clustering import cluster_faces
from mmagent.utils.video_processing import process_video_clip

processing_config = json.load(open("configs/processing_config.json"))
face_app = FaceAnalysis(name="buffalo_l")  # RetinaFace + ArcFace
face_app.prepare(ctx_id=-1)
cluster_size = processing_config["cluster_size"]
logger = logging.getLogger(__name__)

class Face:
    def __init__(self, frame_id, bounding_box, face_emb, cluster_id, extra_data):
        self.frame_id = frame_id
        self.bounding_box = bounding_box
        self.face_emb = face_emb
        self.cluster_id = cluster_id
        self.extra_data = extra_data

def get_face(frames):
    extracted_faces = extract_faces(face_app, frames)
    faces = [Face(frame_id=f['frame_id'], bounding_box=f['bounding_box'], face_emb=f['face_emb'], cluster_id=f['cluster_id'], extra_data=f['extra_data']) for f in extracted_faces]
    return faces

def cluster_face(faces):
    faces_json = [{'frame_id': f.frame_id, 'bounding_box': f.bounding_box, 'face_emb': f.face_emb, 'cluster_id': f.cluster_id, 'extra_data': f.extra_data} for f in faces]
    clustered_faces = cluster_faces(faces_json, 20, 0.5)
    faces = [Face(frame_id=f['frame_id'], bounding_box=f['bounding_box'], face_emb=f['face_emb'], cluster_id=f['cluster_id'], extra_data=f['extra_data']) for f in clustered_faces]
    return faces

def process_faces(video_graph, base64_frames, save_path, preprocessing=[]):
    """
    Process video frames to detect, cluster and track faces.

    Args:
        video_graph: Graph object to store face embeddings and relationships
        base64_frames (list): List of base64 encoded video frames to process

    Returns:
        dict: Mapping of face IDs to lists of face detections, where each face detection contains:
            - frame_id (int): Frame number where face was detected
            - bounding_box (list): Face bounding box coordinates [x1,y1,x2,y2]
            - face_emb (list): Face embedding vector
            - cluster_id (int): ID of face cluster from initial clustering
            - extra_data (dict): Additional face detection metadata
            - matched_node (int): ID of matched face node in video graph

    The function:
    1. Splits frames into batches and processes them in parallel to detect faces
    2. Clusters detected faces to group similar faces together
    3. Converts face detections to JSON format
    4. Updates video graph with face embeddings and relationships
    5. Returns mapping of face IDs to face detections
    """
    batch_size = max(len(base64_frames) // cluster_size, 4)
    
    def _process_batch(params):
        """
        Process a batch of video frames to detect faces.

        Args:
            params (tuple): A tuple containing:
                - frames (list): List of video frames to process
                - offset (int): Frame offset to add to detected face frame IDs

        Returns:
            list: List of detected faces with adjusted frame IDs

        The function:
        1. Extracts frames and offset from input params
        2. Creates face detection request for the batch
        3. Gets face detection response from service
        4. Adjusts frame IDs of detected faces by adding offset
        5. Returns list of detected faces
        """
        frames = params[0]
        offset = params[1]
        faces = get_face(frames)
        for face in faces:
            face.frame_id += offset
        return faces

    def get_embeddings(base64_frames, batch_size):
        num_batches = (len(base64_frames) + batch_size - 1) // batch_size
        batched_frames = [
            (base64_frames[i * batch_size : (i + 1) * batch_size], i * batch_size)
            for i in range(num_batches)
        ]

        faces = []

        # parallel process the batches
        with ThreadPoolExecutor(max_workers=num_batches) as executor:
            for batch_faces in tqdm(
                executor.map(_process_batch, batched_frames), total=num_batches
            ):
                faces.extend(batch_faces)

        faces = cluster_face(faces)
        return faces

    def establish_mapping(faces, key="cluster_id", filter=None):
        mapping = {}
        for face in faces:
            if key not in face.keys():
                raise ValueError(f"key {key} not found in faces")
            if filter and not filter(face):
                continue
            id = face[key]
            if id not in mapping:
                mapping[id] = []
            mapping[id].append(face)
        # sort the faces in each cluster by detection score and quality score
        max_faces = processing_config["max_faces_per_character"]
        for id in mapping:
            mapping[id] = sorted(
                mapping[id],
                key=lambda x: (
                    float(x["extra_data"]["face_detection_score"]),
                    float(x["extra_data"]["face_quality_score"]),
                ),
                reverse=True,
            )[:max_faces]
        return mapping

    def filter_score_based(face):
        dthresh = processing_config["face_detection_score_threshold"]
        qthresh = processing_config["face_quality_score_threshold"]
        return float(face["extra_data"]["face_detection_score"]) > dthresh and float(face["extra_data"]["face_quality_score"]) > qthresh

    def update_videograph(video_graph, tempid2faces):
        id2faces = {}
        for tempid, faces in tempid2faces.items():
            if tempid == -1:
                continue
            if len(faces) == 0:
                continue
            face_info = {
                "embeddings": [face["face_emb"] for face in faces],
                "contents": [face["extra_data"]["face_base64"] for face in faces],
            }
            matched_nodes = video_graph.search_img_nodes(face_info)
            if len(matched_nodes) > 0:
                matched_node = matched_nodes[0][0]
                video_graph.update_node(matched_node, face_info)
                for face in faces:
                    face["matched_node"] = matched_node
            else:
                matched_node = video_graph.add_img_node(face_info)
                for face in faces:
                    face["matched_node"] = matched_node
            if matched_node not in id2faces:
                id2faces[matched_node] = []
            id2faces[matched_node].extend(faces)
        
        max_faces = processing_config["max_faces_per_character"]
        for id, faces in id2faces.items():
            id2faces[id] = sorted(
                faces,
                key=lambda x: (
                    float(x["extra_data"]["face_detection_score"]),
                    float(x["extra_data"]["face_quality_score"]),
                ),
                reverse=True
            )[:max_faces]

        return id2faces
    
    # Check if intermediate results exist
    try:
        with open(save_path, "r") as f:
            faces_json = json.load(f)
    except Exception as e:
        faces = get_embeddings(base64_frames, batch_size)

        faces_json = [
            {
                "frame_id": face.frame_id,
                "bounding_box": face.bounding_box,
                "face_emb": face.face_emb,
                "cluster_id": int(face.cluster_id),
                "extra_data": face.extra_data,
            }
            for face in faces
        ]

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
        with open(save_path, "w") as f:
            json.dump(faces_json, f)
            
    if "face" in preprocessing:
        return

    if len(faces_json) == 0:
        return {}

    tempid2faces = establish_mapping(faces_json, key="cluster_id", filter=filter_score_based)
    if len(tempid2faces) == 0:
        return {}

    id2faces = update_videograph(video_graph, tempid2faces)

    return id2faces

def main():
    _, frames, _ = process_video_clip(
        "/mnt/hdfs/foundation/longlin.kylin/mmagent/data/video_clips/CZ_2/-OCrS_r5GHc/11.mp4"
    )
    process_faces(None, frames, "data/temp/face_detection_results.json", preprocessing=["face"])

if __name__ == "__main__":
    main()