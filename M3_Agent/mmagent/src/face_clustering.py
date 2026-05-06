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
import numpy as np
import hdbscan

def cluster_faces(faces, min_cluster_size=2, distance_threshold=0.5):
    face_embeddings = []
    face_types = []
    face_indices = []
    face_detection_scores = []
    face_quality_scores = []

    for i, face in enumerate(faces):
        face_embeddings.append(face["face_emb"])
        face_types.append(face["extra_data"]["face_type"])
        face_indices.append(i)
        face_detection_scores.append(float(face["extra_data"]["face_detection_score"]))
        face_quality_scores.append(float(face["extra_data"]["face_quality_score"]))

    face_embeddings = np.array(face_embeddings)

    detection_threshold = 0.8
    quality_threshold = 20
    good_mask = [(face_detection_scores[i] >= detection_threshold and face_quality_scores[i] >= quality_threshold) for i in range(len(face_types))]
    bad_mask = [(face_detection_scores[i] < detection_threshold or face_quality_scores[i] < quality_threshold) for i in range(len(face_types))]

    good_embeddings = face_embeddings[good_mask]
    bad_embeddings = face_embeddings[bad_mask]

    all_labels = [-1] * len(face_types)
    max_label = -1

    if len(good_embeddings) >= min_cluster_size:
        good_similarity = np.dot(good_embeddings, good_embeddings.T)
        good_distances = 1 - good_similarity
        good_distances = np.maximum(good_distances, 0).astype(np.float64)

        good_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, metric="precomputed"
        )
        good_labels = good_clusterer.fit_predict(good_distances)
        max_label = (
            max(good_labels)
            if len(good_labels) > 0 and max(good_labels) > -1
            else -1
        )

        good_idx = 0
        for i, is_good in enumerate(good_mask):
            if is_good:
                all_labels[i] = good_labels[good_idx]
                good_idx += 1

    result_faces = []
    for i, face in enumerate(faces):
        face_copy = face.copy()
        face_copy["cluster_id"] = all_labels[i]
        result_faces.append(face_copy)

    return result_faces