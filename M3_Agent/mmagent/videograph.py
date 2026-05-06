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
"""
This module defines the VideoGraph class, which is used to represent the video graph.
"""
import random
import logging

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import json
from .memory_processing import parse_video_caption

processing_config = json.load(open("configs/processing_config.json"))
MAX_RETRIES = processing_config["max_retries"]
# Configure logging
logger = logging.getLogger(__name__)

class VideoGraph:
    """
    This class defines the VideoGraph class, which is used to represent the video graph.
    """
    def __init__(self, max_img_embeddings=10, max_audio_embeddings=20, img_matching_threshold=0.3, audio_matching_threshold=0.6):
        """Initialize a video graph with nodes for faces, voices and text events.
        
        Args:
            max_img_embeddings: Maximum number of image embeddings per face node
            max_audio_embeddings: Maximum number of audio embeddings per voice node
        """
        self.nodes = {}  # node_id -> node object
        self.edges = {}  # (node_id1, node_id2) -> edge weight
        # Maintain ordered text nodes
        self.text_nodes = []  # List of text node IDs in insertion order
        
        self.text_nodes_by_clip = {}
        self.event_sequence_by_clip = {}
        
        self.max_img_embeddings = max_img_embeddings
        self.max_audio_embeddings = max_audio_embeddings
        
        self.img_matching_threshold = img_matching_threshold
        self.audio_matching_threshold = audio_matching_threshold
        
        self.next_node_id = 0

    class Node:
        def __init__(self, node_id, node_type):
            self.id = node_id
            self.type = node_type  # 'img', 'voice', 'episodic' or 'semantic'
            self.embeddings = []
            self.metadata = {}
            
    def _average_similarity(self, embeddings1, embeddings2):
        """Calculate average cosine similarity between two lists of embeddings."""
        if not embeddings1 or not embeddings2:
            return 0
            
        # Convert lists to numpy arrays
        emb1_array = np.array(embeddings1)
        emb2_array = np.array(embeddings2)
        
        # Calculate pairwise cosine similarities between all embeddings
        similarities = cosine_similarity(emb1_array, emb2_array)
        
        # Return mean of all pairwise similarities
        return np.mean(similarities)
    
    def _cluster_semantic_nodes(self, nodes, threshold=0.9):
        # cluster the nodes using cosine similarity
        # return a list of clusters, each cluster is a list of node ids
        # each node id is a string
        
        # calculate pairwise cosine similarities between all nodes
        embeddings = [self.nodes[node_id].embeddings[0] for node_id in nodes]

        similarities = cosine_similarity(embeddings)
        
        # Convert similarity matrix to distance matrix
        # For cosine similarity s, distance = 1 - s
        # This ensures that:
        # - Similar vectors (s close to 1) have small distance
        # - Dissimilar vectors (s close to 0) have large distance
        distances = 1 - similarities
        # filter out negative distances
        distances[distances < 0] = 0

        # print(distances)
        
        # cluster the nodes using DBSCAN
        # eps should now be the maximum allowed distance (1 - threshold)
        dbscan_model = DBSCAN(eps=(1 - threshold), min_samples=1, metric='precomputed')
        
        # get the clusters
        clusters = dbscan_model.fit_predict(distances)
        
        # return the clusters
        return clusters

    # Modification functions
    
    def add_img_node(self, imgs):
        """Add a new face node with initial image embedding(s).
        
        Args:
            img_embedding: Single embedding or list of embeddings
        """
        node = self.Node(self.next_node_id, 'img')
        
        img_embeddings = imgs['embeddings']
        node.embeddings.extend(img_embeddings[:self.max_img_embeddings])
        
        node.metadata['contents'] = imgs['contents']
        
        self.nodes[self.next_node_id] = node
        self.next_node_id += 1

        logger.debug(f"Image node added with ID {node.id}")

        return node.id

    def add_voice_node(self, audios):
        """Add a new voice node with initial audio embedding(s).
        
        Args:
            audio_embedding: Single embedding or list of embeddings
        """
        node = self.Node(self.next_node_id, 'voice')
        
        audio_embeddings = audios['embeddings']
        node.embeddings.extend(audio_embeddings[:self.max_audio_embeddings])
        
        node.metadata['contents'] = audios['contents']
        
        self.nodes[self.next_node_id] = node
        self.next_node_id += 1

        logger.debug(f"Voice node added with ID {node.id}")

        return node.id

    def add_text_node(self, text, clip_id, text_type='episodic'):
        """Add a new text node with episodic or semantic content.
        
        Args:
            text: Text content
            text_type: Type of text node ('episodic' or 'semantic')
        """
        if text_type not in ['episodic', 'semantic']:
            raise ValueError("text_type must be either 'episodic' or 'semantic'")

        node = self.Node(self.next_node_id, text_type)
        node.embeddings = text['embeddings']
        node.metadata['contents'] = text['contents']
        node.metadata['timestamp'] = clip_id
        
        self.nodes[self.next_node_id] = node
        self.text_nodes.append(node.id)  # Add to ordered list
        if clip_id not in self.text_nodes_by_clip:
            self.text_nodes_by_clip[clip_id] = []
        self.text_nodes_by_clip[clip_id].append(node.id)
        if text_type == 'episodic':
            if clip_id not in self.event_sequence_by_clip:
                self.event_sequence_by_clip[clip_id] = []
            self.event_sequence_by_clip[clip_id].append(node.id)

        self.next_node_id += 1

        logger.debug(f"Text node of type {text_type} added with ID {node.id} and contents: {text['contents']}")

        return node.id

    def update_node(self, node_id, update_info):
        """Update an existing node.
        
        Args:
            node_id: ID of target node
            update_info: Dictionary of update information
            
        Returns:
            Boolean indicating success
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")

        node = self.nodes[node_id]
        
        node.metadata['contents'].extend(update_info['contents'])
        
        embeddings = update_info['embeddings']

        if node.type == 'img':
            max_emb = self.max_img_embeddings
        elif node.type == 'voice':
            max_emb = self.max_audio_embeddings
        else:
            raise ValueError("Node type must be either 'img' or'voice' to add embeddings")

        # Combine existing and new embeddings
        all_embeddings = node.embeddings + embeddings

        # If exceeding max limit, randomly select embeddings
        if len(all_embeddings) > max_emb:
            node.embeddings = random.sample(all_embeddings, max_emb)
        else:
            node.embeddings = all_embeddings
        
        logger.debug(f"Node {node_id} updated with {len(embeddings)} embeddings")

        return True

    def add_edge(self, node_id1, node_id2, weight=1.0):
        """Add or update bidirectional weighted edges between two nodes.
        Text-to-text connections are not allowed between same type text nodes."""
        if (node_id1 in self.nodes and node_id2 in self.nodes and not (self.nodes[node_id1].type == self.nodes[node_id2].type and self.nodes[node_id1].type in ['episodic', 'semantic'])):
            # Add both directions with same weight
            self.edges[(node_id1, node_id2)] = weight
            self.edges[(node_id2, node_id1)] = weight
            logger.debug(f"Edge added between {node_id1} and {node_id2}")
            return True
        return False

    def update_edge_weight(self, node_id1, node_id2, delta_weight):
        """Update weight of existing bidirectional edge."""
        if (node_id1, node_id2) in self.edges:
            # Update both directions
            self.edges[(node_id1, node_id2)] += delta_weight
            self.edges[(node_id2, node_id1)] += delta_weight
            # if the weight is less than or equal to 0, remove the edge
            if self.edges[(node_id1, node_id2)] <= 0:
                del self.edges[(node_id1, node_id2)]
                del self.edges[(node_id2, node_id1)]
                logger.debug(f"Edge removed between {node_id1} and {node_id2}")
            return True
        return False

    def reinforce_node(self, node_id, delta_weight=1):
        """Reinforce all edges connected to the given node.
        
        Args:
            node_id: ID of the node to reinforce
            delta_weight: Amount to increase edge weights by (default: 1)
            
        Returns:
            int: Number of edges reinforced
        """
        if node_id not in self.nodes:
            return 0
            
        reinforced_count = 0
        for (n1, n2) in list(self.edges.keys()):  # Create a list to avoid modification during iteration
            if n1 == node_id or n2 == node_id:
                self.update_edge_weight(n1, n2, delta_weight)
                reinforced_count += 1

        logger.debug(f"{reinforced_count} edges reinforced for node {node_id}")
                
        return reinforced_count

    def weaken_node(self, node_id, delta_weight=1):
        """Weaken all edges connected to the given node.
        
        Args:
            node_id: ID of the node to weaken
            delta_weight: Amount to decrease edge weights by (default: 1)
            
        Returns:
            int: Number of edges weakened
        """
        if node_id not in self.nodes:
            return 0
            
        weakened_count = 0
        for (n1, n2) in list(self.edges.keys()):  # Create a list to avoid modification during iteration
            if n1 == node_id or n2 == node_id:
                self.update_edge_weight(n1, n2, -delta_weight)  # Use negative delta_weight to decrease
                weakened_count += 1

        logger.debug(f"{weakened_count} edges weakened for node {node_id}")
                
        return weakened_count
    
    # def summarize(self):
    #     new_semantic_memory = []
    #     for node in self.nodes.values():
    #         if node.type != "img" and node.type != "voice":
    #             continue
    #         connected_text_nodes = self.get_connected_nodes(node.id, type=['episodic', 'semantic'])
    #         connected_text_nodes_contents = [self.nodes[text_id].metadata['contents'][0] for text_id in connected_text_nodes]
    #         node_id = '<face_'+str(node.id)+'>' if node.type == 'img' else '<voice_'+str(node.id)+'>'
    #         input = [
    #             {
    #                 "type": "text",
    #                 "content": prompt_node_summarization.format(node_id=node_id, history_information=connected_text_nodes_contents),
    #             }
    #         ]
    #         messages = generate_messages(input)
    #         model = "gpt-4o-2024-11-20"
    #         summary = None
    #         for i in range(MAX_RETRIES):
    #             summary_string = get_response_with_retry(model, messages)[0]
    #             summary = validate_and_fix_python_list(summary_string)
    #             if summary is not None:
    #                 break
    #         if summary is None:
    #             raise Exception("Failed to generate summary")

    #         new_semantic_memory.extend(summary)
            
    #     process_captions(self, new_semantic_memory, type='semantic')
    
    def fix_collisions(self, node_id, mode='eq_only'):
        # detect collisions through clustering (one-node-cluster is allowed)
        # mode: argmax, dropout
        # argmax: select the node with the highest edge weight from each cluster
        # dropout: drop nodes by specific probability (according to relative edge weights) from each cluster
        
        # get all connected semantic nodes
        connected_nodes = self.get_connected_nodes(node_id, type=['semantic'])
        
        if len(connected_nodes) == 0:
            return []
        
        filtered_nodes = []
        
        if mode == 'eq_only':
            voice_face_mapping_node = None
            max_edge_weight = 0
            
            for node in connected_nodes:
                if self.nodes[node].metadata['contents'][0].lower().startswith("equivalence"):
                    equal_nodes = parse_video_caption(self, self.nodes[node].metadata['contents'][0])
                    # get the other node in the equal_nodes
                    equal_nodes = [n for n in equal_nodes if n[1] != node]
                    if not any(n[0] == 'face' for n in equal_nodes):
                        filtered_nodes.append(node)
                    else:
                        # find the node with the highest edge weight
                        edge_weight = self.edges[(node_id, node)]
                        if edge_weight > max_edge_weight:
                            max_edge_weight = edge_weight
                            voice_face_mapping_node = node
                        elif edge_weight == max_edge_weight:
                            # if the edge weight is the same, randomly select one
                            if random.random() < 0.5:
                                voice_face_mapping_node = node
                else:
                    filtered_nodes.append(node)
            if voice_face_mapping_node is not None:
                filtered_nodes.append(voice_face_mapping_node)
            return filtered_nodes

        # cluster the connected nodes
        clusters = self._cluster_semantic_nodes(connected_nodes)
        
        cluster_ids = list(set(clusters))
        cluster_ids.sort()
        
        for cluster_id in cluster_ids:
            cluster_nodes = [connected_nodes[i] for i in range(len(connected_nodes)) if clusters[i] == cluster_id]

            if len(cluster_nodes) == 1:
                filtered_nodes.append(cluster_nodes[0])
                continue
            if mode == 'argmax':
                # select the node with the highest edge weight
                max_edge_weight = 0
                max_edge_weight_node = None
                for node in cluster_nodes:
                    if self.edges[(node_id, node)] > max_edge_weight:
                        max_edge_weight = self.edges[(node_id, node)]
                        max_edge_weight_node = node
                filtered_nodes.append(max_edge_weight_node)
            elif mode == 'dropout':
                # Take each node with probability proportional to its edge weight
                edge_weights = [self.edges[(node_id, node)] for node in cluster_nodes]
                max_edge_weight = max(edge_weights)
                probabilities = [edge_weights[i] / max_edge_weight for i in range(len(edge_weights))]
                for i, node in enumerate(cluster_nodes):
                    if random.random() < probabilities[i]:
                        filtered_nodes.append(node)
            else:
                raise ValueError("Unknown mode")
            
            logger.debug('=' * 80)
            logger.debug(f"Cluster {cluster_id} has {len(cluster_nodes)} nodes: {cluster_nodes}")
            for node in cluster_nodes:
                logger.debug('-' * 80)
                logger.debug(f"Node {node} [edge weight]: {self.edges[(node_id, node)]}")
                logger.debug(f"Node {node} [content]: {self.nodes[node].metadata['contents'][0]}")

            logger.debug('*' * 80)
            logger.debug(f"Cluster {cluster_id} has {len(filtered_nodes)} nodes after filtering: {filtered_nodes}")
            for node in filtered_nodes:
                logger.debug('-' * 80)
                logger.debug(f"Node {node} [edge weight]: {self.edges[(node_id, node)]}")
                logger.debug(f"Node {node} [content]: {self.nodes[node].metadata['contents'][0]}")
                
        return filtered_nodes
    
    def refresh_equivalences(self):
        # Initialize disjoint set data structure
        parent = {}
        rank = {}
        
        def find(x):
            # Find root/representative of set with path compression
            if x not in parent:
                parent[x] = x
                rank[x] = 0
                return x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
            
        def union(x, y):
            # Union by rank
            px, py = find(x), find(y)
            if px == py:
                return
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
        
        # Process each face/voice node
        filtered_equivalence_nodes = []
        
        for node_id in self.nodes:
            if self.nodes[node_id].type == 'voice':        
                # every voice node should have no more than one voice-face mapping        
                # get filtered semantic nodes and their contents
                filtered_semantic_nodes = self.fix_collisions(node_id, mode='eq_only')
                
                filtered_equivalence_nodes.extend([node for node in filtered_semantic_nodes if self.nodes[node].metadata['contents'][0].lower().startswith("equivalence")])
            elif self.nodes[node_id].type == 'img':
                # no filtering for face nodes
                connected_semantic_nodes = self.get_connected_nodes(node_id, type=['semantic'])
                
                filtered_equivalence_nodes.extend([node for node in connected_semantic_nodes if self.nodes[node].metadata['contents'][0].lower().startswith("equivalence")])
            else:
                continue
            
        # remove duplicates and get the equivalences (equivalences can also be generated by LLM, not implemented here)
        filtered_equivalence_nodes = list(set(filtered_equivalence_nodes))
        equivalences = [self.nodes[node].metadata['contents'][0] for node in filtered_equivalence_nodes]
        
        # Add equivalent nodes to disjoint sets
        for equivalence in equivalences:
            entities = parse_video_caption(self, equivalence)
            if len(entities) >= 2:
                # Union all entities in this equivalence group
                anchor_node = entities[0][1]  # Get ID of first entity
                for entity in entities[1:]:
                    union(anchor_node, entity[1])

        # Group nodes by their representative (character)
        character_mappings = {}
        character_count = 0
        root_to_character = {}
        
        # Find all nodes that are in the disjoint sets
        for x in parent:
            root = find(x)
            tag = f"face_{x}" if self.nodes[x].type == 'img' else f"voice_{x}"
            if root not in root_to_character:
                root_to_character[root] = f"character_{character_count}"
                character_count += 1
            character = root_to_character[root]
            if character not in character_mappings:
                character_mappings[character] = []
            character_mappings[character].append(tag)
        
        for x in self.nodes:
            if x in parent or self.nodes[x].type not in ['img', 'voice']:
                continue
            root = find(x)
            tag = f"face_{x}" if self.nodes[x].type == 'img' else f"voice_{x}"
            if root not in root_to_character:
                root_to_character[root] = f"character_{character_count}"
                character_count += 1
            character = root_to_character[root]
            if character not in character_mappings:
                character_mappings[character] = []
            character_mappings[character].append(tag)
        
        # create reverse mapping
        reverse_character_mappings = {}
        for character, tags in character_mappings.items():
            for tag in tags:
                reverse_character_mappings[tag] = character
            
        self.character_mappings = character_mappings
        self.reverse_character_mappings = reverse_character_mappings

        logger.info(f"Found {character_count} characters")
                
    def order_character(self):
        """
        ablation study, without equivalence
        """
        character_mappings = {}
        reverse_character_mappings = {}
        character_count = 0

        for node_id in self.nodes:
            node_type = self.nodes[node_id].type
            if node_type not in ['img', 'voice']:
                continue

            character = f"character_{character_count}"
            character_count += 1

            tag = f"face_{node_id}" if node_type == 'img' else f"voice_{node_id}"

            character_mappings[character] = [tag]
            reverse_character_mappings[tag] = character

        self.character_mappings = character_mappings
        self.reverse_character_mappings = reverse_character_mappings

        logger.info(f"Assigned {character_count} characters (no equivalence used)")

    
    # Retrieval functions

    def get_connected_nodes(self, node_id, type=['img', 'voice', 'episodic', 'semantic']):
        """Get all nodes connected to given node."""
        connected = set()  # Use set to avoid duplicates due to bidirectional edges
        for (n1, n2), _ in self.edges.items():
            if n1 == node_id and self.nodes[n2].type in type:
                connected.add(n2)
            elif n2 == node_id and self.nodes[n1].type in type:
                connected.add(n1)
        return list(set(connected))

    def search_text_nodes(self, query_embeddings, range_nodes=[], mode="max"):
        """Search for text nodes using text embeddings.
        
        Args:
            query_embeddings: Query embeddings
            range_nodes: Optional list of nodes to restrict search to
            mode: Similarity calculation mode ('mean', 'sum', 'max', 'min')
            
        Returns:
            List of (node_id, similarity_score) tuples sorted by score
        """
        # Get target nodes
        if range_nodes:
            text_nodes = []
            for node_id in range_nodes:
                text_nodes.extend(self.get_connected_nodes(node_id, type=['episodic', 'semantic']))
            text_nodes = list(set(text_nodes))
        else:
            text_nodes = self.text_nodes
        target_nodes = [(node_id, self.nodes[node_id].embeddings) for node_id in text_nodes]
        
        # Calculate similarities in parallel using numpy
        node_ids, node_embeddings = zip(*target_nodes) if target_nodes else ([], [])
        if not node_ids:
            return []
            
        # Convert to numpy arrays for vectorized operations
        node_embeddings = np.array(node_embeddings)
        query_embeddings = np.array(query_embeddings)
        
        # Get shape parameters for better readability
        n_queries = query_embeddings.shape[0]
        n_nodes = node_embeddings.shape[0]
        n_embeddings = node_embeddings.shape[1]
        embedding_dim = node_embeddings.shape[-1]
        
        # Calculate similarities
        similarities = cosine_similarity(query_embeddings.reshape(-1, embedding_dim), node_embeddings.reshape(-1, embedding_dim))
        # Reshape back to (n_queries, n_nodes, n_embeddings)
        similarities = similarities.reshape(n_queries, n_nodes, n_embeddings)
        
        # Apply the specified mode
        if mode == "sum":
            # For sum mode: first average across embeddings, then sum across queries
            similarities = np.sum(np.mean(similarities, axis=2), axis=0)
        else:
            # For other modes: apply directly to all similarities
            if mode == "mean":
                similarities = np.mean(similarities, axis=(0, 2))
            elif mode == "max":
                similarities = np.max(similarities, axis=(0, 2))
            elif mode == "min":
                similarities = np.min(similarities, axis=(0, 2))
            else:
                raise ValueError(f"Invalid mode: {mode}")
        
        # Create results
        results = [(node_id, sim) for node_id, sim in zip(node_ids, similarities)]
        return sorted(results, key=lambda x: x[1], reverse=True)

    def search_img_nodes(self, img_info):
        """Search for face nodes using image embeddings."""
        # Get target nodes
        target_nodes = [(node_id, node.embeddings) for node_id, node in self.nodes.items() if node.type == 'img']
        
        # Calculate similarities in parallel using numpy
        node_ids, node_embeddings = zip(*target_nodes) if target_nodes else ([], [])
        if not node_ids:
            return []
            
        # Convert query embeddings to numpy array
        query_embeddings = np.array(img_info["embeddings"])
        embedding_dim = query_embeddings.shape[-1]
        
        # Calculate similarities for each node separately
        node_similarities = []
        for node_emb in node_embeddings:
            # Convert node embeddings to numpy array
            node_emb = np.array(node_emb)
            # Calculate similarities for this node
            node_sims = cosine_similarity(query_embeddings.reshape(-1, embedding_dim), node_emb.reshape(-1, embedding_dim))
            # Average across both query and embeddings
            node_sim = np.mean(node_sims)
            node_similarities.append(node_sim)
        
        # Create results with threshold
        results = [(node_id, sim) for node_id, sim in zip(node_ids, node_similarities) if sim >= self.img_matching_threshold]
        return sorted(results, key=lambda x: x[1], reverse=True)

    def search_voice_nodes(self, audio_info):
        """Search for voice nodes using audio embeddings."""
        # Get target nodes
        target_nodes = [(node_id, node.embeddings) for node_id, node in self.nodes.items() if node.type == 'voice']
        
        # Calculate similarities in parallel using numpy
        node_ids, node_embeddings = zip(*target_nodes) if target_nodes else ([], [])
        if not node_ids:
            return []
            
        # Convert query embeddings to numpy array
        query_embeddings = np.array(audio_info["embeddings"])
        embedding_dim = query_embeddings.shape[-1]
        
        # Calculate similarities for each node separately
        node_similarities = []
        for node_emb in node_embeddings:
            # Convert node embeddings to numpy array
            node_emb = np.array(node_emb)
            # Calculate similarities for this node
            node_sims = cosine_similarity(query_embeddings.reshape(-1, embedding_dim), node_emb.reshape(-1, embedding_dim))
            # Average across both query and embeddings
            node_sim = np.mean(node_sims)
            node_similarities.append(node_sim)
        
        # Create results with threshold
        results = [(node_id, sim) for node_id, sim in zip(node_ids, node_similarities) if sim >= self.audio_matching_threshold]
        return sorted(results, key=lambda x: x[1], reverse=True)

    def get_entity_info(self, anchor_nodes, drop_threshold=0.9):
        """Get information about entities by retrieving connected episodic and semantic nodes.
        
        This function takes a list of anchor nodes and finds all connected image and voice nodes as entity nodes.
        For each entity node, it retrieves all connected episodic nodes and semantic nodes. The semantic nodes
        are filtered to remove redundant information by comparing similarity between node embeddings.

        Args:
            anchor_nodes (list): List of node IDs to use as anchor points for finding entities
            drop_threshold (float): Similarity threshold above which semantic nodes are considered redundant (default: 0.9)
            
        Returns:
            list: List of node IDs for episodic and filtered semantic nodes connected to all found entities
            
        Raises:
            ValueError: If any found entity node ID is not found or is not an image/voice node
        """
        entity_nodes = set()
        for anchor_node in anchor_nodes:
            entity_nodes.update(self.get_connected_nodes(anchor_node, type=['voice', 'img']))
            
        entity_nodes = list(entity_nodes)
        
        info_nodes = []
        
        for entity_id in entity_nodes:
            if entity_id not in self.nodes or (self.nodes[entity_id].type not in ['img', 'voice']):
                raise ValueError(f"Node {entity_id} is not an image or voice node")
            connected_episodic_nodes = self.get_connected_nodes(entity_id, type=['episodic'])
            info_nodes.extend(connected_episodic_nodes)
            connected_semantic_nodes = self.get_connected_nodes(entity_id, type=['semantic'])
            
            # Filter semantic nodes by iteratively removing nodes with high similarity
            while True:
                # Check all pairs of remaining semantic nodes
                nodes_to_remove = set()
                for i, node_id1 in enumerate(connected_semantic_nodes):
                    for node_id2 in connected_semantic_nodes[i+1:]:
                        # Calculate similarity between node embeddings
                        similarity = self._average_similarity(
                            self.nodes[node_id1].embeddings,
                            self.nodes[node_id2].embeddings
                        )
                        
                        # If similarity exceeds threshold, remove the node with lower edge weight
                        if similarity > drop_threshold:
                            edge_weight1 = self.edges.get((entity_id, node_id1), 0)
                            edge_weight2 = self.edges.get((entity_id, node_id2), 0)
                            if edge_weight1 < edge_weight2:
                                nodes_to_remove.add(node_id1)
                            else:
                                nodes_to_remove.add(node_id2)
                
                # If no nodes need to be removed, we're done
                if not nodes_to_remove:
                    break
                    
                # Remove the identified nodes
                connected_semantic_nodes = [n for n in connected_semantic_nodes if n not in nodes_to_remove]
                
            info_nodes.extend(connected_semantic_nodes)
            
        return info_nodes
    
    # Visualization functions
    
    def print_faces(self, img_nodes, print_num=5):
        """Print faces for given image nodes in a grid layout with 9 faces per row.
        
        Args:
            img_nodes (list): List of image node IDs to display faces for
        """
        # Skip if no nodes to display
        if not img_nodes:
            return
            
        # Get all face images from the nodes with their node IDs
        face_images = []
        node_ids = []
        for node_id in img_nodes:
            if node_id not in self.nodes or self.nodes[node_id].type != 'img':
                continue
            face_base64_list = self.nodes[node_id].metadata['contents'][:print_num]
            for face_base64 in face_base64_list:
                # Convert base64 to PIL Image
                face_bytes = base64.b64decode(face_base64)
                face_img = Image.open(BytesIO(face_bytes))
                face_images.append(face_img)
                node_ids.append(node_id)
                
        # Skip if no faces found
        if not face_images:
            return
            
        # Calculate grid dimensions
        n_faces = len(face_images)
        n_cols = 9
        n_rows = (n_faces + n_cols - 1) // n_cols  # Ceiling division
        
        # Create figure and subplots
        _, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
            
        # Plot faces with node IDs as titles
        for idx, (img, node_id) in enumerate(zip(face_images, node_ids)):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].imshow(img)
            axes[row, col].set_title(f'Node {node_id}')
            axes[row, col].axis('off')
            
        # Hide empty subplots
        for idx in range(len(face_images), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
            
        plt.tight_layout()
        plt.show()
        
    def print_voice_nodes(self):
        for node_id, node in self.nodes.items():
            if node.type != 'voice':
                continue
            print("-"*50 + f"Voice Node {node_id}" + "-"*50)
            print(f"Contents: {node.metadata['contents']}")
            
            connected_text_nodes = self.get_connected_nodes(node_id, type=['episodic', 'semantic'])
            print(f"Connected Nodes: {connected_text_nodes}")
            connected_texts = [self.nodes[text_id].metadata['contents'] for text_id in connected_text_nodes]
            print(f"Connected Nodes Contents: {connected_texts}")
    
    def print_img_nodes(self, node_id=None):
        if node_id is not None:
            if self.nodes[node_id].type!= 'img':
                return

            print("-"*50 + f"Image Node {node_id}" + "-"*50)

            connected_text_nodes = self.get_connected_nodes(node_id, type=['episodic', 'semantic'])
            print(f"Connected Nodes: {connected_text_nodes}")
            connected_texts = [self.nodes[text_id].metadata['contents'] for text_id in connected_text_nodes]
            print(f"Connected Nodes Contents: {connected_texts}")

            self.print_faces([node_id])
        else:
            for node_id, node in self.nodes.items():
                if node.type != 'img':
                    continue
                print("-"*50 + f"Image Node {node_id}" + "-"*50)

                connected_text_nodes = self.get_connected_nodes(node_id, type=['episodic', 'semantic'])
                print(f"Connected Nodes: {connected_text_nodes}")
                connected_texts = [self.nodes[text_id].metadata['contents'] for text_id in connected_text_nodes]
                print(f"Connected Nodes Contents: {connected_texts}")

                self.print_faces([node_id])
            
    def visualize(self):
        """Visualize the video graph."""
        self.print_img_nodes()
        self.print_voice_nodes()
    
    def expand_route(self, route):
        if len(route) == 0:
            while True:
                # sample a random node
                node_id = random.choice(list(self.nodes.keys()))
                # if node_id is an an img or voice node, or an isolated text node, then continue
                if self.nodes[node_id].type in ['episodic', 'semantic']:
                    entities = parse_video_caption(self, self.nodes[node_id].metadata['contents'][0])
                    if len(entities) > 0:
                        return [node_id]
        # select a random node from the route
        node_id = random.choice(route)
        entities = parse_video_caption(self, self.nodes[node_id].metadata['contents'][0])
        anchor_entity = random.choice(entities)
        anchor_node_id = anchor_entity[1]
        new_node_ids = self.get_connected_nodes(anchor_node_id, type=['episodic', 'semantic'])
        new_node_ids = [node_id for node_id in new_node_ids if node_id not in route]
        if len(new_node_ids) == 0:
            return route
        new_node_id = random.choice(new_node_ids)
        return route + [new_node_id]
        
    
    def sample_a_route(self, length=3):
        route = []
        for i in range(length):
            route = self.expand_route(route)
        route_contents = [self.nodes[node_id].metadata['contents'][0] for node_id in route]
        return route, route_contents

    def truncate_memory_by_clip(self, clip_id, refresh=True):
        # truncate the memory by clip_id
        # remove all nodes that are after the clip_id
        # return the truncated memory
        
        # find the last node with clip_id
        last_node_id = None
        for node_id, node in self.nodes.items():
            if node.type in ['episodic', 'semantic'] and node.metadata['timestamp'] == clip_id:
                last_node_id = node_id
        if last_node_id is None:
            return
        # remove all nodes that are after the last_node_id
        to_del = []
        for node_id in self.nodes.keys():
            if node_id > last_node_id:
                to_del.append(node_id)
        for node_id in to_del:
            del self.nodes[node_id]
        # remove all edges that are after the last_node_id
        to_del = []
        for edge in self.edges.keys():
            if edge[0] > last_node_id or edge[1] > last_node_id:
                to_del.append(edge)
        for edge in to_del:
            del self.edges[edge]
        # remove all text nodes that are after the last_node_id
        to_del = []
        for node_id in self.text_nodes:
            if node_id > last_node_id:
                to_del.append(node_id)
        for node_id in to_del:
            self.text_nodes.remove(node_id)
        # update the text_nodes_by_clip
        to_del = []
        for clip, _ in self.text_nodes_by_clip.items():
            if clip > clip_id:
                to_del.append(clip)
        for clip in to_del:
            del self.text_nodes_by_clip[clip]
        # update the event_sequence_by_clip
        to_del = []
        for clip, _ in self.event_sequence_by_clip.items():
            if clip > clip_id:
                to_del.append(clip)
        for clip in to_del:
            del self.event_sequence_by_clip[clip]
        # update the equivalences
        if refresh:
            self.refresh_equivalences()
        return
    
    def prune_memory_by_node_type(self, node_type='semantic'):
        del_nodes = []
        for node_id, node in self.nodes.items():
            if node.type == node_type:
                del_nodes.append(node_id)
        for node_id in del_nodes:
            del self.nodes[node_id]
        # update the edges
        to_del = []
        for edge in self.edges.keys():
            if edge[0] in del_nodes or edge[1] in del_nodes:
                to_del.append(edge)
        for edge in to_del:
            del self.edges[edge]
        # update the text_nodes
        to_del = []
        for node_id in self.text_nodes:
            if node_id in del_nodes:
                to_del.append(node_id)
        for node_id in to_del:
            self.text_nodes.remove(node_id) 
        # update the text_nodes_by_clip
        for clip_id, text_nodes in self.text_nodes_by_clip.items():
            to_del = [node_id for node_id in text_nodes if node_id in del_nodes]
            for node_id in to_del:
                text_nodes.remove(node_id)
        # update the event_sequence_by_clip
        for clip_id, event_sequence in self.event_sequence_by_clip.items():
            to_del = [node_id for node_id in event_sequence if node_id in del_nodes]
            for node_id in to_del:
                event_sequence.remove(node_id)
        # update the equivalences
        self.refresh_equivalences()
        return