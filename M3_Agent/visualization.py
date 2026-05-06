import pickle
import argparse
from mmagent.retrieve import translate
from typing import Optional

ONLY = None    
MAX_LEN = None   
SHOW_FACES = False    

def truncate(text: str, max_len: Optional[int]) -> str:
    if not max_len or len(text) <= max_len:
        return text
    return text[:max_len] + "…"

def print_clip_full(vg, clip_id: int,
                    only: Optional[str] = None,
                    max_len: Optional[int] = None,
                    show_faces: bool = True, file=None) -> None:

    node_ids = vg.text_nodes_by_clip.get(clip_id)
    if not node_ids:
        file.write(f"[Warning] clip_id={clip_id} does not exist or the clip does not have a text node\n")
        return

    file.write(f"\n======= Clip {clip_id} Memory =======\n")


    for nid in node_ids:
        node = vg.nodes[nid]
        if only and node.type != only:
            continue
        contents = node.metadata.get("contents", [])
        contents = translate(vg, contents)
        contents = [truncate(c, max_len) for c in contents]
        metadata_str = ", ".join([f"{key}: {value}" for key, value in node.metadata.items()])
        file.write(f"[{node.type:^8}] id={nid:<4}| " + metadata_str + "\n")


    face_nodes, voice_nodes = set(), set()
    for nid in node_ids:
        face_nodes.update(vg.get_connected_nodes(nid, type=['img']))
        voice_nodes.update(vg.get_connected_nodes(nid, type=['voice']))

    face_nodes, voice_nodes = sorted(face_nodes), sorted(voice_nodes)


    if face_nodes:
        file.write(f"\n======= Clip {clip_id} Face ({len(face_nodes)} face nodes in total) =======\n")
        if show_faces:
            vg.print_faces(face_nodes, print_num=3)
        else:
            for fid in face_nodes:
                imgs = vg.nodes[fid].metadata.get("contents", [])
                file.write(f"[face] id={fid:<4} | face_num={len(imgs)}"
                           f"| base64: {imgs[0][:50]+'…' if imgs else 'N/A'}\n")
    else:
        file.write("\n(no related face)\n")


    if voice_nodes:
        file.write(f"\n======= Clip {clip_id} Voice ({len(voice_nodes)} voice nodes in total) =======\n")
        for vid in voice_nodes:
            audios = vg.nodes[vid].metadata.get("contents", [])
            if audios:
                file.write(f"[voice] id={vid:<4} | voice_num={len(audios)}\n")
                for audio in audios:
                    file.write(f"| Content: {truncate(str(audio), max_len)}\n")  
            else:
                file.write(f"[voice] id={vid:<4} | voice_num=0 | Content: N/A\n")
    else:
        file.write("\n(no related voice)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mem_path", type=str, default="/data/memory_graphs/robot/bedroom_01.pkl")
    args = parser.parse_args()


    with open(args.mem_path, "rb") as f:
        graph = pickle.load(f)
        graph.refresh_equivalences()

    with open("YPzzGPXW8fs.txt", "w") as file:

        for clip_id in range(len(graph.text_nodes_by_clip)):  
            print_clip_full(graph,
                            clip_id=clip_id,
                            only=ONLY,
                            max_len=MAX_LEN,
                            show_faces=SHOW_FACES,
                            file=file)

        attributes_to_write = {
            'edges': graph.edges,
            'text_nodes': graph.text_nodes,
            'text_nodes_by_clip': graph.text_nodes_by_clip,
            'event_sequence_by_clip': graph.event_sequence_by_clip,
            'max_img_embeddings': graph.max_img_embeddings,
            'max_audio_embeddings': graph.max_audio_embeddings,
            'img_matching_threshold': graph.img_matching_threshold,
            'audio_matching_threshold': graph.audio_matching_threshold,
            'next_node_id': graph.next_node_id,
            'character_mappings': graph.character_mappings,
            'reverse_character_mappings': graph.reverse_character_mappings,
        }

        for attr, value in attributes_to_write.items():
            file.write(f"{attr}: {value}\n")


