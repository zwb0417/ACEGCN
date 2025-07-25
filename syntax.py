import numpy as np
import networkx as nx
from ltp import LTP


class BatchSyntaxMaskProcessor:
    def __init__(self, ltp_path='./ltp/', custom_dict_path=None):
        self.ltp = LTP(ltp_path)
        if custom_dict_path:
            self.load_custom_dict(custom_dict_path)

    def load_custom_dict(self, file_path):
        custom_words = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                custom_words = [line.strip().split()[0] for line in f if line.strip()]
            for word in custom_words:
                self.ltp.add_words(word)
        except Exception as e:
            print(f"加载词典失败: {e}")

    def process_knowledge(self, sentence, knowledge_input):
        processed = ''.join([t for t in knowledge_input if t not in ['[CLS]', '[PAD]']])
        added = self.find_added_content(sentence, processed)
        return added, processed

    def find_added_content(self, original, processed):
        o_chars, p_chars = list(original), list(processed)
        added, current = [], []
        o_idx = 0
        for i, c in enumerate(p_chars):
            if o_idx < len(o_chars) and c == o_chars[o_idx]:
                if current:
                    added.append(''.join(current))
                    current = []
                o_idx += 1
            else:
                current.append(c)
        if current:
            added.append(''.join(current))
        return added if added else [""]

    def segment_text(self, texts):
        if not texts:
            return [[]]
        return [self.ltp.pipeline([t], tasks=("cws",)).cws[0] for t in texts]

    def find_entities(self, processed_seg, added_content, original_words):
        entities, processed_text = [], ''.join(processed_seg)
        for content in added_content:
            if not content:
                entities.append(None)
                continue
            pos = processed_text.find(content)
            if pos < 0:
                entities.append(None)
                continue
            prev_pos = pos - 1
            if prev_pos < 0:
                entities.append(None)
                continue
            found = False
            for word in processed_seg:
                w_start = processed_text.find(word)
                w_end = w_start + len(word) - 1
                if w_start <= prev_pos <= w_end:
                    entities.append(word if word in original_words else None)
                    found = True
                    break
            if not found:
                entities.append(None)
        return entities

    def generate_syntax_mask(self, sentence, max_threshold=3):
        result = self.ltp.pipeline([sentence], tasks=("cws", "pos", "dep"))
        words, dep = result.cws[0], result.dep[0]
        heads, rels = dep['head'], dep['label']

        G = nx.Graph()
        for i, w in enumerate(words):
            G.add_node(i + 1, word=w)
        for i, (h, r) in enumerate(zip(heads, rels)):
            if h != 0:
                G.add_edge(h, i + 1, relation=r)

        paths = dict(nx.all_pairs_shortest_path_length(G))
        n = len(words)
        mask = np.full((max_threshold, n, n), -99999.9999)

        for k in range(max_threshold):
            threshold = k + 1
            for i in range(n):
                for j in range(n):
                    u, v = i + 1, j + 1
                    if v in paths.get(u, {}):
                        dist = paths[u][v]
                        if dist <= threshold:
                            mask[k, i, j] = 0.0

        return mask, words, paths

    def insert_into_mask(self, mask_matrix, original_words, entities, added_segs, insert_val=6):

        max_threshold = mask_matrix.shape[0]
        original_n = mask_matrix.shape[1]

        total_insertions = sum(len(seg) for seg in added_segs)
        final_n = original_n + total_insertions

        final_matrix = np.zeros((max_threshold, final_n, final_n))

        final_words = original_words.copy()

        insertion_points = []
        current_offset = 0

        for entity, seg in zip(entities, added_segs):
            if not seg:
                continue

            if entity is None:
                insert_pos = 0
            else:
                if entity in final_words:
                    insert_pos = final_words.index(entity) + 1
                else:
                    insert_pos = len(final_words)

            insertion_points.append((insert_pos, len(seg)))

            for word in reversed(seg):
                final_words.insert(insert_pos, word)

        for k in range(max_threshold):
            layer = mask_matrix[k]

            new_layer = np.full((final_n, final_n), insert_val)

            pos_map = {}
            for i in range(original_n):
                new_i = i
                for pos, size in insertion_points:
                    if i >= pos:
                        new_i += size
                pos_map[i] = new_i

            for i in range(original_n):
                new_i = pos_map[i]
                for j in range(original_n):
                    new_j = pos_map[j]
                    new_layer[new_i, new_j] = layer[i, j]

            final_matrix[k] = new_layer

        return final_matrix, final_words

    def process_threshold_layers(self, processed_matrix, original_words, final_words, entities, added_segs):
        entity_list = []
        knowledge_list = []

        for i, word in enumerate(final_words):
            if word in original_words:
                entity_list.append(i)
            else:
                knowledge_list.append(i)

        knowledge_to_entity = {}
        knowledge_groups = []

        current_knowledge_start = len(original_words)

        for entity, seg in zip(entities, added_segs):
            if not seg:
                continue

            if entity is None:
                entity_pos = None
            else:
                entity_pos = final_words.index(entity) if entity in final_words else None

            group_indices = []
            for word in seg:
                if word in final_words:
                    knowledge_pos = final_words.index(word)
                    knowledge_to_entity[knowledge_pos] = entity_pos
                    group_indices.append(knowledge_pos)

            if group_indices:
                knowledge_groups.append(group_indices)

            current_knowledge_start += len(seg)

        processed = np.copy(processed_matrix)

        for k in range(processed.shape[0]):
            layer = processed[k]

            for i in range(layer.shape[0]):
                if i in entity_list:
                    for j in knowledge_list:
                        entity_j = knowledge_to_entity.get(j)

                        if entity_j is not None:
                            if layer[i, entity_j] == 0:
                                layer[i, j] = 0.0
                            else:
                                layer[i, j] = -99999.9999

                elif i in knowledge_list:
                    entity_i = knowledge_to_entity.get(i)

                    for j in range(layer.shape[1]):
                        if j == entity_i:
                            layer[i, j] = 0.0
                        elif j in knowledge_list:
                            same_group = False
                            for group in knowledge_groups:
                                if i in group and j in group:
                                    layer[i, j] = 0.0
                                    same_group = True
                                    break
                            if not same_group:
                                layer[i, j] = -99999.9999
                        else:
                            layer[i, j] = -99999.9999
            layer[layer == 6] = -99999.9999

            processed[k] = layer

        return processed

    def convert_words_to_tokens(self, refined_matrix, final_words, max_matrix_size=None):

        tokens = []
        word_to_token = {}

        for i, word in enumerate(final_words):
            if len(word) == 1:
                tokens.append(word)
                word_to_token[i] = [len(tokens) - 1]
            else:
                chars = list(word)
                token_indices = []
                for char in chars:
                    tokens.append(char)
                    token_indices.append(len(tokens) - 1)
                word_to_token[i] = token_indices

        actual_token_count = len(tokens)

        if max_matrix_size is None:
            matrix_size = actual_token_count
        else:
            matrix_size = max_matrix_size

        num_thresholds = refined_matrix.shape[0]
        token_matrix = np.full((num_thresholds, matrix_size, matrix_size), -99999.9999)

        for k in range(num_thresholds):
            for i, word_idx in enumerate(final_words):
                token_indices = word_to_token[i]

                for j in range(len(final_words)):
                    target_tokens = word_to_token[j]

                    original_val = refined_matrix[k, i, j]

                    for ti in token_indices:
                        if ti >= matrix_size:
                            continue
                        for tj in target_tokens:
                            if tj >= matrix_size:
                                continue
                            if ti != tj:
                                token_matrix[k, ti, tj] = original_val

            np.fill_diagonal(token_matrix[k], -99999.9999)

        if len(tokens) > matrix_size:
            tokens = tokens[:matrix_size]
        elif len(tokens) < matrix_size:
            pad_count = matrix_size - len(tokens)
            tokens.extend([f'[PAD{i}]' for i in range(pad_count)])

        return token_matrix, tokens

    def print_mask(self, mask, words, threshold, title="掩码矩阵"):
        max_word_len = max(len(str(w)) for w in words)
        max_val_len = 9
        cell_width = max(max_word_len, max_val_len) + 2

        total_width = cell_width * (len(words) + 1) + 1

        print(f"\n{title} (阈值={threshold}):")
        print("-" * total_width)
        print(f"{'':<{cell_width}}|", end="")
        for w in words:
            print(f"{w:^{cell_width}}|", end="")
        print()

        print("-" * total_width)

        for i, row in enumerate(mask):
            print(f"{words[i]:<{cell_width}}|", end="")
            for val in row:
                print(f"{val:^{cell_width}.4f}|", end="")
            print()
        print("-" * total_width)

    def process_batch(self, sentences, knowledge_batch, max_threshold=3, max_matrix_size=30):

        batch_size = len(sentences)
        if batch_size != len(knowledge_batch):
            raise ValueError("句子数量与知识批次数量不匹配")

        batch_matrices = []
        all_tokens = []

        for i in range(batch_size):
            added_content, proc_knowledge = self.process_knowledge(sentences[i], knowledge_batch[i])
            added_seg = self.segment_text(added_content)

            proc_seg = self.segment_text([proc_knowledge])[0]
            orig_seg = self.segment_text([sentences[i]])[0]
            entities = self.find_entities(proc_seg, added_content, orig_seg)
            mask_matrix, words, _ = self.generate_syntax_mask(sentences[i], max_threshold)

            processed_matrix, final_words = self.insert_into_mask(
                mask_matrix, words, entities, added_seg, insert_val=6
            )

            refined_matrix = self.process_threshold_layers(
                processed_matrix,
                words,
                final_words,
                entities,
                added_seg
            )

            token_matrix, tokens = self.convert_words_to_tokens(
                refined_matrix,
                final_words,
                max_matrix_size
            )

            batch_matrices.append(token_matrix)
            all_tokens.append(tokens)

        final_batch_matrix = np.stack(batch_matrices, axis=0)

        return final_batch_matrix, all_tokens

def build_multi_sentence_mask_matrices(sentences, knowledge, max_threshold=3, max_matrix_size=30,
                                       ltp_path='./ltp/', custom_dict_path=None):

    processor = BatchSyntaxMaskProcessor(ltp_path=ltp_path, custom_dict_path=custom_dict_path)

    batch_matrix, all_tokens = processor.process_batch(
        sentences,
        knowledge,
        max_threshold=max_threshold,
        max_matrix_size=max_matrix_size
    )

    return batch_matrix, all_tokens
