from __future__ import annotations
import itertools
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import config
from sentence_transformers import SentenceTransformer, util
from thefuzz import fuzz
import torch
from text_utils import fuzzy_match, normalize_text

@dataclass(frozen=True, order=True)
class Line:
    """
    Uma classe imutável que representa uma linha de texto composta por
    boxes de texto (spans).
    """
    idx: int
    boxes: List[TextBox] = field(compare=False)
    table_id: int = field(default=None, compare=False)

    @property
    def text(self) -> str:
        return " ".join([b.text for b in self.boxes])

    @property
    def y0(self) -> float:
        return min(b.y0 for b in self.boxes)

    @property
    def y1(self) -> float:
        return max(b.y1 for b in self.boxes)

    def __len__(self):
        return len(self.boxes)

    def serialize_layout(self) -> str:
        s = f"[TABLE {self.table_id} ROW] " if self.table_id is not None else ""
        for i,b in enumerate(self.boxes):
            s += f"{b.text} "
            if b.col_index is not None:
                s += f"{{col: {b.col_index}}} "
            if i < len(self.boxes)-1:
                s+= "| "

        return s


@dataclass(frozen=True, order=True)
class TextBox:
    """
    Uma classe imutável e "sortable" que representa a menor unidade
    atômica de texto com propriedades de layout uniformes (um 'span' do Fitz).
    """
    # Define 'y0' como o principal critério de ordenação para 'order=True'
    y0: float 
    x0: float
    y1: float
    x1: float
    text: str = field(compare=False)
    font_size: float = field(compare=False)
    font: str = field(compare=False)
    key: str = field(default=None, compare=False)
    line_idx: int = field(default=None, compare=False)
    value: str = field(default=None, compare=False)
    col_index: int = field(default=None, compare=False)

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def shrinked_y0(self) -> float:
        return self.center_y + 0.6 * (self.y0 - self.center_y)

    @property
    def shrinked_y1(self) -> float:
        return self.center_y + 0.6 * (self.y1 - self.center_y)

    @property
    def shrinked_x0(self) -> float:
        return self.center_x + 0.6 * (self.x0 - self.center_x)

    @property
    def shrinked_x1(self) -> float:
        return self.center_x + 0.6 * (self.x1 - self.center_x)

    def is_right_of(self, other: TextBox) -> bool:
        """Verifica se esta caixa está à direita de outra."""
        # IMPROVEMENT: Adicionar uma tolerância vertical (ex: +/- 5px) para 
        # permitir "quase" na mesma linha.
        return self.shrinked_x0 > other.shrinked_x1 and abs(self.center_y - other.center_y) < config.ROW_TOLERANCE

    def is_below(self, other: TextBox) -> bool:
        """Verifica se esta caixa está abaixo de outra."""
        # IMPROVEMENT: Adicionar uma tolerância horizontal para permitir 
        # "quase" na mesma coluna.
        eps = min((other.x1-other.x0)/len(other.text), (self.x1-self.x0)/len(self.text))
        return (self.shrinked_y0 > other.shrinked_y1 and (
                abs(self.center_x - other.center_x) < (other.shrinked_x1 - other.shrinked_x0)
                or self.x0 - eps <= other.x0 <= self.x0 + eps
                or self.x1 - eps <= other.x1 <= self.x1 + eps
        ))

    def is_similar(self, other: TextBox) -> bool:
        """Verifica se esta caixa é similar a outra (baseado em posição e tamanho da fonte)."""
        return abs(self.font_size - other.font_size) <= 0.5 and self.font==other.font

    def merge_with(self, other: TextBox) -> TextBox:
        """Retorna uma nova TextBox que é a fusão desta com outra."""
        new_text = self.text + " " + other.text
        new_x0 = min(self.x0, other.x0)
        new_y0 = min(self.y0, other.y0)
        new_x1 = max(self.x1, other.x1)
        new_y1 = max(self.y1, other.y1)
        new_font_size = self.font_size
        new_font = self.font

        return TextBox(
            text=new_text,
            x0=new_x0,
            y0=new_y0,
            x1=new_x1,
            y1=new_y1,
            font_size=new_font_size,
            font=new_font,
            line_idx=self.line_idx
        )


class Document:
    """
    Representa um documento de página única, pré-processado com
    informações de layout do Fitz.
    """
    def __init__(self, fitz_page_dict: dict):
        self.boxes = self._parse_spans(fitz_page_dict)
        self.lines = self._group_spans_into_lines(self.boxes)
        self._merge_broken_boxes()
        if not self.boxes:
            raise ValueError("Documento não contém blocos de texto.")
        
        # Armazena as caixas ordenadas por ordem de leitura (top-down, left-right)
        self.full_text = " ".join([b.text for b in self.boxes])
        
        # Propriedades de layout calculadas uma vez
        self.page_height = fitz_page_dict.get("height", None)
        self.page_width = fitz_page_dict.get("width", None)
        self.max_font_size = max(b.font_size for b in self.boxes)
        self.min_font_size = min(b.font_size for b in self.boxes)
        self.tables_dict = {}  # Mapeia table_id para listas de índices de linhas
    def _parse_spans(self, page_dict: dict) -> List[TextBox]:
        """Converte o 'dict' do Fitz em uma lista plana de objetos TextBox."""
        boxes = []
        for block in page_dict.get("blocks", []):
            if block.get("type") == 0: # 0 é um bloco de texto
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if not span.get("text", "").strip():
                            continue # Ignora spans vazios
                            
                        bbox = span.get("bbox", None)
                        boxes.append(TextBox(
                            text=span.get("text", "").strip(),
                            x0=bbox[0],
                            y0=bbox[1],
                            x1=bbox[2],
                            y1=bbox[3],
                            font_size=round(span.get("size", None), 2),
                            font=span.get("font", None),
                        ))
        return boxes

    def _group_spans_into_lines(self, boxes: List[TextBox]) -> List[Line]:
        """
        Agrupa spans (TextBoxes) em Linhas baseado em interseção vertical.
        Espera que 'boxes' já esteja ordenado por y0, x0.
        """
        if not boxes:
            return []
        sorted_boxes = sorted(boxes, key=lambda b: (b.shrinked_y0, b.shrinked_x0))

        lines: List[Line] = []
        current_line_boxes: List[TextBox] = [sorted_boxes[0]]
        
        # Define o BBox vertical da linha atual
        current_line_y0 = sorted_boxes[0].shrinked_y0
        current_line_y1 = sorted_boxes[0].shrinked_y1
        object.__setattr__(sorted_boxes[0], "line_idx", 0)  # Define o índice da linha"

        for box in sorted_boxes[1:]:
            # Testa a interseção vertical usando sua lógica:
            # (max(y0_a, y0_b) < min(y1_a, y1_b))
            does_intersect = max(current_line_y0, box.shrinked_y0) < min(current_line_y1, box.shrinked_y1)

            if does_intersect:
                # A caixa pertence à linha atual. Adiciona e expande o BBox vertical da linha.
                current_line_boxes.append(box)
                current_line_y0 = min(current_line_y0, box.shrinked_y0)
                current_line_y1 = max(current_line_y1, box.shrinked_y1)
            else:
                # A caixa é de uma nova linha. Finaliza a linha anterior.
                current_line_boxes.sort(key=lambda b: b.x0)  # Ordena por x0 dentro da linha
                new_line = Line(idx=len(lines), boxes=current_line_boxes)
                lines.append(new_line)
                
                # Começa a nova linha
                current_line_boxes = [box]
                current_line_y0 = box.shrinked_y0
                current_line_y1 = box.shrinked_y1
        
            object.__setattr__(box, "line_idx", len(lines))  # Define o índice da linha
        # Finaliza a última linha
        if current_line_boxes:
            new_line = Line(idx=len(lines), boxes=current_line_boxes)
            lines.append(new_line)
            
        return lines

    def _merge_broken_boxes(self):
        for lines in self.lines:
            line_boxes = lines.boxes
            boxes_to_remove = []
            boxes_to_add = []
            for b1,b2 in itertools.permutations(line_boxes, r=2):
                if b2.is_below(b1):
                    boxes_to_add.append(b1.merge_with(b2))
                    boxes_to_remove.append((b1,b2))
            for i,box in enumerate(boxes_to_remove):
                idx1 = self.boxes.index(box[0])
                idx2 = self.boxes.index(box[1])
                idx1_line = line_boxes.index(box[0])
                idx2_line = line_boxes.index(box[1])
                self.boxes.remove(box[0])
                self.boxes.remove(box[1])
                line_boxes.remove(box[0])
                line_boxes.remove(box[1])
                idx_to_insert = idx1 if idx1 < idx2 else idx1-1
                self.boxes.insert(idx_to_insert, boxes_to_add[i])
                idx_to_insert = idx1_line if idx1_line < idx2_line else idx1_line-1
                line_boxes.insert(idx_to_insert, boxes_to_add[i])
                del box


    def serialize_layout_for_llm(self) -> str:
        """
        Cria uma representação textual do layout para o prompt do LLM.
        
        IMPROVEMENT: Isso pode ser muito verboso. Uma versão melhor poderia
        enviar apenas caixas dentro de uma 'janela' relevante, ou
        simplificar a serialização (ex: "Linha 1: [Texto 1] [Texto 2]").
        """
        return "\n".join([
            f"[text: '{b.text}', x0: {b.x0:.0f}, y0: {b.y0:.0f}, size: {b.font_size:.0f}]" 
            for b in self.boxes
        ])

    def serialize_layout(self) -> str:
        """
        Cria uma representação textual do layout para o prompt do LLM.
        
        IMPROVEMENT: Isso pode ser muito verboso. Uma versão melhor poderia
        enviar apenas caixas dentro de uma 'janela' relevante, ou
        simplificar a serialização (ex: "Linha 1: [Texto 1] [Texto 2]").
        """
        s = ""
        for line in self.lines:
            s+= line.serialize_layout() + "\n"

        return s


    # --- Métodos de Extração (O Motor de Regras) ---

    def find_close_box(self, box: TextBox, direction: str = 'right') -> Optional[TextBox]:
        """
        Encontra o valor mais próximo na 'direction' especificada.
        """
            
        # IMPROVEMENT: Esta é uma busca O(N). Para documentos enormes,
        # um K-D Tree ou R-Tree pré-calculado sobre as caixas
        # tornaria esta busca O(log N). Para uma página, O(N) é aceitável.
        candidates = []
        if direction == 'right':
            candidates = [b for b in self.boxes if b.is_right_of(box)]
            if not candidates: return None
            # Retorna o mais próximo horizontalmente
            return min(candidates, key=lambda b: b.x0)
            
        elif direction == 'below':
            candidates = [b for b in self.boxes if b.is_below(box) and box.line_idx<=b.line_idx<= box.line_idx + 2]
            if not candidates: return None
            # Retorna o mais próximo verticalmente
            return min(candidates, key=lambda b: b.y0)
        
        return None # Direção não suportada

    def find_text_by_font(self, op: str = 'largest') -> Optional[TextBox]:
        """Encontra o texto baseado no tamanho da fonte (maior ou menor)."""
        if op == 'largest':
            target_size = self.max_font_size
        elif op == 'smallest':
            target_size = self.min_font_size
        else:
            return None
            
        # Retorna o primeiro span que bate com o tamanho
        # IMPROVEMENT: Pode haver múltiplos. Isso poderia retornar uma
        # lista ou concatenar todos os spans de texto com esse tamanho.
        for box in self.boxes:
            if box.font_size == target_size:
                return box
        return None

    def find_box_by_position(self, corner: str) -> Optional[TextBox]:
        """Encontra a caixa de texto mais próxima de um canto da página."""
        page_w = self.page_bbox[2]
        page_h = self.page_bbox[3]
        
        if corner == 'top_left':
            # Pondera y ligeiramente mais para priorizar a linha superior
            return min(self.boxes, key=lambda b: b.x0 * 0.8 + b.y0)
        elif corner == 'bottom_right':
            # Pondera a distância de x/y do canto oposto
            return min(self.boxes, key=lambda b: (page_w - b.x1) + (page_h - b.y1))
        
        # IMPROVEMENT: Adicionar 'top_right' e 'bottom_left'
        return None

    def find_table_row_values(self, header_box: TextBox) -> List[TextBox]:
        """
        Encontra uma linha de cabeçalho e retorna os valores da linha abaixo.
        Esta é a lógica de tabela mais complexa.
        """
        # 1. Encontra todas as caixas na mesma linha do cabeçalho
        header_row = [b for b in self.boxes if abs(b.center_y - header_box.center_y) < config.ROW_TOLERANCE]
        header_row.sort() # Ordena por x0
        
        # 2. Define as "colunas" com base nos centros x dos cabeçalhos
        columns_x_centers = [b.center_x for b in header_row]
        
        # 3. Encontra a próxima linha de texto abaixo
        first_box_below = None
        min_y_dist = float('inf')
        
        for b in self.boxes:
            if b.y0 > header_box.y1 + config.ROW_TOLERANCE / 2: # Se está abaixo
                dist = b.y0 - header_box.y1
                if dist < min_y_dist:
                    min_y_dist = dist
                    first_box_below = b
        
        if not first_box_below:
            return []
            
        # 4. Pega todas as caixas nessa linha de valor
        value_row = [b for b in self.boxes if abs(b.center_y - first_box_below.center_y) < config.ROW_TOLERANCE]
        
        # 5. Mapeia os valores para as colunas de cabeçalho
        # IMPROVEMENT: Esta é uma heurística simples. Uma lógica mais
        # robusta usaria 'find_nearest' para o centro x, ou lidaria
        # com valores que abrangem múltiplas colunas.
        mapped_values = []
        for val_box in value_row:
            # Encontra a coluna (índice) à qual este valor pertence
            col_index = min(range(len(columns_x_centers)), 
                            key=lambda i: abs(val_box.center_x - columns_x_centers[i]))
            mapped_values.append((col_index, val_box))
            
        # Retorna as caixas de valor ordenadas pela coluna
        mapped_values.sort()
        return [box for _, box in mapped_values]
    

    def find_tables(self):
        for line in self.lines:
            boxes = line.boxes
            if len(line) < 2:
                continue
            below_boxes = [self.find_close_box(box, "below") for box in boxes]
            line_indices = [below_box.line_idx if below_box is not None else -1 for below_box in below_boxes]
            most_common_line = max(set(line_indices), key=line_indices.count)
            if line_indices.count(most_common_line)/len(line_indices) < 0.5 or line_indices.count(-1)/len(line_indices) > 0.5:
                continue
            line_size_ratio = len(self.lines[most_common_line])/len(line)
            if line_size_ratio < 0.7 or line_size_ratio > 1.5:
                continue 
            if line.table_id is None:
                object.__setattr__(line, 'table_id', len(self.tables_dict))
                object.__setattr__(self.lines[most_common_line], 'table_id', len(self.tables_dict))
                self.tables_dict[len(self.tables_dict)] = [line.idx, most_common_line]
            else: 
                object.__setattr__(self.lines[most_common_line], 'table_id', line.table_id)
                self.tables_dict[line.table_id].append(most_common_line)
            col_index = 0
            for box in boxes:
                below_box = self.find_close_box(box, "below")
                if below_box is not None and below_box in self.lines[most_common_line].boxes:
                    col_index = box.col_index if box.col_index is not None else col_index
                    object.__setattr__(box, 'col_index', col_index)
                    object.__setattr__(below_box, 'col_index', col_index)
                    col_index += 1

    def embed_spans(self, embedding_model: SentenceTransformer):
        """
        Calcula e armazena os embeddings para cada TextBox (span)
        individual no documento.
        """
        if not self.boxes: # self.boxes é a lista plana de todos os spans
            self.span_embeddings = None
            print("AVISO: Documento não contém spans (TextBoxes) para embedar.")
            return
        
        # Pega o texto de cada span individual
        span_texts = [box.text for box in self.boxes]
        
        self.span_embeddings = embedding_model.encode(
            span_texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            device=config.DEVICE
        )

    def get_semantic_matching_lines(self, embedding_model: SentenceTransformer, query: str, k=config.SNIPPET_K_SEMANTIC) -> set[str]:
        """
        Encontra os 'k' spans mais semanticamente similares à query e
        retorna o texto das LINHAS pais a que eles pertencem.
        """
        
        if self.span_embeddings is None or not self.boxes:
            return set()

        query_embedding = embedding_model.encode(
            query, 
            convert_to_tensor=True, 
            show_progress_bar=False,
            device=config.DEVICE
        )
        
        # Busca por similaridade de cosseno contra os SPANS
        cos_scores = util.cos_sim(query_embedding, self.span_embeddings)[0]
        
        # Pega os 'k' melhores resultados (índices dos spans)
        top_results = torch.topk(cos_scores, k=min(k, len(self.boxes)))
        
        # Agora, mapeia os spans de volta para suas linhas pai
        matched_lines = set()
        for idx in top_results[1]:
            if cos_scores[idx] < config.SEMANTIC_MATCH_THRESHOLD:
                continue
            matched_span = self.boxes[idx]
            
            # Usa nosso mapa para encontrar a linha pai
            parent_line = self.lines[matched_span.line_idx]
            if parent_line is not None:
                matched_lines.add(parent_line)

        return matched_lines

    def get_keyword_matching_lines(self, key: str) -> (set[str], int):
        """
        Encontra snippets de texto usando fuzzy matching da chave contra as LINHAS.
        Retorna (set_de_textos_de_linha, score_maximo).
        """
        k = config.SNIPPET_K_KEYWORDS
        
        normalized_key = key.replace("_", " ")
        snippets = set()
        
        # Pega os textos das linhas para a busca
        scored_boxes = []
        for box in self.boxes:
            box_text = normalize_text(box.text)
            key_candidates = []
            segments = box_text.split(":")
            max_score = 0
            for i,segment in enumerate(segments):
                # First segment should be the label, last the value
                # Segments in between should be half label, half value
                if i==0:
                    key_candidates = [segment]
                elif i==len(segments)-1:
                    break
                else:
                    subsegments = segment.split(" ")
                    for i in range(1, len(subsegments)):
                        key_candidate = " ".join(subsegments[i:])
                        key_candidates.append(key_candidate)
                max_score = max(max_score, max(fuzz.ratio(normalized_key, kc) for kc in key_candidates))
                
            if max_score >= 80:
                scored_boxes.append((box, max_score))
        if not scored_boxes:
            return snippets
        
        scored_boxes.sort(key=lambda x: x[1], reverse=True)
        top_matches = scored_boxes[:k]
        top_lines = set(self.lines[box.line_idx] for (box, _) in top_matches)
                
        return top_lines

    def get_context_for_line(self, line: Line) -> Set[int]:
        """
        "Infla" uma única linha âncora para seu bloco de contexto lógico,
        parando em grandes lacunas verticais ou fronteiras de tabela.
        
        Retorna: Um conjunto (Set) de strings, onde cada string é o texto
                 de uma linha no bloco de contexto.
        """
        
        # LÓGICA 1: Se a linha está em uma tabela, retorne a tabela inteira.
        if line.table_id is not None:
            try:
                # Encontra o objeto Table e serializa ele
                table_lines = self.tables_dict.get(line.table_id, [])
                # Retorna o texto de todas as linhas da tabela
                return {l for l in table_lines}
            except Exception as e:
                print(f"AVISO: Falha ao buscar tabela para linha {line.idx}. {e}")
                # Falha: continue para a lógica de janela,
                # mas trate como uma linha normal (table_id = None).
                pass
        
        # --- LÓGICA 2 e 3: ADAPTATIVAS (Sua nova proposta) ---
        
        # Constante para a heurística de espaçamento
        # (Um gap > 150% da altura da linha é uma quebra de bloco)
        GAP_MULTIPLIER = 1.5
        
        context_lines_set = {line} # Começa com a própria âncora
        current_index = line.idx
        
        # --- 1. Crescer para BAIXO ---
        # Itera pelas linhas abaixo da âncora
        for i in range(current_index + 1, len(self.lines)):
            prev_line = self.lines[i - 1]
            neighbor_line = self.lines[i]
            
            # LÓGICA 3: Parar se entrarmos em uma tabela
            if neighbor_line.table_id is not None:
                break
            
            # LÓGICA 2: Heurística de Espaçamento
            gap = neighbor_line.y0 - prev_line.y1
            line_height = prev_line.y1 - prev_line.y0
            if line_height <= 0: line_height = 1 # Evita divisão por zero
            
            if gap > (line_height * GAP_MULTIPLIER):
                break # Encontrou uma grande lacuna; fim do bloco
            
            # Se passou: a linha é conectada
            context_lines_set.add(neighbor_line)

        # --- 2. Crescer para CIMA ---
        # Itera pelas linhas acima da âncora (em ordem reversa)
        for i in range(current_index - 1, -1, -1):
            next_line = self.lines[i + 1]
            neighbor_line = self.lines[i]
            
            # LÓGICA 3: Parar se entrarmos em uma tabela
            if neighbor_line.table_id is not None:
                break
                
            # LÓGICA 2: Heurística de Espaçamento
            gap = next_line.y0 - neighbor_line.y1
            line_height = neighbor_line.y1 - neighbor_line.y0
            if line_height <= 0: line_height = 1
            
            if gap > (line_height * GAP_MULTIPLIER):
                break # Encontrou uma grande lacuna; fim do bloco
            
            # Se passou: a linha é conectada
            context_lines_set.add(neighbor_line)
            
        # Retorna o texto de todas as linhas do bloco
        return {l.idx for l in context_lines_set}

    def _find_boxes_in_area(self, area: str) -> List[TextBox]:
        """
        Helper privado para encontrar todas as caixas (spans) em uma
        área geométrica da página (ex: 'top_left_quadrant').
        """
        # Define os limites das áreas. 0.4 (40%) é um bom
        # limiar para "canto" sem ser muito restritivo.
        x_left_limit = self.page_width * 0.25
        x_right_limit = self.page_width * 0.75
        y_top_limit = self.page_height * 0.25
        y_bottom_limit = self.page_height * 0.75

        found_boxes = []
        if area == 'top_left':
            found_boxes = [
                b for b in self.boxes 
                if b.center_x < x_left_limit and b.center_y < y_top_limit
            ]
        elif area == 'top_right':
            found_boxes = [
                b for b in self.boxes 
                if b.center_x > x_right_limit and b.center_y < y_top_limit
            ]
        elif area == 'bottom_left':
            found_boxes = [
                b for b in self.boxes 
                if b.center_x < x_left_limit and b.center_y > y_bottom_limit
            ]
        elif area == 'bottom_right':
            found_boxes = [
                b for b in self.boxes 
                if b.center_x > x_right_limit and b.center_y > y_bottom_limit
            ]
        elif area == 'top':
             found_boxes = [b for b in self.boxes if b.center_y < y_top_limit]
        elif area == 'bottom':
             found_boxes = [b for b in self.boxes if b.center_y > y_bottom_limit]
        
        # Ordena as caixas encontradas pela ordem de leitura
        return sorted(found_boxes)

    def get_positional_matching_lines(self, description: str) -> set[str]:
        """
        Analisa a descrição em busca de palavras-chave posicionais
        (ex: "canto superior esquerdo") e retorna as linhas
        encontradas nessas áreas.
        """
        norm_desc = normalize_text(description)
        target_area = None
        
        # Mapeia palavras-chave da descrição para nossas áreas
        if "canto superior esquerdo" in norm_desc:
            target_area = 'top_left'
        elif "canto superior direito" in norm_desc:
            target_area = 'top_right'
        elif "canto inferior esquerdo" in norm_desc:
            target_area = 'bottom_left'
        elif "canto inferior direito" in norm_desc:
            target_area = 'bottom_right'
        elif "parte superior" in norm_desc or "no topo" in norm_desc:
            target_area = 'top'
        elif "parte inferior" in norm_desc or "rodape" in norm_desc:
            target_area = 'bottom'
            
        if not target_area:
            return set() # Nenhuma instrução posicional encontrada

        # Encontra as caixas (spans) nessa área
        matched_boxes = self._find_boxes_in_area(target_area)
        
        # Converte as caixas em seus textos de Linha pai
        matched_lines = set()
        for box in matched_boxes:
            parent_line_id = box.line_idx
            if parent_line_id is not None:
                matched_lines.add(self.lines[parent_line_id])
        
        return matched_lines
