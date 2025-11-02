import config
from text_utils import normalize_text
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util
from thefuzz import process as fuzzy_process
import torch

class Document:
    """
    Representa um único documento de texto, lidando com todo o
    pré-processamento (chunking, embedding) de forma encapsulada.
    """
    def __init__(self, raw_text: str, embedding_model: SentenceTransformer):
        self.raw_text = raw_text
        self.normalized_text = normalize_text(raw_text)
        self.normalized_lines = self.normalized_text.splitlines()
        self.raw_lines = self.raw_text.splitlines()

        # 1. Chunking
        self.chunks, self.chunk_lines = self._chunk_text(raw_text)
        if not self.chunks:
            raise ValueError("Documento está vazio ou não pôde ser dividido.")
        
        # 2. Heurística Posicional
        self.first_chunk = self.chunks[0]
        
        # 3. Chunks Semânticos (para busca)
        self.semantic_chunks = self.chunks[1:]
        
        # 4. Embeddings (calculados na inicialização)
        self.embedding_model = embedding_model
        self.chunk_embeddings = None
        if self.semantic_chunks:
            self.chunk_embeddings = self.embedding_model.encode(
                self.semantic_chunks,
                convert_to_tensor=True,
                show_progress_bar=False,
                device=config.DEVICE
            )

    def _chunk_text(self, text: str) -> list[str]:
        """Divide o texto em chunks, mantendo a integridade estrutural."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        
        chunks = text_splitter.split_text(text)

        chunk_lines = []
        current_index = 0

        for chunk in chunks:
            start_index = text.find(chunk, current_index)
            end_index = start_index + len(chunk)
            start_line = text.count('\n', 0, start_index)
            end_line = text.count('\n', 0, end_index)
            current_index = start_index # Update current_index to search for the next chunk
            chunk_lines.append((start_line, end_line))


        return chunks, chunk_lines

    def get_keyword_line_numbers(self, key: str) -> (set[str], int):
        """
        Encontra snippets de texto usando fuzzy matching da chave.
        Retorna (set_de_snippets, score_maximo).
        """
        normalized_key = key.replace("_", " ")
        snippets = set()
        max_score = 0
        
        matches = fuzzy_process.extractBests(
            normalized_key, 
            self.normalized_lines, 
            limit=config.SNIPPET_K_KEYWORDS, 
            score_cutoff=70
        )
        
        if not matches:
            return set(), 0

        max_score = matches[0][1]
        
        line_numbers = []
        for (line, score) in matches:
            try:
                line_number = self.normalized_lines.index(line)
                line_numbers.append(line_number)
            except:
                continue

        return line_numbers, max_score

    def get_intersecting_chunks(self, line_numbers: list[int], tolerance: int = 1) -> set[str]:
        """Retorna chunks que intersectam com as linhas fornecidas."""
        intersecting_chunks = []
        intersecting_chunks_indices = []
        for chunk_idx, (chunk, (start_line, end_line)) in enumerate(zip(self.semantic_chunks, self.chunk_lines[1:])):
            for line_number in line_numbers:
                if start_line <= line_number + tolerance and line_number - tolerance <= end_line:
                    intersecting_chunks.append(chunk)
                    intersecting_chunks_indices.append(chunk_idx)
                    break

        return intersecting_chunks, intersecting_chunks_indices

    def get_semantic_chunk_indices(self, query: str, line_numbers: list[int]) -> set[str]:
        """Encontra chunks de texto usando busca por similaridade semântica."""
        if self.chunk_embeddings is None or not self.semantic_chunks:
            return set()


        query_embedding = self.embedding_model.encode(
            query, 
            convert_to_tensor=True, 
            show_progress_bar=False,
            device=config.DEVICE
        )

        if len(line_numbers) == 0:
            indices = list(range(len(self.semantic_chunks)))
            embeddings = self.chunk_embeddings
        else:
            indices = self.get_intersecting_chunks(line_numbers)[1]
            embeddings = self.chunk_embeddings[indices,:]

        cos_scores = util.cos_sim(query_embedding, embeddings)[0]
        
        top_results = torch.topk(cos_scores, k=min(config.SNIPPET_K_SEMANTIC, len(embeddings)))

        return [indices[idx] for idx in top_results[1]]
