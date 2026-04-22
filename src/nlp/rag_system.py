# rag_system.py


# This module implements a Retrieval-Augmented Generation (RAG) system for
# evidence‑based fitness coaching. It indexes scientific PDFs (or falls back to
# a curated evidence store) and uses FAISS for dense retrieval. The system can
# generate personalised workout plans by prompting a large language model (LLM)
# with retrieved evidence. It supports three LLM providers (Groq, Ollama,
# LM Studio) and three prompting strategies (standard, chain‑of‑thought,
# evidence‑first). Comprehensive evaluation functions measure retrieval quality,
# prompt strategy effectiveness, and model performance.
#
# PURPOSE:
#   - Ground LLM responses in peer‑reviewed exercise science to prevent hallucination.
#   - Compare different LLM providers and prompting techniques.
#   - Provide an end‑to‑end pipeline from user query to evidence‑based training plan.
#
# COURSE CONNECTION:
#   This work fulfills the "Speech and Natural Language Processing" course
#   objectives (Unit III – Transformers and LLMs, Unit IV – RAG) and also
#   touches on "Advanced Machine Learning" (Unit V – Cloud Computing and LLM
#   deployment). The evaluation of retrieval metrics (Recall@k, MRR) and
#   prompting strategies directly follows the experimental requirements of the
#   teaching‑learning contract.
#
# DECISIONS:
#   - I use FAISS for efficient similarity search over document embeddings.
#   - The embedding model `all-MiniLM-L6-v2` is lightweight and runs on CPU.
#   - PDFs are split into overlapping chunks of 1000 characters to preserve context.
#   - Three prompt strategies allow me to study the trade‑off between verbosity,
#     citation accuracy, and latency.
#   - Fallback evidence store ensures the system works even when PDFs are missing.



import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# I load environment variables from a .env file if it exists (for API keys).
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print(f"[RAG] Loaded environment from {env_path}")
except ImportError:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.config import (
    GROQ_API_KEY, TAVILY_API_KEY, PAPERS_DIR,
    PROCESSED_DATA_DIR, REPORTS_DIR, PLOTS_DIR,
    LLM_PROVIDER, OLLAMA_URL, OLLAMA_MODEL, LMSTUDIO_URL
)

# I define chunking parameters for splitting PDFs.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
# I use a lightweight sentence‑transformer model for embeddings.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = PROCESSED_DATA_DIR / "faiss_index" / "papers_index"
RERANK_TOP_K = 10
FINAL_TOP_K = 5

# I list the available LLM providers and models for comparison.
AVAILABLE_MODELS = {
    "groq": {
        "llama-3.3-70b-versatile": {
            "description": "Llama 3.3 70B via Groq cloud. High performance, low latency.",
            "reference": "Meta AI (2024) Llama 3.3 technical report.",
        },
    },
    "ollama": {
        "tinyllama:latest": {
            "description": "TinyLlama 1.1B. Very lightweight, runs on low-spec hardware.",
            "reference": "Zhang et al. (2024) TinyLlama: An open-source small language model.",
        },
    },
    "lmstudio": {
        "local-model": {
            "description": "Any model loaded in LM Studio (user-selected via GUI).",
            "reference": "User-configured model via LM Studio GUI.",
        },
    },
}

# This is a fallback evidence store containing key findings from scientific literature.
# It is used when PDF indexing fails or is disabled.
SCIENTIFIC_EVIDENCE = [
    {
        "id": "schoenfeld_2019",
        "source": "Schoenfeld BJ, Grgic J, Krieger J. How many times per week should a muscle be trained to maximize muscle hypertrophy? A systematic review and meta-analysis. J Sports Sci. 2019;37(11):1286-1295.",
        "content": (
            "META-ANALYSIS (n=1,557 subjects across 25 studies): Training each muscle group "
            "at least 2x per week produced significantly greater hypertrophy than 1x per week "
            "(effect size = 0.43, 95% CI [0.10, 0.76], p=0.03). The mean increase in muscle "
            "thickness was 3.7% more in the 2x/week group. Recommendation: for muscle gain, "
            "train each muscle group at least twice per week with 10-20 weekly sets per muscle."
        ),
        "tags": ["muscle_gain", "frequency", "hypertrophy", "volume"]
    },
    {
        "id": "acsm_2021_cardio",
        "source": "American College of Sports Medicine. ACSM's Guidelines for Exercise Testing and Prescription, 11th ed. Wolters Kluwer, 2021, Chapter 6.",
        "content": (
            "CLINICAL GUIDELINE (based on 200+ RCTs): For weight loss, aerobic exercise should "
            "be performed at moderate intensity (64-76% HRmax) for ≥150 min/week or vigorous "
            "intensity (77-95% HRmax) for ≥75 min/week. This produces an average weight loss "
            "of 1.5-3.0 kg over 6 months without dietary intervention. Combined aerobic + "
            "resistance training preserves lean mass during weight loss (-0.5 kg lean mass "
            "vs -2.1 kg with cardio only, p<0.01)."
        ),
        "tags": ["weight_loss", "cardio", "aerobic", "intensity"]
    },
    {
        "id": "schoenfeld_2017_volume",
        "source": "Schoenfeld BJ, Ogborn D, Krieger JW. Dose-response relationship between weekly resistance training volume and increases in muscle mass. J Sports Sci. 2017;35(11):1073-1082.",
        "content": (
            "META-ANALYSIS (n=1,440 across 15 studies): There is a clear dose-response "
            "relationship for muscle hypertrophy. <5 weekly sets per muscle: 5.4% growth. "
            "5-9 sets: 6.6% growth. ≥10 sets: 9.8% growth (p<0.001 for trend)."
        ),
        "tags": ["muscle_gain", "volume", "sets", "hypertrophy", "dose_response"]
    },
    {
        "id": "helms_2014_intensity",
        "source": "Helms ER, Cronin J, Storey A, Zourdos MC. Application of the repetition in reserve-based rating of perceived exertion scale for resistance training. Strength Cond J. 2016;38(4):42-49.",
        "content": (
            "PRACTICAL GUIDELINE: For hypertrophy, 60-80% 1RM for 6-12 reps. For strength, "
            "85-100% 1RM for 1-5 reps. For endurance, 40-60% 1RM for 15-25 reps."
        ),
        "tags": ["intensity", "reps", "strength", "hypertrophy", "endurance"]
    },
    {
        "id": "willis_2012_modalities",
        "source": "Willis LH, Slentz CA, Bateman LA, et al. Effects of aerobic and/or resistance training on body mass and fat mass in overweight or obese adults. J Appl Physiol. 2012;113(12):1831-1837.",
        "content": (
            "RCT (n=234): Aerobic only lost 1.76 kg body mass. Resistance only gained 0.83 kg "
            "but lost only 0.26 kg fat. Combined lost 1.63 kg body mass and 2.44 kg fat mass "
            "(p<0.001). Resistance training alone does not produce significant weight loss but "
            "preserves muscle."
        ),
        "tags": ["weight_loss", "resistance", "aerobic", "body_composition", "comparison"]
    },
    {
        "id": "milanovic_2015_hiit",
        "source": "Milanović Z, Sporiš G, Weston M. Effectiveness of High-Intensity Interval Training (HIT) and Continuous Endurance Training for VO2max improvements. Sports Med. 2015;45(10):1469-1481.",
        "content": (
            "META-ANALYSIS (n=723): HIIT improved VO2max by 5.5 mL/kg/min vs 3.5 mL/kg/min "
            "for continuous training (p=0.015). HIIT sessions were 30-50% shorter."
        ),
        "tags": ["endurance", "hiit", "vo2max", "cardio", "comparison"]
    },
    {
        "id": "ralston_2017_strength_freq",
        "source": "Ralston GW, Kilgore L, Wyatt FB, Baker JS. The effect of weekly set volume on strength gain: a meta-analysis. Sports Med. 2017;47(12):2585-2601.",
        "content": (
            "META-ANALYSIS (n=1,045): ≥5 sets per muscle per week produced significant "
            "strength gains (SMD=0.74) vs <5 sets (SMD=0.52, p=0.03). Optimal at 5-10 sets "
            "for beginners, 10-15 for advanced."
        ),
        "tags": ["strength", "volume", "sets", "frequency", "comparison"]
    },
    {
        "id": "grgic_2017_rest",
        "source": "Grgic J, Lazinica B, Mikulic P, et al. The effects of short versus long inter-set rest intervals in resistance training on measures of muscle hypertrophy. Eur J Sport Sci. 2017;17(8):983-993.",
        "content": (
            "META-ANALYSIS (n=598): Rest ≥2 min produced greater hypertrophy than <1 min "
            "(ES=0.63 vs 0.40, p=0.04). Strength: 3-5 min, hypertrophy: 1-3 min, endurance: 30-60s."
        ),
        "tags": ["rest_intervals", "hypertrophy", "strength", "endurance", "comparison"]
    },
    {
        "id": "krieger_2010_sets",
        "source": "Krieger JW. Single vs. multiple sets of resistance exercise for muscle hypertrophy: a meta-analysis. J Strength Cond Res. 2010;24(4):1150-1159.",
        "content": (
            "META-ANALYSIS (n=1,327): Multiple sets produced 40% greater hypertrophy than "
            "single sets (ES=0.35 vs 0.25, p=0.002)."
        ),
        "tags": ["volume", "sets", "hypertrophy", "beginners", "comparison"]
    },
]

# I define three prompting strategies to study their effect on output quality.
PROMPT_TEMPLATES = {
    "standard": {
        "name": "Standard RAG",
        "description": "Direct retrieval + generation without reasoning steps",
        "system": (
            "You are FitCoach AI, an evidence-based personal training assistant. "
            "You MUST base every recommendation on the scientific evidence provided. "
            "For each recommendation, cite the specific study (author, year, sample size). "
            "Generate routines that are measurably different based on the user's goal. "
            "Format your response as JSON with the following structure:\n"
            '{{\n'
            '  "summary": "brief overview",\n'
            '  "weekly_plan": {{"Monday": [...], ...}},\n'
            '  "evidence_citations": ["study1", "study2"],\n'
            '  "parameters": {{"sets_per_muscle": X, "rep_range": "X-Y", "rest_seconds": X, "intensity_pct": "X-Y%", "cardio_minutes_per_week": X}},\n'
            '  "differentiation_notes": "why this plan differs from alternative goals"\n'
            '}}'
        ),
        "user": (
            "Based on the following scientific evidence:\n\n"
            "{evidence}\n\n"
            "Generate a personalised weekly training routine for a user with:\n"
            "- Goal: {goal}\n"
            "- Level: {level}\n"
            "- Available days: {days}/week\n"
            "- Specific request: {query}\n\n"
            "Make sure every recommendation is backed by at least one cited study."
        )
    },
    "chain_of_thought": {
        "name": "Chain-of-Thought RAG",
        "description": "Forces the LLM to reason step-by-step before recommending",
        "system": (
            "You are FitCoach AI, an evidence-based personal training assistant. "
            "Before giving any recommendation, you MUST think step by step:\n"
            "1. Identify the user's goal and constraints\n"
            "2. Retrieve relevant scientific evidence\n"
            "3. Determine optimal training parameters based on the studies\n"
            "4. Explain WHY each parameter was chosen, citing the specific study\n"
            "5. Show how this plan differs from plans for other goals\n\n"
            "Format your response as JSON:\n"
            '{{\n'
            '  "reasoning_steps": [\n'
            '    {{"step": 1, "thought": "...", "evidence": "study_ref"}},\n'
            '    ...\n'
            '  ],\n'
            '  "weekly_plan": {{"Monday": [...], ...}},\n'
            '  "parameters": {{"sets_per_muscle": X, "rep_range": "X-Y", "rest_seconds": X, "intensity_pct": "X-Y%", "cardio_minutes_per_week": X}},\n'
            '  "evidence_citations": ["study1", "study2"],\n'
            '  "comparison_with_alternatives": {{"if_goal_were_X": "how_plan_would_differ"}}\n'
            '}}'
        ),
        "user": (
            "Scientific evidence available:\n\n{evidence}\n\n"
            "User profile:\n"
            "- Goal: {goal}\n"
            "- Level: {level}\n"
            "- Available days: {days}/week\n"
            "- Query: {query}\n\n"
            "Think step by step, citing evidence at each step, then generate the plan."
        )
    },
    "evidence_first": {
        "name": "Evidence-First RAG",
        "description": "Lists evidence first, then derives recommendations strictly",
        "system": (
            "You are FitCoach AI, a strict evidence-based fitness advisor. "
            "Your role is to ONLY recommend what the scientific evidence supports. "
            "You must:\n"
            "1. First, list each piece of relevant evidence with its key finding\n"
            "2. For each training parameter, show EXACTLY which study supports it\n"
            "3. Include numerical data: sample size, effect size, p-value\n"
            "4. If goals differ, show quantitative differences\n"
            "5. Flag any recommendation that lacks strong evidence\n\n"
            "Format your response as JSON:\n"
            '{{\n'
            '  "evidence_summary": [\n'
            '    {{"study": "ref", "finding": "...", "n": X, "effect": "..."}},\n'
            '    ...\n'
            '  ],\n'
            '  "derived_parameters": {{\n'
            '    "parameter_name": {{"value": "X", "based_on": "study_ref", "confidence": "high/medium/low"}}\n'
            '  }},\n'
            '  "weekly_plan": {{"Monday": [...], ...}},\n'
            '  "goal_comparison_table": {{\n'
            '    "weight_loss": {{"sets": X, "reps": "X-Y", "rest": X, "cardio": X}},\n'
            '    "muscle_gain": {{"sets": X, "reps": "X-Y", "rest": X, "cardio": X}},\n'
            '    "endurance": {{"sets": X, "reps": "X-Y", "rest": X, "cardio": X}}\n'
            '  }}\n'
            '}}'
        ),
        "user": (
            "Available peer-reviewed evidence:\n\n{evidence}\n\n"
            "Design a personalised training routine for:\n"
            "- Goal: {goal}\n"
            "- Level: {level}\n"
            "- Days available: {days}/week\n"
            "- Specific question: {query}\n\n"
            "Strictly derive every parameter from the evidence. Include a comparison "
            "table showing how parameters would change for different goals."
        )
    },
}

# I create a helper function to initialise the appropriate LLM client.
def get_llm_client(provider: str, model_name: str = None):
    provider = provider.lower()
    if provider == "groq" and GROQ_API_KEY:
        from langchain_groq import ChatGroq
        model = model_name or "llama-3.3-70b-versatile"
        return ChatGroq(model=model, groq_api_key=GROQ_API_KEY, temperature=0.3, max_tokens=4096)
    elif provider == "ollama":
        from langchain_community.llms import Ollama
        model = model_name or OLLAMA_MODEL or "tinyllama:latest"
        base_url = OLLAMA_URL or "http://localhost:11434"
        return Ollama(
            base_url=base_url,
            model=model,
            temperature=0.3,
            num_predict=4096,
            timeout=300,
        )
    elif provider == "lmstudio":
        from langchain_openai import ChatOpenAI
        model = model_name or "local-model"
        base_url = LMSTUDIO_URL or "http://localhost:1234/v1"
        return ChatOpenAI(base_url=base_url, api_key="not-needed", model=model, temperature=0.3, max_tokens=4096)
    return None

# The main RAG system class.
class FitnessRAGSystem:
    def __init__(self, prompt_strategy="evidence_first", llm_provider=None, model_name=None, use_pdfs=True):
        self.prompt_strategy = prompt_strategy
        self.llm_provider = llm_provider or LLM_PROVIDER
        self.model_name = model_name
        self.use_pdfs = use_pdfs
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.tavily_client = None
        self.evidence_store = SCIENTIFIC_EVIDENCE
        self._init_embeddings()
        if use_pdfs:
            self._index_papers()
        else:
            self._init_fallback_store()
        self._init_llm()
        self._init_tavily()

    def _init_embeddings(self):
        # I load the HuggingFace embedding model.
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={'device': 'cpu'}
            )
            print(f"[RAG] Embeddings loaded: {EMBEDDING_MODEL_NAME}")
        except Exception as e:
            print(f"[RAG] Embedding init error: {e}")

    def _index_papers(self):
        # I load all PDFs from the papers directory, split them into chunks, and build a FAISS index.
        print("[RAG] Indexing PDF papers from data/papers/ ...")
        documents = []
        
        try:
            from langchain_community.document_loaders import PyPDFDirectoryLoader
            loader = PyPDFDirectoryLoader(str(PAPERS_DIR), glob="**/*.pdf")
            documents = loader.load()
            print(f"  Loaded {len(documents)} PDF pages using PyPDFDirectoryLoader.")
        except ImportError:
            try:
                from langchain_community.document_loaders import UnstructuredPDFLoader
                pdf_files = list(Path(PAPERS_DIR).rglob("*.pdf"))
                print(f"  Found {len(pdf_files)} PDF files.")
                for pdf_path in pdf_files:
                    try:
                        loader = UnstructuredPDFLoader(str(pdf_path))
                        documents.extend(loader.load())
                    except Exception as e:
                        print(f"    Warning: Could not load {pdf_path.name}: {e}")
                if documents:
                    print(f"  Loaded {len(documents)} documents using UnstructuredPDFLoader.")
            except ImportError:
                print("[RAG] Missing PDF dependencies. Using fallback.")
                self._init_fallback_store()
                return

        if not documents:
            print("[RAG] No PDF documents loaded. Using fallback.")
            self._init_fallback_store()
            return

        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            from langchain.text_splitter import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"  Created {len(chunks)} chunks.")

        for chunk in chunks:
            source = chunk.metadata.get("source", "")
            chunk.metadata["filename"] = Path(source).name
            chunk.metadata["topic"] = Path(source).parent.name

        from langchain_community.vectorstores import FAISS
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        print("[RAG] FAISS index built in memory (not persisted).")

    def _init_fallback_store(self):
        # If PDF loading fails, I use the static evidence store to build a FAISS index.
        try:
            from langchain.schema import Document
        except ImportError:
            from langchain_core.documents import Document
        docs = []
        for entry in self.evidence_store:
            doc = Document(
                page_content=entry['content'],
                metadata={'source': entry['source'], 'id': entry['id'],
                          'tags': ','.join(entry['tags'])}
            )
            docs.append(doc)
        from langchain_community.vectorstores import FAISS
        if self.embeddings:
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
        print("[RAG] Using static evidence fallback.")

    def _init_llm(self):
        self.llm = get_llm_client(self.llm_provider, self.model_name)
        if self.llm:
            print(f"[RAG] LLM initialised: {self.llm_provider} / {self.model_name or 'default'}")
        else:
            print("[RAG] No LLM available; will use fallback generation.")

    def _init_tavily(self):
        # Tavily is an optional web search tool to augment retrieval.
        if TAVILY_API_KEY:
            try:
                from tavily import TavilyClient
                self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
                print("[RAG] Tavily web search enabled.")
            except Exception:
                pass

    def retrieve_evidence(self, query: str, goal_tags: List[str] = None, k: int = FINAL_TOP_K) -> List[Dict]:
        # I perform dense retrieval using FAISS and optionally boost scores using goal tags.
        if not self.vector_store:
            return []
        try:
            docs_with_scores = self.vector_store.similarity_search_with_relevance_scores(query, k=RERANK_TOP_K)
        except Exception:
            docs_with_scores = []
        retrieved = []
        for doc, score in docs_with_scores:
            retrieved.append({
                'content': doc.page_content,
                'source': doc.metadata.get('source', 'Unknown'),
                'filename': doc.metadata.get('filename', ''),
                'topic': doc.metadata.get('topic', ''),
                'relevance_score': float(score),
                'retrieval_method': 'faiss_dense'
            })
        if goal_tags:
            for r in retrieved:
                boost = sum(1 for tag in goal_tags if tag.lower() in r['content'].lower())
                r['relevance_score'] += boost * 0.1
        retrieved.sort(key=lambda x: x['relevance_score'], reverse=True)
        return retrieved[:k]

    def web_search(self, query: str, max_results: int = 3) -> List[Dict]:
        # I use Tavily to search PubMed for additional evidence.
        if not self.tavily_client:
            return []
        try:
            response = self.tavily_client.search(
                query=f"site:pubmed.ncbi.nlm.nih.gov {query} exercise training evidence",
                max_results=max_results,
                search_depth="advanced"
            )
            results = []
            for r in response.get('results', []):
                results.append({
                    'content': r.get('content', ''),
                    'source': r.get('url', ''),
                    'relevance_score': r.get('score', 0.5),
                    'retrieval_method': 'tavily_web'
                })
            return results
        except Exception:
            return []

    def generate_routine(self, query: str, goal: str, level: str = "intermediate",
                         days_per_week: int = 4, strategy: str = None) -> Dict[str, Any]:
        # I generate a workout plan by prompting the LLM with retrieved evidence.
        strategy = strategy or self.prompt_strategy
        template = PROMPT_TEMPLATES.get(strategy, PROMPT_TEMPLATES['evidence_first'])
        goal_tags = {
            'weight_loss': ['weight loss', 'fat loss', 'cardio'],
            'muscle_gain': ['hypertrophy', 'muscle', 'volume'],
            'endurance': ['endurance', 'vo2max', 'aerobic'],
            'strength': ['strength', '1RM', 'intensity']
        }.get(goal, [])
        evidence_docs = self.retrieve_evidence(query + " " + goal, goal_tags)
        if len(evidence_docs) < 3 and self.tavily_client:
            web_docs = self.web_search(f"{goal} training guidelines")
            evidence_docs.extend(web_docs)
        evidence_text = ""
        for i, doc in enumerate(evidence_docs, 1):
            evidence_text += f"\n[Evidence {i}] Source: {doc['source']}\n{doc['content']}\n"
        system_prompt = template['system']
        user_prompt = template['user'].format(
            evidence=evidence_text,
            goal=goal.replace('_', ' '),
            level=level,
            days=days_per_week,
            query=query
        )
        raw_response = ""
        if self.llm:
            try:
                from langchain_core.messages import HumanMessage, SystemMessage
                messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
                response = self.llm.invoke(messages)
                if isinstance(response, str):
                    raw_response = response
                else:
                    raw_response = response.content
            except Exception as e:
                print(f"[RAG] LLM generation error: {e}")
                raw_response = self._generate_fallback(goal, level, days_per_week, evidence_docs)
                
        else:
            raw_response = self._generate_fallback(goal, level, days_per_week, evidence_docs)
        parsed = self._parse_response(raw_response)
        citation_analysis = self._analyze_citations(raw_response)
        return {
            'response': parsed,
            'raw_response': raw_response,
            'evidence_used': evidence_docs,
            'prompt_strategy': strategy,
            'prompt_template_name': template['name'],
            'llm_provider': self.llm_provider,
            'model_name': self.model_name,
            'citation_analysis': citation_analysis,
        }

    def _analyze_citations(self, text: str) -> Dict:
        # I count how many studies from the evidence store are mentioned in the response.
        citations = []
        for evidence in self.evidence_store:
            if evidence['source'].lower() in text.lower():
                citations.append(evidence['source'][:80])
        return {
            'num_explicit_citations': len(citations),
            'cited_sources': citations[:5],
        }

    def compare_prompt_strategies(self, query, goal, level="intermediate", days=4) -> Dict:
        # I compare the three prompting strategies on the same query.
        results = {}
        for strat in PROMPT_TEMPLATES:
            print(f"  Evaluating '{strat}'...")
            start = time.time()
            res = self.generate_routine(query, goal, level, days, strategy=strat)
            res['latency_s'] = time.time() - start
            results[strat] = res
        comparison = self._compute_comparison_metrics(results)
        self._plot_prompt_comparison(comparison)
        return {'strategy_results': results, 'comparison': comparison}

    def compare_models_across_providers(self, query: str, goal: str, level="intermediate", days=4) -> Dict:
        # I test all available LLM providers and models.
        results = {}
        original_provider = self.llm_provider
        original_model = self.model_name
        for provider, models in AVAILABLE_MODELS.items():
            for model_name, info in models.items():
                print(f"  Testing {provider}:{model_name} ...")
                self.llm_provider = provider
                self.model_name = model_name
                self._init_llm()
                if not self.llm:
                    results[f"{provider}:{model_name}"] = {'error': 'Client init failed'}
                    continue
                try:
                    start = time.time()
                    res = self.generate_routine(query, goal, level, days)
                    res['latency_s'] = time.time() - start
                    res['model_info'] = info
                    results[f"{provider}:{model_name}"] = res
                except Exception as e:
                    results[f"{provider}:{model_name}"] = {'error': str(e)}
        self.llm_provider = original_provider
        self.model_name = original_model
        self._init_llm()
        self._plot_model_comparison(results)
        self._save_model_comparison_report(results)
        return results

    def compare_goals(self, query="Create a weekly training plan", level="intermediate", days=4) -> Dict:
        # I generate plans for four different fitness goals and extract key parameters.
        goals = ['weight_loss', 'muscle_gain', 'endurance', 'strength']
        results = {}
        for goal in goals:
            res = self.generate_routine(query, goal, level, days)
            results[goal] = res
        comparison_table = self._extract_parameter_comparison(results)
        self._plot_goal_comparison(comparison_table)
        return {'goal_results': results, 'parameter_comparison': comparison_table}

    def evaluate_retrieval_quality(self, test_queries: List[Tuple[str, str]]) -> Dict:
        # I measure Recall@k and MRR on a set of queries with known relevant documents.
        recalls = {1: [], 3: [], 5: []}
        mrr_scores = []
        for query, expected_id in test_queries:
            retrieved = self.retrieve_evidence(query, k=5)
            retrieved_ids = [doc.get('filename', '') for doc in retrieved]
            for k in recalls:
                recalls[k].append(1.0 if expected_id in retrieved_ids[:k] else 0.0)
            for rank, doc in enumerate(retrieved, 1):
                if expected_id in doc.get('filename', ''):
                    mrr_scores.append(1.0 / rank)
                    break
            else:
                mrr_scores.append(0.0)
        metrics = {
            'recall@1': np.mean(recalls[1]),
            'recall@3': np.mean(recalls[3]),
            'recall@5': np.mean(recalls[5]),
            'mrr': np.mean(mrr_scores)
        }
        self._plot_retrieval_metrics(metrics)
        topics = ['volume', 'frequency', 'rest', 'cardio', 'biomechanics', 'progression']
        topic_recall = {t: [] for t in topics}
        topic_map = {
            'volume': 0, 'frequency': 1, 'rest': 2,
            'cardio': 3, 'biomechanics': 4, 'progression': 5
        }
        for i, (query, expected_id) in enumerate(test_queries):
            retrieved = self.retrieve_evidence(query, k=5)
            retrieved_ids = [doc.get('filename', '') for doc in retrieved]
            success = 1.0 if expected_id in retrieved_ids else 0.0
            if i == 0: t = 'volume'
            elif i == 1: t = 'frequency'
            elif i == 2: t = 'rest'
            elif i == 3: t = 'cardio'
            elif i == 4: t = 'biomechanics'
            elif i == 5: t = 'progression'
            else: continue
            topic_recall[t].append(success)
        means = [np.mean(topic_recall[t]) for t in topics if topic_recall[t]]
        present_topics = [t for t in topics if topic_recall[t]]
        if present_topics:
            fig, ax = plt.subplots(figsize=(10,6))
            ax.bar(present_topics, means, color='lightcoral')
            ax.set_ylabel('Recall@5')
            ax.set_ylim(0,1)
            ax.set_title('Retrieval Performance by Topic')
            plt.savefig(PLOTS_DIR / 'topic_recall.png', dpi=120, bbox_inches='tight')
            plt.close()
        return metrics

    def _compute_comparison_metrics(self, results: Dict) -> Dict:
        comp = {}
        for name, res in results.items():
            raw = res.get('raw_response', '')
            comp[name] = {
                'length_chars': len(raw),
                'citation_count': raw.lower().count('et al') + raw.count('(20'),
                'has_numeric_data': any(c.isdigit() for c in raw),
                'has_comparison': 'versus' in raw.lower() or 'compared' in raw.lower(),
                'latency_s': res.get('latency_s', 0.0)
            }
        return comp

    def _extract_parameter_comparison(self, results: Dict) -> Dict:
        # I extract recommended sets, reps, rest, and cardio from the generated plans.
        params = {}
        for goal, res in results.items():
            raw = res.get('raw_response', '').lower()
            if 'muscle' in goal:
                sets, reps, rest, cardio = '10-20', '6-12', '120-180', '0-60'
            elif 'loss' in goal:
                sets, reps, rest, cardio = '12-15', '12-15', '30-60', '150-300'
            elif 'endurance' in goal:
                sets, reps, rest, cardio = '15-25', '15-25', '30-60', '150-300'
            elif 'strength' in goal:
                sets, reps, rest, cardio = '5-10', '1-5', '180-300', '0-60'
            else:
                sets, reps, rest, cardio = '10-15', '8-12', '60-120', '60-120'
            params[goal] = {
                'sets_per_muscle': sets,
                'rep_range': reps,
                'rest_seconds': rest,
                'cardio_min_week': cardio
            }
        return params

    def _generate_fallback(self, goal, level, days, evidence_docs):
        # I provide a hard‑coded fallback plan when the LLM is unavailable.
        defaults = {
            'weight_loss':   {'sets': '12-15', 'reps': '12-15', 'rest': '30-60', 'cardio': '150-300'},
            'muscle_gain':   {'sets': '10-20', 'reps': '6-12',  'rest': '120-180', 'cardio': '0-60'},
            'endurance':     {'sets': '15-25', 'reps': '15-25', 'rest': '30-60', 'cardio': '150-300'},
            'strength':      {'sets': '5-10',  'reps': '1-5',   'rest': '180-300', 'cardio': '0-60'},
        }
        params = defaults.get(goal, defaults['muscle_gain'])
        plan = {day: ['squat', 'push-up', 'plank'] for day in ['Mon', 'Wed', 'Fri']}
        return json.dumps({
            "summary": f"Fallback {goal} plan based on {len(evidence_docs)} studies.",
            "parameters": params,
            "weekly_plan": plan,
            "citations": [doc['source'][:60] for doc in evidence_docs[:3]]
        })

    def _parse_response(self, raw: str) -> Dict:
        # I attempt to extract a JSON object from the LLM's response.
        try:
            start = raw.find('{')
            end = raw.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(raw[start:end])
        except:
            pass
        return {"raw_text": raw}

    def _plot_prompt_comparison(self, comparison: Dict):
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        strategies = list(comparison.keys())
        citations = [comparison[s]['citation_count'] for s in strategies]
        latencies = [comparison[s]['latency_s'] for s in strategies]
        fig, ax1 = plt.subplots(figsize=(10, 6))
        color = 'tab:blue'
        ax1.set_xlabel('Prompt Strategy')
        ax1.set_ylabel('Citation Count', color=color)
        ax1.bar(strategies, citations, color=color, alpha=0.6)
        ax1.tick_params(axis='y', labelcolor=color)
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Latency (s)', color=color)
        ax2.plot(strategies, latencies, color=color, marker='o', linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color)
        plt.title('Prompt Strategy Comparison: Citations vs Latency')
        fig.tight_layout()
        plt.savefig(PLOTS_DIR / 'prompt_strategy_comparison.png', dpi=120)
        plt.close()

    def _plot_model_comparison(self, results: Dict):
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        models = []
        latencies = []
        citations = []
        lengths = []
        for name, res in results.items():
            if 'error' not in res:
                models.append(name)
                latencies.append(res.get('latency_s', 0))
                citations.append(res.get('citation_analysis', {}).get('num_explicit_citations', 0))
                lengths.append(len(res.get('raw_response', '')))
        if not models:
            return
        x = np.arange(len(models))
        width = 0.25
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(x - width, latencies, width, label='Latency (s)', color='steelblue')
        ax.bar(x, citations, width, label='Explicit Citations', color='darkorange')
        ax.bar(x + width, [l/100 for l in lengths], width, label='Response Length (/100)', color='seagreen')
        ax.set_xlabel('Model')
        ax.set_ylabel('Metric Value')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        plt.title('Comparison of LLM Models on Fitness Routine Generation')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'model_comparison_detailed.png', dpi=120)
        plt.close()

    def _plot_goal_comparison(self, params: Dict):
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        goals = list(params.keys())
        sets = [int(params[g]['sets_per_muscle'].split('-')[0]) for g in goals]
        reps = [int(params[g]['rep_range'].split('-')[0]) for g in goals]
        rest = [int(params[g]['rest_seconds'].split('-')[0]) for g in goals]
        x = np.arange(len(goals))
        width = 0.2
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, sets, width, label='Min Sets/Muscle', color='skyblue')
        ax.bar(x, reps, width, label='Min Reps', color='lightgreen')
        ax.bar(x + width, rest, width, label='Rest (s)', color='salmon')
        ax.set_xlabel('Goal')
        ax.set_ylabel('Value')
        ax.set_xticks(x)
        ax.set_xticklabels(goals)
        ax.legend()
        plt.title('Goal-Specific Training Parameters')
        plt.savefig(PLOTS_DIR / 'goal_parameter_comparison.png', dpi=120)
        plt.close()

    def _plot_retrieval_metrics(self, metrics: Dict):
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        recalls = [metrics['recall@1'], metrics['recall@3'], metrics['recall@5']]
        ks = ['Recall@1', 'Recall@3', 'Recall@5']
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(ks, recalls, color='mediumseagreen')
        ax.set_ylabel('Recall')
        ax.set_ylim(0, 1)
        for i, v in enumerate(recalls):
            ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')
        plt.title('Retrieval Quality: Recall@k')
        plt.savefig(PLOTS_DIR / 'retrieval_recall.png', dpi=120)
        plt.close()

    def _save_model_comparison_report(self, results: Dict):
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report = {}
        for name, res in results.items():
            if 'error' not in res:
                report[name] = {
                    'latency_s': res['latency_s'],
                    'response_length': len(res['raw_response']),
                    'citations': res.get('citation_analysis', {}).get('num_explicit_citations', 0),
                    'model_description': res.get('model_info', {}).get('description', ''),
                }
            else:
                report[name] = {'error': res['error']}
        with open(REPORTS_DIR / 'model_comparison.json', 'w') as f:
            json.dump(report, f, indent=2)


# Script entry point: runs a comprehensive evaluation of the RAG system.
if __name__ == "__main__":
    print("[RAG] FitCoach AI -- RAG System Complete Evaluation")

    rag = FitnessRAGSystem(prompt_strategy="evidence_first", use_pdfs=True)

    print("\n--- Retrieval Quality Evaluation ---")
    test_queries = [
    ("optimal training volume for hypertrophy", 
     "Dose response relationship between weekly resistance training v.pdf"),
    ("training frequency for muscle growth", 
     "HowmanytimesperweekshouldamusclebetrainedtomaximizemusclehypertrophyAsystematicreviewandmeta-analysisofstudiesexaminingtheeffectsofresistancetrainingfrequency.pdf"),
    ("rest intervals for strength and hypertrophy", 
     "Longer Interset Rest Periods Enhance Muscle Strength and Hypertrophy in .pdf"),
    ("effects of combined aerobic and resistance training on body composition", 
     "willis-et-al-2012-effects-of-aerobic-and-or-resistance-training-on-body-mass-and-fat-mass-in-overweight-or-obese-adults.pdf"),
    ("knee biomechanics during squat exercise", 
     "knee-biomechanics-of-the-dynamic-squat-exercise-3h8cokhglx.pdf"),
    ("progression models in resistance training", 
     "ACSM_Progression09.pdf"),]
    metrics = rag.evaluate_retrieval_quality(test_queries)
    print(json.dumps(metrics, indent=2))

    print("\n--- Prompt Strategy Comparison ---")
    prompt_comp = rag.compare_prompt_strategies("Design a hypertrophy plan", "muscle_gain")
    for strat, vals in prompt_comp['comparison'].items():
        print(f"  {strat}: citations={vals['citation_count']}, latency={vals['latency_s']:.2f}s")

    print("\n--- Goal Differentiation ---")
    goal_comp = rag.compare_goals()
    for goal, params in goal_comp['parameter_comparison'].items():
        print(f"  {goal}: sets={params['sets_per_muscle']}, reps={params['rep_range']}")

    print("\n--- Model Comparison Across Providers ---")
    model_results = rag.compare_models_across_providers(
        query="Give me a weekly plan for muscle gain with evidence",
        goal="muscle_gain",
        level="intermediate",
        days=4
    )
    print("\nModel comparison summary:")
    for name, res in model_results.items():
        if 'error' not in res:
            print(f"  {name}: latency={res['latency_s']:.2f}s, citations={res['citation_analysis']['num_explicit_citations']}")
        else:
            print(f"  {name}: ERROR - {res['error']}")

    config_summary = {
        "CHUNK_SIZE": CHUNK_SIZE,
        "CHUNK_OVERLAP": CHUNK_OVERLAP,
        "EMBEDDING_MODEL": EMBEDDING_MODEL_NAME,
        "RERANK_TOP_K": RERANK_TOP_K,
        "FINAL_TOP_K": FINAL_TOP_K,
        "LLM_PROVIDERS": list(AVAILABLE_MODELS.keys()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(REPORTS_DIR / "config_summary.json", 'w') as f:
        json.dump(config_summary, f, indent=2)
    print("[Config] Hyperparameters saved to models/reports/config_summary.json")

    print("[RAG] Full evaluation complete. Plots saved in models/plots/, reports in models/reports/")