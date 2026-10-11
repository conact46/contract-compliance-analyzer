"""
Contract Compliance Analyzer (Phase 1)
---------------------------------------
Research prototype for a master's thesis in Information Systems
(King Abdulaziz University). Screens business contract clauses against
six general Saudi laws using a knowledge graph, sentence embeddings
(semantic similarity) and fixed rule-based reasoning. No generative
language model is used in this phase; LLM integration is future work.

This file is a direct, cleaned-up port of the research notebook
`contract_analyzer_phase4.ipynb` for deployment as a Hugging Face Space,
so that domain experts can upload real contracts and evaluate the
system's outputs (correctness, clarity, traceability, usability) for the
thesis's expert-based evaluation chapter.
"""

import os
import re
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import networkx as nx

import PyPDF2
import pdfplumber
from docx import Document as DocxDocument

from sentence_transformers import SentenceTransformer
import faiss
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler

import plotly.graph_objects as go

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.enums import TA_RIGHT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import arabic_reshaper
from bidi.algorithm import get_display

import gradio as gr

# Register a bundled Arabic-capable font (Amiri, SIL OFL licensed, see
# fonts/Amiri-OFL-LICENSE.txt) so the PDF report can render Arabic text
# correctly instead of as unmapped glyph boxes.
_FONT_DIR = Path(__file__).parent / "fonts"
_ARABIC_FONT = "Amiri"
_ARABIC_FONT_BOLD = "Amiri-Bold"
try:
    pdfmetrics.registerFont(TTFont(_ARABIC_FONT, str(_FONT_DIR / "Amiri-Regular.ttf")))
    pdfmetrics.registerFont(TTFont(_ARABIC_FONT_BOLD, str(_FONT_DIR / "Amiri-Bold.ttf")))
    _ARABIC_FONT_OK = True
except Exception as e:
    print(f"Warning: could not load Amiri font ({e}); Arabic text in PDF reports may not render.")
    _ARABIC_FONT_OK = False


def _ar(text: str) -> str:
    """Reshapes and reorders Arabic text for correct right-to-left rendering
    in reportlab, which does not do this automatically. Safe to call on
    mixed Arabic/Latin/number text."""
    if not text:
        return text
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)


# ============================================================
# 1. Enums and data classes
# ============================================================

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def emoji(self):
        return {"low": "\U0001F7E2", "medium": "\U0001F7E1", "high": "\U0001F7E0", "critical": "\U0001F534"}[self.value]

    @property
    def arabic(self):
        return {"low": "منخفض", "medium": "متوسط", "high": "عالي", "critical": "حرج"}[self.value]


class ComplianceStatus(Enum):
    COMPLIANT = "compliant"
    PARTIAL = "partial"
    NON_COMPLIANT = "non_compliant"
    NOT_APPLICABLE = "not_applicable"

    @property
    def icon(self):
        return {"compliant": "✅", "partial": "⚠️", "non_compliant": "❌", "not_applicable": "➖"}[self.value]


LAWS = {
    "PDPL": {"ar": "نظام حماية البيانات الشخصية", "en": "Personal Data Protection Law", "weight": 1.2},
    "ECC": {"ar": "نظام التجارة الإلكترونية", "en": "E-Commerce Law", "weight": 1.0},
    "LABOR": {"ar": "نظام العمل", "en": "Labor Law", "weight": 1.1},
    "CYBER": {"ar": "نظام مكافحة الجرائم المعلوماتية", "en": "Anti-Cyber Crime Law", "weight": 1.3},
    "COMMERCIAL": {"ar": "نظام الشركات", "en": "Companies Law", "weight": 1.0},
    "ANTI_FRAUD": {"ar": "نظام مكافحة الاحتيال", "en": "Anti-Fraud Law", "weight": 1.1},
}


@dataclass
class LawArticle:
    id: str
    law_id: str
    number: str
    title: str
    content: str
    keywords: List[str] = field(default_factory=list)
    related_articles: List[str] = field(default_factory=list)
    risk_weight: float = 1.0
    embedding: Optional[np.ndarray] = None


@dataclass
class ContractClause:
    id: str
    contract_id: str
    number: int
    title: str
    content: str
    clause_type: str
    keywords: List[str] = field(default_factory=list)
    entities: Dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    label: Optional[str] = None
    risk_label: Optional[str] = None


@dataclass
class AnalysisResult:
    status: ComplianceStatus
    risk_score: float
    reasoning: str
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    confidence: float = 0.7


@dataclass
class LawScore:
    law_id: str
    compliance_pct: float
    risk_score: float
    risk_level: RiskLevel
    total_checked: int
    compliant: int
    partial: int
    non_compliant: int
    reasoning: str
    findings: List[str] = field(default_factory=list)


@dataclass
class FullReport:
    contract_id: str
    contract_name: str
    timestamp: str
    total_clauses: int
    laws_checked: List[str]
    law_scores: Dict[str, LawScore]
    overall_compliance: float
    overall_risk: float
    overall_level: RiskLevel
    summary: str
    risks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    learned_keywords: List[str] = field(default_factory=list)


@dataclass
class TrainingData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    label_encoder: LabelEncoder
    scaler: StandardScaler


# ============================================================
# 2. Knowledge graphs
# ============================================================

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


class LawKnowledgeGraph:
    """Knowledge graph of the loaded laws and their articles."""

    def __init__(self, embedder: SentenceTransformer):
        self.embedder = embedder
        self.graph = nx.DiGraph()
        self.articles: Dict[str, LawArticle] = {}
        self.by_law: Dict[str, List[str]] = {law: [] for law in LAWS}
        self.index = None
        self.id_map: List[str] = []

        for law_id, info in LAWS.items():
            self.graph.add_node(law_id, node_type="law", **info)

    def add_article(self, article: LawArticle):
        text = f"{article.title} {article.content}"
        article.embedding = self.embedder.encode(text)

        self.articles[article.id] = article
        if article.law_id in self.by_law:
            self.by_law[article.law_id].append(article.id)

        self.graph.add_node(article.id, node_type="article", title=article.title)
        self.graph.add_edge(article.law_id, article.id, relation="contains")

        self.index = None

    def build_index(self):
        if not self.articles:
            return
        embeddings = []
        self.id_map = []
        for aid, art in self.articles.items():
            if art.embedding is not None:
                embeddings.append(art.embedding)
                self.id_map.append(aid)
        if embeddings:
            arr = np.array(embeddings).astype('float32')
            faiss.normalize_L2(arr)
            self.index = faiss.IndexFlatIP(arr.shape[1])
            self.index.add(arr)

    def search(self, query: str, top_k: int = 5, law_filter: List[str] = None) -> List[Tuple[LawArticle, float]]:
        if self.index is None:
            self.build_index()
        if self.index is None or not self.id_map:
            return []

        q = self.embedder.encode([query]).astype('float32')
        faiss.normalize_L2(q)

        k = min(top_k * 3, len(self.id_map))
        scores, indices = self.index.search(q, k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            article = self.articles[self.id_map[idx]]
            if law_filter and article.law_id not in law_filter:
                continue
            results.append((article, float(score)))
            if len(results) >= top_k:
                break
        return results


class ContractKnowledgeGraph:
    """Knowledge graph of the contract currently being analyzed."""

    def __init__(self, embedder: SentenceTransformer):
        self.embedder = embedder
        self.graph = nx.DiGraph()
        self.contracts: Dict[str, Dict] = {}
        self.clauses: Dict[str, ContractClause] = {}
        self.learned_patterns: Dict[str, List[str]] = {}

    def add_contract(self, contract_id: str, name: str, metadata: Dict = None):
        self.contracts[contract_id] = {"name": name, **(metadata or {})}
        self.graph.add_node(contract_id, node_type="contract", name=name)

    def add_clause(self, clause: ContractClause):
        text = f"{clause.title} {clause.content}"
        clause.embedding = self.embedder.encode(text)

        self.clauses[clause.id] = clause
        self.graph.add_node(clause.id, node_type="clause", title=clause.title)
        self.graph.add_edge(clause.contract_id, clause.id, relation="contains")

        ctype = clause.clause_type
        if ctype not in self.learned_patterns:
            self.learned_patterns[ctype] = []
        for kw in clause.keywords:
            if kw not in self.learned_patterns[ctype]:
                self.learned_patterns[ctype].append(kw)

    def get_clauses(self, contract_id: str) -> List[ContractClause]:
        return [c for c in self.clauses.values() if c.contract_id == contract_id]


# ============================================================
# 3. Contract parsing
# ============================================================

class ContractParser:
    """Extracts clauses from a contract (Arabic or English)."""

    CLAUSE_PATTERNS = [
        r'(?:المادة|البند|الفقرة)\s*(?:رقم)?\s*[:\-]?\s*(\d+)',
        r'(?:Article|Clause|Section)\s*[:\-]?\s*(\d+)',
        r'^(\d+)[.\-\)]\s+',
    ]

    CLAUSE_TYPES = {
        'privacy': ['خصوصية', 'بيانات', 'سرية', 'موافقة', 'شخصية'],
        'payment': ['دفع', 'سداد', 'مقابل', 'مالي', 'ريال'],
        'liability': ['مسؤولية', 'ضمان', 'تعويض', 'إخلاء'],
        'termination': ['إنهاء', 'فسخ', 'انتهاء', 'إلغاء'],
        'security': ['أمن', 'حماية', 'اختراق', 'سيبراني'],
        'dispute': ['نزاع', 'تحكيم', 'محكمة', 'خلاف'],
        'compliance': ['امتثال', 'التزام', 'قانون', 'لائحة'],
    }

    def parse_text(self, text: str) -> List[Dict]:
        clauses = []
        lines = text.split('\n')
        current = None
        num = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            is_header = any(re.match(p, line) for p in self.CLAUSE_PATTERNS)
            is_header = is_header or (len(line) < 80 and ':' in line)

            if is_header:
                if current and current['content'].strip():
                    clauses.append(current)
                num += 1
                current = {
                    'number': num,
                    'title': line[:100],
                    'content': '',
                    'type': self._detect_type(line)
                }
            elif current:
                current['content'] += ' ' + line
            else:
                num += 1
                current = {'number': num, 'title': f'بند {num}', 'content': line, 'type': 'general'}

        if current and current['content'].strip():
            clauses.append(current)
        return clauses

    def _detect_type(self, text: str) -> str:
        text_lower = text.lower()
        for ctype, keywords in self.CLAUSE_TYPES.items():
            if any(kw in text_lower for kw in keywords):
                return ctype
        return 'general'

    def extract_keywords(self, text: str) -> List[str]:
        words = re.findall(r'[\w؀-ۿ]{3,}', text)
        stopwords = {'من', 'إلى', 'على', 'في', 'أن', 'هذا', 'التي', 'الذي'}
        return list(set(w for w in words if w.lower() not in stopwords))[:15]

    def parse_pdf(self, path: str) -> Tuple[str, List[Dict]]:
        text = ""
        try:
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    text += (page.extract_text() or "") + "\n"
        except Exception:
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += (page.extract_text() or "") + "\n"
        return text, self.parse_text(text)

    def parse_docx(self, path: str) -> Tuple[str, List[Dict]]:
        doc = DocxDocument(path)
        text = "\n".join([p.text for p in doc.paragraphs])
        return text, self.parse_text(text)


# ============================================================
# 4. Reasoning engine (fixed rule-based formula)
# ============================================================

class ReasoningEngine:
    def __init__(self, embedder: SentenceTransformer):
        self.embedder = embedder

    def analyze(self, clause: ContractClause, articles: List[LawArticle], law_id: str) -> AnalysisResult:
        law_name = LAWS.get(law_id, {}).get('ar', law_id)

        if not articles:
            return AnalysisResult(
                status=ComplianceStatus.NOT_APPLICABLE,
                risk_score=25.0,
                reasoning=f"لا توجد مواد ذات صلة من {law_name}",
                confidence=0.5
            )

        clause_text = f"{clause.title} {clause.content}".lower()
        clause_words = set(re.findall(r'[\w؀-ۿ]{3,}', clause_text))

        total_kw, matched_kw = 0, 0
        issues, recommendations = [], []

        for art in articles:
            art_kw = set(art.keywords)
            total_kw += len(art.keywords)
            matched_kw += len(clause_words & art_kw)

            missing = [k for k in art.keywords if k.lower() not in clause_text]
            if missing:
                issues.append(f"المادة {art.number}: مصطلحات مفقودة ({', '.join(missing[:2])})")
                recommendations.append(f"إضافة بنود عن: {', '.join(missing[:2])}")

        clause_emb = self.embedder.encode(f"{clause.title} {clause.content}")
        similarities = []
        for art in articles:
            if art.embedding is not None:
                sim = np.dot(clause_emb, art.embedding) / (
                    np.linalg.norm(clause_emb) * np.linalg.norm(art.embedding) + 1e-8
                )
                similarities.append(float(sim))

        avg_sim = float(np.mean(similarities)) if similarities else 0.3
        kw_ratio = matched_kw / max(total_kw, 1)
        combined = 0.4 * kw_ratio + 0.6 * avg_sim

        if combined > 0.55:
            status = ComplianceStatus.COMPLIANT
            risk = 15 + (1 - combined) * 20
        elif combined > 0.35:
            status = ComplianceStatus.PARTIAL
            risk = 40 + (1 - combined) * 25
        else:
            status = ComplianceStatus.NON_COMPLIANT
            risk = 65 + (1 - combined) * 30

        reasoning = f"تحليل ضد {law_name}: تشابه {avg_sim*100:.0f}%، مصطلحات {kw_ratio*100:.0f}%، مجمع {combined*100:.0f}%"

        return AnalysisResult(
            status=status,
            risk_score=min(100, max(0, risk)),
            reasoning=reasoning,
            issues=issues[:5],
            recommendations=list(set(recommendations))[:5],
            confidence=0.6 + 0.3 * combined
        )

    def generate_summary(self, overall_risk: float, name: str) -> str:
        if overall_risk < 30:
            status, action = "العقد جيد", "يمكن المتابعة"
        elif overall_risk < 50:
            status, action = "يحتاج تحسينات", "مراجعة البنود المحددة"
        elif overall_risk < 75:
            status, action = "مخاطر ملحوظة", "استشارة قانونية"
        else:
            status, action = "مخاطر عالية", "تعديلات جوهرية مطلوبة"

        return f"العقد: {name}\nالحالة: {status}\nالمخاطر: {overall_risk:.0f}%\nالتوصية: {action}"


# ============================================================
# 5. Main analyzer
# ============================================================

class ContractAnalyzer:
    def __init__(self, embedder: SentenceTransformer):
        self.embedder = embedder
        self.law_kg = LawKnowledgeGraph(embedder)
        self.contract_kg = ContractKnowledgeGraph(embedder)
        self.parser = ContractParser()
        self.engine = ReasoningEngine(embedder)

    def add_law_article(self, article: LawArticle):
        self.law_kg.add_article(article)

    def build_law_index(self):
        self.law_kg.build_index()

    def analyze(self, contract_text: str, contract_name: str, laws: List[str]) -> FullReport:
        contract_id = f"C_{hashlib.md5(contract_text[:50].encode()).hexdigest()[:6]}"
        self.contract_kg.add_contract(contract_id, contract_name)

        parsed = self.parser.parse_text(contract_text)
        clauses = []
        for p in parsed:
            clause = ContractClause(
                id=f"{contract_id}_CL{p['number']}",
                contract_id=contract_id,
                number=p['number'],
                title=p['title'],
                content=p['content'],
                clause_type=p['type'],
                keywords=self.parser.extract_keywords(p['content'])
            )
            self.contract_kg.add_clause(clause)
            clauses.append(clause)

        law_results: Dict[str, List[AnalysisResult]] = {law: [] for law in laws}

        for clause in clauses:
            for law_id in laws:
                relevant = self.law_kg.search(clause.content, top_k=3, law_filter=[law_id])
                articles = [art for art, _ in relevant]
                result = self.engine.analyze(clause, articles, law_id)
                law_results[law_id].append(result)

        law_scores: Dict[str, LawScore] = {}
        for law_id in laws:
            results = law_results[law_id]
            if not results:
                continue

            compliant = sum(1 for r in results if r.status == ComplianceStatus.COMPLIANT)
            partial = sum(1 for r in results if r.status == ComplianceStatus.PARTIAL)
            non_comp = sum(1 for r in results if r.status == ComplianceStatus.NON_COMPLIANT)
            total = len(results)

            compliance_pct = ((compliant + partial * 0.5) / total) * 100 if total else 0
            avg_risk = float(np.mean([r.risk_score for r in results]))

            level = (RiskLevel.LOW if avg_risk < 30 else
                     RiskLevel.MEDIUM if avg_risk < 50 else
                     RiskLevel.HIGH if avg_risk < 75 else RiskLevel.CRITICAL)

            findings = [i for r in results for i in r.issues]

            law_scores[law_id] = LawScore(
                law_id=law_id, compliance_pct=compliance_pct, risk_score=avg_risk,
                risk_level=level, total_checked=total, compliant=compliant,
                partial=partial, non_compliant=non_comp,
                reasoning=f"فحص {total} بند: {compliant}✅ {partial}⚠️ {non_comp}❌",
                findings=list(set(findings))[:5]
            )

        if law_scores:
            total_weight = sum(LAWS[lid]['weight'] for lid in law_scores)
            overall_compliance = sum(law_scores[lid].compliance_pct * LAWS[lid]['weight'] for lid in law_scores) / total_weight
            overall_risk = sum(law_scores[lid].risk_score * LAWS[lid]['weight'] for lid in law_scores) / total_weight
        else:
            overall_compliance, overall_risk = 50, 50

        overall_level = (RiskLevel.LOW if overall_risk < 30 else
                         RiskLevel.MEDIUM if overall_risk < 50 else
                         RiskLevel.HIGH if overall_risk < 75 else RiskLevel.CRITICAL)

        summary = self.engine.generate_summary(overall_risk, contract_name)

        all_risks = [i for r_list in law_results.values() for r in r_list for i in r.issues]
        all_recs = [r for r_list in law_results.values() for res in r_list for r in res.recommendations]

        return FullReport(
            contract_id=contract_id, contract_name=contract_name,
            timestamp=datetime.now().isoformat(), total_clauses=len(clauses),
            laws_checked=laws, law_scores=law_scores,
            overall_compliance=overall_compliance, overall_risk=overall_risk,
            overall_level=overall_level, summary=summary,
            risks=list(set(all_risks))[:10], recommendations=list(set(all_recs))[:10],
            learned_keywords=list(set(
                kw for kws in self.contract_kg.learned_patterns.values() for kw in kws
            ))[:20]
        )


# ============================================================
# 6. Exploratory ML layer (kept disabled by default; see thesis Ch. 4)
# ============================================================

class TrainingDataManager:
    def __init__(self, test_size: float = 0.15, val_size: float = 0.15, random_state: int = 42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

    def prepare_data(self, X: np.ndarray, y: List[str], scale: bool = True) -> TrainingData:
        y_encoded = self.label_encoder.fit_transform(y)

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_encoded, test_size=self.test_size, random_state=self.random_state,
            stratify=y_encoded if len(set(y_encoded)) > 1 else None
        )
        val_ratio = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=self.random_state,
            stratify=y_temp if len(set(y_temp)) > 1 else None
        )

        if scale:
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
            X_test = self.scaler.transform(X_test)

        return TrainingData(
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train=y_train, y_val=y_val, y_test=y_test,
            label_encoder=self.label_encoder, scaler=self.scaler
        )


class ModelTrainer:
    MODELS = {
        "logistic": lambda: LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
        "gradient_boosting": lambda: GradientBoostingClassifier(n_estimators=100, random_state=42),
        "svm": lambda: SVC(kernel='rbf', probability=True, random_state=42),
    }

    def __init__(self):
        self.trained_models: Dict[str, any] = {}
        self.results: Dict[str, Dict] = {}

    def train(self, data: TrainingData, model_name: str = "random_forest") -> Dict:
        model = self.MODELS[model_name]()
        model.fit(data.X_train, data.y_train)
        results = {
            "model_name": model_name,
            "train": self._evaluate(model, data.X_train, data.y_train),
            "val": self._evaluate(model, data.X_val, data.y_val),
            "test": self._evaluate(model, data.X_test, data.y_test),
        }
        self.trained_models[model_name] = model
        self.results[model_name] = results
        return results

    def _evaluate(self, model, X, y) -> Dict:
        y_pred = model.predict(X)
        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y, y_pred, average='weighted', zero_division=0),
            "f1": f1_score(y, y_pred, average='weighted', zero_division=0),
            "confusion_matrix": confusion_matrix(y, y_pred),
        }

    def predict(self, X: np.ndarray, model_name: str = "random_forest") -> np.ndarray:
        return self.trained_models[model_name].predict(X)


class MLContractAnalyzer:
    """Wraps ContractAnalyzer with an optional, exploratory ML risk classifier.

    Per the thesis (Chapter 4), this classifier is exploratory: it is not
    trained by default in this deployment, and no headline metric is
    reported from it. ``analyze_with_ml`` falls back to the rule-based
    result alone (``ml_prediction=None``) until a model is explicitly
    trained on labeled data.
    """

    def __init__(self, embedder: SentenceTransformer):
        self.analyzer = ContractAnalyzer(embedder)
        self.data_manager = TrainingDataManager()
        self.trainer = ModelTrainer()
        self.is_trained = False

    def analyze_with_ml(self, contract_text: str, contract_name: str, laws: List[str]) -> Tuple[FullReport, Optional[str]]:
        report = self.analyzer.analyze(contract_text, contract_name, laws)

        ml_prediction = None
        if self.is_trained:
            clauses = self.analyzer.contract_kg.get_clauses(report.contract_id)
            embeddings = [c.embedding for c in clauses if c.embedding is not None]
            if embeddings:
                X = self.data_manager.scaler.transform(np.array(embeddings))
                predictions = self.trainer.predict(X)
                from collections import Counter
                ml_prediction = Counter(predictions).most_common(1)[0][0]
                ml_prediction = self.data_manager.label_encoder.inverse_transform([ml_prediction])[0]

        return report, ml_prediction


# ============================================================
# 7. PDF report export
# ============================================================

def save_pdf_report(report: FullReport, path: str) -> str:
    """Writes a one-page bilingual (Arabic/English) PDF summary of a
    FullReport, using a bundled Amiri font so Arabic text (contract name,
    issues, recommendations, which are produced in Arabic by the parser
    and reasoning engine) renders correctly instead of as glyph boxes."""
    font = _ARABIC_FONT if _ARABIC_FONT_OK else "Helvetica"
    font_bold = _ARABIC_FONT_BOLD if _ARABIC_FONT_OK else "Helvetica-Bold"

    doc = SimpleDocTemplate(path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('TitleAr', parent=styles['Title'], fontName=font_bold,
                                  alignment=TA_RIGHT, fontSize=16)
    h2_style = ParagraphStyle('H2Ar', parent=styles['Heading2'], fontName=font_bold,
                               alignment=TA_RIGHT, fontSize=12)
    h3_style = ParagraphStyle('H3Ar', parent=styles['Heading3'], fontName=font_bold,
                               alignment=TA_RIGHT, fontSize=11)
    body_style = ParagraphStyle('BodyAr', parent=styles['Normal'], fontName=font,
                                 alignment=TA_RIGHT, fontSize=10, leading=15)
    italic_style = ParagraphStyle('ItalicAr', parent=styles['Normal'], fontName=font,
                                   alignment=TA_RIGHT, fontSize=8, leading=12,
                                   textColor=colors.grey)

    story.append(Paragraph(_ar("تقرير فحص امتثال العقد — Contract Compliance Report"), title_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(_ar(f"العقد / Contract: {report.contract_name}"), body_style))
    story.append(Paragraph(f"Generated: {report.timestamp}", body_style))
    story.append(Paragraph(_ar(f"عدد البنود المحللة / Clauses analyzed: {report.total_clauses}"), body_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        _ar(f"الامتثال الإجمالي / Overall compliance: {report.overall_compliance:.0f}%   "
            f"المخاطر الإجمالية / Overall risk: {report.overall_risk:.0f}% ({report.overall_level.arabic})"),
        h2_style
    ))
    story.append(Spacer(1, 12))

    cell_style = ParagraphStyle('CellAr', parent=body_style, fontSize=8.5, leading=11, alignment=TA_RIGHT)
    header_style = ParagraphStyle('HeaderAr', parent=body_style, fontSize=9, leading=11,
                                   alignment=TA_RIGHT, textColor=colors.white, fontName=font_bold)

    table_data = [[
        Paragraph(_ar("القانون / Law"), header_style),
        Paragraph("Compliance %", header_style),
        Paragraph("Risk %", header_style),
        Paragraph(_ar("المستوى"), header_style),
        Paragraph(_ar("متوافق/جزئي/غير متوافق"), header_style),
    ]]
    for lid, s in report.law_scores.items():
        law_label = f"{LAWS.get(lid, {}).get('en', lid)} / " + _ar(LAWS.get(lid, {}).get('ar', lid))
        table_data.append([
            Paragraph(law_label, cell_style),
            f"{s.compliance_pct:.0f}",
            f"{s.risk_score:.0f}",
            _ar(s.risk_level.arabic),
            f"{s.compliant} / {s.partial} / {s.non_compliant}",
        ])
    table = Table(table_data, hAlign='CENTER', colWidths=[185, 65, 50, 60, 100])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 0), (-1, -1), font),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    if report.risks:
        story.append(Paragraph(_ar("أبرز الملاحظات / Issues found (sample):"), h3_style))
        for r in report.risks[:10]:
            story.append(Paragraph(_ar(f"- {r}"), body_style))
        story.append(Spacer(1, 8))

    if report.recommendations:
        story.append(Paragraph(_ar("التوصيات / Recommendations (sample):"), h3_style))
        for r in report.recommendations[:10]:
            story.append(Paragraph(_ar(f"- {r}"), body_style))

    story.append(Spacer(1, 16))
    story.append(Paragraph(
        _ar(
            "هذه النتيجة إشارة فحص آلية للمراجعة وليست حكمًا قانونيًا. هذا نموذج أولي بحثي "
            "(المرحلة الأولى من خارطة طريق تؤجل دمج نموذج لغوي توليدي لمرحلة لاحقة) أُعدّ لرسالة "
            "ماجستير في نظم المعلومات بجامعة الملك عبدالعزيز. — This output is an automated "
            "screening signal for review, not a legal determination. Research prototype, "
            "Phase 1 (no generative LLM); produced for a master's thesis in Information "
            "Systems at King Abdulaziz University."
        ),
        italic_style
    ))

    doc.build(story)
    return path


# ============================================================
# 8. Bootstrap: embedder, analyzer, sample law content
# ============================================================

print("Loading sentence-transformer model (this runs once at startup)...")
_embedder = SentenceTransformer(EMBED_MODEL_NAME)
ml_analyzer = MLContractAnalyzer(_embedder)

# NOTE: these are illustrative sample articles (2-3 per law), matching the
# thesis's documented functional-validation content. They are NOT the full,
# official texts of the six laws. Replace this block by loading full law
# texts (e.g. via a LawLoader over PDFs) before using this deployment for
# conclusive, publishable expert evaluation of legal correctness.
SAMPLE_LAWS = {
    "PDPL": [
        ("5", "الموافقة", "يجب الحصول على موافقة صريحة قبل معالجة البيانات الشخصية", ["موافقة", "بيانات", "معالجة"]),
        ("10", "الحقوق", "لصاحب البيانات حق الوصول والتصحيح والحذف", ["حقوق", "وصول", "حذف"]),
        ("15", "النقل", "يُحظر نقل البيانات خارج المملكة بدون ضمانات", ["نقل", "خارج", "ضمانات"]),
    ],
    "CYBER": [
        ("3", "الاختراق", "يُعاقب من يدخل نظاماً بدون إذن", ["اختراق", "دخول", "نظام"]),
        ("5", "التنصت", "يُعاقب من يتنصت على البيانات", ["تنصت", "بيانات", "شبكة"]),
    ],
    "ECC": [
        ("3", "الإفصاح", "يجب توفير معلومات واضحة عن مقدم الخدمة", ["إفصاح", "معلومات", "هوية"]),
        ("7", "المستهلك", "يحق للمستهلك الإلغاء خلال 7 أيام", ["مستهلك", "إلغاء", "استرجاع"]),
    ],
    "LABOR": [
        ("74", "العقد", "يجب أن يكون عقد العمل مكتوباً ويتضمن الأجر", ["عقد", "أجر", "عمل"]),
        ("84", "المكافأة", "يستحق العامل مكافأة نهاية الخدمة", ["مكافأة", "خدمة"]),
    ],
    "COMMERCIAL": [
        ("2", "الشكل القانوني", "يجب تسجيل الشركة وفق الشكل القانوني المحدد نظاماً", ["تسجيل", "شركة", "شكل"]),
    ],
    "ANTI_FRAUD": [
        ("4", "البيانات المضللة", "يُحظر تقديم بيانات أو تعهدات مضللة في التعاقد", ["مضللة", "تعهدات", "تعاقد"]),
    ],
}

for law_id, articles in SAMPLE_LAWS.items():
    for num, title, content, keywords in articles:
        ml_analyzer.analyzer.add_law_article(LawArticle(
            id=f"{law_id}_{num}", law_id=law_id, number=num,
            title=title, content=content, keywords=keywords
        ))
ml_analyzer.analyzer.build_law_index()
print(f"Loaded {len(ml_analyzer.analyzer.law_kg.articles)} sample law articles across {len(LAWS)} laws.")


# ============================================================
# 9. Gradio interface
# ============================================================

NOTE_MD = """
### About this prototype

This is a **Phase 1** research prototype for a master's thesis in Information Systems
(King Abdulaziz University): a knowledge-graph and semantic-matching framework that
screens business-contract clauses against six general Saudi laws, with fixed
rule-based reasoning (no generative LLM in this phase).

- The law content currently loaded is a **small set of illustrative example
  articles** per law, not the complete official legal texts.
- Results are an automated **screening signal**, not a legal determination.
- The exploratory ML risk classifier (Chapter 4 of the thesis) is **not trained**
  in this deployment; only the rule-based result is shown.

**هذا نموذج أولي بحثي (المرحلة الأولى)** لرسالة ماجستير في نظم المعلومات (جامعة
الملك عبدالعزيز): يفحص بنود العقود التجارية مقابل ستة أنظمة سعودية عامة باستخدام
رسم معرفي ومطابقة دلالية وقواعد منطقية ثابتة، دون نموذج لغوي توليدي في هذه
المرحلة. القوانين المحمّلة حاليًا مواد توضيحية محدودة وليست النصوص الرسمية
الكاملة، والنتائج إشارة فحص آلية وليست حكمًا قانونيًا.
"""


def _read_uploaded_file(contract_file) -> Optional[str]:
    """Supports both the str path (modern Gradio) and object-with-.name (older)."""
    if contract_file is None:
        return None
    path = contract_file if isinstance(contract_file, str) else getattr(contract_file, "name", None)
    if not path:
        return None
    try:
        if path.lower().endswith('.pdf'):
            text, _ = ml_analyzer.analyzer.parser.parse_pdf(path)
        elif path.lower().endswith('.docx'):
            text, _ = ml_analyzer.analyzer.parser.parse_docx(path)
        else:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        return text
    except Exception as e:
        raise RuntimeError(f"تعذرت قراءة الملف: {e}")


def analyze_ui(contract_text, contract_name, selected_laws, contract_file):
    if contract_file is not None:
        try:
            text = _read_uploaded_file(contract_file)
            if text:
                contract_text = text
        except RuntimeError as e:
            return str(e), None, None

    if not contract_text or not contract_text.strip():
        return "❌ أدخل نص العقد أو ارفع ملف (PDF/DOCX/TXT)", None, None
    if not selected_laws:
        return "❌ اختر قانونًا واحدًا على الأقل", None, None

    report, ml_pred = ml_analyzer.analyze_with_ml(contract_text, contract_name or "عقد", selected_laws)

    result = f"{report.summary}\n\n"
    for lid, s in report.law_scores.items():
        result += f"{s.risk_level.emoji} {LAWS[lid]['ar']} ({LAWS[lid]['en']}): توافق {s.compliance_pct:.0f}%، مخاطر {s.risk_score:.0f}%\n"
    if report.risks:
        result += "\nأبرز الملاحظات:\n" + "\n".join(f"- {r}" for r in report.risks[:6])
    if ml_pred:
        result += f"\n\n\U0001F916 تنبؤ ML (استكشافي): {ml_pred}"

    fig = None
    if report.law_scores:
        laws_en = [LAWS[lid]['en'] for lid in report.law_scores]
        compliance = [report.law_scores[lid].compliance_pct for lid in report.law_scores]
        risk = [report.law_scores[lid].risk_score for lid in report.law_scores]
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Compliance %', x=laws_en, y=compliance, marker_color='#2ecc71'))
        fig.add_trace(go.Bar(name='Risk %', x=laws_en, y=risk, marker_color='#e74c3c'))
        fig.update_layout(barmode='group', height=360, template='plotly_white',
                           margin=dict(l=10, r=10, t=30, b=10))

    pdf_path = save_pdf_report(report, f"/tmp/report_{report.contract_id}.pdf")

    return result, fig, pdf_path


with gr.Blocks(title="Contract Compliance Analyzer", theme=gr.themes.Soft(primary_hue="purple")) as demo:
    gr.Markdown("# \U0001F4DC Contract Compliance Analyzer — محلل امتثال العقود (Phase 1)")
    gr.Markdown(NOTE_MD)

    with gr.Row():
        with gr.Column():
            contract_name = gr.Textbox(label="اسم العقد / Contract name", placeholder="عقد خدمات...")
            contract_text = gr.Textbox(label="نص العقد / Contract text", lines=10)
            contract_file = gr.File(label="\U0001F4C4 رفع ملف / Upload file (PDF, DOCX, TXT)",
                                     file_types=[".pdf", ".docx", ".txt"])
            selected_laws = gr.CheckboxGroup(
                choices=[(f"{LAWS[k]['ar']} / {LAWS[k]['en']}", k) for k in LAWS],
                value=["PDPL", "CYBER"],
                label="القوانين المطلوب الفحص عليها / Laws to check against"
            )
            analyze_btn = gr.Button("\U0001F50D تحليل / Analyze", variant="primary")

        with gr.Column():
            result_text = gr.Textbox(label="النتائج / Results", lines=14)
            result_chart = gr.Plot(label="الرسم البياني / Chart")
            pdf_file = gr.File(label="\U0001F4E5 تقرير PDF / PDF report")

    analyze_btn.click(
        analyze_ui,
        [contract_text, contract_name, selected_laws, contract_file],
        [result_text, result_chart, pdf_file]
    )

    gr.Markdown(
        "---\n"
        "This deployment is used to collect expert feedback for the thesis's evaluation "
        "chapter (correctness, clarity, traceability, usability). Thank you for testing it."
    )

if __name__ == "__main__":
    demo.queue().launch()
