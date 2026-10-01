"""Offline tests for grounded answer generation and API safety."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient

from app import llm
from app.config import settings
from app.llm import (
    GeminiAPIError,
    GeminiGroundedGenerator,
    GroundingValidationError,
    LLMConfigurationError,
)
from app.rag_pipeline import ChatAnswerResponse, ChatSource, RAGPipeline
from app.retrieval import retrieve_with_evaluation


EXPECTED_SOURCES = {
    "HOD IT": ("Prospectus 2025-26.pdf", 15),
    "HOD AIML": ("Prospectus 2025-26.pdf", 12),
    "HOD Computer Engineering": ("Prospectus 2025-26.pdf", 11),
    "fee structure 2026-27": ("fee structure 2026-27.pdf", 1),
    "Pragati/Saksham scholarship": ("Pragati_Saksham_scholarship scheme 2026-27.pdf", 1),
    "academic calendar": ("MHSSCOE-SSR.pdf", 21),
}


class GroundedMock:
    def __init__(self):
        self.calls = []

    def __call__(self, *, question, evidence, answerability, conflict_information):
        call = {
            "question": question,
            "evidence": evidence,
            "answerability": answerability,
            "conflict_information": conflict_information,
        }
        self.calls.append(call)
        refs = []
        if conflict_information.get("conflicting_evidence"):
            claims = []
            for expected_doc in ("Prospectus 2025-26.pdf", "Student Grievance Redressal Committee.pdf"):
                match = next((item for item in evidence if item["document"] == expected_doc), None)
                if match:
                    refs.append(f"[{match['source_id']}]")
                    record = next((item for item in conflict_information["records"]
                                   if item["document"] == expected_doc and item["page"] == match["page"]), None)
                    if record and record["name"].lower() in match["text"].lower():
                        claims.append(f"{record['name']} is listed as {record['role']} {refs[-1]}")
            if len(refs) < 2:
                return "The records conflict, but I cannot identify both source pages."
            if len(claims) < 2:
                raise AssertionError("Conflict claims must be stated in the cited evidence text.")
            answer = "The supplied records say " + " and ".join(claims) + "."
            call["answer"] = answer
            return answer
        excerpt = " ".join(evidence[0]["text"].split())[:120]
        answer = f'The retrieved text states “{excerpt}” [{evidence[0]["source_id"]}].'
        call["answer"] = answer
        return answer


class FakeModels:
    def __init__(self, text=None, error=None):
        self.text = text
        self.error = error
        self.contents = None
        self.kwargs = None

    def generate_content(self, **kwargs):
        self.kwargs = kwargs
        self.contents = kwargs["contents"]
        if self.error:
            raise self.error
        return type("Response", (), {"text": self.text})()


class FakeClient:
    def __init__(self, models):
        self.models = models


class LLMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_reranker_state = settings.CROSS_ENCODER_ENABLED
        settings.CROSS_ENCODER_ENABLED = False

    @classmethod
    def tearDownClass(cls):
        settings.CROSS_ENCODER_ENABLED = cls.previous_reranker_state

    def test_ten_retrieval_cases_and_grounded_generation_gates(self):
        fake = GroundedMock()
        pipeline = RAGPipeline(answer_generator=fake)
        cases = [
            ("Who is the HOD of Information Technology?", "HOD IT", True),
            ("Who is the HOD of AIML?", "HOD AIML", True),
            ("Who is the HOD of Computer Engineering?", "HOD Computer Engineering", True),
            ("What is the fee structure for 2026-27?", "fee structure 2026-27", True),
            ("Who can apply for the Pragati scholarship?", "Pragati/Saksham scholarship", True),
            ("What is the examination date?", None, False),
            ("Am I personally eligible for this scholarship?", None, False),
            ("Who is the principal of MHSSCE?", "conflict", True),
            ("What is the academic calendar?", "academic calendar", True),
            ("What is the hostel mess menu?", None, False),
        ]
        for question, expected, should_generate in cases:
            with self.subTest(question=question):
                response = pipeline.ask(question)
                if expected is None:
                    self.assertEqual(response.status, "INSUFFICIENT_KNOWLEDGE")
                    self.assertEqual(response.generation, "abstention")
                    if "personally eligible" in question:
                        self.assertTrue(response.requires_personal_context)
                        self.assertIn("personal scholarship eligibility", response.confidence_note.lower())
                elif expected == "conflict":
                    self.assertEqual(response.status, "CONFLICTING_EVIDENCE")
                    self.assertTrue(response.conflicting_evidence)
                    docs = {source.document for source in response.sources}
                    self.assertIn("Prospectus 2025-26.pdf", docs)
                    self.assertIn("Student Grievance Redressal Committee.pdf", docs)
                else:
                    self.assertEqual(response.status, "SUFFICIENT")
                    document, page = EXPECTED_SOURCES[expected]
                    self.assertTrue(any(source.document == document and source.page == page for source in response.sources))
                    call = next(item for item in fake.calls if item["question"] == question)
                    self.assertIn(" ".join(call["evidence"][0]["text"].split())[:120], response.answer)
                self.assertEqual(should_generate, response.generation == "gemini")

        self.assertEqual(len(fake.calls), sum(case[2] for case in cases))
        for call in fake.calls:
            self.assertLessEqual(len(call["evidence"]), 5)
            self.assertTrue(all(item["text"] for item in call["evidence"]))
            self.assertNotIn("chunks.json", str(call["evidence"]))
        conflict_call = next(call for call in fake.calls if call["conflict_information"]["conflicting_evidence"])
        conflict_docs = {record["document"] for record in conflict_call["conflict_information"]["records"]}
        self.assertIn("Prospectus 2025-26.pdf", conflict_docs)
        self.assertIn("Student Grievance Redressal Committee.pdf", conflict_docs)

    def test_citations_are_rendered_from_metadata_and_evidence_is_bounded(self):
        models = FakeModels("The IT HOD evidence is present [S1].")
        client = FakeClient(models)
        evidence = [{
            "source_id": "S1",
            "document": "Prospectus 2025-26.pdf",
            "document_type": "prospectus",
            "page": 15,
            "academic_year": "2025-26",
            "chunk_id": "MHSSCE_00123",
            "text": "Dr. Zainab Mirza HOD IT",
        }]
        answer = GeminiGroundedGenerator(client=client, model="test-model").generate_grounded_answer(
            "Who is the HOD of IT?", evidence, {"status": "SUFFICIENT"}, {}
        )
        self.assertIn("[Source: Prospectus 2025-26.pdf, p. 15]", answer)
        self.assertNotIn("[S1]", answer)
        self.assertIn("2025-26", models.contents)
        self.assertIn("MHSSCE_00123", models.contents)
        self.assertNotIn("secret corpus text", models.contents)
        self.assertIn("Answer only from the supplied evidence", models.kwargs["config"].system_instruction)

    def test_missing_citations_and_fabricated_citations_are_rejected(self):
        evidence = [{"source_id": "S1", "document": "Fee.pdf", "page": 1, "text": "Fee evidence"}]
        for answer in ("The fee is 10000.", "The fee is 10000 [S9].", "The fee is 10000 [Source: Other.pdf, p. 9]."):
            models = FakeModels(answer)
            with self.subTest(answer=answer), self.assertRaises(GroundingValidationError):
                GeminiGroundedGenerator(client=FakeClient(models)).generate_grounded_answer(
                    "What is the fee?", evidence, {"status": "SUFFICIENT"}
                )

        conflicting_evidence = [
            {"source_id": "S1", "document": "Prospectus.pdf", "page": 7, "text": "Dr. Ganesh is In-charge Principal."},
            {"source_id": "S2", "document": "Committee.pdf", "page": 1, "text": "Dr. Pathan is Principal."},
        ]
        models = FakeModels("The records differ [S1].")
        with self.assertRaisesRegex(GroundingValidationError, "two conflicting sources"):
            GeminiGroundedGenerator(client=FakeClient(models)).generate_grounded_answer(
                "Who is Principal?",
                conflicting_evidence,
                {"status": "CONFLICTING_EVIDENCE"},
                {"conflicting_evidence": True, "records": [
                    {"document": "Prospectus.pdf", "page": 7},
                    {"document": "Committee.pdf", "page": 1},
                ]},
            )
        models = FakeModels("One record names Dr. Ganesh as in-charge [S1], while another names Dr. Pathan as principal [S2].")
        answer = GeminiGroundedGenerator(client=FakeClient(models)).generate_grounded_answer(
            "Who is Principal?",
            conflicting_evidence,
            {"status": "CONFLICTING_EVIDENCE"},
            {"conflicting_evidence": True, "records": [
                {"document": "Prospectus.pdf", "page": 7},
                {"document": "Committee.pdf", "page": 1},
            ]},
        )
        self.assertIn("[Source: Prospectus.pdf, p. 7]", answer)
        self.assertIn("[Source: Committee.pdf, p. 1]", answer)

    def test_missing_key_and_gemini_api_failures_are_controlled(self):
        with patch.object(llm.settings, "GEMINI_API_KEY", None):
            with self.assertRaisesRegex(LLMConfigurationError, "GEMINI_API_KEY"):
                GeminiGroundedGenerator().generate_grounded_answer(
                    "question", [{"source_id": "S1", "document": "Doc.pdf", "page": 1, "text": "evidence"}],
                    {"status": "SUFFICIENT"},
                )
        models = FakeModels(error=RuntimeError("offline"))
        with self.assertRaises(GeminiAPIError):
            GeminiGroundedGenerator(client=FakeClient(models)).generate_grounded_answer(
                "question", [{"source_id": "S1", "document": "Doc.pdf", "page": 1, "text": "evidence"}],
                {"status": "SUFFICIENT"},
            )

    def test_api_maps_configuration_and_provider_errors(self):
        from app import main as api

        client = TestClient(api.app)
        success = ChatAnswerResponse(
            question="Who is HOD IT?",
            answer="The retrieved evidence supports this [Source: Prospectus 2025-26.pdf, p. 15].",
            status="SUFFICIENT",
            sources=[ChatSource(document="Prospectus 2025-26.pdf", page=15, academic_year="2025-26", chunk_id="MHSSCE_00123")],
        )
        with patch.object(api.pipeline, "ask", return_value=success):
            response = client.post("/api/ask", json={"question": "Who is HOD IT?"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "SUFFICIENT")
            self.assertEqual(response.json()["sources"][0]["page"], 15)
            self.assertNotIn("retrieval_score", response.json()["sources"][0])
        with patch.object(api.pipeline, "ask", side_effect=LLMConfigurationError("missing test key")):
            response = client.post("/api/ask", json={"question": "Who is the HOD of IT?"})
            self.assertEqual(response.status_code, 503)
            self.assertIn("missing test key", response.json()["detail"])
        with patch.object(api.pipeline, "ask", side_effect=GeminiAPIError("provider unavailable")):
            response = client.post("/api/ask", json={"question": "Who is the HOD of IT?"})
            self.assertEqual(response.status_code, 502)


if __name__ == "__main__":
    unittest.main(verbosity=2)
