"""Offline grounded-answer tests and the optional live OpenRouter check."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient

from app import llm
from app.config import settings
from app.llm import (
    LLMConfigurationError,
    OpenRouterAPIError,
    OpenRouterGroundedGenerator,
)
from app.rag_pipeline import ChatAnswerResponse, ChatSource, RAGPipeline

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
        if conflict_information.get("conflicting_evidence"):
            claims = []
            for expected_doc in ("Prospectus 2025-26.pdf", "Student Grievance Redressal Committee.pdf"):
                match = next((item for item in evidence if item["document"] == expected_doc), None)
                if match:
                    record = next(
                        (item for item in conflict_information["records"]
                         if item["document"] == expected_doc and item["page"] == match["page"]),
                        None,
                    )
                    if record and record["name"].lower() in match["text"].lower():
                        claims.append(f"{record['name']} is listed as {record['role']} [{match['source_id']}]")
            if len(claims) < 2:
                return "The records conflict, but I cannot identify both source pages."
            return "The supplied records say " + " and ".join(claims) + "."

        excerpt = " ".join(evidence[0]["text"].split())[:120]
        return f'The retrieved text states “{excerpt}” [{evidence[0]["source_id"]}].'


class FakeCompletions:
    def __init__(self, text=None, error=None, outcomes=None):
        self.text = text
        self.error = error
        self.outcomes = outcomes or {}
        self.kwargs = None
        self.calls = 0
        self.requests = []

    def create(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        self.requests.append(kwargs)
        outcome = self.outcomes.get(kwargs["model"])
        if isinstance(outcome, Exception):
            raise outcome
        if outcome is not None:
            if isinstance(outcome, str):
                text = outcome
            else:
                return outcome
        else:
            if self.error:
                raise self.error
            text = self.text
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
        )


class FakeClient:
    def __init__(self, completions):
        self.chat = SimpleNamespace(completions=completions)


class LLMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_reranker_state = settings.CROSS_ENCODER_ENABLED
        settings.CROSS_ENCODER_ENABLED = False

    @classmethod
    def tearDownClass(cls):
        settings.CROSS_ENCODER_ENABLED = cls.previous_reranker_state

    def test_ten_retrieval_cases_and_generation_gates(self):
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
                self.assertEqual(should_generate, response.generation == "openrouter")

        self.assertEqual(len(fake.calls), sum(case[2] for case in cases))
        for call in fake.calls:
            self.assertLessEqual(len(call["evidence"]), 5)
            self.assertTrue(all(item["text"] for item in call["evidence"]))
            self.assertNotIn("chunks.json", str(call["evidence"]))
        conflict_call = next(call for call in fake.calls if call["conflict_information"]["conflicting_evidence"])
        conflict_docs = {record["document"] for record in conflict_call["conflict_information"]["records"]}
        self.assertIn("Prospectus 2025-26.pdf", conflict_docs)
        self.assertIn("Student Grievance Redressal Committee.pdf", conflict_docs)

    def test_openai_compatible_message_content_is_returned_unchanged(self):
        answer_content = "The IT HOD evidence is present."
        completions = FakeCompletions(answer_content)
        evidence = [{
            "document": "Prospectus 2025-26.pdf",
            "document_type": "prospectus",
            "page": 15,
            "academic_year": "2025-26",
            "chunk_id": "MHSSCE_00123",
            "text": "Dr. Zainab Mirza HOD IT",
        }]
        answer = OpenRouterGroundedGenerator(
            client=FakeClient(completions),
            model="test-model",
        ).generate_grounded_answer(
            "Who is the HOD of IT?", evidence, {"status": "SUFFICIENT"}, {}
        )

        self.assertEqual(answer, answer_content)
        request = completions.kwargs
        self.assertEqual(request["model"], "test-model")
        self.assertEqual(request["temperature"], settings.OPENROUTER_TEMPERATURE)
        self.assertEqual(request["max_tokens"], settings.OPENROUTER_MAX_OUTPUT_TOKENS)
        self.assertEqual(request["extra_body"], {"reasoning": {"effort": "low"}})
        self.assertEqual([message["role"] for message in request["messages"]], ["system", "user"])
        self.assertIn("Answer only from the supplied retrieved evidence", request["messages"][0]["content"])
        self.assertIn("2025-26", request["messages"][1]["content"])
        self.assertIn("MHSSCE_00123", request["messages"][1]["content"])
        self.assertNotIn('"source_id"', request["messages"][1]["content"])
        self.assertNotIn("secret corpus text", request["messages"][1]["content"])

    def test_response_content_is_not_rejected_for_citation_format_or_reasoning_fields(self):
        answer_content = "The fee is 10000."
        response = SimpleNamespace(
            id="response-test",
            choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    role="assistant",
                    content=answer_content,
                    reasoning="private reasoning",
                    reasoning_details={"text": "private reasoning details"},
                ),
            )],
        )
        evidence = [{"document": "Fee.pdf", "page": 1, "text": "Fee evidence"}]
        generator = OpenRouterGroundedGenerator(
            client=FakeClient(FakeCompletions(outcomes={settings.OPENROUTER_MODEL: response}))
        )
        answer = generator.generate_grounded_answer(
            "What is the fee?", evidence, {"status": "SUFFICIENT"}
        )
        self.assertEqual(answer, answer_content)

    def test_missing_key_provider_error_and_empty_content_are_controlled(self):
        with patch.object(llm.settings, "OPENROUTER_API_KEY", None):
            with self.assertRaisesRegex(LLMConfigurationError, "OPENROUTER_API_KEY"):
                OpenRouterGroundedGenerator().generate_grounded_answer(
                    "question", [{"source_id": "S1", "document": "Doc.pdf", "page": 1, "text": "evidence"}],
                    {"status": "SUFFICIENT"},
                )

        with self.assertRaises(OpenRouterAPIError):
            OpenRouterGroundedGenerator(
                client=FakeClient(FakeCompletions(error=RuntimeError("offline")))
            ).generate_grounded_answer(
                "question", [{"source_id": "S1", "document": "Doc.pdf", "page": 1, "text": "evidence"}],
                {"status": "SUFFICIENT"},
            )

        with self.assertRaisesRegex(OpenRouterAPIError, "no usable answer content"):
            OpenRouterGroundedGenerator(
                client=FakeClient(FakeCompletions(text=None))
            ).generate_grounded_answer(
                "question", [{"source_id": "S1", "document": "Doc.pdf", "page": 1, "text": "evidence"}],
                {"status": "SUFFICIENT"},
            )

    def test_provider_error_categories_and_rate_limit_without_fallback(self):
        response = SimpleNamespace(status_code=401, headers={}, request=None)
        cases = [
            (AuthenticationError("auth", response=response, body=None), 502, False),
            (RateLimitError("rate", response=response, body=None), 429, True),
            (APIStatusError(
                "server",
                response=SimpleNamespace(status_code=503, headers={}, request=None),
                body=None,
            ), 502, True),
            (APITimeoutError(request=None), 504, True),
            (APIConnectionError(request=None), 502, True),
        ]
        for provider_error, http_status, retryable in cases:
            with self.subTest(error=type(provider_error).__name__):
                mapped, can_retry = OpenRouterGroundedGenerator._provider_error(provider_error)
                self.assertEqual(mapped.http_status, http_status)
                self.assertEqual(can_retry, retryable)

        rate_limit = RateLimitError("rate", response=response, body=None)
        completions = FakeCompletions(error=rate_limit)
        evidence = [{"source_id": "S1", "document": "Doc.pdf", "page": 1, "text": "evidence"}]
        with patch.object(settings, "OPENROUTER_FALLBACK_MODELS", ""):
            with patch.object(llm.time, "sleep") as sleep:
                with self.assertRaises(OpenRouterAPIError) as raised:
                    OpenRouterGroundedGenerator(
                        client=FakeClient(completions),
                    ).generate_grounded_answer(
                        "question", evidence, {"status": "SUFFICIENT"}
                    )
        self.assertEqual(raised.exception.http_status, 429)
        self.assertEqual(completions.calls, 1)
        sleep.assert_not_called()

    def test_rate_limit_uses_fallback_with_identical_prompt_and_evidence(self):
        rate_response = SimpleNamespace(
            status_code=429,
            headers={"retry-after": "60"},
            request=None,
        )
        primary_rate_limit = RateLimitError("rate", response=rate_response, body=None)
        completions = FakeCompletions(
            outcomes={
                settings.OPENROUTER_MODEL: primary_rate_limit,
                "fallback/model:free": "The retrieved evidence supports this [S1].",
            }
        )
        evidence = [{
            "source_id": "S1",
            "document": "Prospectus 2025-26.pdf",
            "page": 15,
            "academic_year": "2025-26",
            "chunk_id": "MHSSCE_00123",
            "text": "Dr. Zainab Mirza HOD IT",
        }]
        with patch.object(settings, "OPENROUTER_FALLBACK_MODELS", " fallback/model:free, , fallback/model:free "):
            with patch.object(llm.time, "sleep") as sleep:
                answer = OpenRouterGroundedGenerator(
                    client=FakeClient(completions),
                ).generate_grounded_answer(
                    "Who is HOD IT?", evidence, {"status": "SUFFICIENT"}, {}
                )

        self.assertEqual(
            [request["model"] for request in completions.requests],
            [settings.OPENROUTER_MODEL, "fallback/model:free"],
        )
        primary_request, fallback_request = completions.requests
        self.assertEqual(primary_request["messages"], fallback_request["messages"])
        self.assertEqual(primary_request["temperature"], fallback_request["temperature"])
        self.assertEqual(primary_request["max_tokens"], fallback_request["max_tokens"])
        self.assertEqual(sleep.call_args_list, [unittest.mock.call(llm.MAX_RETRY_AFTER_WAIT_SECONDS)])
        self.assertEqual(answer, "The retrieved evidence supports this [S1].")

    def test_transient_error_and_empty_content_advance_to_next_fallback(self):
        rate_response = SimpleNamespace(status_code=429, headers={}, request=None)
        first_limit = RateLimitError("rate", response=rate_response, body=None)
        completions = FakeCompletions(
            outcomes={
                settings.OPENROUTER_MODEL: first_limit,
                "fallback/empty": SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
                ),
                "fallback/answer": "The retrieved evidence supports this [S1].",
            }
        )
        evidence = [{"source_id": "S1", "document": "Doc.pdf", "page": 1, "text": "evidence"}]
        with patch.object(
            settings,
            "OPENROUTER_FALLBACK_MODELS",
            "fallback/empty,fallback/answer",
        ):
            answer = OpenRouterGroundedGenerator(client=FakeClient(completions)).generate_grounded_answer(
                "question", evidence, {"status": "SUFFICIENT"}
            )
        self.assertEqual(
            [request["model"] for request in completions.requests],
            [settings.OPENROUTER_MODEL, "fallback/empty", "fallback/answer"],
        )
        self.assertEqual(answer, "The retrieved evidence supports this [S1].")

    def test_authentication_failure_does_not_try_fallback(self):
        auth_response = SimpleNamespace(status_code=401, headers={}, request=None)
        authentication_error = AuthenticationError("auth", response=auth_response, body=None)
        completions = FakeCompletions(error=authentication_error)
        evidence = [{"source_id": "S1", "document": "Doc.pdf", "page": 1, "text": "evidence"}]
        with patch.object(settings, "OPENROUTER_FALLBACK_MODELS", "fallback/model"):
            with self.assertRaises(OpenRouterAPIError):
                OpenRouterGroundedGenerator(client=FakeClient(completions)).generate_grounded_answer(
                    "question", evidence, {"status": "SUFFICIENT"}
                )
        self.assertEqual(completions.calls, 1)

    def test_api_maps_configuration_and_provider_errors(self):
        from app import main as api

        client = TestClient(api.app)
        success = ChatAnswerResponse(
            question="Who is HOD IT?",
            answer="The retrieved evidence supports this [Source: Prospectus 2025-26.pdf, p. 15].",
            status="SUFFICIENT",
            sources=[ChatSource(
                document="Prospectus 2025-26.pdf",
                page=15,
                academic_year="2025-26",
                chunk_id="MHSSCE_00123",
            )],
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

        with patch.object(api.pipeline, "ask", side_effect=OpenRouterAPIError("provider unavailable")):
            response = client.post("/api/ask", json={"question": "Who is the HOD of IT?"})
            self.assertEqual(response.status_code, 502)

        with patch.object(api.pipeline, "ask", side_effect=OpenRouterAPIError("request timed out", 504)):
            response = client.post("/api/ask", json={"question": "Who is the HOD of IT?"})
            self.assertEqual(response.status_code, 504)


def run_live_checks() -> bool:
    """Exercise the reported queries against live retrieval and OpenRouter."""
    print("Testing the four reported live RAG queries...")
    cases = [
        ("Who is the HOD of Information Technology?", "SUFFICIENT"),
        ("Show me information about the grievance redressal committee.", "SUFFICIENT"),
        ("What is the fee structure for 2026-27?", "SUFFICIENT"),
        ("What is the Pragati scholarship?", "SUFFICIENT"),
    ]

    pipeline = RAGPipeline()
    success = True
    for question, expected_status in cases:
        try:
            result = pipeline.ask(question)
            passed = result.status == expected_status
            if expected_status == "SUFFICIENT":
                passed = passed and result.generation == "openrouter"
                passed = passed and bool(result.sources)
                passed = passed and bool(result.answer.strip())
            elif expected_status == "INSUFFICIENT_KNOWLEDGE":
                passed = passed and result.generation == "abstention"
            print(f"{'SUCCESS' if passed else 'FAILURE'}: {question} [{result.status}]")
            success = success and passed
        except Exception as error:
            print(f"FAILURE: {question} ({type(error).__name__}: {error})")
            success = False
    return success


if __name__ == "__main__":
    if "--live" in sys.argv:
        sys.argv.remove("--live")
        raise SystemExit(0 if run_live_checks() else 1)
    unittest.main(verbosity=2)
