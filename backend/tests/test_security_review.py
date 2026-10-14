import asyncio
import io
import unittest
import zipfile
from unittest.mock import patch

from bson import ObjectId
from fastapi import UploadFile
from pydantic import ValidationError
from fastapi.testclient import TestClient

from app.core.config import _float_setting, _integer_setting
from app.core.security import hash_password, verify_password
from app.main import app
from app.schemas.auth import RegisterRequest
from app.routes.auth import get_current_user, get_database
from app.services import answer_service
from app.services.document_service import (
    InvalidDocumentError,
    _document_filter,
    get_document,
    validate_upload,
)
from app.services.document_service import DocumentNotFoundError
from app.services.vector_search_service import search_chunks


class FakeChunks:
    def __init__(self):
        self.pipeline = None

    def aggregate(self, pipeline, **kwargs):
        self.pipeline = pipeline
        return []


class BackendSecurityReviewTests(unittest.TestCase):
    def test_configuration_settings_reject_malformed_or_out_of_range_values(self):
        with patch.dict("os.environ", {"TEST_SETTING": "not-a-number"}):
            with self.assertRaisesRegex(ValueError, "TEST_SETTING must be an integer"):
                _integer_setting("TEST_SETTING", "5", minimum=1)
        with patch.dict("os.environ", {"TEST_SETTING": "0"}):
            with self.assertRaisesRegex(ValueError, "TEST_SETTING must be between 1 and 20"):
                _integer_setting("TEST_SETTING", "5", minimum=1, maximum=20)
        with patch.dict("os.environ", {"TEST_SETTING": "nan"}):
            with self.assertRaisesRegex(ValueError, "TEST_SETTING must be finite"):
                _float_setting("TEST_SETTING", "0.5", minimum=0.0, maximum=1.0)

    def test_public_registration_cannot_select_role_or_other_fields(self):
        request = RegisterRequest(name="New Employee", email="new@example.com", password="password123")
        self.assertEqual(request.name, "New Employee")
        with self.assertRaises(ValidationError):
            RegisterRequest(name="New Employee", email="new@example.com", password="password123", role="ADMIN")

    def test_password_hash_is_not_plaintext_and_verifies(self):
        encoded = hash_password("correct horse battery staple")
        self.assertNotEqual(encoded, "correct horse battery staple")
        self.assertTrue(verify_password("correct horse battery staple", encoded))
        self.assertFalse(verify_password("wrong password", encoded))

    def test_employee_is_forbidden_from_admin_user_listing(self):
        app.dependency_overrides[get_current_user] = lambda: {"_id": ObjectId(), "role": "EMPLOYEE"}
        app.dependency_overrides[get_database] = lambda: {"users": object()}
        try:
            with TestClient(app) as client:
                response = client.get("/api/users")
            self.assertEqual(response.status_code, 403)
        finally:
            app.dependency_overrides.clear()

    def test_admin_can_access_admin_user_listing(self):
        app.dependency_overrides[get_current_user] = lambda: {"_id": ObjectId(), "role": "ADMIN"}
        app.dependency_overrides[get_database] = lambda: {"users": object()}
        try:
            with patch("app.routes.users.list_users", return_value=[]):
                with TestClient(app) as client:
                    response = client.get("/api/users")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), [])
        finally:
            app.dependency_overrides.clear()

    def test_employee_document_filter_is_always_scoped_to_current_owner(self):
        owner_id, other_id = ObjectId(), ObjectId()
        user = {"_id": owner_id}
        base = _document_filter(current_user=user, is_admin=False, requested_uploader=None, file_type=None)
        requested = _document_filter(
            current_user=user,
            is_admin=False,
            requested_uploader=str(other_id),
            file_type=None,
        )
        self.assertEqual(base, {"uploaded_by": owner_id})
        self.assertEqual(requested, {"uploaded_by": owner_id})

    def test_employee_cannot_fetch_another_users_document(self):
        owner_id = ObjectId()
        other_document_id = ObjectId()

        class Documents:
            def __init__(self):
                self.query = None

            def find_one(self, query):
                self.query = query
                return None

        documents = Documents()
        with self.assertRaises(DocumentNotFoundError):
            get_document(
                documents,
                str(other_document_id),
                current_user={"_id": owner_id},
                is_admin=False,
            )
        self.assertEqual(documents.query, {"_id": other_document_id, "uploaded_by": owner_id})

    def test_vector_search_filters_candidates_by_employee_owner(self):
        owner_id = ObjectId()
        chunks = FakeChunks()
        with patch("app.services.vector_search_service.embed_query", return_value=[0.0] * 384):
            self.assertEqual(
                search_chunks(
                    chunks,
                    query="policy",
                    top_k=5,
                    similarity_threshold=0.0,
                    current_user={"_id": owner_id},
                    is_admin=False,
                ),
                [],
            )
        vector_stage = chunks.pipeline[0]["$vectorSearch"]
        self.assertEqual(vector_stage["filter"], {"uploaded_by": owner_id})

    def test_admin_vector_search_does_not_apply_employee_filter(self):
        chunks = FakeChunks()
        with patch("app.services.vector_search_service.embed_query", return_value=[0.0] * 384):
            search_chunks(
                chunks,
                query="policy",
                top_k=5,
                similarity_threshold=0.0,
                current_user={"_id": ObjectId()},
                is_admin=True,
            )
        self.assertNotIn("filter", chunks.pipeline[0]["$vectorSearch"])

    def test_answer_service_reuses_m10_retrieval_and_keeps_source_metadata(self):
        source = {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "filename": "policy.pdf",
            "score": 0.9,
            "text": "Relevant policy text",
        }
        with (
            patch.object(answer_service, "retrieve_context", return_value={"context": "Relevant policy text", "sources": [source]}) as retrieval,
            patch.object(answer_service, "generate_answer", return_value="Grounded answer") as generate,
        ):
            result = answer_service.answer_question(
                object(), query="Question", current_user={"_id": ObjectId()}, is_admin=False
            )
        retrieval.assert_called_once()
        generate.assert_called_once_with("Question", "Relevant policy text")
        self.assertEqual(result["answer"], "Grounded answer")
        self.assertEqual(result["sources"], [{key: source[key] for key in ("chunk_id", "document_id", "filename", "score")}])

    def test_docx_zip_bomb_is_rejected_before_extraction(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("[Content_Types].xml", "types")
            archive.writestr("word/document.xml", "x" * (2 * 1024 * 1024))
        stream.seek(0)
        upload = UploadFile(filename="document.docx", file=stream, headers={"content-type": "application/octet-stream"})
        with self.assertRaises(InvalidDocumentError):
            asyncio.run(validate_upload(upload))


if __name__ == "__main__":
    unittest.main()
