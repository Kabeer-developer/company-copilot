import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from bson import ObjectId
from fastapi.testclient import TestClient
from pymongo.errors import PyMongoError

from app.main import app
from app.routes.auth import get_current_user, get_database
from app.services import llm_service, retrieval_service


class FakeCursor:
    def __init__(self, records):
        self.records = list(records)

    def sort(self, fields):
        for name, direction in reversed(fields):
            self.records.sort(key=lambda record: record.get(name), reverse=direction < 0)
        return self

    def skip(self, amount):
        self.records = self.records[amount:]
        return self

    def limit(self, amount):
        if amount > 0:
            self.records = self.records[:amount]
        return self

    def __iter__(self):
        return iter(self.records)


class FakeCollection:
    def __init__(self):
        self.records = []
        self.indexes = []
        self.fail_assistant_insert = False

    def create_index(self, spec, name=None):
        self.indexes.append((spec, name))
        return name

    @staticmethod
    def _matches(record, query):
        for key, expected in query.items():
            actual = record.get(key)
            if isinstance(expected, dict) and "$ne" in expected:
                if actual == expected["$ne"]:
                    return False
            elif actual != expected:
                return False
        return True

    def insert_one(self, record):
        if record.get("role") == "assistant" and self.fail_assistant_insert:
            raise PyMongoError("simulated assistant persistence error")
        record.setdefault("_id", ObjectId())
        self.records.append(record)
        return SimpleNamespace(inserted_id=record["_id"])

    def find(self, query):
        return FakeCursor([record for record in self.records if self._matches(record, query)])

    def find_one(self, query):
        return next((record for record in self.records if self._matches(record, query)), None)

    def count_documents(self, query):
        return sum(self._matches(record, query) for record in self.records)

    def update_one(self, query, update):
        record = self.find_one(query)
        if record is None:
            return SimpleNamespace(matched_count=0, modified_count=0)
        record.update(update.get("$set", {}))
        return SimpleNamespace(matched_count=1, modified_count=1)

    def delete_many(self, query):
        before = len(self.records)
        self.records = [record for record in self.records if not self._matches(record, query)]
        return SimpleNamespace(deleted_count=before - len(self.records))

    def delete_one(self, query):
        record = self.find_one(query)
        if record is None:
            return SimpleNamespace(deleted_count=0)
        self.records.remove(record)
        return SimpleNamespace(deleted_count=1)


class FakeDatabase:
    def __init__(self):
        self.collections = {
            "conversations": FakeCollection(),
            "messages": FakeCollection(),
            "document_chunks": object(),
        }

    def __getitem__(self, name):
        return self.collections[name]


class ConversationEndpointTests(unittest.TestCase):
    def setUp(self):
        self.owner_id = ObjectId()
        self.other_id = ObjectId()
        self.current_user = {"_id": self.owner_id, "role": "EMPLOYEE"}
        self.database = FakeDatabase()
        app.state.conversation_indexes_ready = False
        app.dependency_overrides[get_current_user] = lambda: self.current_user
        app.dependency_overrides[get_database] = lambda: self.database
        self.client = TestClient(app)
        self.search_patch = patch.object(retrieval_service, "search_chunks", return_value=[{
            "chunk_id": str(ObjectId()),
            "document_id": str(ObjectId()),
            "filename": "leave-policy.pdf",
            "text": "Employees receive 20 days of annual leave.",
            "score": 0.87,
        }])
        self.search_chunks = self.search_patch.start()
        self.provider_payloads = []

        def mock_ollama(url, *, json, timeout):
            self.provider_payloads.append(json)
            return SimpleNamespace(
                status_code=200,
                raise_for_status=lambda: None,
                json=lambda: {"message": {"content": "Employees receive 20 days of annual leave."}},
            )

        self.ollama_patch = patch.object(llm_service.httpx, "post", side_effect=mock_ollama)
        self.ollama_patch.start()

    def tearDown(self):
        self.ollama_patch.stop()
        self.search_patch.stop()
        app.dependency_overrides.clear()
        app.state.conversation_indexes_ready = False

    def create_conversation(self):
        response = self.client.post("/api/conversations")
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def test_create_and_list_only_the_authenticated_users_conversations(self):
        created = self.create_conversation()
        self.assertEqual(self.provider_payloads, [])
        other_conversation = {
            "_id": ObjectId(), "user_id": self.other_id, "title": "Private",
            "created_at": datetime.now(timezone.utc), "updated_at": datetime.now(timezone.utc),
        }
        self.database["conversations"].records.append(other_conversation)

        response = self.client.get("/api/conversations?page=1&page_size=10")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["total"], 1)
        self.assertEqual(response.json()["items"][0]["id"], created["id"])
        self.assertEqual(response.json()["items"][0]["title"], "New conversation")
        self.assertEqual(len(self.database["conversations"].indexes), 1)
        self.assertEqual(len(self.database["messages"].indexes), 1)

    def test_message_persists_answer_and_original_retrieval_sources(self):
        conversation = self.create_conversation()

        response = self.client.post(
            f"/api/conversations/{conversation['id']}/messages",
            json={"query": "What is the annual leave policy?"},
        )

        self.assertEqual(response.status_code, 200, response.text)
        exchange = response.json()
        self.assertEqual(exchange["user_message"]["role"], "user")
        self.assertEqual(exchange["assistant_message"]["role"], "assistant")
        self.assertEqual(exchange["assistant_message"]["content"], "Employees receive 20 days of annual leave.")
        self.assertEqual(exchange["assistant_message"]["sources"][0]["filename"], "leave-policy.pdf")
        self.assertEqual(len(self.database["messages"].records), 2)
        self.assertNotIn("embedding", self.database["messages"].records[0])
        self.assertNotIn("sources", self.database["messages"].records[0])
        self.assertEqual(self.search_chunks.call_args.kwargs["current_user"]["_id"], self.owner_id)
        self.assertFalse(self.search_chunks.call_args.kwargs["is_admin"])
        self.assertEqual(self.database["conversations"].records[0]["title"], "What is the annual leave policy?")

    def test_follow_up_uses_recent_history_for_retrieval_and_ollama(self):
        conversation = self.create_conversation()
        path = f"/api/conversations/{conversation['id']}/messages"
        self.client.post(path, json={"query": "What is the annual leave policy?"})

        response = self.client.post(path, json={"query": "Can I carry those days forward?"})

        self.assertEqual(response.status_code, 200)
        retrieval_query = self.search_chunks.call_args.kwargs["query"]
        self.assertIn("What is the annual leave policy?", retrieval_query)
        self.assertIn("Can I carry those days forward?", retrieval_query)
        followup_prompt = self.provider_payloads[-1]["messages"][1]["content"]
        self.assertIn("What is the annual leave policy?", followup_prompt)
        self.assertIn("Employees receive 20 days of annual leave.", followup_prompt)
        self.assertIn("Can I carry those days forward?", followup_prompt)
        self.assertIn("Previous assistant answers are not authoritative company facts", followup_prompt)

    def test_recent_history_limit_is_respected(self):
        conversation = self.create_conversation()
        path = f"/api/conversations/{conversation['id']}/messages"
        with patch("app.routes.conversations.CONVERSATION_HISTORY_LIMIT", 2):
            for number in range(3):
                result = self.client.post(path, json={"query": f"Question {number}"})
                self.assertEqual(result.status_code, 200, result.text)
        last_prompt = self.provider_payloads[-1]["messages"][1]["content"]
        self.assertIn("Question 1", last_prompt)
        self.assertIn("Question 2", last_prompt)
        self.assertNotIn("Question 0", last_prompt)

    def test_detail_returns_messages_chronologically(self):
        conversation = self.create_conversation()
        path = f"/api/conversations/{conversation['id']}/messages"
        self.client.post(path, json={"query": "First question"})
        self.client.post(path, json={"query": "Second question"})

        response = self.client.get(f"/api/conversations/{conversation['id']}")

        self.assertEqual(response.status_code, 200)
        messages = response.json()["messages"]
        self.assertEqual([message["role"] for message in messages], ["user", "assistant", "user", "assistant"])
        self.assertEqual(messages[0]["content"], "First question")
        self.assertEqual(messages[2]["content"], "Second question")

    def test_delete_removes_owned_conversation_and_messages(self):
        conversation = self.create_conversation()
        self.client.post(f"/api/conversations/{conversation['id']}/messages", json={"query": "Question"})

        response = self.client.delete(f"/api/conversations/{conversation['id']}")

        self.assertEqual(response.status_code, 204)
        self.assertEqual(self.database["conversations"].records, [])
        self.assertEqual(self.database["messages"].records, [])

    def test_another_user_cannot_read_message_or_delete_conversation(self):
        conversation = self.create_conversation()
        self.current_user = {"_id": self.other_id, "role": "ADMIN"}

        get_response = self.client.get(f"/api/conversations/{conversation['id']}")
        message_response = self.client.post(
            f"/api/conversations/{conversation['id']}/messages", json={"query": "Read private data"},
        )
        delete_response = self.client.delete(f"/api/conversations/{conversation['id']}")

        self.assertEqual(get_response.status_code, 404)
        self.assertEqual(message_response.status_code, 404)
        self.assertEqual(delete_response.status_code, 404)
        self.assertEqual(len(self.database["conversations"].records), 1)

    def test_invalid_and_nonexistent_conversation_ids(self):
        invalid = self.client.get("/api/conversations/not-an-object-id")
        missing = self.client.get(f"/api/conversations/{ObjectId()}")

        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(missing.status_code, 404)

    def test_empty_query_is_rejected_before_persistence(self):
        conversation = self.create_conversation()

        response = self.client.post(
            f"/api/conversations/{conversation['id']}/messages", json={"query": "  "},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.database["messages"].records, [])

    def test_no_authentication_returns_401(self):
        app.dependency_overrides.pop(get_current_user)

        response = self.client.get("/api/conversations")

        self.assertEqual(response.status_code, 401)

    def test_invalid_authentication_returns_401(self):
        app.dependency_overrides.pop(get_current_user)

        response = self.client.get(
            "/api/conversations", headers={"Authorization": "Bearer invalid-token"},
        )

        self.assertEqual(response.status_code, 401)

    def test_client_cannot_supply_context_or_sources(self):
        conversation = self.create_conversation()

        response = self.client.post(
            f"/api/conversations/{conversation['id']}/messages",
            json={"query": "Question", "context": "private client context", "sources": []},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.database["messages"].records, [])

    def test_llm_failure_keeps_user_question_without_fabricated_answer(self):
        conversation = self.create_conversation()
        with patch.object(llm_service.httpx, "post", side_effect=httpx.ConnectError("private endpoint detail")):
            response = self.client.post(
                f"/api/conversations/{conversation['id']}/messages", json={"query": "Question"},
            )

        self.assertEqual(response.status_code, 502)
        self.assertNotIn("private endpoint detail", response.text)
        stored = self.database["messages"].records
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["role"], "user")

    def test_failure_to_save_answer_does_not_claim_exchange_was_saved(self):
        conversation = self.create_conversation()
        self.database["messages"].fail_assistant_insert = True

        response = self.client.post(
            f"/api/conversations/{conversation['id']}/messages", json={"query": "Question"},
        )

        self.assertEqual(response.status_code, 500)
        self.assertIn("could not be saved completely", response.json()["detail"])
        self.assertEqual([message["role"] for message in self.database["messages"].records], ["user"])


if __name__ == "__main__":
    unittest.main()
