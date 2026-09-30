"""A conversation as an adapter records it: a thread, the turns inside it and an attachment."""

from __future__ import annotations

import pytest

from tests.agent.conftest import BOOTSTRAP, sdk
from trellis.memory import MemoryError

pytestmark = pytest.mark.e2e

NOTES = b"Release review moved to Thursday. Owner: Priya Raman. Risk: the migration window.\n"


async def _harness(app, tenant_id: str = "acme"):
    platform = sdk(app, BOOTSTRAP)
    tenant = await platform.admin.create_tenant(tenant_id.title(), tenant_id=tenant_id)
    admin = sdk(app, tenant.admin_key.token)
    service = await admin.tenant.keys.issue("service", f"{tenant_id}-harness")
    return sdk(app, service.token)


@pytest.mark.covers(
    "threads.create_thread",
    "threads.get_thread",
    "messages.create_message",
    "messages.get_message",
    "messages.list_messages",
    "threads.delete_thread",
)
async def test_an_adapter_records_a_whole_turn_and_then_deletes_the_thread(app, running) -> None:
    harness = await _harness(app)
    # A message names its whole conversational lineage: the thread, the session and the turn
    # are the caller's to assign (sdk/python/README.md's first example binds all three, and
    # ConversationService refuses a message without them).
    chat = harness.bind(
        user_id="u1",
        thread_id="thr-quarterly-review",
        session_id="ses-quarterly-review",
        turn_id="trn-1",
    )

    thread = await chat.chat.create(title="Quarterly review", project="payments")
    assert thread.thread_id == "thr-quarterly-review" and thread.title == "Quarterly review"
    # Creating the same thread twice is the same thread, even when the second call carries
    # different metadata: an adapter that reconnects must not have to remember whether it
    # already did this, and it never sent an Idempotency-Key to be held to.
    assert (await chat.chat.create(title="Quarterly review")).thread_id == thread.thread_id

    asked = await chat.chat.user("When is the release review?")
    answered = await chat.chat.assistant("Thursday at 15:00.")
    # An INTERNAL AGENT message belongs to an agent, so it is written from an agent context:
    # working chatter is the agent's, not the user's, and the service says so.
    thought = await chat.agent("scheduler-bot").chat.internal("checking the calendar tool")
    assert asked.sequence < answered.sequence < thought.sequence
    assert asked.thread_id == answered.thread_id == thread.thread_id
    assert asked.turn_id and asked.session_id

    # A retried send is one message, at the same sequence: the SDK keys the retry off the
    # content, so a network wobble mid-turn does not duplicate the user's question.
    again = await chat.chat.user("When is the release review?")
    assert again.message_id == asked.message_id and again.sequence == asked.sequence

    one = await chat.chat.message(answered.message_id)
    assert one.role == "ASSISTANT" and one.content == "Thursday at 15:00."
    assert one.kind == "VISIBLE"

    visible = await chat.history()
    assert [m.message_id for m in visible] == [asked.message_id, answered.message_id]
    everything = await chat.history(include_internal=True)
    assert thought.message_id in {m.message_id for m in everything}

    assert (await chat.chat.thread()).thread_id == thread.thread_id

    await chat.chat.delete_thread()
    # A soft-deleted thread is gone to every reader, not an empty one: the listing answers
    # "not found", which is what stops an adapter from carrying on writing into it.
    with pytest.raises(MemoryError) as deleted:
        await chat.history()
    assert deleted.value.status == 404


@pytest.mark.covers("documents.upload_document", "documents.get_document")
async def test_an_agent_attaches_a_document_and_waits_for_it_to_be_retrievable(
    app, running
) -> None:
    harness = await _harness(app)
    chat = harness.bind(user_id="u1", thread_id="thr-attachments")

    handle = await chat.advanced.documents.add(
        (("release-notes.txt", NOTES, "text/plain")), title="Release notes", source="drive"
    )
    assert handle.document_id and handle.size_bytes == len(NOTES)
    assert handle.checksum and not handle.deduplicated

    # The same bytes under the same name are the same document, not a second copy.
    replay = await chat.advanced.documents.add(("release-notes.txt", NOTES, "text/plain"))
    assert replay.document_id == handle.document_id

    document = await chat.advanced.documents.wait_ready(handle.document_id, max_wait=30.0)
    assert document.document_id == handle.document_id
    assert document.status in ("READY", "PARSING", "INDEXING", "PENDING"), document.status
    assert document.filename == "release-notes.txt" and document.media_type == "text/plain"
    assert document.thread_id == "thr-attachments"


@pytest.mark.covers_error(
    "threads.get_thread",
    "messages.get_message",
    "documents.get_document",
    "threads.delete_thread",
    "messages.list_messages",
    "messages.create_message",
    "documents.upload_document",
    "threads.create_thread",
)
async def test_another_tenant_reaches_none_of_this_conversation(app, running) -> None:
    harness = await _harness(app, "acme")
    other = await _harness(app, "globex")
    mine = harness.bind(
        user_id="u1", thread_id="thr-private", session_id="ses-private", turn_id="trn-1"
    )
    ack = await mine.chat.user("The board pack is in the finance drive.")
    handle = await mine.advanced.documents.add(("board.txt", NOTES, "text/plain"))

    theirs = other.bind(
        user_id="u1", thread_id="thr-private", session_id="ses-private", turn_id="trn-1"
    )
    for call in (
        theirs.chat.thread(),
        theirs.chat.message(ack.message_id),
        theirs.advanced.documents.document(handle.document_id),
        theirs.chat.delete_thread("thr-private"),
        theirs.history(limit=5),
    ):
        with pytest.raises(MemoryError) as refused:
            await call
        assert refused.value.status in (403, 404), refused.value

    # Naming another tenant in the header is refused outright, on every write of this group.
    claiming = other.bind(
        tenant_id="acme",
        user_id="u1",
        thread_id="thr-private",
        session_id="ses-private",
        turn_id="trn-2",
    )
    with pytest.raises(MemoryError) as message:
        await claiming.chat.user("hello")
    assert message.value.status == 403
    with pytest.raises(MemoryError) as thread:
        await claiming.chat.create(title="mine now")
    assert thread.value.status == 403
    with pytest.raises(MemoryError) as upload:
        await claiming.advanced.documents.add(("x.txt", b"x", "text/plain"))
    assert upload.value.status == 403
