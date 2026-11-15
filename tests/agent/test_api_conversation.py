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
    "threads.patch_thread",
    "threads.get_thread",
    "messages.create_messages",
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

    thread = await chat.history.update(title="Quarterly review", metadata={"project": "payments"})
    assert thread.thread_id == "thr-quarterly-review" and thread.title == "Quarterly review"
    # Setting the title again is the same thread: an adapter that reconnects must not have to
    # remember whether it already did this.
    assert (await chat.history.update(title="Quarterly review")).thread_id == thread.thread_id

    [asked] = await chat.history.add([("USER", "When is the release review?")])
    [answered] = await chat.history.add([("ASSISTANT", "Thursday at 15:00.")])
    # An INTERNAL AGENT message belongs to an agent, so it is written from an agent context:
    # working chatter is the agent's, not the user's, and the service says so.
    [thought] = await chat.agent("scheduler-bot").history.add(
        [{"role": "AGENT", "kind": "INTERNAL", "content": "checking the calendar tool"}]
    )
    assert asked.sequence < answered.sequence < thought.sequence
    assert asked.thread_id == answered.thread_id == thread.thread_id
    assert asked.turn_id and asked.session_id

    # A retried send is one message, at the same sequence: the SDK keys the retry off the
    # content, so a network wobble mid-turn does not duplicate the user's question.
    [again] = await chat.history.add([("USER", "When is the release review?")])
    assert again.message_id == asked.message_id and again.sequence == asked.sequence

    one = await chat.history.message(answered.message_id)
    assert one.role == "ASSISTANT" and one.content == "Thursday at 15:00."
    assert one.kind == "VISIBLE"

    visible = await chat.history()
    assert [m.message_id for m in visible] == [asked.message_id, answered.message_id]
    everything = await chat.history(include_internal=True)
    assert thought.message_id in {m.message_id for m in everything}

    assert (await chat.history.thread()).thread_id == thread.thread_id

    await chat.history.delete()
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
    "messages.create_messages",
    "documents.upload_document",
    "threads.patch_thread",
)
async def test_another_tenant_reaches_none_of_this_conversation(app, running) -> None:
    harness = await _harness(app, "acme")
    other = await _harness(app, "globex")
    mine = harness.bind(
        user_id="u1", thread_id="thr-private", session_id="ses-private", turn_id="trn-1"
    )
    [ack] = await mine.history.add([("USER", "The board pack is in the finance drive.")])
    handle = await mine.advanced.documents.add(("board.txt", NOTES, "text/plain"))

    theirs = other.bind(
        user_id="u1", thread_id="thr-private", session_id="ses-private", turn_id="trn-1"
    )
    for call in (
        theirs.history.thread(),
        theirs.history.message(ack.message_id),
        theirs.advanced.documents.document(handle.document_id),
        theirs.history.delete(),
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
        await claiming.history.add([("USER", "hello")])
    assert message.value.status == 403
    with pytest.raises(MemoryError) as thread:
        await claiming.history.update(title="mine now")
    assert thread.value.status == 403
    with pytest.raises(MemoryError) as upload:
        await claiming.advanced.documents.add(("x.txt", b"x", "text/plain"))
    assert upload.value.status == 403


BADGE_NOTE = b"Visitor badge B7731 opens the north gate.\n"
TEAM_NOTE = b"The north gate closes at 22:00 for everyone.\n"


def _count(app, running, sql: str, **params: str) -> int:
    """``SELECT count(*) ...`` straight from the store."""
    from sqlalchemy import text  # noqa: PLC0415

    async def read() -> int:
        async with app.state.container.database.engine.begin() as conn:
            return int((await conn.execute(text(sql), params)).scalar_one())

    return running.portal.call(read)


def _messages_left(app, running, thread_id: str) -> int:
    """The thread's message rows not marked deleted, from the store."""
    return _count(
        app,
        running,
        "SELECT count(*) FROM messages WHERE thread_id = :t AND deleted_at IS NULL",
        t=thread_id,
    )


@pytest.mark.covers("threads.delete_thread", "documents.upload_document")
@pytest.mark.covers_error("documents.get_document")
async def test_a_deleted_threads_documents_and_messages_are_read_by_nothing(app, running) -> None:
    """After ``DELETE /v1/threads/{id}`` nothing of the conversation is returned: a document
    attached in it (readable only inside it) left ``/v1/context``, search and the document
    read, and its messages are marked deleted. A document the user shared wider (``USER``)
    is held by that audience still, and stays."""
    harness = await _harness(app)
    chat = harness.bind(user_id="u1", thread_id="thr-gate")
    await chat.history.add([("USER", "Here is my visitor badge note.")])
    attached = await chat.advanced.documents.add(("badge.txt", BADGE_NOTE, "text/plain"))
    shared = await chat.advanced.documents.add(
        ("gate.txt", TEAM_NOTE, "text/plain"), visibility="USER"
    )
    container = app.state.container
    running.portal.call(container.tasks.drain)
    user = harness.bind(user_id="u1")
    assert "B7731" in (await chat.context("visitor badge north gate")).rendered

    await chat.history.delete()
    running.portal.call(container.tasks.drain)

    for reader in (user, harness.bind(user_id="u1", thread_id="thr-next")):
        assert "B7731" not in (await reader.context("visitor badge north gate")).rendered
        found = await reader.search("visitor badge north gate", kinds=["chunk"], limit=20)
        assert not any("B7731" in item.text for item in found)
        with pytest.raises(MemoryError) as gone:
            await reader.advanced.documents.document(attached.document_id)
        assert gone.value.status == 404
    assert (await user.advanced.documents.document(shared.document_id)).document_id
    assert _messages_left(app, running, "thr-gate") == 0


@pytest.mark.covers("threads.delete_thread", "messages.create_messages", "documents.get_document")
async def test_a_document_another_live_thread_attached_outlives_the_first_thread(
    app, running
) -> None:
    """A document is shared across threads by attaching it to a message of another one: it
    is removed only when no live thread holds it."""
    harness = await _harness(app)
    first = harness.bind(user_id="u1", thread_id="thr-first")
    handle = await first.advanced.documents.add(("badge.txt", BADGE_NOTE, "text/plain"))
    attachment = {
        "filename": "badge.txt",
        "media_type": "text/plain",
        "size_bytes": handle.size_bytes,
        "checksum": handle.checksum,
        "document_id": handle.document_id,
    }
    second = harness.bind(user_id="u1", thread_id="thr-second")
    await second.history.add(
        [{"role": "USER", "content": "same note", "attachments": [attachment]}]
    )
    running.portal.call(app.state.container.tasks.drain)

    await first.history.delete()
    assert (await second.advanced.documents.document(handle.document_id)).document_id
    await second.history.delete()  # the last thread that held it
    with pytest.raises(MemoryError) as gone:
        await harness.bind(user_id="u1").advanced.documents.document(handle.document_id)
    assert gone.value.status == 404


NORTHWIND_NOTE = (
    b"Northwind Systems acquired Westfalen Labs in March 2025.\n"
    b"Westfalen Labs provides payment software to retailers in Germany.\n"
)
CONTOSO_NOTE = (
    b"Contoso Holdings acquired Fabrikam Industries in June 2024.\n"
    b"Westfalen Labs provides freight software to carriers in France.\n"
)


def _graph_rows(app, running, document_id: str) -> tuple[int, int]:
    """What the store's graph holds from ``document_id``: its relations, and the entities that
    cite it (its own document entity, or evidence taken from it)."""
    relations = _count(
        app,
        running,
        "SELECT count(*) FROM graph_relations WHERE document_id = :d",
        d=document_id,
    )
    entities = _count(
        app,
        running,
        "SELECT count(*) FROM graph_entities WHERE canonical_name = 'doc:' || :d"
        " OR evidence @> jsonb_build_array(jsonb_build_object('document_id', :d))",
        d=document_id,
    )
    return relations, entities


@pytest.mark.covers("threads.delete_thread", "documents.upload_document", "graph.search_entities")
@pytest.mark.covers_error("graph.entity_profile")
async def test_a_deleted_threads_documents_leave_nothing_in_the_graph(app, running) -> None:
    """MEM-05. The graph facts and entities built from a document only the deleted thread
    held go with it: no graph read returns them, from any thread or none, and the store holds
    none; an entity both documents name stays with only what the kept one says. A document
    another live thread still holds keeps its facts."""
    harness = await _harness(app)
    chat = harness.bind(user_id="u1", thread_id="thr-deals")
    released = await chat.advanced.documents.add(("northwind.txt", NORTHWIND_NOTE, "text/plain"))
    kept = await chat.advanced.documents.add(("contoso.txt", CONTOSO_NOTE, "text/plain"))
    holder = harness.bind(user_id="u1", thread_id="thr-holder")
    attachment = {
        "filename": "contoso.txt",
        "media_type": "text/plain",
        "size_bytes": kept.size_bytes,
        "checksum": kept.checksum,
        "document_id": kept.document_id,
    }
    await holder.history.add([{"role": "USER", "content": "the deal", "attachments": [attachment]}])
    container = app.state.container
    running.portal.call(container.tasks.drain)
    user = harness.bind(user_id="u1")  # every thread of theirs in scope
    [northwind] = await user.advanced.graph.entities("Northwind Systems", limit=1)
    [westfalen] = await user.advanced.graph.entities("Westfalen Labs", limit=1)
    assert "payment software" in westfalen.summary and "freight software" in westfalen.summary
    kept_before = _graph_rows(app, running, kept.document_id)
    assert all(_graph_rows(app, running, released.document_id)) and all(kept_before)

    await chat.history.delete()
    running.portal.call(container.tasks.drain)

    for reader in (user, holder, harness.bind(user_id="u1", thread_id="thr-new")):
        assert not await reader.advanced.graph.entities("Northwind Systems")
        with pytest.raises(MemoryError) as gone:
            await reader.advanced.graph.entity(northwind.entity_id)
        assert gone.value.status == 404
    assert _graph_rows(app, running, released.document_id) == (0, 0)
    assert _graph_rows(app, running, kept.document_id) == kept_before
    [contoso] = await user.advanced.graph.entities("Contoso Holdings", limit=1)
    assert (await user.advanced.graph.entity(contoso.entity_id)).relations
    both = await user.advanced.graph.entity(westfalen.entity_id)
    facts = both.relations + both.history
    told = " ".join([both.entity.summary, *(f"{f.object} {f.fact_text}" for f in facts)])
    assert "freight software" in told
    assert not any(word in told for word in ("payment software", "Germany", "Northwind"))
    assert released.document_id not in {e.document_id for e in both.evidence}
