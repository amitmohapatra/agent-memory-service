# ADR 0044: A later chat recalls what was said in the user's earlier chats, the assistant's answers included

Date: 2026-11-01. Status: accepted. Amends ADR 0036 (the assistant's replies keep their
default audience: they still do, and are now read back as what was said) and the past
conversations of the retrieval guide (`message` and `episode` stay inside the request's
workspace).

## Context

"What was that reorder number you worked out on Monday?" asked in Friday's chat. The user's
memories hold their own words, in every conversation of theirs (ADR 0036), and never the
assistant's text, by design: an answer is the model's output, not a fact the user stated. So a
figure the assistant computed in an earlier chat was reachable in one way only, a
`memory_search` the model had to make with `kinds: ["message"]`. The holistic Haiku evaluation's
break-and-fix rounds named it ("a figure the agent computed in an earlier chat is reachable only
by message search", held-out case `bfh-mixed-01`): a model that searched with the default kinds
(memory and document passages) found the user's question and not the answer.

The two products this follows do it in two ways, and do both:

* **Claude, "search past chats"**: the model searches the user's earlier conversations as a
  tool call, visible in the run, when the user refers to one.
* **ChatGPT, "reference chat history"**: what an earlier chat said is brought into the context
  without a tool call, for the question that needs it.

Both limit it in the same way: only that user's own chats; within the project the chat is in;
a deleted chat is never used, and saved memories stay; a temporary chat is never used and never
remembered.

## Decision

1. **A question that points to an earlier chat is recognised by its words**
   (`router.points_to_earlier_chat`, the route's `earlier_chat` signal): "last time", "the other
   day", "you worked out / calculated / told me / said / mentioned", "I told / gave you", "we
   talked about", "what did you say", "remind me what", "do you remember", and the same in
   German, Spanish, Arabic and Hindi, read whatever the question's language, as the history cue
   of ADR 0041 is. A question about the past of the world ("sales last week") is not one: it
   needs no earlier chat. The cue words are those of the capability eval's `later` phrasings
   and their translations. The route's type is unchanged, so nothing else about the read moves.
2. **The pushed context carries the assistant's best matching earlier answers**
   (`Searcher.earlier_chats`, `ContextBuilder._earlier_chats`) for such a question, and only
   for it: under `## Earlier conversations` in `rendered` and as `earlier_chats` in the full
   form. They are read from the messages already stored (the same bounded, indexed read as the
   message search: the user's 100 most recently active threads, 200 newest messages sharing a
   word with the question), the assistant's messages of conversations other than this one,
   ranked by the question words they share, each word weighted by its inverse document
   frequency over the messages read (a word every message shares, "the", says nothing). At most
   three (`EARLIER_CHAT_LINES`), each within half the best one's score (`EARLIER_CHAT_SHARE`),
   a long one as its start and its end (`EARLIER_CHAT_CHARS`, 1,000), within half of what is left
   of the token budget after the pinned sections. Each line says when (date and weekday), in
   which conversation, and that the assistant said it, and "corrected by the user's next
   message: use the correction" when the user's next turn there was a correction (ADR 0037,
   the mark search already gave, PR #69). The user's own words in those chats are not repeated:
   they are memories already (ADR 0036). An earlier answer is no evidence for `/v1/verify`:
   it is what the model said then, not proof.
3. **`memory_search` reads what was said when such a question names no kinds**
   (`search.default_kinds`): memory, chunk and message. The tool's description says message
   includes the assistant's own earlier answers. The call stays a tool call in the run, as any
   other. No new tool: the existing one could always do it; the model had to know to ask.
4. **The limits are those of the two products.**
   * Only the user's own conversations: threads whose owner is the request's user, read for
     that user and the agents acting for them; another user of the same agent reads none of it
     (an agent principal is bound to its user, ADR 0036).
   * Within the workspace: a message search and an earlier conversation (`episode`) read only the
     conversations of the request's workspace, a request with none only those with none, as a
     project's chats are (`owned_recent(workspace_id=)`, `threads.owned_in`). The user's
     memories still follow them everywhere (ADR 0036): only the recall of earlier chats is
     scoped, as ChatGPT's project memory and Claude's project search are.
   * Never a deleted conversation: its messages are not read (as before), and deleting it moves
     its owner's revision, so a context cached in another of their conversations that quoted it
     is rebuilt without it.
   * Never a temporary chat: the harness's `without=["memory"]` (per run or per agent) sends a
     run nothing and records nothing in this service (no context, no messages, no feedback:
     agent-harness `tests/integration/test_selection.py`), so its chat is not here to be read.
     That is the per-run "do not remember" the harness already has; no thread-level flag is
     added. A deployment that calls this service directly keeps a temporary chat out the same
     way, by not sending its messages.
5. **Nothing new is stored.** No migration, no new index, no new setting: the messages, their
   threads and the correction labels were all there.

## Consequences

* Measured: the numbers are in the pull request that carries this ADR (the capability eval
  before and after, the break-and-fix cross-chat cases with Haiku, and context tokens per turn).
* A question that does not point to an earlier chat reads nothing more and its context is the
  same size: the cue is a regular expression over the question, checked before any read.
* A question that does reads one more indexed read beside the retrieval (the message search's
  own, measured in the retrieval guide) and carries at most three answers.
* The cue is words. A question that points back without them ("and the number from before?")
  gets no lines; the model can still search, and does in the run. The other way round, a
  question with the words that does not need an earlier chat gets lines that share its words,
  which cost tokens and are labelled as what was said then.
* Message search ranking now weights words by their rarity among the messages read, for the
  tool and the context alike; with one word or equal counts the order is the one it was.
* An earlier conversation of another workspace is no longer found by `memory_search` or
  `/v1/recall` (`message`, `episode`). A deployment that used one workspace per chat and relied
  on recall across them passes the shared workspace, or none, on the request.
