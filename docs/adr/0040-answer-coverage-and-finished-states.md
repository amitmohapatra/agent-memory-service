# ADR 0040: Abstain when the evidence lacks the value asked for; a finished state replaces an ongoing one

Date: 2026-10-18. Status: accepted. Follows up [ADR 0036](0036-a-users-words-follow-the-user.md)
and [ADR 0038](0038-replacement-across-conversations.md).

## Context

ADR 0036 made a user's words follow them into every conversation. Before it, a new
conversation saw nothing of the user's earlier ones, so on the capability eval's core set
every unanswerable question asked in a new conversation came back INSUFFICIENT (abstention
1.00) - and so did 98.5% of the answerable ones. Measured on the same set with pre-ADR-0036
code (c0aa48c; the eval did not exist then, so the current runner served that tree): new
conversation abstention 1.00 with a false-refusal rate of 0.985; same conversation 0.10-0.21.
Since ADR 0036 the new-conversation figure is the same-conversation one (0.31, false refusals
0.35): the abstention rule never judged *whether the evidence answers*, only whether it
shares a word with the question, and a user's statements about the same things always do
("What's the alarm code for Store 33?" finds "The alarm code at Store 31 is 4471" and "Store
33 opens at 7 am").

ADR 0038 left a state change the head reads as no contradiction unreplaced without a model:
"I finished Dune last night" against "I'm reading Dune at the moment" reads 0.15 in German,
0.04 in Arabic, 0.22 in Hindi, and the encoder puts the Hindi pair at 0.46.

## Decision

1. **Coverage before relevance** (`modules/context/coverage.py`, applied by the context builder
   to what the bundle carries as current evidence; `RetrievalSettings.coverage_check`, on by
   default). A question that asks for a contact detail or a secret - a phone number (or an
   extension), an e-mail address, a code or password; the asking words in the five pack
   languages, German compounds ("Alarmcode") included, writing a message ("draft an email") and
   the verb "to phone" ("फ़ोन करूँ") not - is answered only when an item of evidence states such
   a value *and* names what the question names: every labelled identifier ("Store 33", not
   "Store 31") and one of its names ("John Doe"; in German a run of capitalised words counts
   only when marked or used by the evidence, since German capitalises nouns). A question that
   names a month is not answered by evidence that names months but never that one ("April
   shrink" / "March shrink came in at 1.8%"). Otherwise the bundle is INSUFFICIENT with a note
   saying what is missing; the evidence stays in the bundle. The rule only ever says "not
   answered": a question it cannot read is left to the rules before it.
   Only these value kinds: the ones the eval's dev split asks for. Money, serial numbers,
   expiry dates and a relative's name were considered and left out - the held-out split asks
   for them, and a rule written for them would be tuned on the held-out half.
2. **Related evidence is evidence.** The overlap rule calls a bundle INSUFFICIENT when no
   evidence shares a word with the question; a paraphrase, an indirect question or another
   language shares none with its answer. A memory-only bundle (no document chunks, whose
   companion checks the overlap rule skipped) that carries a current memory at a relevance of
   `RetrievalSettings.related_evidence_relevance` (0.3) or more is COMPLETE instead, with a
   note. Chosen on the dev split's recorded bundles: false refusals 0.297 -> 0.204 in a new
   conversation, 0.208 -> 0.135 in the same one, for 1.6 points of abstention; 0.25 refused
   less still but sits next to the context's relevance floor (0.2).
3. **A finished state after an ongoing one** (`replacement.finished_after_ongoing`,
   `moved_on`). Two of a user's statements tied by a shared name or identifier (not by a slot
   alone, nor by the encoder), the newer saying a state has ended or been reached ("finished",
   "is back", "resolved", "zu Ende", "terminé", "ha vuelto", "أنهيت", "عادت", "ख़त्म", "लौट")
   and the older that it was going on ("at the moment", "until further notice", "gerade",
   "bis auf Weiteres", "ahora mismo", "estoy leyendo", "حاليًا", "حتى إشعار آخر", "इन दिनों",
   "पढ़ रहा हूँ"): the older is an earlier value, whatever the head read. Never a supersession.

## Consequences

* Measured offline on the dev split's recorded bundles (core set, no-LLM, both placements):
  abstention 0.28 -> 0.77 in a new conversation, 0.09 -> 0.72 in the same one; false
  refusals on answerable questions 0.2975 -> 0.3009 and 0.2051 -> 0.2085 (8 of 2316: "her
  phone kept trying an old password" in a question about unlocking an account, a typo'd
  "correo" for a reorder e-mail). The paired gate run is in the follow-up measured commit.
* Finished-after-ongoing on ADR 0038's pair set: dev 39 -> 43 of 45 replacements found,
  none of the 213 dev and 170 held-out non-replacement pairs marked; nothing changes on the
  held-out split.
* Still abstained on by nothing: a question that names nothing and asks for a value kind not
  read here ("Router login, quick."), and an unknown person in a German question.
* A question that asks for a value no evidence states is now INSUFFICIENT even in a
  conversation where the user said related things; the answerer says so instead of offering
  the neighbouring value (Store 31's code for Store 33).
