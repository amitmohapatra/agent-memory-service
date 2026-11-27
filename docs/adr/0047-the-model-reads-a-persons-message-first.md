# ADR 0047: Where a model is allowed, it alone reads a person's message, and says whether the message is kept

Date: 2026-11-27. Status: accepted; its quality with a real model is not measured yet (see
Measurements). Supersedes ADR 0045's order (the rules first, then one call for what they
missed) and, for a message the model reads, ADR 0037's reading of a sentence as a rule.

## Context

A person's message was read by the word patterns first (`native`: "always" / "never" a rule,
"my email is" an attribute, "I prefer" a preference), and with a model allowed one call then
added what they missed (ADR 0045). The model could add, update or contradict; it could not
take back what a pattern had read.

A pattern reads words, not whose they are. Reproduced on a local service (no model):

- a skill's text pasted into a chat ("# Read-Only Mode ... Never call a tool that writes.
  Always tell the user which plan you read.") became two lasting rules of the person who
  pasted it, in their profile in every later conversation, the pasted turn listed among
  their standing rules;
- a colleague's email pasted to be summarised ("I am the regional manager at Castor Supply.
  My email is priya@example.com. I prefer daily digests.") replaced the person's own email,
  role and preference in their profile: "What is my email address?" in a new conversation
  was answered from the colleague's;
- "for now skip consent" was kept for good with nothing saying it was for the moment.

Whether a sentence is the speaker's own statement or material they share to be worked on,
and whether an instruction is for every time or for now, is a judgement about meaning. No
pattern makes it: a pasted "Never call a tool that writes." and the same words typed as a
rule are one string.

## Decision

With a model allowed for the message (`contextual_extraction`, as before: a person's own
message, no Hindsight server reading it) the call is made first, and what it returns is all
that is read from the message:

- `ObservationPipeline._read` asks the model before any pattern runs. The call is given the
  message, the four turns before it and the speaker's eight closest memories - not a reading
  of the message by anything else.
- Its instructions say whose words count: only what the speaker states in their own voice is
  theirs; pasted or forwarded text, a document, a skill's or a policy's text, code and
  quotations are material, from which nothing is the speaker's rule, preference or profile
  and nothing known about the speaker changes (a fact the material states about the person or
  thing it names may be added with that subject). A rule is an instruction for every time;
  an instruction for now, for this task or conversation, and a request to do something, are
  not memory.
- A new type, `rule`, beside fact, profile, preference, event, plan, procedure and opinion.
  What is read of the speaker is filed where a context reads it from, as the patterns'
  reading of the same sentence is: `rule` a lasting preference with category `rule`,
  `profile` an attribute, `preference` a preference; anything else a source fact, as before.
- The message as it was said is kept beside the reading (`NativeMemoryIntelligence.said`): it
  carries the change it makes to earlier statements (a correction, a status, a start or an
  end: the replacement path and the search read it) and nothing about its speaker - a turn of
  pasted instructions is not a rule turn. A piece of the turn a lasting reading says in the
  same words is not kept beside it, as before.
- **The model says what the message is** (`kind`): a `statement` (the speaker tells
  something worth keeping for their later conversations, with or without a request beside
  it), a `request` (it only asks for something, or gives an instruction for now, for this
  task or this conversation) or `material` (text shared to be worked on). Only a statement is
  kept beyond its conversation, as the turn it was said in. A request and material stay in
  the transcript, where the conversation's window reads them, and form no memory: in a real
  trace "ok for now skip the consent stage and call create_allocation directly", "delete
  this plan" and a pasted block of SVG came back as things the user said, in a later
  conversation, beside the remembered instruction to ask for consent before every create. A
  reply that does not say, or says something else, keeps the turn, as before.
- **No pattern stands in for a model that is allowed.** A message the model does not answer
  for - not reachable, a failed call, a reply that is not what was asked for - is read by
  nothing: it stays unprocessed and is read again 1 minute, 5 minutes, 30 minutes, 2 hours,
  6 hours and 24 hours later (`pipeline.LATER`, about 33 hours in all; `memory.unread`,
  `memory_unread_total{outcome="deferred"}`). After the last it is kept as it was said, with
  nothing read from it (status `UNREAD`, `outcome="gave_up"`). A message by its shape not
  asked about (a short opening message) is kept as said and read by nothing either. The
  first version of this change let the patterns read a message its model did not answer
  for: every outage would have made pasted text the person's rules again.
- The patterns read a person's message only where no model is allowed for it (no key, a
  policy without the use, a Hindsight server reading it), and a message with nothing said
  (only questions), from which they form nothing. An empty list is an answer: the message
  teaches nothing.
- Every check on an op stays (an exact quote, no number the quote does not state, the
  library's day, no duplicate of a memory held, an update only of a memory shown). The
  checks that compared the model's facts with the patterns' reading of the same message are
  removed with that reading.

No setting, use name, route or SDK call changed. The cost is the call a tenant with a model
already paid for each message.

## What it does not change

- A tenant with no model: the patterns read every message, and the three cases above stay as
  they were.
- An event, an import and a feedback note are read by the patterns as before, with or
  without a model: a rule-like line in an event still becomes a rule of its user.
- What was stored before: rules and attributes already made from pasted text stay until they
  are forgotten.
- A job's status: the job that put a message off succeeds, and the message's memories are
  not there yet. `GET /v1/jobs/{id}` on the first job says it ran, not that the message was
  read.
- The statement labeller still says what change a turn makes (correction, status, lifecycle).
- A very long message: the call's input is cut at 8,000 characters (the turns before it
  included), as before. What lies past the cut is in the message as said and is read by
  nothing; the patterns used to read all of it.

## Measurements

None with a real model yet: it needs a paid run. The tests script the model
(`tests/integration/test_model_reads_first.py`, `tests/unit/test_gapfill.py`): they prove the
order, the filing and the fallback, not the model's judgement. ADR 0045 measured rules alone
against rules plus the call on the extraction eval (recall 0.23 against 0.94-0.95, no false
fact in either); that eval has no pasted material and no instruction for the moment, and the
arm "the model alone" is new. To measure: the extraction eval with this order, with cases of
pasted skill text, a forwarded email and an instruction for now added, and LoCoMo's J on the
same conversations.

## Consequences

- With a model, text a person pastes is no longer read as their own rules, attributes or
  preferences, no longer replaces what they said of themselves, and is not kept beyond its
  conversation; nor is a request or an instruction for the moment - as far as the model
  judges it right. A wrong judgement that drops a real rule costs a restatement; one that
  keeps a pasted one is what the patterns did every time.
- What a message teaches now depends on the model answering the same way twice. The checks
  bound what it can write, not what it leaves out. A statement the model calls a request is
  not kept beyond its conversation; it is still in the transcript.
- A model that is allowed and never answers (a wrong key, a provider that is gone) costs a
  tenant its readings: each message waits its 33 hours and is then kept as said. The
  counter and the log line say so; no pattern hides it.
- The patterns and the labeller's rule reading are the no-model path and the fallback, not
  removed.
