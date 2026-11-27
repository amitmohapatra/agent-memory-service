# ADR 0047: With a model, the model reads a person's message first; the word patterns read it only when no model answers

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
- When the model does not answer - not allowed, not reachable, a failed call, a reply that
  is not the list of operations asked for - or the message is by its shape not asked about
  (only questions; a short opening message), the patterns read it exactly as they do without
  a model. An empty list is an answer: the message teaches nothing.
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
- The message as said stays searchable in the person's later conversations, a request or an
  instruction for the moment included.
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
  preferences, and no longer replaces what they said of themselves - as far as the model
  judges it right. A wrong judgement that drops a real rule costs a restatement; one that
  keeps a pasted one is what the patterns did every time.
- What a message teaches now depends on the model answering the same way twice. The checks
  bound what it can write, not what it leaves out.
- The patterns and the labeller's rule reading are the no-model path and the fallback, not
  removed.
