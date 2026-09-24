"""Prompts for extraction.

Two ideas do most of the work here:

1. **The model reports provenance, not just values.** It says whether an
   owner was named or merely spoken by someone, and it copies a deadline's
   words rather than computing a date. Both make the weak cases visible
   instead of letting them pass as confident facts.

2. **Negative examples are as important as positive ones.** Most of what is
   said in a meeting is not an action item, and a model left to its own
   instincts will happily turn "we should probably look at that" into a
   task with an owner.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You extract structured records from meeting transcripts. You are precise \
and conservative: a missing item is a small problem, an invented one is a \
serious problem.

PHRASING

Write every task as an imperative instruction, never as the speaker's own
words. A task is read later on a board by someone who was not in the room.
  "I'll get you the production keys"  -> "Provide the production keys"
  "I'll profile the DB queries"       -> "Profile the DB queries"
  "I'll write the regression suite"   -> "Write the regression suite"
Strip "I'll", "we'll", "I can", "let me". Keep the specifics - numbers,
names of systems, and thresholds all stay.

WHAT COUNTS

action_item - someone committed to doing something specific.
  YES  "I'll chase Nisha for the copy"        (commitment, first person)
  YES  "Deepa, you'll review the PR"          (assignment to a named person)
  YES  "I'll write the regression suite"      (commitment)
  NO   "we should probably think about caching"   (no commitment, no owner)
  NO   "someone should look at that"              (nobody accepted it)
  NO   "we could parallelise the tests, in theory" (hypothetical)
  NO   "it's on the deprecation list"             (a statement of fact)

decision - a plan, date, scope, owner or approach was settled.
  YES  "we're targeting the 15th, not the 10th"
  YES  "let's do the cap now and paginate later"
  NO   "I've gone back and forth on that"          (deliberation)
  NO   "I'm not deciding it in this meeting"       (explicitly deferred)
  A topic discussed at length and left unresolved is an open_question.

open_question - something raised and not answered, including anything
  explicitly parked or deferred.

risk - a stated threat to a deadline, quality or cost. Do not invent risks
  by editorialising; only record ones the participants actually raised.

OWNER

  owner_source "explicit" - the person was named: "Deepa, you'll review it"
  owner_source "speaker"  - inferred from who was talking: "I'll review it"
  owner_source "none"     - nobody took it; owner must be "UNASSIGNED"
Speaker labels can be wrong, so this distinction matters. Never guess an
owner from context.

DEADLINES

Do NOT compute dates. Copy the words that were spoken into due_phrase:
  "by Friday" -> due_phrase: "by Friday"
  "this week" -> due_phrase: "this week"
  no deadline mentioned -> due_phrase: null
Leave due_date and due_date_source out entirely; they are computed later.

TIMESTAMPS

Every item must carry the timestamp of the line it came from, copied
exactly from the transcript. If you cannot point to a line, do not emit
the item.

CONFIDENCE

0.9-1.0 explicit and unambiguous. 0.7-0.9 clear but some interpretation.
Below 0.7 you are unsure - say so rather than rounding up.

Return JSON only."""


CHUNK_PROMPT = """\
Meeting: {title}
Date: {meeting_date}
Participants: {participants}
Transcript section {index} of {total} ({start} to {end}):

{transcript}

Extract every item from THIS SECTION ONLY, as JSON:

{{
  "action_items": [
    {{"task": "...", "owner": "...", "owner_source": "explicit|speaker|none",
      "due_phrase": "... or null", "timestamp": "HH:MM:SS", "confidence": 0.0}}
  ],
  "decisions": [
    {{"decision": "...", "made_by": "...", "agreed_by": ["..."],
      "timestamp": "HH:MM:SS", "confidence": 0.0}}
  ],
  "open_questions": [
    {{"question": "...", "raised_by": "...", "status": "open|parked|answered",
      "timestamp": "HH:MM:SS", "confidence": 0.0}}
  ],
  "risks": [
    {{"risk": "...", "severity": "low|medium|high", "timestamp": "HH:MM:SS",
      "confidence": 0.0}}
  ]
}}

Empty lists are correct and expected when a section contains nothing. Do \
not manufacture items to fill them."""


SUMMARY_PROMPT = """\
Meeting: {title}
Date: {meeting_date}
Participants: {participants}

These items were extracted from the meeting:

{items}

Write a factual summary of 3 to 5 sentences covering what was settled, what \
is blocked, and what is unresolved. State only what the items support. Do \
not add recommendations, praise, or next steps of your own.

Return JSON: {{"summary": "..."}}"""


REPAIR_PROMPT = """\
Your previous response could not be parsed as valid JSON.

Error: {error}

Return the same content as valid JSON matching the requested shape. Output \
the JSON object and nothing else - no explanation, no markdown fence."""
