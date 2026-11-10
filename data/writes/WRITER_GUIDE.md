# Writer guide: Simplified Technical English (STE) rewrites

You rewrite an existing answer (`original`) to a user's question (`prompt`) so that it follows the 8 rules below.
Same facts, better writing. The STE checker (`ste_rlhf.ste.cli`) scores every answer 0–100 (12.5 points per rule).
**Every rewrite must score ≥ 87.5 (at least 7 of 8 rules); aim for 100.**

## The 8 rules

| # | Rule | Checker threshold |
|---|---|---|
| R1 | **Short sentences.** Instruction/procedure sentences ≤ 20 words; descriptive sentences ≤ 25 words. | ≥ 90% of sentences within the limit |
| R2 | **Active voice.** "Turn the valve", "The pump moves the water" — not "The valve is turned", "Water is moved by the pump". | ≤ 10% passive sentences |
| R3 | **One instruction per sentence.** "Open the file. Save it." — not "Open the file and save it." | ≥ 90% of instruction sentences |
| R4 | **Lists.** Numbered steps for procedures; bullets for sets of items. | needed when there are ≥ 3 steps/items |
| R5 | **Short paragraphs.** Prose paragraphs ≤ 3 sentences, one topic each. | 0 long paragraphs |
| R6 | **Exact technical terms, used consistently. No vague words**: some, various, a lot of, lots of, stuff, things, kind of, sort of, several, many, numerous, quite, fairly, really, very, pretty much, etc., and so on, whatever, somehow. | ≤ 1 vague term |
| R7 | **Common words**: use (not utilise/utilize), help (not facilitate), start (not commence), end (not terminate), about (not approximately), then/after (not subsequently), before (not prior to), also (not in addition), enough (not sufficient), show (not demonstrate), try (not endeavour), get (not obtain), need (not require). | ≤ 1 complex word |
| R8 | **No filler**: basically, essentially, it should be noted that, it is important to note, in order to, as a matter of fact, at the end of the day, needless to say, in conclusion, overall, I hope this helps, feel free to, great question, sure!, certainly!, of course, let me know if… | 0 filler phrases |

## Rewrite rules

1. **Keep every fact, number, name, command and caveat** from the original. **Add nothing new.** Remove only filler and repetition. If the original is wrong, keep its content (we change style, not facts).
2. Answer the user's question directly in the first sentence.
3. Use numbered steps for procedures and bullets for lists. Use a short bold title line (`**Title**`) only when the answer has 2 or more sections.
4. Keep code blocks and exact commands unchanged.
5. Target length: **40–80% of the original's word count.**
6. Plain Markdown only. No preamble ("Here is the rewrite"), no notes to the reader about the rules.

## Example

Original (excerpt): *"In order to reset your router, it is important to note that you should first locate the reset button, which is usually found on the back of the device. Once it has been located, the button should be pressed and held for approximately 10 seconds, and then the lights will start to flash."*

Rewrite:
```
To reset the router:

1. Find the reset button on the back of the router.
2. Push and hold the button for about 10 seconds.
3. Release the button when the lights flash.
```
