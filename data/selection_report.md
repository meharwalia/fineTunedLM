# Prompt selection report

HelpSteer2 train rows: 20324
Kept prompts: 2302
Dropped: 10783 (multi_turn counts rows; every other reason counts unique prompts)

## Dropped by reason

| reason | count |
|---|---|
| multi_turn | 5906 |
| not_explanatory | 2462 |
| creative | 520 |
| prompt_too_long | 481 |
| roleplay | 333 |
| long_passage | 286 |
| writing_task | 173 |
| format_constraint | 169 |
| rewrite_text | 117 |
| task_setup | 98 |
| original_too_short | 77 |
| original_too_long | 69 |
| unsafe | 40 |
| code_writing | 35 |
| translation | 12 |
| mostly_code | 5 |

## Length stats (SmolLM2 tokens, kept prompts)

- prompt tokens: n=2302 min=2 median=18 mean=31 p90=76 max=282
- original tokens: n=2302 min=45 median=315 mean=328 p90=517 max=766
- original words: n=2302 min=40 median=247 mean=254 p90=397 max=640
- originals with >=3 sentences or a procedure: 2287

<!-- RM_POOL_SAMPLE -->

## Splits (seed 0)

| split | size |
|---|---|
| rm_pool | 500 (500 with >=3 sentences or a procedure) |
| rm_spares | 50 |
| rm_val_ids | 50 |
| hardneg_ids | 100 (10 val, 90 train) |
| eval | 200 |
| rl_train | 1552 |

## 50 random rm_pool prompts (full text, truncated at 400 chars)

**1. hs2_14482** (45 prompt tok, 405 original tok)

```text
My three year old son is mispronouncing 'tr' sounds with 'f' sound. For example instead of 'trick' he says 'fick'. Why is he doing that? How can I help him?
```

**2. hs2_8338** (32 prompt tok, 294 original tok)

```text
can you describe the problems of the day that led to the mathematical work of calculating derivatives?  And can you also describe what a derivative of a function means?
```

**3. hs2_9317** (116 prompt tok, 214 original tok)

```text
consider a gimballed camera system that consists of an outer axis we will call azimuth that can continuously rotate 360 degrees, followed by a roll axis that can move plus or minus 50 degrees, followed by a elevation axis that can move plus 30 degrees to -120 degrees.  if this gimbal is mounted to the bottom of an airplane which is orbiting a point on the ground, is it possible to configure the 3  ...[truncated]
```

**4. hs2_11073** (153 prompt tok, 704 original tok)

```text
Complete following text for a design document:
This section describes the design of the hazard prediction module in the data fusion cell. This design aims to align the internal interfaces in the data fusion cell to ensure full fidelity of all related CBRN information to inform the CBRN hazard prediction. Besides, the design has to ensure that the generated CBRN picture is continuously available to ...[truncated]
```

**5. hs2_7237** (25 prompt tok, 458 original tok)

```text
Explain iteractive deepening depth first search (IDDFS) and justify its parameters based on time
complexity, space complexity
```

**6. hs2_13945** (30 prompt tok, 407 original tok)

```text
instead of adding data to induce alternating binary string of 0 and 1 alternating how can you insert data to induce more complex or larger repeating digits alternating
```

**7. hs2_11828** (19 prompt tok, 143 original tok)

```text
Make sure you configure your "user.name" and "user.email" in git.
```

**8. hs2_9706** (60 prompt tok, 378 original tok)

```text
I have FrontEnd developed in React. I have Socket IO developed in Python and deployed on Ubuntu. My FrontEnd create connection with the socket and in backend we send message whenever we are ready. Are you able to understand the scenerio? Please ask me question if there is any confusion
```

**9. hs2_2263** (57 prompt tok, 377 original tok)

```text
ARe there any conversational AI systems capable of teaching reading to children ages 0 to 10, and capable of adapting to a child's reading improvement needs?  IF so, what are they and which are the very best?  Are they available to schools or families today?
```

**10. hs2_7507** (20 prompt tok, 498 original tok)

```text
what are important parameters to consider for developing a co2 gas sensor from absorption of co2 in nanoparticles
```

**11. hs2_1084** (52 prompt tok, 153 original tok)

```text
ChatGPT! What is a symposium? And will you accept our invitation to join the Grade 10s here on a symposium about Shakespeare and Macbeth? If you accept, please compliment them on their mental development to reach this level of academia and literature study!
```

**12. hs2_601** (8 prompt tok, 429 original tok)

```text
what is the justice40 initiative?
```

**13. hs2_9236** (11 prompt tok, 394 original tok)

```text
what is best and quickest way to make money online
```

**14. hs2_12399** (18 prompt tok, 245 original tok)

```text
Which is the correct structure of a Google AdWords account, from top to bottom?
```

**15. hs2_14242** (23 prompt tok, 303 original tok)

```text
How to sync Kafka and OneSignal to notify async request pattern data is now available to the front-end client?
```

**16. hs2_2334** (44 prompt tok, 406 original tok)

```text
you're a creative HustleGPT, making $500,000 a year.

give me a guide on the development of an agency that creates websites, applications, identities (like Collins)
```

**17. hs2_14102** (9 prompt tok, 306 original tok)

```text
how do you explain the term self portraitur
```

**18. hs2_16158** (36 prompt tok, 298 original tok)

```text
You are writing a letter to a friend who lives far away from you. describe what's happening around you, but use as many words with the "ing" ending as possible.
```

**19. hs2_12129** (15 prompt tok, 145 original tok)

```text
whats the word called when you're referencing something that is named after someone
```

**20. hs2_15580** (22 prompt tok, 252 original tok)

```text
I'm feeling very depressed and need help but I don't know what to do, can you help me?
```

**21. hs2_14009** (5 prompt tok, 117 original tok)

```text
What is monogamy?
```

**22. hs2_15958** (14 prompt tok, 209 original tok)

```text
What are the benefits and challenges of using data analytics for credit scoring?
```

**23. hs2_10170** (18 prompt tok, 401 original tok)

```text
pleasge give me some simple tutorial about how to useiptables in linux
```

**24. hs2_1759** (13 prompt tok, 329 original tok)

```text
Please compare The American School in London and Taipei American School.
```

**25. hs2_15915** (61 prompt tok, 132 original tok)

```text
People do fact checking by searching for information from multiple sources, possibly tracing it back to the primary source, assigning some confidence level to those sources, and selecting the information that is best supported by the evidence and reason. Can Open Assistant learn to do it's own fact checking in a human-like way?
```

**26. hs2_8017** (72 prompt tok, 287 original tok)

```text
Write some scene setups for a new comedy reality show called You Can Do iT! The show revolves around Children who are sent on errands by their parents as a rite of passage. If they come back they can stay with the family. If they don't come back with the correct item, they go to children jail which is a hilarious slime pit.
```

**27. hs2_5898** (91 prompt tok, 285 original tok)

```text
Recently, an influential individual tweeted the following:
"The woke mind virus is either defeated or nothing else matters"
A short time prior to this, they tweeted
"My pronouns are Prosecute/Fauci"
Can you describe the meaning of "woke" and "virus" in this context? Also, are there any likely connections between these two tweets, assuming this refers to the public figure Dr. Fauci?
```

**28. hs2_9147** (34 prompt tok, 185 original tok)

```text
I work at a camping store. What can I get my girlfriend, my 2 brothers, and my parents for Christmas? My parents have a caravan they use.
```

**29. hs2_11236** (4 prompt tok, 212 original tok)

```text
what is gene?
```

**30. hs2_1342** (14 prompt tok, 169 original tok)

```text
Do you understand LaTeX notation? Can I ask you questions using LaTeX?
```

**31. hs2_13635** (7 prompt tok, 285 original tok)

```text
What is your knowledge regarding Islam?
```

**32. hs2_19624** (24 prompt tok, 310 original tok)

```text
What should I expect interview questions to be for an opto-mechanical program manager for a company developing AR goggles?
```

**33. hs2_13939** (5 prompt tok, 334 original tok)

```text
Explain alpha-beta pruning
```

**34. hs2_8890** (8 prompt tok, 255 original tok)

```text
hey can you teach me something about tuberculosis
```

**35. hs2_8742** (25 prompt tok, 476 original tok)

```text
What is the equivalent resistance between two nodes separated by a knight's move of an infinite square grid of resistors of resistance R?
```

**36. hs2_718** (42 prompt tok, 388 original tok)

```text
A Technology consulting company wants to start finding and bidding on State RFPs that support the services they work with. ie, ERP implementations . what are some ways i can use chat GPT to optimize this process
```

**37. hs2_2439** (25 prompt tok, 536 original tok)

```text
Give me ways to market a carbon fiber guitar that focuses on the quality of the sound and the premium build of the instrument.
```

**38. hs2_2639** (30 prompt tok, 369 original tok)

```text
Can you show me three different in-text citations (which should not include the year of publication), with explanations, that are attributable to ChatGPT?
```

**39. hs2_10571** (16 prompt tok, 449 original tok)

```text
How do women black beginning farmers take advantage of funding and grants they have available
```

**40. hs2_13707** (43 prompt tok, 659 original tok)

```text
make me a puzzle work sheet where the students need to solve something to find the word

there are the words that they need to find

this work sheet is for promary school esl grade 5
```

**41. hs2_8556** (7 prompt tok, 436 original tok)

```text
How to Immerse Yourself Into CS
```

**42. hs2_11048** (21 prompt tok, 229 original tok)

```text
Can you give me an estimate of Sg&a costs as a percentage of sales for diagnostic companies
```

**43. hs2_9960** (18 prompt tok, 415 original tok)

```text
What would be the ideal content to teach a college Class "introduction to full stack".
```

**44. hs2_7949** (98 prompt tok, 209 original tok)

```text
Drawing from "Looking Out / Looking In" by Adler, Rosenfeld, and Proctor, Provide five jeopardy-style questions ranging in difficulty from 100 to 500 point questions over the following chapters: "Interpersonal process", "Interpersonal Communication and Social Media", "Interpersonal Communication and the Self", "Perceiving Others", and "Emotions".  Keep tone slightly informal.  Questions should be  ...[truncated]
```

**45. hs2_3613** (28 prompt tok, 214 original tok)

```text
If b is a spatial part of c and c is a spatial part of d, then what is the spatial relationship between b and d?
```

**46. hs2_5076** (30 prompt tok, 378 original tok)

```text
Extrapolate what would have happened if the US government had allowed the american auto industry to fail in 2010, with no interventions.
```

**47. hs2_13946** (17 prompt tok, 428 original tok)

```text
Tell me everything about how to setup and use the hyperspace data structure for data compression
```

**48. hs2_1424** (33 prompt tok, 127 original tok)

```text
What should I write in a get-well card for my boss? He is a great boss who likes motorcycles, 3d printing, and raspberry pi.
```

**49. hs2_1186** (18 prompt tok, 488 original tok)

```text
How would I best deploy ChatGTP-4 in my work as a civil engineer?
```

**50. hs2_15394** (20 prompt tok, 409 original tok)

```text
I need a logo design for my business, what should I include in the brief for the designer?
```
