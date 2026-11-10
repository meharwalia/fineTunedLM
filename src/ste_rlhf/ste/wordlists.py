"""Word lists for the STE checker (pure data, plus small form generators)."""
from __future__ import annotations

# Abbreviations whose trailing period must not end a sentence (lowercase, no final dot).
ABBREVIATIONS = [
    "e.g", "i.e", "etc", "vs", "dr", "mr", "mrs", "ms", "prof", "u.s", "u.k", "approx",
    "fig", "cf", "inc", "ltd", "jr", "sr", "st", "a.m", "p.m", "viz",
]

# Verbs that start an imperative (instruction) sentence.
IMPERATIVES = set("""
add allow answer apply ask avoid back begin bake boil bookmark break bring build call change check choose
chop clean clear click close combine compare compile configure confirm connect consider convert copy count
create cut define delete describe determine dice disable do download drag draw drop dry edit enable enter
ensure execute exit expand explain export extract fill find finish fix follow format get give go grab heat
hold identify ignore import include increase insert install keep launch learn leave let lift list load
locate log look lower make map measure merge mix move name navigate note open order paste pick place plan
plug pour press print provide pull push put read record refer remember remove rename repeat replace reply
report request reset restart restore return review rinse rotate run save scroll search season see select
send serve set show sign slice slide sort split start stir stop store submit subtract switch take tap tell
test think tighten toggle trim try tune turn type uninstall unplug unzip update upgrade upload use validate
verify visit wait wash watch wipe wrap write
""".split())

# Words that may precede an imperative verb without changing it.
LEAD_ADVERBS = {
    "first", "firstly", "second", "secondly", "third", "next", "then", "finally", "lastly", "now",
    "also", "afterwards", "please", "always", "never", "just", "again", "later", "once", "done",
}

# Irregular past participles (the regular -ed ones are caught by suffix).
IRREGULAR_PARTICIPLES = set("""
begun bitten blown broken brought built bought caught chosen come done drawn driven eaten fallen fed
found flown forgotten forgiven frozen given grown held hidden hit hurt kept known laid led left lent
lost made meant met paid put read ridden rung risen run said seen sold sent set shaken shown shut sung
sunk slept spoken spent split spread stolen struck sworn swept taken taught thrown told thought torn
understood woken won worn written withdrawn overwritten rewritten undone beaten bent bound burnt
cast cut dealt dug fit flung ground hung lit overtaken shot shrunk sought stuck stung stood
sewn sown spun swum wound wrung forbidden proven mistaken broadcast
""".split())

# -ed/-en words that behave as adjectives after "be" (not counted as passive).
PASSIVE_ADJ_EXCLUDE = set("""
interested located excited tired worried concerned surprised pleased scared bored confused married
supposed limited advanced related unlimited complicated detailed skilled experienced determined
committed dedicated sacred hundred naked wicked indeed speed seed feed need bleed
""".split())

# Vague quantifiers / filler nouns (R6). 'numerous' lives in R7 only (no double counting).
VAGUE = [
    "some", "various", "a lot of", "lots of", "a lot", "stuff", "things", "thing", "kind of", "sort of",
    "several", "many", "etc", "and so on", "and so forth", "whatever", "somehow", "quite",
    "fairly", "really", "very", "pretty much", "somewhat", "a bit", "a few", "a number of",
    "a variety of", "a range of", "certain", "and more", "or so", "plenty of", "loads of", "tons of",
    "countless", "myriad",
]

# Filler phrases (R8).
FILLERS = [
    "basically", "essentially", "it should be noted that", "it is important to note", "it's important to note",
    "it is worth noting", "it's worth noting", "in order to", "as a matter of fact", "at the end of the day",
    "needless to say", "in conclusion", "to sum up", "in summary", "i hope this helps", "hope this helps",
    "feel free to", "great question", "of course", "let me know if", "please note that", "as you can see",
    "when it comes to", "it goes without saying", "at this point in time", "in my opinion", "to be honest",
    "to be clear", "without further ado", "i would be happy to", "i'd be happy to", "happy to help",
    "don't hesitate to", "do not hesitate to", "obviously", "simply put",
]
# Regex-only fillers: exclamation openers and sentence-initial "Overall,".
FILLER_REGEXES = [r"\b(?:sure|certainly|absolutely|definitely)!", r"(?:^|[.!?]\s+)overall,"]

# (complex, simple, kind, extra_forms). kind 'v' = verb (inflected), 'x' = exact (plus plural s).
COMPLEX = [
    ("utilize", "use", "v", ["utilization", "utilisation"]),
    ("facilitate", "help", "v", ["facilitation"]),
    ("commence", "start", "v", ["commencement"]),
    ("terminate", "end", "v", ["termination"]),
    ("approximately", "about", "x", []),
    ("subsequently", "then / after", "x", ["subsequent"]),
    ("prior to", "before", "x", []),
    ("in addition", "also", "x", ["in addition to"]),
    ("sufficient", "enough", "x", []),
    ("demonstrate", "show", "v", ["demonstration"]),
    ("endeavor", "try", "v", ["endeavour"]),
    ("numerous", "many", "x", []),
    ("obtain", "get", "v", []),
    ("require", "need", "v", ["requirement"]),
    ("purchase", "buy", "v", []),
    ("assist", "help", "v", ["assistance"]),
    ("attempt", "try", "v", []),
    ("additional", "more / extra", "x", []),
    ("consequently", "so", "x", []),
    ("therefore", "so", "x", []),
    ("thus", "so", "x", []),
    ("initiate", "start", "v", ["initiation"]),
    ("modify", "change", "v", ["modification"]),
    ("regarding", "about", "x", []),
    ("concerning", "about", "x", []),
    ("ensure", "make sure", "v", []),
    ("indicate", "show", "v", ["indication"]),
    ("possess", "have", "v", []),
    ("construct", "build", "v", []),
    ("determine", "find", "v", []),
    ("optimal", "best", "x", []),
    ("leverage", "use", "v", []),
    ("employ", "use", "v", []),
    ("nevertheless", "but", "x", []),
    ("furthermore", "also", "x", []),
    ("moreover", "also", "x", []),
    ("in the event that", "if", "x", []),
    ("in the event of", "if", "x", []),
    ("a majority of", "most", "x", []),
    ("with regard to", "about", "x", []),
    ("due to the fact that", "because", "x", []),
    ("elucidate", "explain", "v", []),
    ("ascertain", "find out", "v", []),
    ("reside", "live", "v", []),
    ("inquire", "ask", "v", []),
    ("comprehend", "understand", "v", []),
    ("acquire", "get", "v", []),
    ("attain", "reach", "v", []),
    ("depict", "show", "v", []),
    ("exhibit", "show", "v", []),
    ("methodology", "method", "x", []),
    ("functionality", "function", "x", []),
    ("aforementioned", "this", "x", []),
    ("discontinue", "stop", "v", []),
    ("cease", "stop", "v", []),
    ("retain", "keep", "v", []),
    ("relocate", "move", "v", []),
    ("magnitude", "size", "x", []),
    ("adequate", "enough", "x", []),
    ("frequently", "often", "x", []),
    ("initially", "first", "x", []),
    ("currently", "now", "x", []),
    ("presently", "now", "x", []),
    ("remainder", "rest", "x", []),
    ("beneficial", "helpful", "x", []),
    ("necessitate", "need", "v", []),
    ("permit", "let", "v", []),
    ("accomplish", "do", "v", []),
    ("whilst", "while", "x", []),
    ("amongst", "among", "x", []),
]


def verb_forms(w: str) -> list[str]:
    """Inflected surface forms of a base verb (over-generating is harmless)."""
    forms = {w}
    if w.endswith("e"):
        forms |= {w + "s", w + "d", w[:-1] + "ing"}
    elif w.endswith("y") and len(w) > 1 and w[-2] not in "aeiou":
        forms |= {w[:-1] + "ies", w[:-1] + "ied", w + "ing"}
    else:
        forms |= {w + "s", w + "es", w + "ed", w + "ing", w + w[-1] + "ed", w + w[-1] + "ing"}
    return sorted(forms)


def entry_forms(complex_word: str, kind: str, extra: list[str]) -> list[str]:
    base = verb_forms(complex_word) if kind == "v" else [complex_word, complex_word + "s"]
    out = set(base) | set(extra)
    for f in list(out):  # British spelling variants
        if "iz" in f:
            out.add(f.replace("iz", "is"))
    return sorted(out)
