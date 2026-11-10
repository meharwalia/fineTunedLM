import json
import time

from ste_rlhf.eval.ste_eval import ste_table
from ste_rlhf.ste import cli
from ste_rlhf.ste.metrics import _split_sentences, ste_report

C1_KEYS = {
    "words", "sentences", "avg_sentence_words", "pct_long_sentences", "pct_passive",
    "multi_instruction_sentences", "list_items", "has_list", "needs_list", "long_paragraphs",
    "vague_terms", "complex_words", "filler_phrases", "flesch_reading_ease", "rules", "ste_score", "passes_80",
}

STE_TEXT = """## Install the tool

You need Python 3.10 or newer. Follow these steps.

1. Open a terminal.
2. Run `pip install mytool`.
3. Type `mytool --version`.
4. Check that the version is 2.1.

The tool stores its settings in a file. You can edit this file at any time.
"""

BAD_TEXT = (
    "Basically, it should be noted that the system is utilized by numerous users in order to facilitate "
    "the process of data that is being handled by a lot of various servers, and the results are subsequently "
    "obtained by the administrators of the organization who are responsible for the maintenance of the "
    "infrastructure that was deployed by the team prior to the commencement of the project.\n\n"
    "Essentially, the configuration was written by someone and it is really quite complicated, so things are "
    "often broken by stuff that was changed by some people. At the end of the day, the logs are reviewed by "
    "the team. Needless to say, the issues are fixed by them. Feel free to ask. I hope this helps."
)


def rules(text):
    return ste_report(text)["rules"]


def test_keys_and_types():
    r = ste_report(STE_TEXT)
    assert set(r) == C1_KEYS
    assert set(r["rules"]) == {f"R{i}" for i in range(1, 9)}
    json.dumps(r)


def test_clear_ste_text_scores_high():
    r = ste_report(STE_TEXT)
    assert r["ste_score"] >= 87.5 and r["passes_80"]
    assert r["list_items"] == 4 and r["has_list"]


def test_bad_text_scores_low():
    r = ste_report(BAD_TEXT)
    assert r["ste_score"] <= 37.5 and not r["passes_80"]


def test_score_arithmetic():
    r = ste_report(BAD_TEXT)
    assert r["ste_score"] == round(100 * sum(r["rules"].values()) / 8, 1)


def test_r1():
    assert rules("Open the file. Save it now.")["R1"]
    long_proc = "Open the file that you want to edit and then look at the very first line of text in the editor now."
    assert not rules(long_proc)["R1"]  # 22 words procedure
    long_desc = ("The file holds all of the settings that the program reads when it starts and it keeps "
                 "them in a plain text format for the user.")  # 26 words, descriptive
    assert not rules(long_desc)["R1"]
    ok_desc = "The file holds all of the settings that the program reads when it starts up today for you."  # 19
    assert rules(ok_desc)["R1"]
    # 21-word descriptive sentence is fine (limit 25) but the same as a procedure is too long
    s21 = "The server reads the file when it starts and then it loads all of the rules into memory for use."
    assert rules(s21)["R1"]


def test_r2():
    assert rules("The team wrote the code. It runs fast.")["R2"]
    assert not rules("The code was written by the team.")["R2"]
    assert not rules("The file has been quickly deleted by the script.")["R2"]
    # adjective false positives are not passive
    assert rules("The user is interested in tests. The office is located in Paris.")["R2"]
    assert rules("The tool is used to running tests.")["R2"]


def test_r3():
    assert rules("Open the file. Save it.")["R3"]
    r = ste_report("Open the file and save it.")
    assert not r["rules"]["R3"] and r["multi_instruction_sentences"] == 1
    assert not rules("Open the file, then close the window.")["R3"]
    assert rules("Mix the flour and sugar.")["R3"]
    assert rules("The tool opens the file and saves it.")["R3"]  # not an instruction sentence
    assert not rules("1. Install Python and run the script.")["R3"]


def test_r4():
    prose = "First, open the file. Then, edit the line. Finally, save the file."
    r = ste_report(prose)
    assert r["needs_list"] and not r["rules"]["R4"]
    assert ste_report("1. Open the file.\n2. Edit the line.\n3. Save the file.")["rules"]["R4"]
    assert ste_report("The sky is blue. Water is wet.")["rules"]["R4"]
    assert not ste_report("Open the file. Edit the line. Save the file.")["rules"]["R4"]
    r = ste_report("Colors:\n- red\n- green\n- blue")
    assert r["list_items"] == 3 and r["rules"]["R4"]


def test_r5():
    ok = "One sentence here. Two sentences here. Three here."
    assert rules(ok)["R5"]
    four = "One is here. Two is here. Three is here. Four is here."
    r = ste_report(four)
    assert r["long_paragraphs"] == 1 and not r["rules"]["R5"]
    seven_lines = "\n".join(f"Line {i} is here." for i in range(7))
    assert not rules(seven_lines)["R5"]
    assert ste_report(four + "\n\nShort one.")["long_paragraphs"] == 1


def test_r6():
    assert rules("Use the tool. It works well.")["R6"]
    assert rules("Some files are big.")["R6"]  # 1 occurrence allowed
    r = ste_report("Some files and various things are in a lot of places.")
    assert r["vague_terms"] >= 3 and not r["rules"]["R6"]


def test_r7():
    assert rules("Use the tool to get the file.")["R7"]
    assert rules("You can utilize the tool.")["R7"]  # one allowed
    r = ste_report("You must utilize the tool and facilitate the work prior to commencing.")
    assert r["complex_words"] == 4 and not r["rules"]["R7"]
    assert ste_report("It utilizes it and utilised it.")["complex_words"] == 2  # inflections


def test_r8():
    assert rules("Run the test.")["R8"]
    for t in ["Basically, run the test.", "It is important to note that this works.",
              "Great question! Run the test.", "Sure! Run the test.", "Run it. I hope this helps.",
              "Run the test in order to check it.", "Run it. Overall, it works."]:
        r = ste_report(t)
        assert r["filler_phrases"] >= 1 and not r["rules"]["R8"], t


def test_markdown_and_code_handling():
    text = ("# Heading That Is Not A Sentence\n\n**Note:** use `a_very_long_function_name()` here.\n\n"
            "```python\nbasically = utilize(some, various, things, stuff)\n# Sentence in code. Another one.\n```\n\n"
            "- first item\n* second item\n+ third item\n")
    r = ste_report(text)
    assert r["list_items"] == 3
    assert r["filler_phrases"] == 0 and r["complex_words"] == 0 and r["vague_terms"] == 0
    assert r["sentences"] == 4  # 1 prose + 3 items; heading and code excluded
    r2 = ste_report("1) Open it.\n2) Close it.")
    assert r2["list_items"] == 2 and r2["sentences"] == 2


def test_abbreviations_decimals_urls():
    assert len(_split_sentences("Use tools, e.g. git, i.e. a VCS. It is fast.")) == 2
    assert len(_split_sentences("Ask Dr. Smith about the U.S. rules vs. the others. Then go.")) == 2
    assert len(_split_sentences("The value is 3.14 and the version is 2.7.1. Done.")) == 2
    assert len(_split_sentences("See https://example.com/a.b for more. Then stop.")) == 2
    assert len(_split_sentences("We have apples, pears, etc. Then we leave.")) == 2
    assert len(_split_sentences("We have apples, etc. and pears.")) == 1
    assert ste_report("Use tools, e.g. git. It is fast.")["sentences"] == 2


def test_flesch_reasonable():
    easy = ste_report("The cat sat on the mat. It was a big cat.")["flesch_reading_ease"]
    hard = ste_report("Notwithstanding considerable organizational complexity, institutional "
                      "responsibilities necessitate comprehensive administrative documentation.")["flesch_reading_ease"]
    assert easy > 80 > 20 > hard - 1000 and easy > hard


def test_empty():
    r = ste_report("")
    assert r["ste_score"] == 0 and not r["passes_80"] and r["words"] == 0


def test_speed():
    text = (STE_TEXT + "\n" + BAD_TEXT)[:2000]
    text = " ".join(text.split()[:300]) if len(text.split()) > 300 else text
    ste_report(text)
    t0 = time.perf_counter()
    n = 50
    for _ in range(n):
        ste_report(text)
    per = (time.perf_counter() - t0) / n * 1000
    assert per < 20, per


def test_cli_and_table(tmp_path, capsys):
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps({"id": "a", "ste": STE_TEXT}) + "\n" + json.dumps({"id": "b", "ste": BAD_TEXT}) + "\n")
    assert cli.main([str(p), "--field", "ste", "--show-fails"]) == 0
    out = capsys.readouterr().out
    assert "a\t" in out and "R8:" in out and "n=2" in out and "mean_ste_score" in out
    assert cli.main(["--text", "Basically, open the file."]) == 0
    assert "R8" in capsys.readouterr().out
    tab = ste_table({"good": [STE_TEXT], "bad": [BAD_TEXT]})
    assert tab["good"]["pct_passes_80"] == 100.0 and tab["bad"]["pct_passes_80"] == 0.0
    assert tab["good"]["n"] == 1 and "flesch_mean" in tab["bad"] and "words_mean" in tab["bad"]
