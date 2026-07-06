#!/usr/bin/env python3
"""GPT-Eval for the Juzhen TinyStories model, following Eldan & Li (2023).

Two phases:
  generate  — feed ~40 held-out-style story openings to our model's /generate
              endpoint (default http://localhost:8127) and save completions.
  grade     — send each (prompt, completion) to Claude and ask it to grade
              Grammar / Creativity / Consistency on a 0-10 scale plus an age
              group, exactly as the paper's GPT-Eval does. Aggregates and
              prints averages next to the paper's hidden=512 / 8-layer row.

The grader is an LLM-as-judge; it uses the Anthropic API (claude-opus-4-8).
Generation needs no API key; grading does.

  python scripts/tinystories_gpteval.py generate
  python scripts/tinystories_gpteval.py grade      # needs ANTHROPIC_API_KEY or `ant auth login`
"""

import json
import os
import sys
import time
import urllib.request

OUT = "/mnt/external_hdd/data/nlp/tinystories/gpteval_completions.json"
SERVER = os.environ.get("TS_SERVER", "http://localhost:8127")
JUDGE_MODEL = "claude-opus-4-8"
OPENAI_JUDGE = os.environ.get("OPENAI_JUDGE", "gpt-4o")  # paper used gpt-4 (2023)

# Held-out-style openings: partial sentences cut so a grammatically nontrivial
# completion is required (the paper's prompt design). Simple 3-4-year-old vocab.
PROMPTS = [
    "Once upon a time, there was a little girl named Lily who loved to",
    "Tom and his dog went to the park. Suddenly, they saw a big",
    "One sunny day, a small bird wanted to learn how to",
    "The little robot was sad because it could not find its",
    "In a warm and cozy house, a cat named Whiskers liked to",
    "Every morning, Ben would wake up and run to the window to see",
    "There was a magic tree in the garden. When the wind blew, it would",
    "Lucy found a shiny red box under her bed. When she opened it, she",
    "The three little rabbits were hungry, so they decided to",
    "A tiny fish lived in the big blue sea. One day it met a",
    "When the rain started to fall, the children ran inside and began to",
    "Grandpa told Mia a story about a brave knight who wanted to",
    "The ice cream truck came down the street, and all the kids",
    "Sam built a tall tower with his blocks, but then his baby sister",
    "On the farm, the little duck could not swim, so the wise old",
    "The moon was bright that night, and the sleepy bear decided to",
    "Katie planted a seed in the ground and waited every day for it to",
    "A friendly dragon lived on the hill. He was not scary at all; he just",
    "The toy soldier fell behind the couch and was afraid he would never",
    "When Jack lost his balloon, it floated up into the sky and",
    "Two best friends, a mouse and an elephant, wanted to play a game but",
    "The old clock in the hall went tick-tock, and at midnight it suddenly",
    "Emma had a red umbrella that she loved. On a windy day it",
    "The baby bird fell out of its nest, and a kind squirrel decided to",
    "At the beach, Leo dug a deep hole in the sand and found a",
    "The snowman in the yard was lonely until a little girl came and",
    "Every night the stars would come out, and the little owl liked to",
    "When the power went out, the family lit some candles and started to",
    "A curious kitten climbed up the tall tree and then realized it could not",
    "The gingerbread man jumped out of the oven and ran as fast as he could to",
    "Nina had a magic paintbrush. Whatever she painted would come to life, so she",
    "The little train tried to climb the big hill, huffing and puffing, saying",
    "On his birthday, Max got a shiny new bike, but he did not know how to",
    "The butterfly landed on Rosa's nose, and she giggled because it",
    "Deep in the forest, a lost puppy was looking for its mother when it",
    "The wind blew the kite higher and higher until it almost touched the",
    "Anna shared her sandwich with a hungry bird, and the bird was so happy that it",
    "The little boat sailed across the pond, but a big frog jumped in and",
    "When the clock struck three, the toys in the room came alive and began to",
    "Oliver was scared of the dark, but then his night light showed him that",
]


def generate():
    completions = []
    for i, p in enumerate(PROMPTS):
        url = f"{SERVER}/generate?length=200&temperature=0.8&topk=40"
        req = urllib.request.Request(url, data=p.encode("utf-8"),
                                     headers={"Content-Type": "text/plain; charset=utf-8"})
        text = urllib.request.urlopen(req, timeout=180).read().decode("utf-8", "replace")
        completions.append({"prompt": p, "completion": text})
        print(f"[{i+1}/{len(PROMPTS)}] {p[:50]}... -> {len(text)} chars")
    with open(OUT, "w") as f:
        json.dump(completions, f, indent=2)
    print(f"\nsaved {len(completions)} completions to {OUT}")


GRADE_INSTRUCTIONS = """\
You are grading a story completion written by a student, in the style of the \
TinyStories GPT-Eval (Eldan & Li 2023). The student was given the beginning of \
a story (before the ***) and wrote the continuation (after the ***). The \
beginning was cut mid-sentence, so a good completion must first finish that \
sentence grammatically, then continue the story.

Grade ONLY the student's continuation (after the ***), on three axes, each an \
integer from 0 to 10:
- grammar: is it grammatically correct English?
- creativity: is the continuation imaginative and varied?
- consistency: is it consistent with the beginning (characters, plot, tone)?
Also estimate the age group the writing reflects, one of: A (3 or under), \
B (4-5), C (6-7), D (8-9), E (10-12), F (13-16).

Story beginning + student completion:
{combined}
"""


def grade():
    import anthropic
    from pydantic import BaseModel

    class Grade(BaseModel):
        grammar: int
        creativity: int
        consistency: int
        age_group: str
        note: str

    with open(OUT) as f:
        items = json.load(f)

    client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY or ant profile
    graded = []
    for i, it in enumerate(items):
        combined = it["prompt"] + " *** " + it["completion"]
        msg = client.messages.parse(
            model=JUDGE_MODEL,
            max_tokens=1024,
            messages=[{"role": "user",
                       "content": GRADE_INSTRUCTIONS.format(combined=combined)}],
            output_format=Grade,
        )
        g = msg.parsed_output
        graded.append({**it, "grade": g.model_dump()})
        print(f"[{i+1}/{len(items)}] gram {g.grammar} crea {g.creativity} "
              f"cons {g.consistency} age {g.age_group}")
        time.sleep(0.2)

    n = len(graded)
    avg = lambda k: sum(x["grade"][k] for x in graded) / n
    ga, ca, coa = avg("grammar"), avg("creativity"), avg("consistency")

    with open(OUT.replace(".json", "_graded.json"), "w") as f:
        json.dump(graded, f, indent=2)

    print("\n=== GPT-Eval results (our TinyStories model, %d prompts) ===" % n)
    print(f"  Grammar     {ga:.2f} / 10")
    print(f"  Creativity  {ca:.2f} / 10")
    print(f"  Consistency {coa:.2f} / 10")
    print("\n--- paper (Eldan & Li 2023), hidden=512 / 8-layer, GPT-4 judge ---")
    print("  Grammar     8.34 / 10")
    print("  Creativity  6.85 / 10")
    print("  Consistency 8.95 / 10")
    print("\n(judge differs: paper used GPT-4, we used %s — treat as indicative,\n"
          " not a controlled head-to-head.)" % JUDGE_MODEL)


def grade_openai():
    """Same rubric/completions as grade(), but judged by GPT-4 (OpenAI) — the
    paper's actual grader, so results are directly comparable to its table."""
    from openai import OpenAI
    from pydantic import BaseModel

    class Grade(BaseModel):
        grammar: int
        creativity: int
        consistency: int
        age_group: str
        note: str

    with open(OUT) as f:
        items = json.load(f)

    client = OpenAI()  # reads OPENAI_API_KEY
    graded = []
    for i, it in enumerate(items):
        combined = it["prompt"] + " *** " + it["completion"]
        resp = client.chat.completions.parse(
            model=OPENAI_JUDGE,
            messages=[{"role": "user",
                       "content": GRADE_INSTRUCTIONS.format(combined=combined)}],
            response_format=Grade,
        )
        g = resp.choices[0].message.parsed
        graded.append({**it, "grade": g.model_dump()})
        print(f"[{i+1}/{len(items)}] gram {g.grammar} crea {g.creativity} "
              f"cons {g.consistency} age {g.age_group}")
        time.sleep(0.2)

    n = len(graded)
    avg = lambda k: sum(x["grade"][k] for x in graded) / n
    ga, ca, coa = avg("grammar"), avg("creativity"), avg("consistency")

    with open(OUT.replace(".json", "_graded_gpt4.json"), "w") as f:
        json.dump(graded, f, indent=2)

    print("\n=== GPT-Eval results (our model, %d prompts, judge=%s) ===" % (n, OPENAI_JUDGE))
    print(f"  Grammar     {ga:.2f} / 10")
    print(f"  Creativity  {ca:.2f} / 10")
    print(f"  Consistency {coa:.2f} / 10")
    print("\n--- paper (Eldan & Li 2023), hidden=512 / 8-layer, GPT-4 judge ---")
    print("  Grammar     8.34 / 10")
    print("  Creativity  6.85 / 10")
    print("  Consistency 8.95 / 10")
    print("\n(same judge family as the paper — this is the apples-to-apples row;\n"
          " %s is a current GPT-4-family model, not the exact 2023 snapshot.)" % OPENAI_JUDGE)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "generate"
    if cmd == "generate":
        generate()
    elif cmd == "grade":
        grade()
    elif cmd == "grade_openai":
        grade_openai()
    else:
        sys.exit("usage: tinystories_gpteval.py [generate|grade|grade_openai]")
