"""
pipeline.py — Step 2: call the APIs once, cache what they say, flatten it into tables.

A pipeline built on someone else's API has two kinds of step, and keeping them
apart is the whole design:

- **Extract steps** call the API — slow, rate-limited, and the only part that can
  fail because of the network. Each one saves the raw JSON it got back to a cache
  file, and the next time it runs it reads that file instead of calling again.
  (Lesson 4-2: cache every call you can.)
- **Transform steps** turn that JSON into DataFrames. They never touch the
  network, so you can run them a hundred times while you get them right.

    reviews ─sentiment_step─▶ sentiment JSON ─sentences_table─▶ a row per sentence
                                            └opinions_table──▶ a row per target
    reviews ─entities_step──▶ entities JSON ─entities_table──▶ a row per entity
    mentions ─subjects_step─▶ a row per mention, with its subject

Each arrow changes the **grain** — review, sentence, target, entity — the idea
from Assignment 05, now coming out of JSON instead of a CSV.

Less scaffolding here: the steps are described, the code is yours.
"""

import json
import os

import pandas as pd

from .apicalls import classify_mentions, get_entities, get_sentiment
from .settings import MENTION_CATEGORIES


def sentiment_step(reviews: pd.DataFrame, cache_file: str) -> list:
    """Sentiment for every review, from the cache if it exists, else from the API.

    Returns a list with one dict per review: the Azure document
    (`response["results"]["documents"][0]`) with one extra key, `review_id`,
    copied from the review's row — so every document says which review it is.

    Steps: if `cache_file` exists, `json.load` it and return it — no API calls.
    Otherwise start an empty list; for each review (loop over
    `reviews.to_dict("records")` to get one dict per row) call `get_sentiment` on
    its `text`, take the document out of the response, set
    `document["review_id"]`, and append it. Then `json.dump` the list to
    `cache_file` and return it.

    Delete the cache file to force fresh calls. The portal caches too, so a
    re-run on the same reviews is free either way — but your cache also works
    offline, and it is what makes the transform steps below fast to debug.
    """
    # if the file is already there, read the cache and stop without API calls
    if os.path.exists(cache_file):
        with open(cache_file) as fh:
            return json.load(fh)

    # if the file is not there, call the API once per review
    documents = []
    for row in reviews.to_dict("records"):
        response = get_sentiment(row["text"])
        document = response["results"]["documents"][0]
        document["review_id"] = row["review_id"]
        documents.append(document)

    # save what the API said so the next run reads the file instead of API
    with open(cache_file, "w") as fh:
        json.dump(documents, fh)
    return documents


def entities_step(reviews: pd.DataFrame, cache_file: str) -> list:
    """Named entities for every review, cached — the same shape as `sentiment_step`.

    One dict per review: the Azure entity-recognition document with `review_id`
    added. Same cache rule, same loop, `get_entities` instead of `get_sentiment`.
    """
    # if the file is already there, read the cache and stop without API calls
    if os.path.exists(cache_file):
        with open(cache_file) as fh:
            return json.load(fh)

    # if the file is not there, call the API once per review
    documents = []
    for row in reviews.to_dict("records"):
        response = get_entities(row["text"])
        document = response["results"]["documents"][0]
        document["review_id"] = row["review_id"]
        documents.append(document)

    # save what the API said so the next run reads the file instead of API
    with open(cache_file, "w") as fh:
        json.dump(documents, fh)
    return documents


def sentences_table(documents: list) -> pd.DataFrame:
    """One row per sentence: the sentiment documents flattened with `json_normalize`.

    Columns, in this order: `review_id`, `sentence_id`, `sentence_text`,
    `sentence_sentiment`, `sentence_offset`, `sentence_length`.

    Steps: `pd.json_normalize(documents, record_path="sentences",
    meta=["review_id"], record_prefix="sentence_")` gives one row per sentence
    with every field prefixed — `sentence_text`, `sentence_sentiment`, ... — and
    the parent's `review_id` copied onto each (lesson 4-1). `sentence_id` is the
    sentence's position in its review, 0, 1, 2, ...: a group-by on `review_id`
    and `.cumcount()`. Keep only the six columns above. Strip the trailing space
    Azure leaves on `sentence_text` with `.str.strip()`.
    """
    # one row per sentence, review_id copied from the parent document
    df = pd.json_normalize(documents, record_path="sentences",
                           meta=["review_id"], record_prefix="sentence_")

    # number the sentences in each review
    df["sentence_id"] = df.groupby("review_id").cumcount()

    # remove the trailing space on each sentence from azure
    df["sentence_text"] = df["sentence_text"].str.strip()

    # keep only the six columns in their order
    return df[["review_id", "sentence_id", "sentence_text",
               "sentence_sentiment", "sentence_offset", "sentence_length"]]


def opinions_table(documents: list) -> pd.DataFrame:
    """One row per opinion target, with the words that carry the opinion.

    Columns, in this order: `review_id`, `sentence_id`, `mention`,
    `mention_sentiment`, `assessments`, `mention_offset`, `source`.

    Each sentence in a sentiment document has `targets` and `assessments`. A
    target ("mocha latte") points at its assessments ("overpriced", "lukewarm")
    through its `relations` list — each relation has a `relationType` and a
    `ref` like `"#/documents/0/sentences/1/assessments/0"`. The number after
    `assessments/` is the assessment's position in that sentence's
    `assessments` list.

    This one is three levels deep with cross-references, which is past what
    `json_normalize` does in one call — so build it with nested loops (lesson
    4-1, "parsing the JSON response"). For each document, for each sentence
    (`enumerate` gives you `sentence_id`), for each target: collect the text of
    every assessment its relations point to (only relations whose
    `relationType` is `"assessment"`), join them with `", "`, and append a row
    dict. `mention` is the target's `text`, `mention_sentiment` its `sentiment`,
    `mention_offset` its `offset`, and `source` is always `"opinion"`. Return
    `pd.DataFrame(rows, columns=[...the seven names...])`, so a set of reviews
    with no targets still gives a table with the right columns.
    """
    rows = []
    for document in documents:
        # enumerate gives each sentence's position: 0, 1, 2...
        for sentence_id, sentence in enumerate(document["sentences"]):
            for target in sentence["targets"]:
                # follow each relation to the assessment it points at
                words = []
                for relation in target["relations"]:
                    if relation["relationType"] == "assessment":
                        position = int(relation["ref"].split("/")[-1])
                        words.append(sentence["assessments"][position]["text"])

                rows.append({
                    "review_id": document["review_id"],
                    "sentence_id": sentence_id,
                    "mention": target["text"],
                    "mention_sentiment": target["sentiment"],
                    "assessments": ", ".join(words),
                    "mention_offset": target["offset"],
                    "source": "opinion",
                })

    return pd.DataFrame(rows, columns=["review_id", "sentence_id", "mention",
                                       "mention_sentiment", "assessments",
                                       "mention_offset", "source"])


def entities_table(documents: list) -> pd.DataFrame:
    """One row per person or product the entity recognizer found.

    Columns, in this order: `review_id`, `entity_text`, `entity_category`,
    `entity_offset`. Only the categories in `MENTION_CATEGORIES` ("Person",
    "Product"); a plain 0, 1, 2, ... index.

    Steps: `json_normalize` again — `record_path="entities"`,
    `meta=["review_id"]`, `record_prefix="entity_"` — then a boolean index with
    `.isin(MENTION_CATEGORIES)` on `entity_category`. A document with no
    entities contributes no rows, which `json_normalize` handles for you.
    """
    # one row per entity, copy review_id from the parent doc
    df = pd.json_normalize(documents, record_path="entities",
                           meta=["review_id"], record_prefix="entity_")

    # keep only people and products
    df = df[df["entity_category"].isin(MENTION_CATEGORIES)]

    # the four columns in order with numbered index
    return df[["review_id", "entity_text", "entity_category",
               "entity_offset"]].reset_index(drop=True)


def subjects_step(reviews: pd.DataFrame, mentions: pd.DataFrame,
                  cache_file: str) -> pd.DataFrame:
    """What each mention is about — service, atmosphere, product, other — cached.

    Returns one row per (review, mention) with columns `review_id`, `mention`,
    `subject`, `subject_confidence`.

    Steps: if `cache_file` exists, `pd.read_csv` it and return it. Otherwise, for
    each review that has mentions: take its mentions in order of
    `mention_offset`, without repeats (`.drop_duplicates()` keeps the first),
    and call `classify_mentions(review_text, that_list)`. The answers come back
    keyed by mention: `response["answers"][mention]["choice"]` and
    `["confidence"]`. Append one row dict per mention. Build the DataFrame,
    `to_csv(cache_file, index=False)`, and return it.

    Always send the mentions in offset order: the portal only answers from its
    cache when the request is exactly the same, so the same reviews must
    produce the same list every time.
    """
    # read the cache with no API calls
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file)

    rows = []
    for review in reviews.to_dict("records"):
        these = mentions[mentions["review_id"] == review["review_id"]]
        if these.empty:
            continue

        # in offset order, no repeats
        names = (these.sort_values("mention_offset")["mention"]
                 .drop_duplicates().tolist())

        # one API call per review, asking about all its mentions at once
        response = classify_mentions(review["text"], names)

        # one row per mention, from the answer for that mention
        for name in names:
            answer = response["answers"][name]
            rows.append({
                "review_id": review["review_id"],
                "mention": name,
                "subject": answer["choice"],
                "subject_confidence": answer["confidence"],
            })

    # save so the next run reads the file instead
    df = pd.DataFrame(rows, columns=["review_id", "mention",
                                     "subject", "subject_confidence"])
    df.to_csv(cache_file, index=False)
    return df
