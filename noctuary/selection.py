"""Bounded, read-only selective passive recall, separate from deliberate recall.

Similarity is retrieval evidence, not confidence. Exact title cues can enter the
judge pool below the semantic floor, but never bypass the deterministic direct
threshold. No candidate or packet generation reinforces the memory graph.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re

from .recall import RecallPacket, _one_paragraph

QUERY_CHARS = 1500
POOL_LIMIT = 12
EXCERPT_CHARS = 500
PACKET_HEADER = (
    "## Noctuary passive recall\n"
    "Recall gists; preserve the original uncertainty. Nothing surfaced does "
    "not mean an event did not happen. Drill down with memory_recall or memory_verify."
)
_TYPES = ("concept", "episode", "surface", "pattern")
# These are not entity evidence, even when a broad page happens to use them as
# its title. Do not give generic navigation/umbrella pages an exact-name boost.
_GENERIC = frozenset("""a an the and or but of for to in on at by with from about
my me i you your we our it its this that these those is are was were be been
have has had do does did can could would should will what when where why how
who which tell know remember recall please again also some any all more now
then today yesterday tomorrow memory memories notes overview summary general
personal life things stuff topic topics page pages project projects work
health travel finance finances interests preferences recent update updates
help discuss need want use using details information question questions""".split())


def _words(text):
    return re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)


def _terms(text):
    return {word for word in _words(text) if len(word) > 2 and word not in _GENERIC}


def _phrase(needle, haystack):
    words = _words(needle)
    return bool(words) and (" " + " ".join(words) + " ") in (
        " " + " ".join(_words(haystack)) + " "
    )


def _named(candidate, query):
    title = candidate["title"]
    return bool(_terms(title)) and _phrase(title, query)


def _setting(engine, key, default, low, high):
    # New keys need not exist in older config DEFAULTS or partial fixtures.
    try:
        value = getattr(engine.cfg, "values", {}).get(key, default)
        if isinstance(value, bool):
            return default
        value = float(value)
        return min(high, max(low, value)) if math.isfinite(value) else default
    except (TypeError, ValueError, AttributeError):
        return default


def _support(candidate, query):
    return _terms(query) & _terms(" ".join([
        candidate["title"], candidate["excerpt"], *candidate["topics"]
    ]))


def _priority(candidate, query):
    return (
        -int(_named(candidate, query)),
        -int(_named(candidate, query) and candidate["type"] == "concept"),
        -len(_support(candidate, query)),
        -candidate["similarity"],
        candidate["id"],
    )


def _candidate(node, similarity):
    return {
        "id": node.id, "type": node.type,
        "title": " ".join((node.title or node.id).split())[:200],
        "excerpt": _one_paragraph(node.body, EXCERPT_CHARS),
        "similarity": similarity,
        "topics": [str(topic)[:100] for topic in node.topics[:8]],
    }


def retrieve_candidates(engine, query, *, exclude_ids=()) -> list[dict]:
    """Search all graph layers and retain at most twelve grounded candidates.

    A title-only graph scan rescues explicitly named subjects that a broad,
    multi-domain embedding query misses. Only twelve such matches are embedded;
    their actual cosine is retained, never replaced by a lexical pseudo-score.
    """
    query = (query or "").strip()[:QUERY_CHARS]
    if not query or not _terms(query):
        return []
    excluded = set(exclude_ids)
    vector = engine.embedder.embed([query])[0]
    floor = _setting(engine, "candidateSimilarity", .45, 0, 1)
    raw = engine.index.search(
        vector, node_types=list(_TYPES), include_pinned=True, top_k=POOL_LIMIT, exclude_ids=excluded
    )
    candidates = {}
    for node_id, _node_type, similarity in raw[:POOL_LIMIT]:
        if node_id in excluded or not isinstance(similarity, (int, float)) or not math.isfinite(similarity):
            continue
        node = engine.store.load_node(node_id)
        if node is None or node.id != node_id or node.type not in _TYPES:
            continue
        candidate = _candidate(node, float(similarity))
        if similarity >= floor or _named(candidate, query):
            candidates[node_id] = candidate

    # The graph has no title index. Scan only for literal title matches, not
    # broad keyword hits, so graph size cannot expand the judge/model payload.
    exact = []
    for node in engine.store.all_nodes():
        if node.type not in _TYPES or node.id in candidates or node.id in excluded:
            continue
        candidate = _candidate(node, 0.0)
        if _named(candidate, query):
            exact.append((node, candidate))
            exact.sort(key=lambda pair: _priority(pair[1], query))
            del exact[POOL_LIMIT:]
    if exact:
        vectors = engine.embedder.embed([node.embed_text() for node, _ in exact])
        for (_, candidate), node_vector in zip(exact, vectors):
            similarity = sum(a * b for a, b in zip(vector, node_vector))
            if math.isfinite(similarity):
                candidate["similarity"] = float(similarity)
                candidates[candidate["id"]] = candidate
    return sorted(candidates.values(), key=lambda c: _priority(c, query))[:POOL_LIMIT]


def _redundant(engine, candidate, selected):
    """Collapse copies and linked abstractions, not every related memory."""
    left = engine.store.load_node(candidate["id"])
    left_text = _terms(candidate["excerpt"])
    for other in selected:
        if candidate["id"] == other["id"]:
            return True
        right = engine.store.load_node(other["id"])
        right_text = _terms(other["excerpt"])
        overlap = len(left_text & right_text)
        if left_text and right_text and overlap / max(len(left_text), len(right_text)) >= .8:
            return True
        if left is None or right is None:
            continue
        for a, b in ((left, right), (right, left)):
            if any(b.id in a.links.get(kind, []) for kind in ("broader", "narrower", "supports")):
                return True
            # A concept and its retelling can have only a related/wiki link.
            linked = b.id in a.links.get("related", []) or f"[[{b.id}]]" in a.body
            if linked and {a.type, b.type} & {"concept", "surface"}:
                if overlap >= 2 and overlap / max(1, min(len(left_text), len(right_text))) >= .5:
                    return True
    return False


def _distinct_cues(candidates, query):
    """Require disjoint explicit topic/title cues for *every* extra domain.

    Merely naming three nodes in one domain is insufficient when their topic
    metadata overlaps; do not infer domains from the judge's enthusiasm.
    """
    domains = []
    topic_sets = []
    for candidate in candidates:
        topics = {_words_key(t) for t in candidate["topics"] if _words_key(t)}
        if any(topics & previous for previous in topic_sets):
            return False
        topic_sets.append(topics)
        cues = set().union(*(
            _terms(t) for t in candidate["topics"] if _phrase(t, query)
        )) if candidate["topics"] else set()
        if not cues and _named(candidate, query):
            cues = _terms(candidate["title"])
        if not cues or any(cues & previous for previous in domains):
            return False
        domains.append(cues)
    return True


def _words_key(text):
    return " ".join(_words(text))


def _judge_choices(response, candidates):
    """Validate independently of the transport; any invalid response is empty."""
    known = {candidate["id"] for candidate in candidates}
    if type(response) is not list or len(response) > POOL_LIMIT:
        return None
    seen = set()
    for item in response:
        if type(item) is not dict or set(item) != {"id", "kind", "relevance"}:
            return None
        node_id, kind, relevance = item["id"], item["kind"], item["relevance"]
        if (type(node_id) is not str or node_id not in known or node_id in seen
                or type(kind) is not str or kind not in ("direct", "association")
                or type(relevance) is not int or not 0 <= relevance <= 3):
            return None
        seen.add(node_id)
    return {item["id"]: item["kind"] for item in response
            if (item["kind"] == "direct" and item["relevance"] == 3)
            or (item["kind"] == "association" and item["relevance"] >= 2)}


def select_packet(engine, query, *, exclude_ids=(), recent_context="", judge=None,
                  allow_association=False) -> RecallPacket:
    """Select without reinforcement; judge failures return no passive recall."""
    empty = RecallPacket("", [])
    query = (query or "").strip()[:QUERY_CHARS]
    if not query:
        return empty
    try:
        excluded = set(exclude_ids)
        retrieval_query = query
        # Expand referential queries only for candidate discovery; the judge
        # still receives current and previous conversation as separate fields.
        if judge is not None and recent_context and re.search(r'\b(that|this|it|those|they|their)\b', query, re.I):
            retrieval_query = query[:650] + '\n' + str(recent_context)[-800:]
        candidates = retrieve_candidates(engine, retrieval_query, exclude_ids=excluded)
        if not candidates:
            return empty
        if judge is None:
            threshold = _setting(engine, "directRecallSimilarity", .68, 0, 1)
            choices = {c["id"]: "direct" for c in candidates
                       if c["similarity"] >= threshold and _support(c, query)}
        else:
            response = judge(query, (recent_context or "")[-6000:], deepcopy(candidates))
            choices = _judge_choices(response, candidates)
            if choices is None:
                return empty
        candidates.sort(key=lambda c: _priority(c, query))
        direct_limit = int(_setting(engine, "maxDirectRecallEntries", 2, 0, POOL_LIMIT))
        multi_limit = int(_setting(engine, "maxMultiDomainRecallEntries", 3, 0, POOL_LIMIT))
        direct = []
        for candidate in candidates:
            if choices.get(candidate["id"]) != "direct" or _redundant(engine, candidate, direct):
                continue
            if len(direct) < direct_limit:
                direct.append(candidate)
            elif (direct_limit > 0 and len(direct) < multi_limit
                  and _distinct_cues(direct + [candidate], query)):
                direct.append(candidate)
        selected = [(c, "direct") for c in direct]
        if allow_association and judge is not None:
            for candidate in candidates:
                if (choices.get(candidate["id"]) == "association"
                        and not _redundant(engine, candidate, direct)):
                    selected.append((candidate, "association"))
                    break
        budget = int(_setting(engine, "recallTokenBudget", 1000, 0, 100000)) * 4
        text, node_ids = PACKET_HEADER, []
        for candidate, kind in selected:
            line = (f"\n- [{kind} | gist | node {candidate['id']} ({candidate['type']})] "
                    f"{candidate['title']} — {candidate['excerpt']}")
            if len(text) + len(line) <= budget:
                text += line
                node_ids.append(candidate["id"])
        return RecallPacket(text, node_ids) if node_ids else empty
    except Exception:
        # An optional passive packet must never degrade the foreground turn or
        # fall back to unjudged memories after a transport/validation failure.
        return empty
