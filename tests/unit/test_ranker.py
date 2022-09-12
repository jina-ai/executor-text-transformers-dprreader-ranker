from pathlib import Path
from typing import List

import pytest
from jina import Document, DocumentArray, Executor

from dpr_reader import DPRReaderRanker


@pytest.fixture(scope='session')
def basic_ranker() -> DPRReaderRanker:
    return DPRReaderRanker()


@pytest.fixture(scope='session')
def basic_ranker_title() -> DPRReaderRanker:
    return DPRReaderRanker(title_tag_key='title')


@pytest.fixture(scope='function')
def example_docs(request) -> DocumentArray:
    """Generate example documents with matches (up to 10)."""
    data_dir = Path(__file__).parent.parent / 'test_data'
    with open(data_dir / 'questions.txt', 'r') as f:
        questions = [line.strip() for line in f.readlines()]

    with open(data_dir / 'text.txt', 'r') as f:
        passages = [line.strip() for line in f.readlines()]

    with open(data_dir / 'titles.txt', 'r') as f:
        titles = [line.strip() for line in f.readlines()]

    docs = DocumentArray()
    for question in questions[: request.param]:
        doc = Document(text=question)
        matches = [
            Document(text=passage, tags={'title': title, 'something_else': 'whatever'})
            for passage, title in zip(passages[:10], titles)
        ]
        doc.matches.extend(matches)
        docs.append(doc)

    return docs


def test_config():
    encoder = Executor.load_config(str(Path(__file__).parents[2] / 'config.yml'))
    assert encoder.batch_size == 32
    assert encoder.access_paths == '@r'
    assert encoder.title_tag_key is None
    assert encoder.num_spans_per_match == 2


def test_empty_documents(basic_ranker: DPRReaderRanker):
    docs = DocumentArray([])
    basic_ranker.rank(docs, {})
    assert len(docs) == 0


def test_no_text_documents(basic_ranker: DPRReaderRanker):
    docs = DocumentArray([Document()])
    basic_ranker.rank(docs, {})
    assert len(docs[0].matches) == 0


def test_documents_no_matches(basic_ranker: DPRReaderRanker):
    docs = DocumentArray([Document(text='I have no matches')])
    basic_ranker.rank(docs, {})
    assert len(docs) == 1
    assert len(docs[0].matches) == 0


def test_matches_no_title(basic_ranker_title: DPRReaderRanker):
    doc = Document(text='A question?')
    doc.matches.append(Document(text='I have no titile.'))
    docs = DocumentArray([doc])

    with pytest.raises(ValueError, match='All matches are required to have'):
        basic_ranker_title.rank(docs, {})


@pytest.mark.parametrize('example_docs', [2], indirect=['example_docs'])
def test_ranking_cpu(basic_ranker: DPRReaderRanker, example_docs: DocumentArray):

    basic_ranker.rank(example_docs, {})

    assert len(example_docs[0].matches) == 20

    # A quirk related to how HF chooses spans/overlapping
    assert len(example_docs[1].matches) == 19

    for match in example_docs[0].matches:
        assert 'relevance_score' in match.scores
        assert 'span_score' in match.scores
        assert 'something_else' in match.tags


@pytest.mark.parametrize('example_docs', [10], indirect=['example_docs'])
def test_spans_title_match(
    example_docs: DocumentArray, basic_ranker_title: DPRReaderRanker
):
    title_text_dict = {}
    for match in example_docs[0].matches:
        title_text_dict[match.tags['title']] = match.text
    basic_ranker_title.rank(example_docs, {})

    for doc in example_docs:
        for match in doc.matches[:1]:
            # Imperfect token decoding
            match_text = (
                match.text.lower().strip('#').replace(' - ', '-').replace(' +', '+')
            )
            assert match_text in title_text_dict[match.tags['title']].lower()


@pytest.mark.gpu
@pytest.mark.parametrize('example_docs', [2], indirect=['example_docs'])
def test_ranking_gpu(example_docs: DocumentArray):

    ranker = DPRReaderRanker(device='cuda')
    assert ranker.model.device.type == 'cuda'
    ranker.rank(example_docs, {})

    assert len(example_docs[0].matches) == 20

    # A quirk related to how HF chooses spans/overlapping
    assert len(example_docs[1].matches) == 19

    for match in example_docs[0].matches:
        assert 'relevance_score' in match.scores
        assert 'span_score' in match.scores


@pytest.mark.parametrize('example_docs', [2], indirect=['example_docs'])
@pytest.mark.parametrize('num_spans_per_match', [1, 2])
def test_num_spans_per_match(num_spans_per_match: int, example_docs: DocumentArray):
    ranker = DPRReaderRanker(num_spans_per_match=num_spans_per_match)
    ranker.rank(example_docs, {})

    for doc in example_docs:
        # A quirk related to how HF chooses spans/overlapping
        exp_len = 10 * num_spans_per_match
        assert len(doc.matches) in [exp_len, exp_len - 1]
        assert 'relevance_score' in doc.matches[0].scores
        assert 'span_score' in doc.matches[0].scores


@pytest.mark.parametrize('example_docs', [10], indirect=['example_docs'])
@pytest.mark.parametrize('batch_size', [1, 2, 4, 8])
def test_batch_size(
    basic_ranker: DPRReaderRanker, batch_size: int, example_docs: DocumentArray
):
    basic_ranker.rank(example_docs, parameters={'batch_size': batch_size})

    for doc in example_docs:
        # A quirk related to how HF chooses spans/overlapping
        assert len(doc.matches) in [20, 19]
        assert 'relevance_score' in doc.matches[0].scores
        assert 'span_score' in doc.matches[0].scores


@pytest.mark.parametrize('example_docs', [10], indirect=['example_docs'])
@pytest.mark.parametrize('access_paths', ['@r','@m','@m,c'])
def test_access_paths(
    basic_ranker: DPRReaderRanker,
    access_paths: List[str],
    example_docs: DocumentArray,
):
    # Set up document structure
    if access_paths == '@r':
        docs = example_docs
    elif access_paths == '@m':
        docs = DocumentArray([Document()])
        docs[0].matches.extend(example_docs)
    elif access_paths == '@m,c':
        docs = DocumentArray([Document()])
        docs[0].matches.extend(example_docs[:5])
        docs[0].chunks.extend(example_docs[5:])

    basic_ranker.rank(docs, parameters={'access_paths': access_paths})

    for doc in docs[access_paths]:
        # A quirk related to how HF chooses spans/overlapping
        assert len(doc.matches) in [20, 19]
        assert 'relevance_score' in doc.matches[0].scores
        assert 'span_score' in doc.matches[0].scores


@pytest.mark.parametrize('example_docs', [3], indirect=['example_docs'])
def test_quality_ranking(
    basic_ranker_title: DPRReaderRanker, example_docs: DocumentArray
):
    """A small test to see that the results make some sense."""
    basic_ranker_title.rank(example_docs, {})

    assert example_docs[0].matches[0].text in 'gay community'
    assert example_docs[1].matches[0].text == 'sparking change in corporate boardrooms'
    assert example_docs[2].matches[0].text == 'protein family of solute carriers'
