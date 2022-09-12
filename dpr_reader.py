from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from docarray import Document, DocumentArray
from jina import Executor, requests
from transformers import DPRReader, DPRReaderTokenizerFast
import warnings

def _logistic_fn(x: np.ndarray) -> List[float]:
    """Compute the logistic function"""
    return (1 / (1 + np.exp(-x))).tolist()


class DPRReaderRanker(Executor):
    """
    This executor first extracts answers (answers spans) from all the matches,
    ranks them according to their relevance score, and then replaces the original
    matches with these extracted answers.

    This executor uses the DPR Reader model to re-rank documents based on
    cross-attention between the question (main document text) and the answer
    passages (text of the matches + their titles, if specified).
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = 'facebook/dpr-reader-single-nq-base',
        base_tokenizer_model: Optional[str] = None,
        title_tag_key: Optional[str] = None,
        num_spans_per_match: int = 2,
        max_length: Optional[int] = None,
        access_paths: str = '@r',
        traversal_paths: Optional[str] = None,
        batch_size: int = 32,
        device: str = 'cpu',
        *args,
        **kwargs,
    ):
        """
        :param pretrained_model_name_or_path: Can be either:
            - the model id of a pretrained model hosted inside a model repo
              on huggingface.co.
            - A path to a directory containing model weights, saved using
              the transformers model's `save_pretrained()` method
        :param base_tokenizer_model: Base tokenizer model. The possible values are
            the same as for the `pretrained_model_name_or_path` parameters. If not
            provided, the `pretrained_model_name_or_path` parameter value will be used
        :param title_tag_key: The key of the tag that contains document title in the
            match documents. Specify it if you want the text of the matches to be combined
            with their titles (to mirror the method used in training of the original model)
        :param num_spans_per_match: Number of spans to extract per match
        :param max_length: Max length argument for the tokenizer
        :param access_paths: Default traversal paths for processing documents,
            used if the traversal path is not passed as a parameter with the request.
        :param traversal_paths: please use access_paths
        :param batch_size: Default batch size for processing documents, used if the
            batch size is not passed as a parameter with the request.
        :param device: The device (cpu or gpu) that the model should be on.
        """
        super().__init__(*args, **kwargs)
        self.title_tag_key = title_tag_key
        self.device = device
        self.max_length = max_length
        self.num_spans_per_match = num_spans_per_match

        if not base_tokenizer_model:
            base_tokenizer_model = pretrained_model_name_or_path

        self.tokenizer = DPRReaderTokenizerFast.from_pretrained(base_tokenizer_model)
        self.model = DPRReader.from_pretrained(pretrained_model_name_or_path)

        self.model = self.model.to(torch.device(self.device)).eval()

        if traversal_paths is not None:
            self.access_paths = traversal_paths
            warnings.warn("'traversal_paths' will be deprecated in the future, please use 'access_paths'.",
                          DeprecationWarning,
                          stacklevel=2)
        else:
            self.access_paths = access_paths
        self.batch_size = batch_size

    @requests
    def rank(
        self, docs: Optional[DocumentArray] = None, parameters: Dict = {}, **kwargs
    ) -> DocumentArray:
        """
        Extracts answers from existing matches, (re)ranks them, and replaces the current
        matches with extracted answers.

        For each match `num_spans_per_match` of answers will be extracted, which
        means that the new matches of the document will have a length of previous
        number of matches times `num_spans_per_match`.

        The new matches will be have a score called `relevance_score` saved under
        their scores. They will also have a tag `span_score`, which refers to their
        span score, which is used to rank answers that come from the same match.

        If you specified `title_tag_key` at initialization, the tag `title` will
        also be added to the new matches, and will equal the title of the match from
        which they were extracted.

        :param docs: Documents whose matches to re-rank (specifically, the matches of
            the documents on the traversal paths will be re-ranked). The document's
            `text` attribute is taken as the question, and the `text` attribute
            of the matches as the context. If you specified `title_tag_key` at
            initialization, the matches must also have a title (under this tag).
        :param parameters: dictionary to define the `access_paths` and the
            `batch_size`. For example
            `parameters={'access_paths': 'r', 'batch_size': 10}`
        """
        access_paths = parameters.get('traversal_paths', None)
        if access_paths is not None:
            import warnings
            warnings.warn(
                f'`traversal_paths` is renamed to `access_paths` with the same usage, please use the latter instead. '
                f'`traversal_paths` will be removed soon.',
                DeprecationWarning,
            )
            parameters['access_paths'] = access_paths

        batch_size = parameters.get('batch_size', self.batch_size)

        for doc in docs[parameters.get('access_paths', self.access_paths)]:
            if not doc.text:
                continue
            new_matches = []
            match_batches_generator = DocumentArray(filter(
                lambda x: bool(x.text),
                doc.matches
            )).batch(batch_size=batch_size)
            for matches in match_batches_generator:
                question, titles = self._prepare_inputs(doc.text, matches)
                with torch.inference_mode():
                    new_matches += self._get_new_matches(question, matches, titles)

            # Make sure answers are sorted by relevance scores
            new_matches.sort(
                key=lambda x: (
                    x.scores['relevance_score'].value,
                    x.scores['span_score'].value,
                ),
                reverse=True,
            )

            # Replace previous matches with actual answers
            doc.matches = new_matches

    def _prepare_inputs(
        self, question: str, matches: DocumentArray
    ) -> Tuple[str, List[str]]:

        titles = None
        if self.title_tag_key:
            titles = matches[:, f'tags__{self.title_tag_key}']

            if len(titles) != len(matches) or None in titles:
                raise ValueError(
                    f'All matches are required to have the {self.title_tag_key}'
                    ' tag, but found some matches without it.'
                )

        return question, titles

    def _get_new_matches(
        self, question: str, matches: DocumentArray, titles: Optional[List[str]]
    ) -> List[Document]:
        texts = matches[:, 'text']
        encoded_inputs = self.tokenizer(
            questions=[question] * len(texts),
            titles=titles,
            texts=texts,
            padding='longest',
            return_tensors='pt',
        ).to(self.device)
        outputs = self.model(**encoded_inputs)

        # For each context, extract num_spans_per_match best spans
        best_spans = self.tokenizer.decode_best_spans(
            encoded_inputs,
            outputs,
            num_spans=self.num_spans_per_match * len(texts),
            num_spans_per_passage=self.num_spans_per_match,
        )

        new_matches = []
        for idx, span in enumerate(best_spans):
            new_match = Document(text=span.text)
            new_match.tags.update(matches[span.doc_id].tags)
            new_match.scores['relevance_score'].value = _logistic_fn(
                span.relevance_score.cpu()
            )
            new_match.scores['span_score'].value = _logistic_fn(span.span_score.cpu())
            if titles:
                new_match.tags['title'] = titles[span.doc_id]
            new_matches.append(new_match)

        return new_matches
