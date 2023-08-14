"""
QuAC: Question Answering in Context
https://arxiv.org/abs/1808.07036

Question Answering in Context (QuAC) is a dataset for modeling, understanding, and
participating in information seeking dialog. Data instances consist of an interactive
dialog between two crowd workers: (1) a student who poses a sequence of freeform
questions to learn as much as possible about a hidden Wikipedia text, and (2)
a teacher who answers the questions by providing short excerpts (spans) from the text.

Homepage: https://quac.ai/
"""
import inspect

import efficiency_benchmark.dependencies.lm_eval.datasets.quac.quac
from efficiency_benchmark.dependencies.lm_eval.base import Task

_CITATION = """
@article{choi2018quac,
    title={Quac: Question answering in context},
    author={Choi, Eunsol and He, He and Iyyer, Mohit and Yatskar, Mark and Yih, Wen-tau and Choi, Yejin and Liang, Percy and Zettlemoyer, Luke},
    journal={arXiv preprint arXiv:1808.07036},
    year={2018}
}
"""


class QuAC(Task):
    VERSION = 0
    DATASET_PATH = inspect.getfile(efficiency_benchmark.dependencies.lm_eval.datasets.quac.quac)
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        raise NotImplementedError("QuAC has no test docs.")

    def _process_doc(self, doc):
        doc["title"] = doc["title"] + " - " + doc["section_title"]
        return doc

    def doc_to_text(self, doc):
        return (
            "TITLE: "
            + doc["title"]
            + "\n"
            + "PARAGRAPH: "
            + doc["paragraph"]
            + "\n\n"
            + "Q: "
            + doc["question"]
            + "\n\n"
            + "A: "
        )

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["paragraph"]

    def doc_to_target(self, doc):
        return doc["answer"]

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        # TODO: implement evaluation.
        raise NotImplementedError("Evaluation not implemented")

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        # TODO: implement evaluation.
        raise NotImplementedError("Evaluation not implemented")

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        # TODO: implement evaluation.
        raise NotImplementedError("Evaluation not implemented")

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        # TODO: implement evaluation.
        raise NotImplementedError("Evaluation not implemented")
