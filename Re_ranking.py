from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          AutoModelForSeq2SeqLM,
                          PreTrainedModel,
                          PreTrainedTokenizer,
                          T5ForConditionalGeneration)
import torch
from typing import List
from copy import deepcopy
from base import Reranker, Query, Text

  
# tokenizer = AutoTokenizer.from_pretrained("castorini/monot5-large-msmarco")

# model = AutoModel.from_pretrained("castorini/monot5-large-msmarco")

class MonoBERT(Reranker):
    def __init__(self,
                 model: PreTrainedModel = None,
                 tokenizer: PreTrainedTokenizer = None,
                 use_amp = False):
        self.model = model or self.get_model()
        self.tokenizer = tokenizer or self.get_tokenizer()
        self.device = next(self.model.parameters(), None).device
        self.use_amp = use_amp

    @staticmethod
    def get_model(pretrained_model_name_or_path: str = 'castorini/monobert-large-msmarco',
                  *args, device: str = None, **kwargs) -> AutoModelForSequenceClassification:
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device(device)
        return AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path,
                                                                  *args, **kwargs).to(device).eval()

    @staticmethod
    def get_tokenizer(pretrained_model_name_or_path: str = 'bert-large-uncased',
                      *args, **kwargs) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=False, *args, **kwargs)

    @torch.no_grad()
    def rescore(self, query: Query, texts: List[Text]) -> List[Text]:
        texts = deepcopy(texts)
        for text in texts:
            ret = self.tokenizer.encode_plus(query.text,
                                             text.text,
                                             max_length=512,
                                             truncation=True,
                                             return_token_type_ids=True,
                                             return_tensors='pt')
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                input_ids = ret['input_ids'].to(self.device)
                tt_ids = ret['token_type_ids'].to(self.device)
                output, = self.model(input_ids, token_type_ids=tt_ids, return_dict=False)
                if output.size(1) > 1:
                    text.score = torch.nn.functional.log_softmax(
                        output, 1)[0, -1].item()
                else:
                    text.score = output.item()

        return texts