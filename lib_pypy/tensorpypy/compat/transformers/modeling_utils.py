import re

import torch.nn as nn

from transformers.loss import loss_utils as _loss

from transformers.integrations.sdpa_attention import sdpa_attention_forward


class AttentionInterface(object):
    """upstream's registry: the implementation named in the config, or the
    model's own eager function when the config says eager."""
    _global_mapping = {"sdpa": sdpa_attention_forward}

    def get_interface(self, attn_implementation, default):
        if attn_implementation == "eager" or attn_implementation is None:
            return default
        f = self._global_mapping.get(attn_implementation)
        if f is None:
            raise NotImplementedError("attention implementation %r"
                                      % attn_implementation)
        return f


ALL_ATTENTION_FUNCTIONS = AttentionInterface()


def _not_ported(name):
    def loss(*args, **kwargs):
        raise NotImplementedError("%s loss" % name)
    return loss


# upstream's LOSS_MAPPING (transformers/loss/loss_utils.py): every key, in
# upstream's order, because the model class name is matched against them in
# that order; the text-model losses are the ported functions
_KEYS = ["ForSemanticSegmentation", "ForCausalLM", "ForMaskedLM",
         "ForQuestionAnswering", "ForSequenceClassification",
         "ForImageClassification", "ForVideoClassification",
         "ForAudioClassification", "ForTokenClassification", "ForSegmentation",
         "ForObjectDetection", "ForConditionalGeneration",
         "DeformableDetrForObjectDetection",
         "ConditionalDetrForObjectDetection", "DabDetrForObjectDetection",
         "GroundingDinoForObjectDetection",
         "MMGroundingDinoForObjectDetection",
         "ConditionalDetrForSegmentation", "RTDetrForObjectDetection",
         "RTDetrV2ForObjectDetection", "DFineForObjectDetection",
         "Deimv2ForObjectDetection", "CsmForConditionalGeneration",
         "LwDetrForObjectDetection", "ParakeetForRNNT", "ParakeetForTDT",
         "RfDetrForObjectDetection", "RfDetrForInstanceSegmentation"]
LOSS_MAPPING = dict((k, _not_ported(k)) for k in _KEYS)
LOSS_MAPPING.update({
    "ForCausalLM": _loss.ForCausalLMLoss,
    "ForMaskedLM": _loss.ForMaskedLMLoss,
    "ForQuestionAnswering": _loss.ForQuestionAnsweringLoss,
    "ForSequenceClassification": _loss.ForSequenceClassificationLoss,
    "ForImageClassification": _loss.ForSequenceClassificationLoss,
    "ForVideoClassification": _loss.ForSequenceClassificationLoss,
    "ForAudioClassification": _loss.ForSequenceClassificationLoss,
    "ForTokenClassification": _loss.ForTokenClassification,
    "ForConditionalGeneration": _loss.ForCausalLMLoss,
    "CsmForConditionalGeneration": _loss.ForCausalLMLoss,
})


class PreTrainedModel(nn.Module):
    """The model base, for a model whose tensors arrive by
    load_state_dict: post_init's weight initialisation and tying have
    nothing to do, the tensors being the checkpoint's."""
    config_class = None
    base_model_prefix = ""

    def __init__(self, config, *inputs, **kwargs):
        self.config = config
        # as upstream's __init__ picks it
        loss_type = self.__class__.__name__
        if loss_type not in LOSS_MAPPING:
            found = re.findall("(%s)" % "|".join(_KEYS),
                               self.__class__.__name__)
            loss_type = found[0] if found else None
        self.loss_type = loss_type

    @property
    def loss_function(self):
        if "_loss_function" in self.__dict__:
            return self._loss_function
        loss_type = getattr(self, "loss_type", None)
        if loss_type is None or loss_type not in LOSS_MAPPING:
            loss_type = "ForCausalLM"
        return LOSS_MAPPING[loss_type]

    def post_init(self):
        pass

    def warn_if_padding_and_no_attention_mask(self, input_ids,
                                              attention_mask):
        """A warning upstream; no effect on what is computed."""

    @property
    def dtype(self):
        return "float32"
