"""The ModelOutput classes the ported models return.  Upstream they are
dataclasses over an OrderedDict; here each lists its fields in upstream's
order, and indexing, to_tuple() and iteration skip the fields left None, as
upstream's do."""


class ModelOutput(object):
    _fields = ()

    def __init__(self, *args, **kwargs):
        for f in self._fields:
            setattr(self, f, None)
        for f, v in zip(self._fields, args):
            setattr(self, f, v)
        for k, v in kwargs.items():
            if k not in self._fields:
                raise TypeError("%s has no field %s"
                                % (type(self).__name__, k))
            setattr(self, k, v)

    def keys(self):
        return [f for f in self._fields if getattr(self, f) is not None]

    def to_tuple(self):
        return tuple(getattr(self, f) for f in self.keys())

    def __getitem__(self, k):
        if isinstance(k, str):
            return getattr(self, k)
        return self.to_tuple()[k]

    def __setitem__(self, k, v):
        setattr(self, k, v)

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())


def _output(name, fields):
    return type(name, (ModelOutput,), {"_fields": tuple(fields)})


BaseModelOutput = _output("BaseModelOutput", [
    "last_hidden_state", "hidden_states", "attentions"])
BaseModelOutputWithPooling = _output("BaseModelOutputWithPooling", [
    "last_hidden_state", "pooler_output", "hidden_states", "attentions"])
BaseModelOutputWithPastAndCrossAttentions = _output(
    "BaseModelOutputWithPastAndCrossAttentions", [
        "last_hidden_state", "past_key_values", "hidden_states",
        "attentions", "cross_attentions"])
BaseModelOutputWithPoolingAndCrossAttentions = _output(
    "BaseModelOutputWithPoolingAndCrossAttentions", [
        "last_hidden_state", "pooler_output", "hidden_states",
        "past_key_values", "attentions", "cross_attentions"])
MaskedLMOutput = _output("MaskedLMOutput", [
    "loss", "logits", "hidden_states", "attentions"])
CausalLMOutputWithCrossAttentions = _output(
    "CausalLMOutputWithCrossAttentions", [
        "loss", "logits", "past_key_values", "hidden_states", "attentions",
        "cross_attentions"])
CausalLMOutputWithPast = _output("CausalLMOutputWithPast", [
    "loss", "logits", "past_key_values", "hidden_states", "attentions"])
SequenceClassifierOutput = _output("SequenceClassifierOutput", [
    "loss", "logits", "hidden_states", "attentions"])
