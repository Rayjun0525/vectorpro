"""Local Gemma enum transport: plain letter choices instead of tool-call JSON."""
import json
from time import monotonic
from experiments.verified_acquisition_gemma import GemmaModel


class GoalChoiceModel(GemmaModel):
    def __init__(self, model_path, compact=False, words=False):
        super().__init__(model_path)
        self.compact = compact
        self.words = words
    def complete(self, messages, tools):
        function = tools[0]["function"]
        properties = function["parameters"]["properties"]
        if len(tools) != 1 or len(properties) != 1:
            return super().complete(messages, tools)
        field, schema = next(iter(properties.items()))
        choices = schema.get("enum")
        if not choices or len(choices) > 26:
            return super().complete(messages, tools)
        labels = {chr(65 + i): value for i, value in enumerate(choices)}
        if self.words:
            words = {"unchanged": "KEEP", "absent": "REMOVE", "present": "EXISTS", "unsupported": "UNSUPPORTED", "unclear": "UNCLEAR"}
            labels = {(words.get(value) or "ORIGINAL_" + value[8:].upper()): value for value in choices}
        reverse = {v: k for k, v in labels.items()}
        options = "\n".join(f"{label}: {value}" for label, value in labels.items())
        adapted = []
        for message in messages:
            if message["role"] == "system":
                role, content = "user", message["content"] + "\nAnswer with ONE letter only.\nChoices:\n" + options
            elif message["role"] == "assistant":
                value = json.loads(message["tool_calls"][0]["function"]["arguments"])[field]
                role, content = "assistant", reverse[value]
            else:
                role, content = "user", message["content"]
            if adapted and adapted[-1]["role"] == role:
                adapted[-1]["content"] += "\n\n" + content
            else:
                adapted.append({"role": role, "content": content})
        if self.compact and function["name"] == "describe_parameter":
            parameter = json.loads(messages[0]["content"].split("Parameter to describe: ")[1])
            meanings = {"unchanged": "Keep this file exactly unchanged", "absent": "Remove this file; it must not exist",
                "present": "Require this file to exist, without specifying its bytes", "unsupported": "The request requires encryption, byte transformation, computation, process or network effects",
                "unclear": "The request does not specify what should happen"}
            readable = "\n".join(f"{label}. " + ("This file contains the original bytes of " + value[8:] if value.startswith("initial:") else meanings[value]) for label, value in labels.items())
            adapted = [{"role": "user", "content":
                "Request: " + messages[-1]["content"] + "\nParameter: " + json.dumps(parameter) +
                "\nWhat final condition does the request require for this parameter? Choose ONE option label.\n" + readable + "\nAnswer:"}]
        formatted = self.formatter(messages=adapted)
        tokens = self.llm.tokenize(formatted.prompt.encode(), add_bos=False, special=True)
        grammar = self.llama_cpp.LlamaGrammar.from_string('root ::= ' + ' | '.join(json.dumps(label) for label in labels), verbose=False)
        start = monotonic()
        output = self.llm.create_completion(tokens, max_tokens=32 if self.words else 4, temperature=0, seed=0,
            stop=["<end_of_turn>", "<eos>", "\n"], grammar=grammar)
        answer = output["choices"][0]["text"].strip()
        self.raw.append({"messages": messages, "tools": tools, "adapted_messages": adapted,
            "choice_map": labels, "raw_output": answer, "seconds": monotonic() - start, "usage": output.get("usage")})
        if answer not in labels:
            raise ValueError("model returned an unknown choice")
        return {"role": "assistant", "tool_calls": [{"id": "choice", "type": "function",
            "function": {"name": function["name"], "arguments": json.dumps({field: labels[answer]})}}]}
