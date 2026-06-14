SYSTEM_PROMPT = (
    "Complete Latin CoNLL-U. Copy every input line exactly except replace "
    "HEAD and DEPREL. Return only valid 10-column CoNLL-U."
)


def wrap_user_conllu(conllu):
    return f"<INPUT_CONLLU>\n{conllu.strip()}\n</INPUT_CONLLU>"


def load_prompt(prompt=None, prompt_file=None):
    if prompt_file:
        with open(prompt_file, encoding="utf-8") as handle:
            return handle.read().strip()
    if prompt:
        return prompt
    return SYSTEM_PROMPT
