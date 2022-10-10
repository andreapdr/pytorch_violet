def convert_to_string(tokenizer, txt):
    sent = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(txt)).replace("[PAD]", "").replace("[CLS]", "").replace("[SEP]", "")
    return sent.lstrip(" ").rstrip(" ")


def task2str(task):
    _task2str = {
        "ir": "inverse_action",
        "ar": "action_recognition",
        # "ps": "prestate_foiling",
        # "pr": "postate_foiling"
        }

    return _task2str[task.lower()]



