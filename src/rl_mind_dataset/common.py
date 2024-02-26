import copy
import sys
import time
import json
import math
import pandas as pd
import numpy as np

from logging import getLogger
from enum import Enum


def convert_config_dict(config_dict):
    r"""This function convert the str parameters to their original type."""
    check_set = (str, int, float, list, tuple, dict, bool, Enum)
    for key in config_dict:
        param = config_dict[key]
        if not isinstance(param, str):
            continue
        try:
            value = eval(
                param
            )  # convert str to int, float, list, tuple, dict, bool. use ',' to split integer values
            if value is not None and not isinstance(value, check_set):
                value = param
        except (NameError, SyntaxError, TypeError, ValueError):
            if isinstance(param, str):
                if param.lower() == "true":
                    value = True
                elif param.lower() == "false":
                    value = False
                else:
                    if "," in param:  # split by ',' if it is a string
                        value = []
                        for v in param.split(","):
                            if len(v) == 0:
                                continue
                            try:
                                v = eval(v)
                            except (NameError, SyntaxError, TypeError, ValueError):
                                v = v
                            value.append(v)
                    else:
                        value = param
            else:
                value = param
        config_dict[key] = value
    return config_dict


def load_cmd_line():
    """
    Load command line arguments
    :return: dict
    """
    cmd_config_dict = {}
    unrecognized_args = []
    if "ipykernel_launcher" not in sys.argv[0]:
        for arg in sys.argv[1:]:
            if not arg.startswith("--") or len(arg[2:].split("=")) != 2:
                unrecognized_args.append(arg)
                continue
            cmd_arg_name, cmd_arg_value = arg[2:].split("=")
            if (
                cmd_arg_name in cmd_config_dict
                and cmd_arg_value != cmd_config_dict[cmd_arg_name]
            ):
                raise SyntaxError(
                    "There are duplicate commend arg '%s' with different value." % arg
                )
            else:
                cmd_config_dict[cmd_arg_name] = cmd_arg_value
    if len(unrecognized_args) > 0:
        logger = getLogger()
        logger.warning(
            f"Unrecognized command line arguments(correct is '--key=value'): {' '.join(unrecognized_args)}"
        )
    cmd_config_dict = convert_config_dict(cmd_config_dict)
    cmd_config_dict["cmd_args"] = copy.deepcopy(cmd_config_dict)
    return cmd_config_dict


def check_empty(value):
    if value in [None, "", "NA", "missing", "nan"] or (
        isinstance(value, float) and math.isnan(value)
    ):
        return True
    return False


def to_format_excel(saved_df, saved_path, column_widths=None, **kwargs):
    """Save the dataframe to an Excel file with the given column widths in a beautiful format.

    :param saved_df: dataframe to be saved
    :param saved_path: path to save the dataframe
    :param column_widths: column widths in the format of {"A:A": 15, "B:C": 80, "D:D": 10, "E:XFD": 80}
    """
    sheet_name = kwargs.get("sheet_name", "overall")
    if column_widths is None:
        column_widths = {
            "A:A": 15,
            "B:C": 80,
            "D:D": 10,
            "E:XFD": 80,  # XFD is the last column in Excel
        }
    writer = pd.ExcelWriter(saved_path, engine="xlsxwriter")
    saved_df.to_excel(writer, sheet_name=sheet_name, index=False)
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]

    # Format for cell: wrap text, align top-left
    cell_format = workbook.add_format(
        {
            "text_wrap": True,
            "valign": "top",
            "align": "left",
        }
    )
    for col_num, value in column_widths.items():
        # Apply the format to the cells with content
        worksheet.set_column(col_num, value, cell_format)
    worksheet.freeze_panes(1, 1)

    # Define a format for the cell content with text wrapping, align to the top and left
    cell_format = workbook.add_format()
    cell_format.set_text_wrap()
    cell_format.set_align("top")
    cell_format.set_align("left")
    # Define and apply a header format
    header_format = workbook.add_format(
        {
            "bold": True,
            "text_wrap": True,
            "valign": "top",
            "align": "left",
            "fg_color": "#D7E4BC",
            "border": 1,
        }
    )

    for col_num, value in enumerate(saved_df.columns.values):
        worksheet.write(0, col_num, value, header_format)
    writer.close()


def chat(prompt, model="gpt-3.5-turbo", max_try=5, **model_params):
    import openai

    """Set up the Openai API key using openai.api_key = api_key"""
    try:
        if "gpt" in model.lower():
            chat_completion = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **model_params,
            )
            content = chat_completion.choices[0].message["content"]
        else:
            import google.generativeai as palm

            content = palm.chat(prompt=[prompt], model=model).last
        if content is None:
            if max_try > 0:
                time.sleep(20)
                content = chat(prompt, model, max_try - 1)
            else:
                content = ""
        return content
    except:
        if max_try > 0:
            time.sleep(20)
            return chat(prompt, model, max_try - 1)
        else:
            raise Exception("Max try exceeded")


def get_history_candidate_prompt(news_df, behavior):
    history_news = news_df[news_df["news_id"].isin(behavior["history"].split())]
    cand_news_index = [i.split("-")[0] for i in behavior["impressions"].split()]
    cand_label = [i.split("-")[1] for i in behavior["impressions"].split()]
    # get candidate news from news_df and save them to a list with the same order as cand_news_index
    candidate_news = [
        news_df[news_df["news_id"] == i].iloc[0]["title"] for i in cand_news_index
    ]
    history_prompt = "\n".join(
        [f"H{i + 1}: {news}" for i, news in enumerate(history_news["title"].values)]
    )
    candidate_prompt = "\n".join(
        [f"C{i + 1}: {news}" for i, news in enumerate(candidate_news)]
    )
    return (
        history_prompt,
        candidate_prompt,
        ",".join([f"C{i + 1}" for i, l in enumerate(cand_label) if int(l)]),
    )


def load_api_key(json_path="openai_key.json"):
    return json.load(open(json_path))["api_key"]


def evaluate_one(labels, predictions):
    """
    Compute nDCG@k and MRR for a single row in recommendation tasks.

    Parameters:
    - labels: list of true labels, like ['C1', 'C3', 'C5']
    - predictions: list of predicted labels in order, like ['C2', 'C5', 'C1', 'C3', 'C4']

    Returns:
    - ndcg5_at_k: nDCG@5 for the row
    - ndcg10_at_k: nDCG@10 for the row
    - mrr: MRR for the row
    """
    # Compute relevance scores for nDCG@k
    relevance = [1 if item in labels else 0 for item in predictions]

    def dcg_at_k(r, k):
        """Compute DCG@k for a single sample.
        Args:
        r: list of relevance scores in the order they were ranked
        k: number of results to consider
        Returns:
        DCG@k
        """
        r = np.asfarray(r)[:k]
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))

    def ndcg_at_k(r, k):
        """Compute nDCG@k for a single sample.
        Args:
        r: list of relevance scores in the order they were ranked
        k: number of results to consider
        Returns:
        nDCG@k
        """
        dcg_max = dcg_at_k(sorted(r, reverse=True), k)
        if not dcg_max:
            return 0.0
        return dcg_at_k(r, k) / dcg_max

    # Calculate nDCG@k values
    ndcg5_at_k = ndcg_at_k(relevance, 5)
    ndcg10_at_k = ndcg_at_k(relevance, 10)

    # Helper function to compute MRR for a single row
    def compute_mrr(labels, predictions):
        for index, item in enumerate(predictions, 1):
            if item in labels:
                return 1.0 / index
        return 0

    # Calculate MRR
    mrr = compute_mrr(labels, predictions)
    return ndcg5_at_k, ndcg10_at_k, mrr
