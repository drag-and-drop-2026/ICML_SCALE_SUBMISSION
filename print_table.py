"""Print a LaTeX table of accuracy@1 from aggregated_*.json files."""

import json
import sys

ROW_FMT = "  {model} & {total} & {th} & {cs} & {sm} & {er} & {er10} & {er15} \\\\"

HEADER = r"""\begin{{table*}}[!t]
  \centering
  \resizebox{{\linewidth}}{{!}}{{%
  \begin{{tabular}}{{|l|c|c|c|c|c||c|c|}}
  \hline
   & Total & Text Highlighting & Cell Selection & Slider Manipulation & Element Resizing & ER acc@10\% & ER acc@15\% \\ \hline"""

FOOTER = r"""  \hline
  \end{{tabular}}%
  }}
  \caption{{Success rate on the test set.}}
  \label{{tab:results-pass10}}
\end{{table*}}"""


def fmt(v):
    return "{:.1f}\\%".format(v * 100)


def main(paths):
    print(HEADER.format())
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        by_domain = data["by_domain"]
        er = by_domain["slide_resize"]["accuracy_by_scale"]
        print(
            ROW_FMT.format(
                model=data["model_id"],
                total=fmt(data["accuracy_by_scale"]["accuracy@1x"]),
                th=fmt(by_domain["text_highlight"]["accuracy_by_scale"]["accuracy@1x"]),
                cs=fmt(by_domain["sheet"]["accuracy_by_scale"]["accuracy@1x"]),
                sm=fmt(by_domain["slider"]["accuracy_by_scale"]["accuracy@1x"]),
                er=fmt(er["accuracy@1x"]),
                er10=fmt(er["accuracy@2x"]),
                er15=fmt(er["accuracy@3x"]),
            )
        )
    print(FOOTER.format())


if __name__ == "__main__":
    main(sys.argv[1:])
