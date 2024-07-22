### Module Installation

`_> cd 1_data_quality` <br>
`_> poetry install`

### Manual Analysis

The data need to be analyzed at first to understand the nature of pre-processing that need to be done. The manual analysis report is kept in `/1_data_quality/analysis/faq_data_quality.ipynb`

Based upon the analysis, we understood that following steps need to be taken:

Following PREPROCESSING steps would have to be done.

- We can drop the column `parent_category` as huge portion of data is already missing
- `category_id` and `category` columns are similar and thus redundant. We can keep either of the two columns.
- Remove the column `que_ans` as it is redundant, since the information is a concatenated string of `question` and `answer` columns.
- Drop the rows with missing answers in the `answer` column.
- Look out for HTML tags in the columns [`question`, `answer`].
- Conversion of Text to lower case can be done for columns [`question`, `answer`]. But it is not suggested for a NLP use case, as words with upper case have a spatial significance and thus mean different as compared to small case words.
- Search for special characters and get the data cleaned from html tags for both columns [`question`,`answer`].

### Running the CLI command to clean the data

I developed a CLI utility to run pre-processing in general:

```
usage: data_cleaner_main.py [-h] [--input_path INPUT_PATH] [--delete_cols DELETE_COLS] [--delete_rows DELETE_ROWS]
                            [--lower_case_cols LOWER_CASE_COLS] [--clean_html_tags CLEAN_HTML_TAGS]
                            [--output_path OUTPUT_PATH]

Custom Clean dataset

optional arguments:
  -h, --help            show this help message and exit
  --input_path INPUT_PATH
                        path to input dataset
  --delete_cols DELETE_COLS
                        Comma separated column names to be deleted
  --delete_rows DELETE_ROWS
                        Comma separated columns name, which if empty, rows to be deleted
  --lower_case_cols LOWER_CASE_COLS
                        Comma separated column names to be converted to lower case
  --clean_html_tags CLEAN_HTML_TAGS
                        Comma separated column names with html tags to be removed
  --output_path OUTPUT_PATH
                        path to pre-processed dataset
```

Eg., command which was used to do cleanup in our case:

```
_> poetry run python data_quality_module/data_cleaner_main.py --input_path dataset/FAQs.csv --delete_cols parent_category,category_id,que_ans --delete_rows answer --lower_case_cols question,answer --clean_html_tags question,answer --output_path dataset/processed_FAQs.csv
```

The output of the data_quality suite is used for model-performance testing.
