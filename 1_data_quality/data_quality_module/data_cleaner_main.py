from data_preprocessing_pipeline import DataPreprocessingPipeline
from preprocessors.delete_cols_preprocessor import DeleteColsPreprocessor
from preprocessors.delete_rows_preprocessor import DeleteRowsPreProcessor
from preprocessors.lower_case_preprocessor import LowerCasePreProcessor
from preprocessors.clean_html_tags_preprocessor import RemoveHTMLTagsPreProcessor
import argparse
import pandas as pd
import sys

def preprocess_data(args):
    df = pd.read_csv(args.input_path)
    
    # Get the list of preprocessors
    preprocessors = []
    if args.delete_cols:
        delete_cols = [col.strip() for col in args.delete_cols.split(',')]
        preprocessors.append(DeleteColsPreprocessor(delete_cols))
    if args.delete_rows:
        delete_rows_for_cols = [col.strip() for col in args.delete_rows.split(',')]
        preprocessors.append(DeleteRowsPreProcessor(delete_rows_for_cols))
    if args.lower_case_cols:
        lower_case_cols = [col.strip() for col in args.lower_case_cols.split(',')]
        preprocessors.append(LowerCasePreProcessor(lower_case_cols))
    if args.clean_html_tags:
        clean_html_tags = [col.strip() for col in args.clean_html_tags.split(',')]
        preprocessors.append(RemoveHTMLTagsPreProcessor(clean_html_tags))

    if preprocessors:
        pipeline = DataPreprocessingPipeline(preprocessors)
        processed_df = pipeline.process(df)
        # Saving the processed dataframe to the output path
        processed_df.to_csv(args.output_path, index=False)
    
    else:
        print("No preprocessors specified. Skipping preprocessing.")
        # Saving the original dataframe to the output path
        df.to_csv(args.output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom Clean dataset")
    parser.add_argument("--input_path", type=str, help="path to input dataset")
    parser.add_argument("--delete_cols", type=str, help="Comma separated column names to be deleted")
    parser.add_argument("--delete_rows", type=str, help="Comma separated columns name, which if empty, rows to be deleted")
    parser.add_argument("--lower_case_cols", type=str, help="Comma separated column names to be converted to lower case")
    parser.add_argument("--clean_html_tags", type=str, help="Comma separated column names with html tags to be removed")
    parser.add_argument("--output_path", type=str, help="path to pre-processed dataset")
    
    args = parser.parse_args()

    if not len(sys.argv) > 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    preprocess_data(args)