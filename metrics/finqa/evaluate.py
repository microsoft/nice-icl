import pandas as pd
from typing import Union, Any
from math import isclose
from sympy.solvers import solve
from sympy import Symbol, Eq
from sympy import simplify
import numpy as np
import json
import math
import re
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='text', help='path to the file to be evaluated')
parser.add_argument('--show', action='store_false')
parser.add_argument('--tolerance', type=float, default=1e-3)
args = parser.parse_args()

def extract_after_substring(input_string, substring):
    index = input_string.lower().rfind(substring.lower())
    if index != -1:
        return input_string[index + len(substring):]
    else:
        raise ValueError(f'Substring "{substring}" not found.')

def extract_largest_numerical_or_boolean_substring(input_string):
    
    input_string = str(input_string)
    input_string = input_string.replace('$', '').replace('%', '')

    if input_string.find('=') >= 0:
        input_string = input_string.split('=')[-1]

    # Remove space between minus sign and numerical value
    input_string = input_string.replace('- ', '-')

    numerical_substrings = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?(?:\s*[/]\s*\d+(?:,\d+)*(?:\.\d+)?)?|(?:\d+:\d+)', input_string)

    if len(numerical_substrings) == 0:
      if "yes" in input_string.lower() or "true" in input_string.lower():
        return "Yes"
      elif "no" in input_string.lower() or "false" in input_string.lower():
        return "No"
      else:
        return ""

    if ':' in input_string:
        ratio_parts = numerical_substrings
        if len(ratio_parts) == 1:
            return ratio_parts[0]

        first_value = float(ratio_parts[0])
        second_value = float(ratio_parts[1])

        try:
          value_to_return = first_value / second_value
          return value_to_return
        except Exception as e:
          return ""

    largest_numerical_substring = max(numerical_substrings, key=len)

    try:
        if '/' in largest_numerical_substring:
            fraction_parts = largest_numerical_substring.split('/')
            numerator = fraction_parts[0].replace(',', '').replace(' ', '')
            denominator = fraction_parts[1].replace(',', '').replace(' ', '')
            numerator_value = float(numerator) if '.' in numerator else int(numerator)
            denominator_value = float(denominator) if '.' in denominator else int(denominator)
            fraction = numerator_value / denominator_value
            return float(fraction)
        else:
            numerical_value = float(largest_numerical_substring.replace(',', '')) if '.' in largest_numerical_substring else int(largest_numerical_substring.replace(',', ''))
            return numerical_value
        
    except ValueError as e:
        return ""

    return ""



def apply_extraction(row):
    try:
        output = str(row['prediction']).lower()

        if "the final answer (in millions) is" in output:
            return extract_after_substring(output, "the final answer (in millions) is")
        elif "the final answer (in thousands) is" in output:
            return extract_after_substring(output, "the final answer (in thousands) is")
        elif "the final answer is" in output:
            return extract_after_substring(output, "the final answer is")
        elif "the answer is" in output:
            return extract_after_substring(output, "the answer is")
        
        else:
          # Add handling for cases where prefixes are not present
            return str(extract_largest_numerical_or_boolean_substring(output))

    except Exception as e:
        error_set_answer_extraction.add(row.name)
        return ""


def get_precision(gt_ans: float) -> int:
    precision = 8  # Set a higher default precision
    if '.' in str(gt_ans):
        precision = len(str(gt_ans).split('.')[-1])
    return precision


def finqa_equal(prediction: Union[bool, float, str],
                reference: Union[float, str],
                include_percentage: bool = True,
                is_close: bool = True, include_million: bool = True) -> bool:

    if prediction is "":
        return False

    
    if reference.lower() == "yes" or reference.lower() == "no" :
        return str(prediction).strip().lower() == reference.strip().lower()
    else:
        try:
            prediction = float(prediction)
            reference = float(reference)

            if include_million and include_percentage:
                gt_result = [reference / 1000000, reference / 100, reference, reference * 1000000, reference * 100]

            else:
                gt_result = [reference]

            for item in gt_result:

                if is_close and isclose(item, prediction, rel_tol=args.tolerance):
                  return True
                precision = min(get_precision(prediction), get_precision(item))
                if round(prediction, precision) == round(item, precision):
                    return True

        except Exception as e:
            print("Exception:", e)


    return False



error_set_answer_extraction=set()
error_set_answer_extraction = set()

df = pd.read_csv(args.path)


df['Extracted answer string'] = df.apply(apply_extraction, axis=1)


for index, row in df.iterrows():
    
    extracted_value = extract_largest_numerical_or_boolean_substring(row['Extracted answer string'])
    answer_value = extract_largest_numerical_or_boolean_substring(row['ground_truth'])
    df.at[index, 'prediction'] = str(extracted_value)
    df.at[index, 'answers'] = str(answer_value)
    
df['correct'] = df.apply(lambda row: finqa_equal(row['prediction'], row['answers']), axis=1)  

df.to_csv(args.path)

acc = df['correct'].mean()

if  args.show:
    print(f"Accuracy:", acc)
    

