# embarked_matching.py

# New functions
import re
import pandas as pd
from operator import itemgetter
from fuzzywuzzy import fuzz

# stopwords manually imported from NLTK package to prevent install and downloads from this library
stop = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
        "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
        'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
        'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll",
        'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
        'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
        'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
        'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
        'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
        'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
        'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
        'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
        'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",
        'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
        "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
        'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

manual_remove = ['', 'raw'] # if titles need to be clean


# Function to apply same cleaning on titles as it is the case on the preprocess file
def cleaning_input(string, manual_remove_list=manual_remove):
    """
    Function to apply on the output of the facebook algorithm to apply the same cleaning executed on MIT titles
    """
    string = string.lower()
    string = re.sub(r'\([^)]*\)', '', string)
    string = string.replace('[^\w\s]', ' ')
    string = ' '.join([item for item in string.split() if item not in stop])
    string = ' '.join([item for item in string.split() if item not in manual_remove_list])

    return string


def info_from_name(ref_df, ref_dict, name):
    """
    Extract information from de pre-processed dataset (ingredients, weights, etc.) based on the title of the recipe
    """

    # get first id from name
    recipe_id = list(ref_dict.keys())[list(ref_dict.values()).index(name)]
    # filter df with id
    sub_df = ref_df[ref_df.id == recipe_id]

    # return name, url, weight_list, ingredient_list, impact_list, total_impact
    recipe_name = sub_df.title_raw.iloc[0]
    recipe_url = sub_df.url.iloc[0]
    recipe_impact = sum(sub_df.ingredient_impact)
    ingredient_weights = list(sub_df.weight_per_ingr)
    ingredient_names = list(sub_df.ingredient_raw)
    ingredient_impacts = list(sub_df.ingredient_impact)

    return recipe_name, recipe_url, recipe_impact, ingredient_weights, ingredient_names, ingredient_impacts


def get_results(img_response, ref_df, ref_dict, ref_list_titles):
    """
    Wrapping function to return recipes (with titles, ingredients, impacts, etc.) based on the title estimated by facebook algorithm
    """

    # - Process input - #

    # Get usable format
    json_output_fb = img_response[0]  # 0 for the first recipe

    # Extract main information
    name = json_output_fb['title']

    clean_name = cleaning_input(name)

    mit_names = [(x, fuzz.token_sort_ratio(clean_name, x)) for x in ref_list_titles]
    closest = max(mit_names, key=itemgetter(1))[0]

    return info_from_name(ref_df, ref_dict, closest)

#
