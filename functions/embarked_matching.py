# embarked_matching.py


def get_total_impact(img_response, df_mapping, weight=300):
    """
    To do
    """

    # - Process input - #

    # Get usable format
    json_output_fb = img_response[0]  # 0 for the first recipe

    # Extract main informations
    list_ingredients = json_output_fb['ingrs']
    list_instructions = json_output_fb['recipe']
    name = json_output_fb['title']

    # Create a single string instructions from its list
    string_instructions = " ".join(list_instructions)
    print(string_instructions)

    # - Get impact from ingredients - #

    # Filter mapping items according to ingredients from recipe
    df_recette = df_mapping[df_mapping.can_match.isin(list_ingredients)]

    # Compute relative impact
    total_pond = sum(df_recette.pond)

    df_recette['rel_weight'] = (df_recette['pond'] / total_pond) * (weight / 100)
    df_recette['rel_impact'] = df_recette['rel_weight'] * df_recette['est_impact']

    # - Get impact from cooking - #

    # To do, looking for "bake, boil, toast, etc." from cooking steps from recipe
    # To do, Looking for verbs with PoS tagging + manual selection ?
    total_cook = 0

    if "Toast" in string_instructions:
        total_cook += 300
    if "boil" in string_instructions:
        total_cook += 200

    string_cook = 'cooking impact: ' + str(total_cook) + 'g.'

    # - Get total impact from both parts - #

    total = round(sum(df_recette['rel_impact']) + total_cook)

    return df_recette, string_cook, total, name

#
