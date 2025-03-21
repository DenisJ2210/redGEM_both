def remove_duplicates(input_list):
    # This function check each possible path from one metabolite to another
    # If in the path there are duplicates, the path is removed (since this path is not really length L but L-1)
    # Convert each sublist into a tuple for comparison
    input_tuples = [tuple(item) for item in input_list]

    # Initialize a set for tracking duplicates and a list for unique items
    unique_items = []
    removed_indices = []

    for index, item in enumerate(input_list):
        if len(set(item)) != len(item):
            removed_indices.append(index)
        else:
            unique_items.append(item)

    # If there are duplicates, return the unique items and removed indices
    if removed_indices:
        return unique_items, removed_indices
    else:
        return input_list, []