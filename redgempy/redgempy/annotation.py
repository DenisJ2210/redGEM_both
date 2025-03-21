def get_seed_ids_for_rxn(rxn):
    """Return the set of unique SEED IDs for metabolites in rxn."""
    seed_ids = set()
    for met in rxn.metabolites.keys():
        # Access the metabolite's annotation for seed.compound.
        # Adjust this according to your model's annotation structure.
        seed_list = met.annotation.get("seed.compound", [])
        if seed_list:
            seed_ids.add(seed_list[0])
    return seed_ids